# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speech-to-Text for Hamsa.

The endpoint is request/response rather than streaming: one POST carries one
utterance and returns its transcript. Two things matter for latency:

* **Send 16 kHz mono.** LiveKit hands the plugin room audio, normally 48 kHz.
  Resampling first cuts the upload to a third and the endpoint wants 16 kHz
  anyway.
* **Reuse the connection.** With a warm keep-alive socket a turn costs roughly
  0.25-0.45 s; paying a fresh TLS handshake each time costs several times that.
  Both this class and :class:`~livekithamsa.eos.HamsaEOS` share a single
  ``aiohttp`` session for the life of the process.

For turn detection, see :class:`~livekithamsa.eos.HamsaEOS`, which uses the
endpoint's own end-of-speech model instead of a fixed silence timer.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass, replace

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._audio import STT_SAMPLE_RATE, to_wav
from ._ws import WS_URL, HamsaWSPool
from .log import logger

HAMSA_STT_BASE_URL = "https://api.tryhamsa.com/v1/realtime/stt"

# What the endpoint returns in place of a transcript when `isEosEnabled` is on
# and its end-of-speech model judges that the speaker has not finished. It is a
# control signal, never user speech -- forwarding it downstream would put the
# literal string "[THINKING]" into the conversation as if the user had said it.
THINKING_SENTINEL = "[THINKING]"

__all__ = ["STT", "HamsaSTTOptions", "HAMSA_STT_BASE_URL", "THINKING_SENTINEL"]


@dataclass
class HamsaSTTOptions:
    language: str
    api_key: str
    base_url: str = HAMSA_STT_BASE_URL
    eos_enabled: bool = False
    eos_threshold: float = 0.3
    transport: str = "http"
    model: str | None = None


class STT(stt.STT):
    """Hamsa speech-to-text.

    Args:
        language: language code, e.g. ``"ar"`` or ``"en"``.
        api_key: Hamsa API key. Falls back to ``HAMSA_API_KEY``.
        base_url: override the endpoint.
        eos_enabled: ask the server's end-of-speech model whether the speaker
            has finished. When it decides they have not, the response carries no
            transcript and this plugin reports an ``INTERIM_TRANSCRIPT`` with no
            alternatives. Leave this off for plain one-shot transcription;
            :class:`~livekithamsa.eos.HamsaEOS` turns it on and acts on it.
        eos_threshold: trailing-silence threshold, in seconds, for the
            end-of-speech model. Only meaningful when ``eos_enabled``.
        transport: ``"http"`` (default) or ``"websocket"``. The socket carries the
            same whole-utterance request/response exchange but skips the
            per-request HTTP overhead, worth roughly 0.1 s per call. It does
            **not** implement the ``[THINKING]`` end-of-speech sentinel, so it
            cannot back :class:`~livekithamsa.eos.HamsaEOS`.
        model: ``"s2"`` (default server-side) or ``"s3"``. Websocket transport
            only -- the HTTP endpoint takes no model field.
        ws_url: override the websocket endpoint.
        ws_pool_size: sockets to keep. A reply is matched to its request by
            arrival order, so each in-flight call needs its own socket; this is
            therefore also the concurrency limit.
        http_session: reuse an existing session instead of LiveKit's shared one.
    """

    def __init__(
        self,
        *,
        language: str = "ar",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = HAMSA_STT_BASE_URL,
        eos_enabled: bool = False,
        eos_threshold: float = 0.3,
        transport: str = "http",
        model: str | None = None,
        ws_url: str = WS_URL,
        ws_pool_size: int = 2,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=True)
        )

        key = api_key if is_given(api_key) else os.environ.get("HAMSA_API_KEY")
        if not key:
            raise ValueError(
                "Hamsa API key is required. Pass api_key= or set HAMSA_API_KEY."
            )

        if transport not in ("http", "websocket"):
            raise ValueError(f"transport must be 'http' or 'websocket', got {transport!r}")
        if model is not None and transport != "websocket":
            logger.warning(
                "Hamsa STT: model=%r is only sent over the websocket transport; "
                "the HTTP endpoint has no model field and will ignore it.",
                model,
            )

        self._opts = HamsaSTTOptions(
            language=language,
            api_key=key,
            base_url=base_url,
            eos_enabled=eos_enabled,
            eos_threshold=eos_threshold,
            transport=transport,
            model=model,
        )
        self._ws_url = ws_url
        self._ws_pool_size = ws_pool_size
        self._ws: HamsaWSPool | None = None
        # False when the pool was handed to us by a sibling: only the owner may
        # close it, or one instance shutting down would drop the other's socket.
        self._owns_ws = True
        # Only a session we created ourselves may be closed in aclose(); the one
        # from http_context is shared with every other plugin in the process.
        self._session = http_session
        self._owns_session = False

    @property
    def language(self) -> str:
        return self._opts.language

    @property
    def transport(self) -> str:
        return self._opts.transport

    async def warmup(self) -> None:
        """Open the connection before the first turn needs it.

        Nothing requires this -- the pool dials on demand -- but the first turn
        would otherwise pay the handshake. Call it from an agent's entrypoint, or
        from LiveKit's ``prewarm_fnc``.
        """
        if self._opts.transport != "websocket":
            return
        pool = self._ensure_ws()
        async with pool.lease():
            pass  # dialed, greeted, and returned to the idle set

    def _ensure_ws(self) -> HamsaWSPool:
        if self._ws is None:
            self._ws = HamsaWSPool(
                api_key=self._opts.api_key,
                session_factory=self._ensure_session,
                url=self._ws_url,
                size=self._ws_pool_size,
            )
        return self._ws

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        eos_enabled: NotGivenOr[bool] = NOT_GIVEN,
        eos_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Change options at runtime, e.g. when an agent switches language."""
        if is_given(language):
            self._opts.language = language
        if is_given(eos_enabled):
            self._opts.eos_enabled = eos_enabled
        if is_given(eos_threshold):
            self._opts.eos_threshold = eos_threshold

    def _sibling(self, **overrides: object) -> STT:
        """A second instance differing only in the given options, sharing this
        one's HTTP session.

        :class:`~livekithamsa.eos.HamsaEOS` needs to issue both eos-on probes and
        eos-off finalizations. Holding two instances keeps each request going
        through the normal ``recognize()`` path -- retries and
        ``metrics_collected`` included -- without one call mutating options out
        from under another that is already in flight.
        """
        opts = replace(self._opts, **overrides)  # type: ignore[arg-type]
        clone = STT(
            language=opts.language,
            api_key=opts.api_key,
            base_url=opts.base_url,
            eos_enabled=opts.eos_enabled,
            eos_threshold=opts.eos_threshold,
            transport=opts.transport,
            model=opts.model,
            ws_url=self._ws_url,
            ws_pool_size=self._ws_pool_size,
            http_session=self._session,
        )
        # Share the session lazily too, so a sibling made before the first
        # request still lands on the same pool.
        clone._ensure_session = self._ensure_session  # type: ignore[method-assign]
        # And share the websocket pool. HamsaEOS holds two siblings; giving each
        # its own pool would double the sockets and mean a turn's probe and its
        # finalization never reuse the same connection.
        if opts.transport == "websocket":
            clone._ws = self._ensure_ws()
            clone._owns_ws = False
        return clone

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = self._opts
        if is_given(language):
            opts = replace(opts, language=language)
        return await self._recognize_with(buffer, opts=opts, conn_options=conn_options)

    async def _recognize_with(
        self,
        buffer: AudioBuffer,
        *,
        opts: HamsaSTTOptions,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """The single request/response cycle, with ``opts`` resolved by the
        caller. :class:`~livekithamsa.eos.HamsaEOS` calls this directly so it can
        flip ``eos_enabled`` per request without mutating shared state."""
        try:
            wav_bytes = to_wav(buffer, sample_rate=STT_SAMPLE_RATE)
        except Exception as e:
            raise APIConnectionError(f"could not encode audio: {e}") from e

        duration = _wav_duration(wav_bytes)
        payload = {
            "audioBase64": base64.b64encode(wav_bytes).decode("ascii"),
            "language": opts.language,
            "isEosEnabled": opts.eos_enabled,
            "eosThreshold": opts.eos_threshold,
        }

        if opts.transport == "websocket":
            if opts.model:
                payload["model"] = opts.model
            text = await self._ensure_ws().request_text(
                payload, timeout=conn_options.timeout
            )
            return self._to_event(
                {"data": {"text": text}}, language=opts.language, duration=duration
            )

        try:
            async with self._ensure_session().post(
                url=opts.base_url,
                json=payload,
                headers={
                    "Authorization": f"Token {opts.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    body = await res.text()
                    raise APIStatusError(
                        message=f"Hamsa STT error: {body}",
                        status_code=res.status,
                        body=body,
                    )
                data = await res.json()
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Hamsa STT request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Hamsa STT connection error: {e}") from e

        return self._to_event(data, language=opts.language, duration=duration)

    def _to_event(
        self, data: object, *, language: str, duration: float
    ) -> stt.SpeechEvent:
        text = ""
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            text = (data["data"].get("text") or "").strip()

        # The server withheld the transcript because the speaker is still going.
        # Report it as an interim with no alternatives: truthful about there
        # being nothing to act on, and never leaks the sentinel into the chat.
        if text == THINKING_SENTINEL:
            logger.debug("Hamsa STT: end-of-speech model says the turn is not over")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[]
            )

        if not text:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[]
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=language,
                    text=text,
                    start_time=0.0,
                    end_time=duration,
                    # The endpoint does not return a per-utterance score.
                    confidence=1.0,
                )
            ],
        )

    async def aclose(self) -> None:
        if self._ws is not None and self._owns_ws:
            await self._ws.aclose()
        self._ws = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        await super().aclose()


def _wav_duration(wav_bytes: bytes) -> float:
    """Duration in seconds of a 16-bit mono WAV produced by :func:`to_wav`."""
    if len(wav_bytes) <= 44:
        return 0.0
    return (len(wav_bytes) - 44) / (2 * STT_SAMPLE_RATE)
