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

"""Text-to-Speech for Hamsa.

The endpoint streams audio back as it is generated, and time-to-first-byte is
about 0.3 s regardless of how much text is sent. Two consequences shape this
implementation:

* **Get the first request out early.** Since a long request is no slower to
  start than a short one, the only way to reduce time-to-first-audio is to stop
  waiting for the LLM. The opening request goes out at the first punctuation
  after a few words -- Arabic punctuation included -- so synthesis starts while
  the model is still writing the rest of the reply, and the fragment sent still
  ends somewhere a voice would naturally pause. A word ceiling keeps that from
  becoming an open-ended wait on text that never gets punctuated.
* **Never let the pipe run dry.** Generation runs several times faster than
  playback, but a new request still costs ~0.3 s before its first byte. The next
  chunk is therefore fetched while the current one is still being handed to
  LiveKit, so that gap falls inside the previous chunk's playback instead of
  becoming an audible pause.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ._audio import MULAW_SAMPLE_RATE, ulaw_to_pcm16
from ._ws import WS_URL, HamsaWSPool
from .log import logger
from .tokenizer import CLAUSE_SEPARATORS, SENTENCE_TERMINATORS

BASE_URL = "https://api.tryhamsa.com/v1/realtime/tts-stream"
NUM_CHANNELS = 1

# Where a chunk after the first is allowed to end. Sentence terminators are the
# ideal break; clause separators keep the word cap from cutting mid-phrase.
_CHUNK_BREAKS = SENTENCE_TERMINATORS + CLAUSE_SEPARATORS

# The endpoint's output format is fixed: 16 kHz signed 16-bit mono PCM, or 8 kHz
# G.711 μ-law when `mulaw` is set. There is no field in the request to ask for
# anything else.
PCM_SAMPLE_RATE = 16000

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "BASE_URL"]


@dataclass
class _TTSOptions:
    speaker: str
    dialect: str
    mulaw: bool
    sample_rate: int
    base_url: str
    api_key: str
    first_chunk_words: int
    first_chunk_max_words: int
    chunk_words: int
    max_chunk_words: int
    prefetch: int
    word_tokenizer: tokenize.WordTokenizer
    transport: str = "http"
    language_id: str = "ar"
    expressiveness: int = 1


class TTS(tts.TTS):
    """Hamsa text-to-speech.

    Args:
        speaker: voice name, e.g. ``"Ali"``, ``"Lana"``, ``"Mariam"``.
        dialect: dialect code, e.g. ``"pls"``, ``"jor"``, ``"egy"``.
        mulaw: request 8 kHz G.711 μ-law instead of 16 kHz PCM. Useful for
            telephony; the bytes are decoded to PCM before being handed to
            LiveKit.
        sample_rate: normally leave unset. The endpoint's rate is fixed
            (16 kHz PCM, 8 kHz μ-law); passing a different value here does not
            change what the server sends, it only mislabels it, which shows up
            as pitch-shifted playback.
        api_key: Hamsa API key. Falls back to ``HAMSA_API_KEY``.
        base_url: override the endpoint.
        first_chunk_words: floor for the opening chunk. Below this many words it
            will not be released at all, so the TTS is never handed a stray word.
        first_chunk_max_words: ceiling for the opening chunk. Between the floor
            and this, the chunk is released at the first punctuation; if none
            appears by the ceiling it is cut there anyway. Releasing on the
            count alone lands mid-phrase and the voice sounds truncated, so the
            punctuation gate is what makes the opening sound finished -- the
            ceiling just bounds how long that gate can wait.
        chunk_words: same floor, for the chunks after the first.
        max_chunk_words: same ceiling, for the chunks after the first. Without
            it, text that never gets punctuated -- which LLMs do produce --
            would buffer until the reply ended.
        prefetch: how many chunks to keep in flight ahead of the one being
            played. 1 is enough to cover the ~0.3 s request latency; 0 disables
            it.
        word_tokenizer: splits incoming text into words. Must keep trailing
            punctuation attached, since that is how sentence boundaries are
            detected.
        transport: ``"http"`` (default) or ``"websocket"``. Both return the same
            16 kHz PCM; the socket skips per-request HTTP overhead. Note the
            socket delivers audio in larger frames, so it is not automatically
            better for time-to-first-audio -- measure before switching.
        language_id: language tag sent on the websocket payload.
        expressiveness: 0-2, how much prosodic variation the voice uses.
            Websocket transport only; the HTTP endpoint has no such field.
        ws_url: override the websocket endpoint.
        ws_pool_size: sockets to keep. Audio frames are matched to their request
            by arrival order, so each in-flight synthesis needs its own socket.
            Keep this at least ``prefetch + 1``.
        http_session: reuse an existing session instead of LiveKit's shared one.
    """

    def __init__(
        self,
        *,
        speaker: str = "Ali",
        dialect: str = "pls",
        mulaw: bool = False,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        first_chunk_words: int = 3,
        first_chunk_max_words: int = 10,
        chunk_words: int = 18,
        max_chunk_words: int = 25,
        prefetch: int = 1,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        transport: str = "http",
        language_id: str = "ar",
        expressiveness: int = 1,
        ws_url: str = WS_URL,
        ws_pool_size: NotGivenOr[int] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if transport not in ("http", "websocket"):
            raise ValueError(f"transport must be 'http' or 'websocket', got {transport!r}")

        native_rate = MULAW_SAMPLE_RATE if mulaw else PCM_SAMPLE_RATE
        if is_given(sample_rate) and sample_rate != native_rate:
            logger.warning(
                "Hamsa TTS returns %d Hz when mulaw=%s; sample_rate=%d will play "
                "back at the wrong speed. Ignoring it.",
                native_rate,
                mulaw,
                sample_rate,
            )
        resolved_rate = native_rate

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=resolved_rate,
            num_channels=NUM_CHANNELS,
        )

        key = api_key if is_given(api_key) else os.environ.get("HAMSA_API_KEY")
        if not key:
            raise ValueError(
                "Hamsa API key is required. Pass api_key= or set HAMSA_API_KEY."
            )

        self._opts = _TTSOptions(
            speaker=speaker,
            dialect=dialect,
            mulaw=mulaw,
            sample_rate=resolved_rate,
            base_url=base_url,
            api_key=key,
            first_chunk_words=max(1, first_chunk_words),
            first_chunk_max_words=max(first_chunk_words, first_chunk_max_words),
            chunk_words=max(1, chunk_words),
            max_chunk_words=max(chunk_words, max_chunk_words),
            prefetch=max(0, prefetch),
            word_tokenizer=(
                word_tokenizer
                if is_given(word_tokenizer)
                # ignore_punctuation=False keeps "هامسا،" and "حالك؟" intact, so
                # the trailing character still marks the boundary.
                else tokenize.basic.WordTokenizer(ignore_punctuation=False)
            ),
            transport=transport,
            language_id=language_id,
            expressiveness=expressiveness,
        )
        self._session = http_session
        self._owns_session = False
        self._ws_url = ws_url
        # One socket per in-flight chunk, plus the one being played.
        self._ws_pool_size = (
            ws_pool_size if is_given(ws_pool_size) else max(2, self._opts.prefetch + 1)
        )
        self._ws: HamsaWSPool | None = None
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def warmup(self) -> None:
        """Open the connection before the first utterance needs it."""
        if self._opts.transport != "websocket":
            return
        async with self._ensure_ws().lease():
            pass

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
        speaker: NotGivenOr[str] = NOT_GIVEN,
        dialect: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Change the voice for subsequent synthesis.

        ``mulaw`` and the sample rate are deliberately not updatable: they change
        the audio format, which LiveKit reads once when the track is created.
        """
        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(dialect):
            self._opts.dialect = dialect

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._ws is not None:
            await self._ws.aclose()
            self._ws = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        await super().aclose()


async def _synthesize(
    *,
    tts_impl: TTS,
    opts: _TTSOptions,
    text: str,
    conn_options: APIConnectOptions,
) -> AsyncIterator[bytes]:
    """Dispatch to whichever transport is configured, yielding decoded PCM."""
    if opts.transport == "websocket":
        payload = {
            "text": text,
            "speaker": opts.speaker,
            "dialect": opts.dialect,
            "languageId": opts.language_id,
            "mulaw": opts.mulaw,
            "expressiveness": opts.expressiveness,
        }
        pool = tts_impl._ensure_ws()
        async for data in pool.stream_audio(payload, timeout=conn_options.timeout):
            if data:
                yield ulaw_to_pcm16(data) if opts.mulaw else data
        return

    async for data in _synthesize_http(
        session=tts_impl._ensure_session(),
        opts=opts,
        text=text,
        conn_options=conn_options,
    ):
        yield data


async def _synthesize_http(
    *,
    session: aiohttp.ClientSession,
    opts: _TTSOptions,
    text: str,
    conn_options: APIConnectOptions,
) -> AsyncIterator[bytes]:
    """POST ``text`` and yield decoded PCM as it arrives, in whole samples.

    ``iter_any()`` rather than ``iter_chunked(n)``: the latter withholds bytes
    until it has a full block, which delays the first frame for no benefit.

    A read can end mid-sample, so a trailing odd byte is carried over to the next
    one. This is not cosmetic: LiveKit's ``AudioByteStream.flush()`` throws away
    its *entire* remaining buffer if the byte count is not a whole number of
    samples, and the emitter flushes on its own whenever audio arrives slower
    than realtime -- which is exactly what happens while an LLM is still typing.
    Handing over a partial sample therefore costs real audio.
    """
    payload = {
        "speaker": opts.speaker,
        "dialect": opts.dialect,
        "text": text,
        "mulaw": opts.mulaw,
    }
    try:
        async with session.post(
            opts.base_url,
            headers={
                "Authorization": f"Token {opts.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_connect=conn_options.timeout,
                sock_read=conn_options.timeout,
            ),
        ) as res:
            if res.status != 200:
                body = await res.text()
                raise APIStatusError(
                    message=f"Hamsa TTS error: {body}",
                    status_code=res.status,
                    body=body,
                )
            # Bytes per sample on the wire: 1 for G.711, 2 for signed 16-bit PCM.
            wire_width = 1 if opts.mulaw else 2
            carry = b""
            async for data in res.content.iter_any():
                if not data:
                    continue
                if carry:
                    data = carry + data
                    carry = b""
                if remainder := len(data) % wire_width:
                    data, carry = data[:-remainder], data[-remainder:]
                    if not data:
                        continue
                yield ulaw_to_pcm16(data) if opts.mulaw else data
            if carry:
                logger.debug(
                    "Hamsa TTS: response ended mid-sample, discarding %d byte(s)",
                    len(carry),
                )
    except asyncio.TimeoutError as e:
        raise APITimeoutError("Hamsa TTS request timed out") from e
    except aiohttp.ClientError as e:
        raise APIConnectionError(f"Hamsa TTS connection error: {e}") from e


class ChunkedStream(tts.ChunkedStream):
    """One-shot synthesis of a complete string."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        async for data in _synthesize(
            tts_impl=self._tts,
            opts=self._opts,
            text=self._input_text,
            conn_options=self._conn_options,
        ):
            output_emitter.push(data)

        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """Incremental synthesis: text arrives token by token from the LLM."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def update_options(
        self,
        *,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        dialect: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(dialect):
            self._opts.dialect = dialect

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        # One segment per flush from LiveKit, i.e. per agent utterance.
        segments = utils.aio.Chan[tokenize.WordStream]()

        async def tokenize_input() -> None:
            stream: tokenize.WordStream | None = None
            async for data in self._input_ch:
                if isinstance(data, str):
                    if stream is None:
                        stream = self._opts.word_tokenizer.stream()
                        segments.send_nowait(stream)
                    stream.push_text(data)
                elif isinstance(data, self._FlushSentinel):
                    if stream is not None:
                        stream.end_input()
                    stream = None
            if stream is not None:
                stream.end_input()
            segments.close()

        async def run_segments() -> None:
            async for stream in segments:
                await self._run_segment(stream, output_emitter)

        tasks = [
            asyncio.create_task(tokenize_input(), name="hamsa-tts-tokenize"),
            asyncio.create_task(run_segments(), name="hamsa-tts-segments"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

        output_emitter.end_input()

    async def _chunk_texts(self, stream: tokenize.WordStream) -> AsyncIterator[str]:
        """Group incoming words into synthesis chunks.

        Every chunk uses the same rule: a word-count floor, then release at the
        first punctuation, with a hard ceiling if none turns up.

            |<-- floor -->|<---- looking for punctuation ---->|<- ceiling
            never break    break at the first , . ? ; :        break anyway

        The floor stops the TTS being handed a stray word or two. The punctuation
        gate is what makes it sound finished rather than truncated -- releasing on
        the count alone lands mid-phrase, and the voice audibly stops in the
        middle of a breath. The ceiling keeps that gate from becoming an
        open-ended wait when a model writes long unpunctuated stretches.

        Counted in words, not characters, because a character threshold is not
        stable across the text this plugin exists to handle: the same four Arabic
        words are 18 characters undiacritized and 28 with tashkeel, and Arabic
        runs shorter per word than Latin, so one number would mean three
        different things.
        """
        words: list[str] = []
        first = True
        async for ev in stream:
            token = ev.token.strip()
            if not token:
                continue
            words.append(token)

            if first:
                floor, ceiling = (
                    self._opts.first_chunk_words,
                    self._opts.first_chunk_max_words,
                )
            else:
                floor, ceiling = self._opts.chunk_words, self._opts.max_chunk_words

            if len(words) < floor:
                continue
            if token[-1] in _CHUNK_BREAKS or len(words) >= ceiling:
                yield " ".join(words)
                words.clear()
                first = False

        if words:
            yield " ".join(words)

    async def _run_segment(
        self, stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        # Two bounds, and they are not the same thing.
        #
        # `slots` caps CONCURRENT REQUESTS at prefetch+1 -- the one being played
        # plus `prefetch` ahead of it. The channel alone could not do this:
        # `_ChunkFetch.__init__` fires its request immediately, so bounding only
        # the buffer let a THIRD request go out at prefetch=1 (one draining, one
        # buffered, one whose `send` was blocked). That extra request matters on
        # the websocket transport, where `ws_pool_size` defaults to prefetch+1 --
        # it went out and then sat waiting for a socket.
        #
        # `pipeline` caps chunks BUFFERED and undrained, preserving the ordered
        # hand-off to LiveKit. A slot is released when its request finishes, which
        # is earlier than when its audio is consumed -- deliberate, since it is
        # the request (and its socket) being rationed, not the buffer.
        slots = asyncio.Semaphore(self._opts.prefetch + 1)
        pipeline = utils.aio.Chan[_ChunkFetch](maxsize=max(1, self._opts.prefetch))
        started = False

        async def produce() -> None:
            nonlocal started
            try:
                async for text in self._chunk_texts(stream):
                    if not started:
                        # Anchors the time-to-first-byte metric at the moment we
                        # first had something to say.
                        self._mark_started()
                        started = True
                    await slots.acquire()
                    await pipeline.send(
                        _ChunkFetch(
                            tts_impl=self._tts,
                            opts=self._opts,
                            text=text,
                            conn_options=self._conn_options,
                            slot=slots,
                        )
                    )
            finally:
                pipeline.close()

        producer = asyncio.create_task(produce(), name="hamsa-tts-produce")
        current: _ChunkFetch | None = None
        # The segment is opened lazily. An utterance that tokenizes to nothing
        # (whitespace or punctuation only) would otherwise open a segment the
        # emitter never closes -- it ignores end_segment when no audio was
        # pushed -- and the next start_segment would then raise.
        opened = False
        try:
            async for fetch in pipeline:
                current = fetch
                async for data in fetch.drain():
                    if not opened:
                        output_emitter.start_segment(segment_id=utils.shortuuid())
                        opened = True
                    output_emitter.push(data)
                # Release the tail of this chunk instead of holding a partial
                # frame until the next chunk's bytes arrive.
                if opened:
                    output_emitter.flush()
                current = None
            await producer
        finally:
            # On the error path the chunk being read, and anything prefetched
            # behind it, still have live HTTP requests to tear down.
            await utils.aio.cancel_and_wait(producer)
            leftovers = [current] if current is not None else []
            while True:
                try:
                    leftovers.append(pipeline.recv_nowait())
                except Exception:
                    break
            for fetch in leftovers:
                await fetch.aclose()

        if opened:
            output_emitter.end_segment()


class _ChunkFetch:
    """A synthesis request whose audio is buffered as it arrives.

    Buffering is what makes prefetch possible: audio must reach LiveKit in order,
    so a chunk fetched early has to park its bytes somewhere until the chunk
    before it has finished.
    """

    def __init__(
        self,
        *,
        tts_impl: TTS,
        opts: _TTSOptions,
        text: str,
        conn_options: APIConnectOptions,
        slot: asyncio.Semaphore | None = None,
    ) -> None:
        self.text = text
        # Acquired by the producer on this fetch's behalf and released once the
        # request is done (success, failure or cancellation), which is what frees
        # the next chunk to start. Guarded so it can never be released twice.
        self._slot = slot
        self._released = False
        self._queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue()
        self._task = asyncio.create_task(
            self._pump(tts_impl=tts_impl, opts=opts, conn_options=conn_options),
            name="hamsa-tts-fetch",
        )

    def _release(self) -> None:
        if self._slot is not None and not self._released:
            self._released = True
            self._slot.release()

    async def _pump(
        self,
        *,
        tts_impl: TTS,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        try:
            async for data in _synthesize(
                tts_impl=tts_impl,
                opts=opts,
                text=self.text,
                conn_options=conn_options,
            ):
                self._queue.put_nowait(data)
            self._queue.put_nowait(None)
        except asyncio.CancelledError:
            self._queue.put_nowait(None)
            raise
        except BaseException as e:  # noqa: BLE001 - handed to the consumer
            self._queue.put_nowait(e)
        finally:
            self._release()

    async def drain(self) -> AsyncIterator[bytes]:
        """Yield this chunk's audio in order, re-raising any request failure.

        Failures propagate rather than being logged and swallowed, so LiveKit's
        retry logic sees them instead of the agent going silently mute.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aclose(self) -> None:
        await utils.aio.cancel_and_wait(self._task)
        # `_pump`'s finally normally does this; belt-and-braces for a task
        # cancelled before it ever ran.
        self._release()
