"""WebSocket transport for the Hamsa realtime endpoint.

`wss://api.tryhamsa.com/v1/realtime/ws` carries the same request/response work as
the HTTP endpoints -- you send a whole utterance or a whole string and get a
transcript or a stream of audio back -- but over a socket that stays open. It is
not incremental audio streaming, so it does not change the latency *model*; it
removes the per-request HTTP overhead, which measures at roughly 0.1 s per STT
call (0.27 s vs 0.38 s median on a warm connection).

Two properties of the protocol shape this module:

* **Responses carry no request id.** A reply is matched to a request purely by
  arrival order, so a socket can serve exactly one request at a time. That is why
  this is a pool of sockets rather than one shared socket: concurrency comes from
  holding several, each leased exclusively for the duration of a request.
* **The socket does not implement the end-of-speech sentinel.** With
  ``isEosEnabled: true`` the HTTP endpoint withholds the transcript and returns
  ``[THINKING]`` while the speaker is mid-utterance; the socket returns a
  transcript regardless (measured 0/15 vs 9/15 on identical audio). Anything that
  depends on that signal -- :class:`~livekithamsa.eos.HamsaEOS` -- must stay on
  HTTP.

Protocol, as documented at docs.tryhamsa.com/websocket and verified live:

    connect   wss://api.tryhamsa.com/v1/realtime/ws   (X-Api-Key header
              or ?api_key=), server greets with {"type":"info",...}
    stt       -> {"type":"stt","payload":{audioBase64,language,
                  isEosEnabled,eosThreshold,model}}
              <- the transcript as a bare string (not JSON-wrapped)
    tts       -> {"type":"tts","payload":{text,speaker,dialect,languageId,
                  mulaw,expressiveness}}
              <- {"type":"ack"}, binary audio frames, {"type":"end"}
    errors    <- {"type":"error","payload":{"message":...}}
    close     4001/4500 auth, 4003 insufficient funds, 1000 idle timeout
              (60 min), 1001 server shutdown
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
from livekit.agents import APIConnectionError, APIStatusError

from .log import logger

__all__ = ["WS_URL", "HamsaWSPool"]

WS_URL = "wss://api.tryhamsa.com/v1/realtime/ws"

# Close codes that mean "do not bother reconnecting, the problem is the account".
_FATAL_CLOSE_CODES = {4001: "authentication failed", 4003: "insufficient funds", 4500: "auth error"}

# Envelope types the server sends that are not the answer to our request.
_CHATTER = ("info", "ack")


class _Conn:
    """One socket, plus the bookkeeping to know whether it is still good."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.healthy = True

    @property
    def usable(self) -> bool:
        return self.healthy and not self.ws.closed

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()


class HamsaWSPool:
    """A small pool of persistent sockets, leased one request at a time.

    Args:
        api_key: Hamsa API key.
        session_factory: returns the ``aiohttp`` session to dial from. Called
            lazily so LiveKit's shared session is picked up inside a job.
        url: override the endpoint.
        size: how many sockets may exist at once. Each in-flight request holds
            one, so this is also the concurrency limit; further callers wait.
            2 covers a voice agent (a probe and a finalization can overlap).
        connect_timeout: seconds to wait for the upgrade and greeting.
    """

    def __init__(
        self,
        *,
        api_key: str,
        session_factory,
        url: str = WS_URL,
        size: int = 2,
        connect_timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._session_factory = session_factory
        self._url = url
        self._size = max(1, size)
        self._connect_timeout = connect_timeout

        self._idle: list[_Conn] = []
        self._created = 0
        # Bounds total sockets. Acquired before a socket is taken or made, so a
        # caller that cannot get one waits here rather than opening an extra.
        self._slots = asyncio.Semaphore(self._size)
        self._lock = asyncio.Lock()
        self._closed = False

    # -- connection management -------------------------------------------

    async def _dial(self) -> _Conn:
        session = self._session_factory()
        try:
            ws = await session.ws_connect(
                self._url,
                headers={"X-Api-Key": self._api_key},
                timeout=aiohttp.ClientWSTimeout(ws_close=self._connect_timeout),
                autoping=True,  # the server pings every 30s
                heartbeat=None,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                message=f"Hamsa websocket handshake failed: {e.message}",
                status_code=e.status,
            ) from e
        except Exception as e:
            raise APIConnectionError(f"Hamsa websocket connect failed: {e}") from e

        conn = _Conn(ws)
        # The server greets with {"type":"info"}. Consume it so the first real
        # reply is not mistaken for it.
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=self._connect_timeout)
            self._check_close(msg)
        except asyncio.TimeoutError:
            logger.debug("Hamsa ws: no greeting arrived, continuing anyway")
        self._created += 1
        logger.debug("Hamsa ws: opened connection (%d live)", self._created)
        return conn

    @staticmethod
    def _check_close(msg: aiohttp.WSMessage) -> None:
        closing = (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        )
        if msg.type not in closing:
            return
        code = getattr(msg, "data", None)
        reason = _FATAL_CLOSE_CODES.get(code if isinstance(code, int) else -1)
        if reason:
            raise APIStatusError(
                message=f"Hamsa websocket closed: {reason} (code {code})",
                status_code=401 if code != 4003 else 402,
            )
        raise APIConnectionError(f"Hamsa websocket closed unexpectedly (code {code})")

    @contextlib.asynccontextmanager
    async def lease(self) -> AsyncIterator[aiohttp.ClientWebSocketResponse]:
        """Borrow a socket for exactly one request/response exchange."""
        if self._closed:
            raise APIConnectionError("Hamsa websocket pool is closed")

        await self._slots.acquire()
        conn: _Conn | None = None
        try:
            async with self._lock:
                while self._idle:
                    candidate = self._idle.pop()
                    if candidate.usable:
                        conn = candidate
                        break
                    await candidate.aclose()
                    self._created -= 1
            if conn is None:
                conn = await self._dial()

            try:
                yield conn.ws
            except BaseException:
                # A failed exchange leaves the socket's read position unknown --
                # the next reply could be this request's. Never reuse it.
                conn.healthy = False
                raise
        finally:
            if conn is not None:
                if conn.usable and not self._closed:
                    async with self._lock:
                        self._idle.append(conn)
                else:
                    await conn.aclose()
                    self._created = max(0, self._created - 1)
            self._slots.release()

    async def aclose(self) -> None:
        self._closed = True
        async with self._lock:
            idle, self._idle = self._idle, []
        for conn in idle:
            await conn.aclose()
        self._created = 0

    # -- request helpers -------------------------------------------------

    async def request_text(self, payload: dict[str, Any], *, timeout: float) -> str:
        """Send an ``stt`` request and return the transcript.

        The transcript arrives as a bare string rather than inside an envelope,
        so anything that fails to parse as a control envelope *is* the answer.
        """
        async with self.lease() as ws:
            await ws.send_str(json.dumps({"type": "stt", "payload": payload}))
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise APIConnectionError("Hamsa websocket STT timed out")
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                if msg.type is aiohttp.WSMsgType.TEXT:
                    envelope = _envelope(msg.data)
                    if envelope is None:
                        return msg.data  # the transcript
                    kind = envelope.get("type")
                    if kind in _CHATTER:
                        continue
                    if kind == "error":
                        raise APIStatusError(
                            message=f"Hamsa websocket STT error: "
                            f"{envelope.get('payload', {}).get('message')}",
                            status_code=400,
                        )
                    # An envelope we do not recognise; treat the raw text as the
                    # answer rather than hanging.
                    return msg.data
                if msg.type is aiohttp.WSMsgType.BINARY:
                    logger.debug("Hamsa ws: unexpected binary frame during STT")
                    continue
                self._check_close(msg)

    async def stream_audio(
        self, payload: dict[str, Any], *, timeout: float
    ) -> AsyncIterator[bytes]:
        """Send a ``tts`` request and yield binary audio frames until ``end``."""
        async with self.lease() as ws:
            await ws.send_str(json.dumps({"type": "tts", "payload": payload}))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise APIConnectionError("Hamsa websocket TTS timed out")
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                if msg.type is aiohttp.WSMsgType.BINARY:
                    # Each frame extends the deadline: a long reply is fine as
                    # long as audio keeps arriving.
                    deadline = loop.time() + timeout
                    yield msg.data
                    continue
                if msg.type is aiohttp.WSMsgType.TEXT:
                    envelope = _envelope(msg.data)
                    if envelope is None:
                        continue
                    kind = envelope.get("type")
                    if kind == "end":
                        return
                    if kind == "error":
                        raise APIStatusError(
                            message=f"Hamsa websocket TTS error: "
                            f"{envelope.get('payload', {}).get('message')}",
                            status_code=400,
                        )
                    continue  # info / ack
                self._check_close(msg)


def _envelope(data: str) -> dict[str, Any] | None:
    """Return the control envelope, or None if this is a bare payload."""
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and "type" in parsed:
        return parsed
    return None
