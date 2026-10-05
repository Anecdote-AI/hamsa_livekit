"""Tests for the websocket transport, with no network.

The property that matters most is that the socket is opened once and reused for
every subsequent turn -- reconnecting per turn would put the handshake back on
the critical path, which is the entire reason the transport exists.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from livekithamsa._ws import HamsaWSPool
from livekithamsa.eos import HamsaEOS
from livekithamsa.stt import STT
from livekithamsa.tts import TTS


class _Msg:
    def __init__(self, type_, data):
        self.type = type_
        self.data = data


class _FakeWS:
    """Scripted socket. Records sends and replies from a per-request script."""

    opened = 0

    def __init__(self, script):
        self._script = script
        self.sent: list[dict] = []
        self.closed = False
        self._pending: list[_Msg] = [
            _Msg(aiohttp.WSMsgType.TEXT, json.dumps({"type": "info", "payload": {}}))
        ]

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))
        self._pending.extend(self._script(json.loads(data)))

    async def receive(self):
        # Yield control before replying. Without a real await point the "gathered"
        # requests would each run to completion in turn, and a pool bug where
        # overlapping calls share one socket would go unnoticed.
        await asyncio.sleep(0.01)
        if self._pending:
            return self._pending.pop(0)
        await asyncio.sleep(3600)  # would hang; tests must not reach this

    async def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, script):
        self._script = script
        self.dials = 0
        self.sockets: list[_FakeWS] = []

    async def ws_connect(self, url, **kwargs):
        self.dials += 1
        ws = _FakeWS(self._script)
        self.sockets.append(ws)
        return ws


def stt_script(transcript: str):
    def script(req):
        assert req["type"] == "stt"
        return [_Msg(aiohttp.WSMsgType.TEXT, transcript)]

    return script


def tts_script(chunks: list[bytes]):
    def script(req):
        assert req["type"] == "tts"
        out = [_Msg(aiohttp.WSMsgType.TEXT, json.dumps({"type": "ack", "payload": {}}))]
        out += [_Msg(aiohttp.WSMsgType.BINARY, c) for c in chunks]
        out.append(_Msg(aiohttp.WSMsgType.TEXT, json.dumps({"type": "end", "payload": {}})))
        return out

    return script


# ------------------------------------------------------- connection reuse


def test_socket_is_opened_once_and_reused_across_turns():
    """The point of the transport: one handshake, many turns."""
    session = _FakeSession(stt_script("مرحبا"))
    pool = HamsaWSPool(api_key="k", session_factory=lambda: session, size=2)

    async def run():
        for _ in range(10):
            text = await pool.request_text({"audioBase64": "x"}, timeout=5)
            assert text == "مرحبا"
        await pool.aclose()

    asyncio.run(run())
    assert session.dials == 1, f"reconnected {session.dials} times instead of once"


def test_pool_opens_a_second_socket_only_for_concurrency():
    """Replies carry no request id, so overlapping calls must not share a socket."""
    session = _FakeSession(stt_script("ok"))
    pool = HamsaWSPool(api_key="k", session_factory=lambda: session, size=3)

    async def run():
        await asyncio.gather(
            *[pool.request_text({"audioBase64": "x"}, timeout=5) for _ in range(3)]
        )
        # after they finish, further calls reuse rather than dial again
        before = session.dials
        for _ in range(5):
            await pool.request_text({"audioBase64": "x"}, timeout=5)
        assert session.dials == before
        await pool.aclose()

    asyncio.run(run())
    assert session.dials == 3


def test_pool_size_caps_sockets():
    session = _FakeSession(stt_script("ok"))
    pool = HamsaWSPool(api_key="k", session_factory=lambda: session, size=2)

    async def run():
        await asyncio.gather(
            *[pool.request_text({"audioBase64": "x"}, timeout=5) for _ in range(6)]
        )
        await pool.aclose()

    asyncio.run(run())
    assert session.dials <= 2, session.dials


def test_broken_socket_is_replaced_not_reused():
    """A failed exchange leaves the read position unknown; the next reply could
    belong to the dead request."""
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            err = json.dumps({"type": "error", "payload": {"message": "boom"}})
            return [_Msg(aiohttp.WSMsgType.TEXT, err)]
        return [_Msg(aiohttp.WSMsgType.TEXT, "recovered")]

    session = _FakeSession(flaky)
    pool = HamsaWSPool(api_key="k", session_factory=lambda: session, size=2)

    async def run():
        with pytest.raises(Exception, match="boom"):
            await pool.request_text({"audioBase64": "x"}, timeout=5)
        text = await pool.request_text({"audioBase64": "x"}, timeout=5)
        await pool.aclose()
        return text

    assert asyncio.run(run()) == "recovered"
    assert session.dials == 2
    assert session.sockets[0].closed, "the broken socket should have been closed"


def test_greeting_is_not_mistaken_for_a_transcript():
    session = _FakeSession(stt_script("النص"))
    pool = HamsaWSPool(api_key="k", session_factory=lambda: session, size=1)

    async def run():
        text = await pool.request_text({"audioBase64": "x"}, timeout=5)
        await pool.aclose()
        return text

    assert asyncio.run(run()) == "النص"


# ------------------------------------------------------------- plumbing


def test_stt_websocket_transport_sends_the_documented_envelope(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "k")
    session = _FakeSession(stt_script("مرحبا بك"))
    stt = STT(language="ar", transport="websocket", model="s3")
    monkeypatch.setattr(stt, "_ensure_session", lambda: session)

    from livekit import rtc

    frame = rtc.AudioFrame(
        data=b"\x00\x01" * 16000, sample_rate=16000, num_channels=1, samples_per_channel=16000
    )

    async def run():
        ev = await stt.recognize([frame])
        await stt.aclose()
        return ev

    ev = asyncio.run(run())
    assert ev.alternatives[0].text == "مرحبا بك"
    sent = session.sockets[0].sent[0]
    assert sent["type"] == "stt"
    assert sent["payload"]["language"] == "ar"
    assert sent["payload"]["model"] == "s3"
    assert "audioBase64" in sent["payload"]


def test_tts_websocket_transport_streams_binary(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "k")
    session = _FakeSession(tts_script([b"\x01\x02" * 100, b"\x03\x04" * 100]))
    tts = TTS(transport="websocket", speaker="Lana", dialect="jor", expressiveness=2)
    monkeypatch.setattr(tts, "_ensure_session", lambda: session)

    async def run():
        total = 0
        async for ev in tts.synthesize("مرحبا"):
            total += len(bytes(ev.frame.data))
        await tts.aclose()
        return total

    assert asyncio.run(run()) > 0
    sent = session.sockets[0].sent[0]
    assert sent["type"] == "tts"
    assert sent["payload"]["speaker"] == "Lana"
    assert sent["payload"]["dialect"] == "jor"
    assert sent["payload"]["expressiveness"] == 2
    assert sent["payload"]["languageId"] == "ar"


def test_update_options_language_reaches_the_next_request_on_the_same_socket(monkeypatch):
    """A bilingual agent's voice follows the caller by retuning between turns. The
    new dialect/languageId must be on the very next request, on the socket that is
    already open -- a reconnect per switch would put the handshake on the turn the
    caller is waiting for."""
    monkeypatch.setenv("HAMSA_API_KEY", "k")
    session = _FakeSession(tts_script([b"\x01\x02" * 100]))
    tts = TTS(transport="websocket", speaker="Lana", dialect="ksa", language_id="ar")
    monkeypatch.setattr(tts, "_ensure_session", lambda: session)

    async def run():
        async for _ in tts.synthesize("مرحبا"):
            pass
        tts.update_options(dialect="en", language_id="en")
        async for _ in tts.synthesize("hello"):
            pass
        await tts.aclose()

    asyncio.run(run())
    assert session.dials == 1
    sent = [m["payload"] for m in session.sockets[0].sent]
    assert [(p["dialect"], p["languageId"]) for p in sent] == [("ksa", "ar"), ("en", "en")]
    assert {p["speaker"] for p in sent} == {"Lana"}


def test_eos_refuses_the_websocket_transport(monkeypatch):
    """Turn detection is built on a signal the socket does not send. Silently
    degrading would make the agent interrupt people."""
    monkeypatch.setenv("HAMSA_API_KEY", "k")

    class _VAD:
        capabilities = None

        def stream(self):
            raise AssertionError("should not get this far")

    with pytest.raises(ValueError, match=r"\[THINKING\]"):
        HamsaEOS(stt=STT(transport="websocket"), vad=_VAD())


def test_eos_siblings_share_one_pool(monkeypatch):
    """Two siblings with two pools would double the sockets and stop a turn's
    probe and finalization reusing the same connection."""
    monkeypatch.setenv("HAMSA_API_KEY", "k")
    base = STT(transport="websocket")
    session = _FakeSession(stt_script("ok"))
    monkeypatch.setattr(base, "_ensure_session", lambda: session)
    a = base._sibling(eos_enabled=True)
    b = base._sibling(eos_enabled=False)
    assert a._ws is b._ws is base._ensure_ws()
    assert not a._owns_ws and not b._owns_ws

    async def run():
        await a.aclose()  # a sibling closing must not drop the shared socket
        assert not base._ensure_ws()._closed
        await b.aclose()
        assert not base._ensure_ws()._closed
        await base.aclose()  # only the owner closes it
        assert base._ws is None

    asyncio.run(run())


def test_invalid_transport_is_rejected(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "k")
    with pytest.raises(ValueError, match="transport"):
        STT(transport="grpc")
    with pytest.raises(ValueError, match="transport"):
        TTS(transport="grpc")
