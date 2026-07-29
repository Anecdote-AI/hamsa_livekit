"""Does the WebSocket honour isEosEnabled the way HTTP does?

HamsaEOS reads the `[THINKING]` sentinel as "the speaker has not finished". If
the socket does not produce it, turn detection silently degrades to "commit the
first thing you hear" -- so this has to be established before the WS transport is
allowed anywhere near HamsaEOS.

Same audio prefixes, same flags, both transports, several repetitions.

    python tests/probe_ws_eos.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WS_URL = "wss://api.tryhamsa.com/v1/realtime/ws"
HTTP_URL = "https://api.tryhamsa.com/v1/realtime/stt"
SENTINEL = "[THINKING]"
REPS = 3


def key() -> str:
    k = os.environ.get("HAMSA_API_KEY")
    if not k:
        raise SystemExit("HAMSA_API_KEY is not set")
    return k

def find_test_wav() -> str:
    """Arabic speech to test with. Not shipped in the repo -- point
    HAMSA_TEST_WAV at any 16-bit mono wav of someone talking."""
    path = os.environ.get("HAMSA_TEST_WAV") or str(
        Path(__file__).resolve().parents[2] / "test-ar.wav"
    )
    if not Path(path).exists():
        raise SystemExit(
            f"test audio not found: {path}\n"
            "Set HAMSA_TEST_WAV to a 16-bit mono wav of Arabic speech."
        )
    return path


def pcm16k() -> bytes:
    import array

    with wave.open(find_test_wav(), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(raw)
    if ch > 1:
        a = array.array("h", a[::ch])
    if sr == 16000:
        return a.tobytes()
    ratio = sr / 16000
    return array.array("h", (a[int(i * ratio)] for i in range(int(len(a) / ratio)))).tobytes()


def wav_of(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return b.getvalue()


def body(pcm: bytes, eos: bool) -> dict:
    return {
        "audioBase64": base64.b64encode(wav_of(pcm)).decode(),
        "language": "ar",
        "isEosEnabled": eos,
        "eosThreshold": 0.3,
    }


async def via_ws(ws, pcm: bytes, eos: bool) -> str:
    await ws.send_str(json.dumps({"type": "stt", "payload": body(pcm, eos)}))
    while True:
        msg = await asyncio.wait_for(ws.receive(), timeout=30)
        if msg.type is not aiohttp.WSMsgType.TEXT:
            return f"<{msg.type.name}>"
        try:
            env = json.loads(msg.data)
        except json.JSONDecodeError:
            return msg.data
        if isinstance(env, dict) and env.get("type") in ("info", "ack"):
            continue
        if isinstance(env, dict) and env.get("type") == "error":
            return f"<error {env.get('payload')}>"
        return msg.data


async def via_http(session, k: str, pcm: bytes, eos: bool) -> str:
    async with session.post(
        HTTP_URL,
        json=body(pcm, eos),
        headers={"Authorization": f"Token {k}", "Content-Type": "application/json"},
    ) as r:
        j = await r.json()
    return (j.get("data") or {}).get("text", "") if isinstance(j, dict) else ""


def tag(text: str) -> str:
    t = (text or "").strip()
    if t == SENTINEL:
        return "THINKING"
    if not t:
        return "empty"
    return "transcript"


async def main() -> int:
    k = key()
    pcm = pcm16k()
    prefixes = [("0.8s", 0.8), ("1.0s", 1.0), ("1.5s", 1.5), ("2.5s", 2.5), ("full", None)]

    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(WS_URL, headers={"X-Api-Key": k})
        await asyncio.wait_for(ws.receive(), timeout=5)  # greeting

        print(f"isEosEnabled=true, {REPS} reps each  (THINKING = 'still talking')\n")
        print(f"{'prefix':>7}  {'websocket':<34} {'http':<34}")
        ws_thinking = http_thinking = 0
        for label, secs in prefixes:
            buf = pcm if secs is None else pcm[: int(16000 * 2 * secs)]
            ws_tags, http_tags = [], []
            for _ in range(REPS):
                ws_tags.append(tag(await via_ws(ws, buf, True)))
                http_tags.append(tag(await via_http(session, k, buf, True)))
            ws_thinking += ws_tags.count("THINKING")
            http_thinking += http_tags.count("THINKING")
            print(f"{label:>7}  {','.join(ws_tags):<34} {','.join(http_tags):<34}")

        total = REPS * len(prefixes)
        print(f"\nTHINKING seen: websocket {ws_thinking}/{total}, http {http_thinking}/{total}")
        if ws_thinking == 0 and http_thinking > 0:
            print(
                "\n=> the socket does NOT implement the end-of-speech sentinel.\n"
                "   HamsaEOS must stay on HTTP; WS is safe for plain STT and TTS."
            )
        elif ws_thinking > 0:
            print("\n=> the socket does honour it; HamsaEOS can use either transport.")
        else:
            print("\n=> neither produced it on this audio; inconclusive.")

        await ws.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
