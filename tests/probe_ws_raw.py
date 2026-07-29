"""Dump every websocket frame verbatim, interpreting nothing.

An earlier probe concluded the socket does not report end-of-speech. That probe
returned any non-{info,ack,error} frame as "the transcript", so a JSON envelope
carrying eos data alongside the text would have been reported as a plain
transcript. The docs also list a `response` server type that the probe never
handled. This one prints raw frames so the question can be settled by looking.

    python tests/probe_ws_raw.py
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
    return array.array(
        "h", (a[int(i * ratio)] for i in range(int(len(a) / ratio)))
    ).tobytes()


def wav_of(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return b.getvalue()


async def dump(ws, pcm: bytes, payload_extra: dict, label: str) -> None:
    payload = {
        "audioBase64": base64.b64encode(wav_of(pcm)).decode(),
        "language": "ar",
        **payload_extra,
    }
    printable = {k: v for k, v in payload.items() if k != "audioBase64"}
    print(f"\n--- {label} ---")
    print(f"  sent payload (minus audio): {printable}")
    await ws.send_str(json.dumps({"type": "stt", "payload": payload}))

    for _ in range(5):
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=25)
        except asyncio.TimeoutError:
            print("  <no more frames>")
            return
        if msg.type is aiohttp.WSMsgType.TEXT:
            print(f"  RAW TEXT: {msg.data!r}")
            # If it is JSON, show the full structure -- that is where an eos
            # field would live.
            try:
                parsed = json.loads(msg.data)
            except (json.JSONDecodeError, ValueError):
                print("    (not JSON -> a bare transcript string)")
                return
            print(f"    parsed JSON: {json.dumps(parsed, ensure_ascii=False)}")
            if isinstance(parsed, dict):
                print(f"    top-level keys: {sorted(parsed.keys())}")
                if isinstance(parsed.get("payload"), dict):
                    print(f"    payload keys : {sorted(parsed['payload'].keys())}")
                if parsed.get("type") in ("info", "ack"):
                    continue  # chatter, keep reading
            return
        print(f"  RAW {msg.type.name}: {getattr(msg, 'data', None)!r}")
        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            return


async def main() -> int:
    k = key()
    pcm = pcm16k()
    mid = pcm[: 16000 * 2 * 3 // 2]  # 1.5s -> HTTP reliably says THINKING here

    print("Reference: what HTTP returns for the same 1.5s prefix")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            HTTP_URL,
            json={
                "audioBase64": base64.b64encode(wav_of(mid)).decode(),
                "language": "ar",
                "isEosEnabled": True,
                "eosThreshold": 0.3,
            },
            headers={"Authorization": f"Token {k}", "Content-Type": "application/json"},
        ) as r:
            body = await r.text()
        print(f"  {body}")

        ws = await session.ws_connect(WS_URL, headers={"X-Api-Key": k})
        greet = await asyncio.wait_for(ws.receive(), timeout=5)
        print(f"\ngreeting: {greet.data!r}")

        # Every combination that might surface an eos field.
        await dump(ws, mid, {"isEosEnabled": True, "eosThreshold": 0.3}, "eos on, thr 0.3")
        await dump(ws, mid, {"isEosEnabled": True, "eosThreshold": 0.7}, "eos on, thr 0.7")
        await dump(ws, mid, {"isEosEnabled": False}, "eos off")
        await dump(
            ws, mid, {"isEosEnabled": True, "eosThreshold": 0.3, "model": "s3"}, "eos on, model s3"
        )
        # Undocumented flags, in case the socket gates richer output behind them.
        await dump(
            ws,
            mid,
            {"isEosEnabled": True, "eosThreshold": 0.3, "returnEos": True, "verbose": True},
            "eos on + returnEos/verbose (guesses)",
        )
        await ws.close()

    print(
        "\nWhat to look for: any frame whose JSON contains an eos/prediction field,\n"
        "or a `response` envelope wrapping the transcript. A bare non-JSON string\n"
        "means the transcript is all the socket sends."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
