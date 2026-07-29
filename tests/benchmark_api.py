"""Reproduces the measurements the implementation choices are based on.

    python tests/benchmark_api.py

Needs HAMSA_API_KEY and network. Numbers move with server load and your distance
to the endpoint; the *relative* differences are the point.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from livekithamsa.stt import HAMSA_STT_BASE_URL, THINKING_SENTINEL  # noqa: E402
from livekithamsa.tts import BASE_URL as TTS_URL  # noqa: E402

KEY = os.environ.get("HAMSA_API_KEY", "")
HDRS = {"Authorization": f"Token {KEY}", "Content-Type": "application/json"}
WAV = os.environ.get("BENCH_WAV", "../test-ar.wav")
N = 3

AR = "مرحبا بك في هامسا، نحن نقدم خدمات الذكاء الاصطناعي للغة العربية."


def read_pcm(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2
        return w.readframes(w.getnframes()), w.getframerate()


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    """Nearest-neighbour: only payload size matters for these measurements."""
    import array

    a = array.array("h")
    a.frombytes(pcm)
    if src == dst:
        return pcm
    ratio = src / dst
    out = array.array("h")
    for i in range(int(len(a) / ratio)):
        out.append(a[int(i * ratio)])
    return out.tobytes()


def wav_bytes(pcm: bytes, sr: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return b.getvalue()


async def stt(session, pcm: bytes, sr: int, *, eos: bool) -> tuple[str, float, int]:
    body = {
        "audioBase64": base64.b64encode(wav_bytes(pcm, sr)).decode(),
        "language": "ar",
        "isEosEnabled": eos,
        "eosThreshold": 0.3,
    }
    size = len(body["audioBase64"])
    t0 = time.monotonic()
    async with session.post(HAMSA_STT_BASE_URL, json=body, headers=HDRS) as r:
        data = await r.json()
    text = (data.get("data") or {}).get("text", "") if isinstance(data, dict) else ""
    return text, time.monotonic() - t0, size


async def tts_ttfb(session, text: str, *, reader: str) -> tuple[float, float]:
    t0 = time.monotonic()
    async with session.post(
        TTS_URL,
        json={"speaker": "Ali", "dialect": "pls", "text": text, "mulaw": False},
        headers=HDRS,
    ) as r:
        r.raise_for_status()
        first = None
        it = r.content.iter_chunked(8192) if reader == "chunked" else r.content.iter_any()
        async for _ in it:
            if first is None:
                first = time.monotonic() - t0
        return first or -1.0, time.monotonic() - t0


def med(xs: list[float]) -> float:
    return statistics.median(xs)


async def main() -> int:
    if not KEY:
        print("HAMSA_API_KEY is not set")
        return 2
    if not os.path.exists(WAV):
        print(f"missing {WAV}; set BENCH_WAV")
        return 2

    raw, native_sr = read_pcm(WAV)
    pcm16 = resample(raw, native_sr, 16000)[: 16000 * 2 * 3]  # 3 s at 16 kHz

    async with aiohttp.ClientSession() as s:
        await stt(s, pcm16[:16000], 16000, eos=False)  # warm the pool
        await tts_ttfb(s, "تجربة", reader="any")

        print("STT payload sample rate  (same 3 s of speech, warm connection)")
        for sr in (16000, 48000):
            pcm = resample(pcm16, 16000, sr) if sr != 16000 else pcm16
            runs = [await stt(s, pcm, sr, eos=False) for _ in range(N)]
            kb = runs[0][2] / 1024
            print(f"  {sr:>5} Hz   payload {kb:6.0f} KB   median {med([r[1] for r in runs]):.2f} s")
        print("  -> resampling to 16 kHz before upload is why _audio.to_wav exists\n")

        print("STT round trip vs utterance length (16 kHz)")
        for secs in (1, 2, 3):
            n = int(16000 * 2 * secs)
            runs = [await stt(s, pcm16[:n], 16000, eos=False) for _ in range(N)]
            print(f"  {secs} s audio    median {med([r[1] for r in runs]):.2f} s")
        print("  -> cheap enough to re-transcribe the turn at every pause\n")

        print("Server end-of-speech signal (isEosEnabled=true)")
        for secs in (1.0, 2.5, 3.0):
            n = min(int(16000 * 2 * secs), len(pcm16))
            text, dt, _ = await stt(s, pcm16[:n], 16000, eos=True)
            verdict = "still talking" if text.strip() == THINKING_SENTINEL else "turn over"
            print(f"  {secs:4.1f} s prefix  {verdict:13}  {dt:.2f} s  {text[:38]!r}")
        print("  -> what HamsaEOS reads instead of waiting out a silence timer\n")

        print("TTS time to first byte vs text length")
        lengths = (
            ("6 words", AR.split("،")[0]),
            ("full sentence", AR),
            ("doubled", AR * 2),
        )
        for label, text in lengths:
            runs = [await tts_ttfb(s, text, reader="any") for _ in range(N)]
            print(
                f"  {label:<14} ttfb median {med([r[0] for r in runs]):.2f} s   "
                f"whole response {med([r[1] for r in runs]):.2f} s"
            )
        print("  -> ttfb barely moves with length, so the win is issuing the")
        print("     first request earlier, not making it smaller\n")

        print("Response reader")
        for reader in ("any", "chunked"):
            runs = [await tts_ttfb(s, AR, reader=reader) for _ in range(N)]
            name = "iter_any()" if reader == "any" else "iter_chunked(8192)"
            print(f"  {name:<20} ttfb median {med([r[0] for r in runs]):.2f} s")
        print("  -> the gap is small and load-dependent, but iter_chunked can only")
        print("     ever be slower: it withholds bytes until a block is full\n")

        print("Chunk concurrency (3 clauses)")
        parts = ["مرحبا بك في هامسا.", "نحن نقدم خدمات الذكاء الاصطناعي.", "كيف أساعدك؟"]
        t0 = time.monotonic()
        for p in parts:
            await tts_ttfb(s, p, reader="any")
        seq = time.monotonic() - t0
        t0 = time.monotonic()
        await asyncio.gather(*[tts_ttfb(s, p, reader="any") for p in parts])
        par = time.monotonic() - t0
        print(f"  sequential {seq:.2f} s   concurrent {par:.2f} s")
        print("  -> headroom the TTS prefetch uses to keep the pipe from drying up")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
