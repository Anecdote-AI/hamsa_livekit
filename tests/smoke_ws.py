"""Live check of the websocket transport.

The claim under test is that the socket is opened once and stays open across
turns, and that this is measurably faster than HTTP for the same work.

    python tests/smoke_ws.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import livekithamsa  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents import APIConnectOptions  # noqa: E402
from livekit.agents import utils as lk_utils  # noqa: E402

CONN = APIConnectOptions(max_retry=0, timeout=30)
TURNS = 6

ok = fail = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok, fail
    if passed:
        ok += 1
        print(f"  PASS  {label}  {detail}")
    else:
        fail += 1
        print(f"  FAIL  {label}  {detail}")


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


def frames() -> list[rtc.AudioFrame]:
    with wave.open(find_test_wav(), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    per = int(sr * 0.02)
    step = per * ch * 2
    return [
        rtc.AudioFrame(
            data=raw[i : i + step],
            sample_rate=sr,
            num_channels=ch,
            samples_per_channel=per,
        )
        for i in range(0, len(raw) - step + 1, step)
    ]


async def main() -> int:
    key = os.environ.get("HAMSA_API_KEY")
    if not key:
        print("HAMSA_API_KEY is not set")
        return 2
    buf = frames()
    print(f"audio: {sum(f.duration for f in buf):.2f}s\n")

    async with lk_utils.http_context.open():
        print("STT over websocket")
        stt = livekithamsa.STT(language="ar", transport="websocket", api_key=key)
        await stt.warmup()
        pool = stt._ensure_ws()
        dialed_after_warmup = pool._created

        times, texts = [], []
        for _ in range(TURNS):
            t0 = time.monotonic()
            ev = await stt.recognize(buf, conn_options=CONN)
            times.append(time.monotonic() - t0)
            texts.append(ev.alternatives[0].text if ev.alternatives else "")

        check(
            "one socket served every turn",
            pool._created == dialed_after_warmup == 1,
            f"sockets opened={pool._created} across {TURNS} turns",
        )
        check(
            "every turn returned a transcript",
            all(texts),
            f"median rtt={statistics.median(times):.2f}s",
        )
        check(
            "warmup removed the handshake from turn 1",
            times[0] < statistics.median(times) * 2.5,
            f"turn1={times[0]:.2f}s median={statistics.median(times):.2f}s",
        )
        print(f"        transcript: {texts[-1][:56]!r}")
        ws_med = statistics.median(times)
        await stt.aclose()

        print("\nSTT over http, same audio")
        http_stt = livekithamsa.STT(language="ar", api_key=key)
        await http_stt.recognize(buf, conn_options=CONN)  # warm the pool
        http_times = []
        for _ in range(TURNS):
            t0 = time.monotonic()
            await http_stt.recognize(buf, conn_options=CONN)
            http_times.append(time.monotonic() - t0)
        http_med = statistics.median(http_times)
        await http_stt.aclose()
        delta = http_med - ws_med
        print(
            f"  websocket median {ws_med:.2f}s   http median {http_med:.2f}s   "
            f"({delta:+.2f}s, {100 * delta / http_med:+.0f}%)"
        )

        print("\nTTS over websocket")
        tts = livekithamsa.TTS(transport="websocket", api_key=key)
        await tts.warmup()
        tpool = tts._ensure_ws()
        durations, ttfbs = [], []
        for _ in range(3):
            t0 = time.monotonic()
            first = None
            total = 0.0
            async for ev in tts.synthesize("مرحبا بك في هامسا", conn_options=CONN):
                if first is None:
                    first = time.monotonic() - t0
                total += ev.frame.duration
            durations.append(total)
            ttfbs.append(first or -1)
        check(
            "tts produced audio on every call",
            all(d > 0.3 for d in durations),
            f"durations={[round(d, 2) for d in durations]}",
        )
        check(
            "tts reused one socket",
            tpool._created == 1,
            f"sockets opened={tpool._created}",
        )
        print(f"        ttfb median {statistics.median(ttfbs):.2f}s")
        await tts.aclose()

        print("\nHamsaEOS guard")
        from livekit.plugins import silero

        vad = silero.VAD.load(min_silence_duration=0.2)
        try:
            livekithamsa.HamsaEOS(
                stt=livekithamsa.STT(transport="websocket", api_key=key), vad=vad
            )
            check("HamsaEOS rejects websocket STT", False, "it was accepted")
        except ValueError as e:
            check("HamsaEOS rejects websocket STT", "THINKING" in str(e), "")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
