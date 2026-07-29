"""End-to-end checks against the real Hamsa API.

Not part of the unit suite: needs HAMSA_API_KEY and a network connection.

    python tests/smoke_live.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
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
AR_SHORT = "مرحبا كيف حالك اليوم؟"
AR_REPLY_TOKENS = [
    "مرحبا بك في هامسا، ",
    "نحن نقدم خدمات ",
    "الذكاء الاصطناعي المتقدمة ",
    "للغة العربية بجميع لهجاتها. ",
    "كيف يمكنني مساعدتك اليوم؟",
]

ok_count = 0
fail_count = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok_count, fail_count
    if passed:
        ok_count += 1
        print(f"  PASS  {label}  {detail}")
    else:
        fail_count += 1
        print(f"  FAIL  {label}  {detail}")


def load_wav_frames(path: str, chunk_ms: int = 20) -> list[rtc.AudioFrame]:
    """Read a wav into LiveKit frames at its native rate."""
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        assert w.getsampwidth() == 2, "expected 16-bit wav"
        raw = w.readframes(w.getnframes())
    per = int(sr * chunk_ms / 1000)
    step = per * ch * 2
    frames = []
    for i in range(0, len(raw) - step + 1, step):
        frames.append(
            rtc.AudioFrame(
                data=raw[i : i + step],
                sample_rate=sr,
                num_channels=ch,
                samples_per_channel=per,
            )
        )
    return frames


def silence_frames(seconds: float, like: rtc.AudioFrame) -> list[rtc.AudioFrame]:
    n = int(seconds * 1000 / 20)
    step = int(like.sample_rate * 0.02)
    return [
        rtc.AudioFrame(
            data=b"\x00\x00" * step * like.num_channels,
            sample_rate=like.sample_rate,
            num_channels=like.num_channels,
            samples_per_channel=step,
        )
        for _ in range(n)
    ]


# ------------------------------------------------------------------ TTS


async def test_chunked(tts: livekithamsa.TTS) -> None:
    """TTS.synthesize() -- the path that raised TypeError on livekit-agents 1.x."""
    t0 = time.monotonic()
    ttfb = None
    frames = 0
    duration = 0.0
    async for ev in tts.synthesize(AR_SHORT, conn_options=CONN):
        if ttfb is None:
            ttfb = time.monotonic() - t0
        frames += 1
        duration += ev.frame.duration
    check(
        "TTS.synthesize (ChunkedStream)",
        frames > 0 and duration > 0.3,
        f"ttfb={ttfb:.2f}s frames={frames} audio={duration:.2f}s",
    )


async def test_stream(tts: livekithamsa.TTS, label: str) -> float:
    """TTS.stream() fed token by token, the way an LLM would."""
    stream = tts.stream(conn_options=CONN)
    t0 = time.monotonic()
    ttfb: float | None = None
    duration = 0.0
    gaps: list[float] = []
    last = None

    async def push() -> None:
        for tok in AR_REPLY_TOKENS:
            stream.push_text(tok)
            await asyncio.sleep(0.12)  # imitate LLM token cadence
        stream.flush()
        stream.end_input()

    pusher = asyncio.create_task(push())
    try:
        async for ev in stream:
            now = time.monotonic()
            if ttfb is None:
                ttfb = now - t0
            if last is not None:
                gaps.append(now - last)
            last = now
            duration += ev.frame.duration
    finally:
        await pusher
        await stream.aclose()

    worst = max(gaps) if gaps else 0.0
    check(
        f"TTS.stream ({label})",
        duration > 3.0 and ttfb is not None,
        f"ttfb={ttfb:.2f}s audio={duration:.2f}s worst_inter_frame_gap={worst:.2f}s",
    )
    return ttfb or -1.0


async def test_unpunctuated(api_key: str) -> None:
    """A reply with no punctuation at all must still produce audio while the LLM
    is generating -- this is the case the word ceilings exist for.

    The ceilings are set low here on purpose: with the defaults (first chunk
    capped at 10 words) a 12-word reply releases its first chunk just as
    generation finishes, which would make the assertion vacuous rather than wrong.
    """
    tts = livekithamsa.TTS(
        api_key=api_key,
        first_chunk_words=3,
        first_chunk_max_words=6,
        chunk_words=4,
        max_chunk_words=8,
    )
    stream = tts.stream(conn_options=CONN)
    words = (
        "أنا سعيد جدا بلقائك اليوم وأتمنى أن نتحدث كثيرا عن هذا الموضوع "
        "المهم للغاية بالنسبة لي ولك أيضا في هذا الوقت"
    ).split()

    t0 = time.monotonic()
    ttfb: float | None = None
    audio_before_done = 0.0
    push_done = False

    async def push() -> None:
        nonlocal push_done
        for w in words:
            stream.push_text(w + " ")
            await asyncio.sleep(0.15)
        push_done = True
        stream.flush()
        stream.end_input()

    pusher = asyncio.create_task(push())
    try:
        async for ev in stream:
            if ttfb is None:
                ttfb = time.monotonic() - t0
            if not push_done:
                audio_before_done += ev.frame.duration
    finally:
        await pusher
        await stream.aclose()
        await tts.aclose()

    check(
        "unpunctuated text streams before the reply ends",
        audio_before_done > 0,
        f"ttfb={ttfb:.2f}s audio_produced_while_generating={audio_before_done:.2f}s",
    )


async def test_mulaw(api_key: str) -> None:
    tts = livekithamsa.TTS(mulaw=True, api_key=api_key)
    check("mulaw declares 8 kHz", tts.sample_rate == 8000, f"rate={tts.sample_rate}")
    duration = 0.0
    async for ev in tts.synthesize(AR_SHORT, conn_options=CONN):
        duration += ev.frame.duration
        assert ev.frame.sample_rate == 8000
    # Same words as the PCM run, so the durations should be close.
    check("mulaw audio decodes to a sane duration", 0.5 < duration < 4.0, f"{duration:.2f}s")
    await tts.aclose()


# ------------------------------------------------------------------ STT


async def test_recognize(stt: livekithamsa.STT, frames: list[rtc.AudioFrame]) -> None:
    t0 = time.monotonic()
    ev = await stt.recognize(frames, conn_options=CONN)
    dt = time.monotonic() - t0
    text = ev.alternatives[0].text if ev.alternatives else ""
    check(
        "STT.recognize",
        bool(text) and livekithamsa.THINKING_SENTINEL not in text,
        f"rtt={dt:.2f}s text={text[:48]!r}",
    )


async def test_eos_stream(api_key: str, frames: list[rtc.AudioFrame]) -> None:
    """Full turn detection: push real speech then silence, expect one committed
    turn with a transcript and no sentinel leakage."""
    from livekit.plugins import silero

    vad = silero.VAD.load(min_silence_duration=0.2)
    eos = livekithamsa.HamsaEOS(
        stt=livekithamsa.STT(language="ar", api_key=api_key),
        vad=vad,
        max_turn_silence=2.0,
    )
    stream = eos.stream(conn_options=CONN)

    events: list[tuple[str, str]] = []
    t0 = time.monotonic()
    committed_at: float | None = None

    async def feed() -> None:
        for f in frames:
            stream.push_frame(f)
            await asyncio.sleep(0.005)  # faster than realtime, but ordered
        for f in silence_frames(2.5, frames[0]):
            stream.push_frame(f)
            await asyncio.sleep(0.005)
        stream.end_input()

    async def collect() -> None:
        nonlocal committed_at
        async for ev in stream:
            text = ev.alternatives[0].text if ev.alternatives else ""
            events.append((ev.type.value, text))
            if ev.type.value == "final_transcript" and text:
                committed_at = time.monotonic() - t0

    feeder = asyncio.create_task(feed())
    collector = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(asyncio.gather(feeder, collector), timeout=60)
    except asyncio.TimeoutError:
        check("HamsaEOS committed a turn", False, "timed out")
    finally:
        await lk_utils.aio.cancel_and_wait(feeder, collector)
        await stream.aclose()
        await eos.aclose()

    finals = [t for kind, t in events if kind == "final_transcript" and t]
    kinds = [k for k, _ in events]

    check("HamsaEOS emitted START_OF_SPEECH", "start_of_speech" in kinds, "")
    check(
        "HamsaEOS committed exactly one turn",
        len(finals) == 1,
        f"finals={finals} at={committed_at}",
    )
    check(
        "HamsaEOS never leaked the sentinel",
        all(livekithamsa.THINKING_SENTINEL not in t for _, t in events),
        "",
    )
    check(
        "HamsaEOS paired END_OF_SPEECH with the final",
        kinds.count("end_of_speech") == len(finals),
        f"eos={kinds.count('end_of_speech')}",
    )
    print(f"        event trace: {kinds}")
    if finals:
        print(f"        transcript:  {finals[0][:70]!r}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="../test-ar.wav", help="16-bit wav of Arabic speech")
    args = ap.parse_args()

    api_key = os.environ.get("HAMSA_API_KEY")
    if not api_key:
        print("HAMSA_API_KEY is not set")
        return 2
    if not os.path.exists(args.wav):
        print(f"missing test audio: {args.wav}")
        return 2

    frames = load_wav_frames(args.wav)
    print(
        f"test audio: {args.wav} "
        f"({sum(f.duration for f in frames):.2f}s @ {frames[0].sample_rate} Hz)\n"
    )

    # Sets up the shared, keep-alive session the plugins pull from. Outside a
    # LiveKit job this has to be opened explicitly.
    async with lk_utils.http_context.open():
        print("TTS")
        tts = livekithamsa.TTS(api_key=api_key)
        await test_chunked(tts)
        ttfb_pre = await test_stream(tts, "prefetch=1")
        await tts.aclose()

        tts_np = livekithamsa.TTS(
            api_key=api_key,
            prefetch=0,
            first_chunk_words=10_000,
            first_chunk_max_words=10_000,
            chunk_words=10_000,
            max_chunk_words=10_000,
        )
        ttfb_whole = await test_stream(tts_np, "no chunking, no prefetch")
        await tts_np.aclose()
        if ttfb_pre > 0 and ttfb_whole > 0:
            print(
                f"        first-audio latency: chunked={ttfb_pre:.2f}s "
                f"vs unchunked={ttfb_whole:.2f}s "
                f"({ttfb_whole - ttfb_pre:+.2f}s)"
            )

        await test_unpunctuated(api_key)
        await test_mulaw(api_key)

        print("\nSTT")
        stt = livekithamsa.STT(language="ar", api_key=api_key)
        await test_recognize(stt, frames)
        await stt.aclose()

        print("\nHamsaEOS (turn detection)")
        await test_eos_stream(api_key, frames)

    print(f"\n{ok_count} passed, {fail_count} failed")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
