"""What exactly does the plugin upload to the STT endpoint?

Intercepts the request body instead of trusting the code path, and reports the
sample rate and size of the WAV actually sent, for both entry points:

  * STT.recognize() directly, given 48 kHz room audio
  * HamsaEOS's streaming path, which resamples once on ingest

    python tests/verify_payload.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import livekithamsa  # noqa: E402
from livekit import rtc  # noqa: E402
from livekithamsa._audio import STT_SAMPLE_RATE  # noqa: E402

captured: list[bytes] = []


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return {"success": True, "data": {"text": "ok"}}

    async def text(self):
        return "ok"


class _FakeSession:
    """Records the JSON body instead of sending it."""

    def post(self, url, *, json=None, headers=None, timeout=None, data=None):  # noqa: A002
        captured.append(json)
        return _FakeResponse()


def frame(rate: int, seconds: float, channels: int = 1) -> rtc.AudioFrame:
    n = int(rate * seconds)
    return rtc.AudioFrame(
        data=b"\x11\x22" * n * channels,
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=n,
    )


def describe(payload: dict) -> tuple[int, float, float]:
    raw = base64.b64decode(payload["audioBase64"])
    with wave.open(io.BytesIO(raw)) as w:
        rate, frames, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    return rate, frames / rate, len(payload["audioBase64"]) / 1024, ch


async def main() -> int:
    print(f"plugin target rate: {STT_SAMPLE_RATE} Hz\n")

    # ---- 1. direct recognize() with 48 kHz room audio -------------------
    stt = livekithamsa.STT(language="ar", api_key="test", http_session=_FakeSession())
    captured.clear()
    await stt.recognize([frame(48000, 1.0)])
    rate, dur, kb, ch = describe(captured[0])
    print("STT.recognize(), given 1.00s of 48 kHz mono:")
    print(f"  uploaded: {rate} Hz, {ch}ch, {dur:.2f}s, {kb:.0f} KB base64")
    ok_direct = rate == STT_SAMPLE_RATE
    print(f"  -> {'16 kHz as intended' if ok_direct else 'WRONG RATE'}")

    # what it would have cost unresampled, for comparison
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x11\x22" * 48000)
    unres = len(base64.b64encode(b.getvalue())) / 1024
    print(f"  (unresampled 48 kHz would have been {unres:.0f} KB -> {unres / kb:.1f}x larger)")

    # ---- 2. stereo 48 kHz, the other common room format -----------------
    captured.clear()
    await stt.recognize([frame(48000, 1.0, channels=2)])
    rate, dur, kb, ch = describe(captured[0])
    print("\nSTT.recognize(), given 1.00s of 48 kHz stereo:")
    print(f"  uploaded: {rate} Hz, {ch}ch, {dur:.2f}s, {kb:.0f} KB base64")
    ok_stereo = rate == STT_SAMPLE_RATE and ch == 1

    # ---- 3. HamsaEOS streaming path ------------------------------------
    seen: list = []

    class _FakeVADStream:
        def push_frame(self, f):
            # The stream's forward_audio pushes here *after* the base
            # RecognizeStream has resampled, so this is what the turn buffer gets.
            seen.append(f)

        def flush(self):
            pass

        def end_input(self):
            pass

        async def aclose(self):
            pass

        def __aiter__(self):
            async def gen():
                if False:
                    yield None

            return gen()

    class _FakeVAD:
        capabilities = None

        def stream(self):
            return _FakeVADStream()

    eos = livekithamsa.HamsaEOS(stt=stt, vad=_FakeVAD())
    stream = eos.stream()
    captured.clear()
    try:
        # Push 48 kHz frames the way a room would; the base RecognizeStream
        # resamples on ingest, so what lands in the turn buffer is already 16 kHz.
        for _ in range(10):
            stream.push_frame(frame(48000, 0.1))
        for _ in range(40):
            await asyncio.sleep(0.02)
            if seen:
                break
        rates = {f.sample_rate for f in seen}
        print("\nHamsaEOS streaming path, fed 48 kHz frames:")
        print(f"  frames reaching the turn buffer: {sorted(rates)} Hz")
        ok_stream = rates == {STT_SAMPLE_RATE}
        print(
            f"  -> {'resampled once on ingest' if ok_stream else 'NOT resampled'}"
            "  (so re-sending the growing turn costs no repeated resampling)"
        )
    finally:
        await stream.aclose()
        await eos.aclose()

    print()
    all_ok = ok_direct and ok_stereo and ok_stream
    print("RESULT:", "16 kHz mono on every path" if all_ok else "MISMATCH — see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
