"""Audio helpers shared by the STT and TTS plugins."""

from __future__ import annotations

import io
import wave

from livekit import rtc
from livekit.agents.utils import AudioBuffer

__all__ = ["STT_SAMPLE_RATE", "MULAW_SAMPLE_RATE", "to_wav", "ulaw_to_pcm16"]

# Hamsa's STT expects 16 kHz mono. LiveKit room audio is normally 48 kHz, so
# sending the buffer untouched triples the upload for no gain in accuracy.
STT_SAMPLE_RATE = 16000

# `mulaw=True` on the TTS endpoint returns 8 kHz G.711 μ-law, one byte per
# sample -- not 16-bit PCM at the configured rate.
MULAW_SAMPLE_RATE = 8000


def _combine(buffer: AudioBuffer) -> rtc.AudioFrame:
    if isinstance(buffer, list):
        return rtc.combine_audio_frames(buffer)
    return buffer


def to_wav(buffer: AudioBuffer, *, sample_rate: int = STT_SAMPLE_RATE) -> bytes:
    """Down-mix and resample ``buffer`` to mono ``sample_rate``, returned as a
    RIFF/WAVE container.

    A fresh resampler is used per call: the filter is stateful and utterances
    are independent, so reusing one would bleed a few samples of the previous
    turn into the start of the next.
    """
    frame = _combine(buffer)

    if frame.sample_rate != sample_rate or frame.num_channels != 1:
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate,
            output_rate=sample_rate,
            num_channels=frame.num_channels,
            quality=rtc.AudioResamplerQuality.QUICK,
        )
        frames = resampler.push(frame)
        frames.extend(resampler.flush())
        if not frames:
            raise ValueError("resampling produced no audio")
        frame = rtc.combine_audio_frames(frames)

    pcm = bytes(frame.data)
    if frame.num_channels > 1:
        pcm = _downmix(pcm, frame.num_channels)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _downmix(pcm: bytes, channels: int) -> bytes:
    """Average interleaved 16-bit channels down to mono."""
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    out = array.array("h", bytes(2 * (len(samples) // channels)))
    for i in range(len(out)):
        total = 0
        base = i * channels
        for c in range(channels):
            total += samples[base + c]
        out[i] = int(total / channels)
    return out.tobytes()


def _build_ulaw_table() -> bytes:
    """G.711 μ-law -> signed 16-bit PCM, as a 512-byte little-endian table.

    Packed explicitly little-endian so the output is byte-order independent,
    which is what LiveKit's audio path expects.
    """
    import struct

    values = []
    for byte in range(256):
        u = ~byte & 0xFF
        t = ((u & 0x0F) << 3) + 0x84
        t <<= (u & 0x70) >> 4
        values.append((0x84 - t) if (u & 0x80) else (t - 0x84))
    return struct.pack("<256h", *values)


_ULAW_TABLE = _build_ulaw_table()


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 μ-law bytes to signed 16-bit little-endian PCM."""
    out = bytearray(len(data) * 2)
    table = _ULAW_TABLE
    for i, byte in enumerate(data):
        out[2 * i] = table[2 * byte]
        out[2 * i + 1] = table[2 * byte + 1]
    return bytes(out)
