"""Headless stand-in for the browser, for running the demo without a microphone.

Joins a room, dispatches the named agent to it, publishes a wav of Arabic speech
as if spoken into a mic, then reports what came back: the agent's transcript, its
audio, the metrics it published, and the wall-clock gap between "stopped
speaking" and "first audio from the agent".

    python simulate_user.py --wav ../../test-ar.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

FRAME_MS = 20


def peak(frame: rtc.AudioFrame) -> int:
    """Peak absolute sample. LiveKit pads a live track with near-silence that is
    not bitwise zero, so "any nonzero byte" would count silence as speech."""
    import array

    a = array.array("h")
    a.frombytes(bytes(frame.data))
    return max((abs(s) for s in a), default=0)


async def synthesize_utterance(
    text: str, out: str, *, trailing_silence: float = 1.2
) -> tuple[str, float]:
    """Render `text` with Hamsa TTS to use as the user's speech.

    A recording excerpt makes a poor test turn: internal pauses let the
    end-of-speech model reasonably commit early, and with no trailing silence
    there is no moment where the user is clearly done. A synthesized utterance
    plus real silence is what an actual turn looks like.
    """
    import livekithamsa
    from livekit.agents import utils as lk_utils

    frames: list[rtc.AudioFrame] = []
    # Outside a LiveKit job the shared HTTP session has to be opened explicitly.
    async with lk_utils.http_context.open():
        # A different voice from the agent's, so the two are easy to tell apart.
        tts = livekithamsa.TTS(speaker="Lana", dialect="jor")
        try:
            async for ev in tts.synthesize(text):
                frames.append(ev.frame)
        finally:
            await tts.aclose()

    combined = rtc.combine_audio_frames(frames)
    pad = b"\x00\x00" * int(combined.sample_rate * trailing_silence)
    with wave.open(out, "wb") as w:
        w.setnchannels(combined.num_channels)
        w.setsampwidth(2)
        w.setframerate(combined.sample_rate)
        w.writeframes(bytes(combined.data) + pad)
    print(f"  synthesized user speech: {text!r} -> {out} "
          f"({combined.duration:.2f}s + {trailing_silence:.1f}s silence)")
    return out, combined.duration


def load_frames(path: str) -> tuple[list[rtc.AudioFrame], int, int]:
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        assert w.getsampwidth() == 2, "expected a 16-bit wav"
        raw = w.readframes(w.getnframes())
    per = int(sr * FRAME_MS / 1000)
    step = per * ch * 2
    frames = [
        rtc.AudioFrame(
            data=raw[i : i + step],
            sample_rate=sr,
            num_channels=ch,
            samples_per_channel=per,
        )
        for i in range(0, len(raw) - step + 1, step)
    ]
    return frames, sr, ch


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--say",
        default="مرحبا، ما هي خدماتكم؟",
        help="synthesize this as the user's turn (default). Pass '' to use --wav.",
    )
    ap.add_argument("--wav", default=str(HERE.parents[1] / "test-ar.wav"))
    ap.add_argument("--room", default=f"hamsa-demo-{secrets.token_hex(3)}")
    ap.add_argument("--listen", type=float, default=25.0, help="seconds to wait for a reply")
    ap.add_argument("--out", default="/tmp/hamsa-demo-agent.wav", help="save agent audio here")
    ap.add_argument(
        "--settings",
        default=None,
        help='JSON of session settings, as the page would send: '
             '\'{"speaker":"Amira","dialect":"uae","stt_transport":"websocket"}\'',
    )
    args = ap.parse_args()

    speech_only = None
    if args.say:
        args.wav, speech_only = await synthesize_utterance(
            args.say, "/tmp/hamsa-demo-user.wav"
        )

    url = os.environ["LIVEKIT_URL"]
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]
    agent_name = os.environ.get("AGENT_NAME")

    frames, sr, ch = load_frames(args.wav)
    speech_secs = len(frames) * FRAME_MS / 1000
    print(f"room     : {args.room}")
    print(f"speech   : {args.wav} ({speech_secs:.2f}s @ {sr} Hz, {ch}ch)")

    token = (
        api.AccessToken(key, secret)
        .with_identity("sim-user")
        .with_name("sim-user")
        .with_grants(api.VideoGrants(room_join=True, room=args.room))
        .to_jwt()
    )

    room = rtc.Room()
    metrics: list[dict] = []
    transcript: list[tuple[str, str]] = []
    agent_audio: list[rtc.AudioFrame] = []
    agent_joined = asyncio.Event()
    first_audio_at: float | None = None
    done_speaking_at: float | None = None
    # Only latch "first reply audio" once we have actually stopped talking.
    # Without this the greeting still arriving in the background gets timed
    # instead, which reads as a negative latency.
    listening = False
    last_agent_audio_at: float = 0.0

    @room.on("participant_connected")
    def _joined(p: rtc.RemoteParticipant) -> None:
        print(f"  + participant joined: {p.identity}")
        agent_joined.set()

    @room.on("data_received")
    def _data(packet: rtc.DataPacket) -> None:
        try:
            msg = json.loads(packet.data.decode())
        except Exception:
            return
        kind = msg.pop("kind", "?")
        if kind == "metrics":
            metrics.append(msg)
            print(f"  · metrics    {msg}")
        elif kind == "transcript":
            transcript.append((msg.get("role", "?"), msg.get("text", "")))
            print(f"  · {msg.get('role','?'):9} {msg.get('text','')[:70]}")
        elif kind == "ready":
            print(f"  · agent ready {msg}")

    collector: asyncio.Task | None = None

    @room.on("track_subscribed")
    def _track(track: rtc.Track, *_):
        nonlocal collector
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        print("  + subscribed to agent audio")

        async def collect() -> None:
            nonlocal first_audio_at, last_agent_audio_at
            stream = rtc.AudioStream(track)
            async for ev in stream:
                frame = ev.frame
                # LiveKit keeps the track alive with silence; only real samples
                # count as the agent speaking.
                if peak(frame) < 250:
                    continue
                now = time.monotonic()
                last_agent_audio_at = now
                if not listening:
                    continue
                if first_audio_at is None:
                    first_audio_at = now
                    print(f"  * first reply audio {now - done_speaking_at:.2f}s after I stopped")
                agent_audio.append(frame)

        collector = asyncio.create_task(collect())

    await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=True))
    print(f"  connected as {room.local_participant.identity}")

    # Publish our mic track before dispatching, so the agent sees audio the
    # moment it joins.
    source = rtc.AudioSource(sr, ch)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    print("  published mic track")

    if agent_name:
        http_url = url.replace("wss://", "https://").replace("ws://", "http://")
        lkapi = api.LiveKitAPI(http_url, key, secret)
        try:
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=args.room,
                    agent_name=agent_name,
                    # Same channel the token server uses, so this exercises the
                    # settings path the browser goes through.
                    metadata=args.settings or "{}",
                )
            )
            extra = f" with settings {args.settings}" if args.settings else ""
            print(f"  dispatched agent {agent_name!r} to the room{extra}")
        finally:
            await lkapi.aclose()

    try:
        await asyncio.wait_for(agent_joined.wait(), timeout=30)
    except asyncio.TimeoutError:
        print("  ! the agent never joined -- is agent.py running?")
        await room.disconnect()
        return 1

    # Wait out the greeting: quiet for 1.5s means it has finished, so the
    # measured turn is not an interruption.
    print("  waiting for the greeting to finish...")
    quiet_deadline = time.monotonic() + 30
    while time.monotonic() < quiet_deadline:
        await asyncio.sleep(0.25)
        if last_agent_audio_at and time.monotonic() - last_agent_audio_at > 1.5:
            break

    # The turn ends when the *words* stop. The trailing silence after that is
    # what lets the agent detect the turn is over, and it legitimately starts
    # replying during it -- so timing from the end of the published buffer would
    # report a latency of roughly zero.
    speech_end = speech_only if speech_only is not None else speech_secs
    print(f"  speaking {speech_secs:.2f}s ({speech_end:.2f}s of words)...")

    t0 = time.monotonic()
    for i, f in enumerate(frames):
        elapsed = (i + 1) * FRAME_MS / 1000
        if done_speaking_at is None and elapsed >= speech_end:
            done_speaking_at = time.monotonic()
            first_audio_at = None
            agent_audio.clear()
            listening = True
            print(f"  words ended at {done_speaking_at - t0:.2f}s; publishing silence")
        await source.capture_frame(f)
        # capture_frame only queues; pace it so the agent's VAD sees realtime.
        await asyncio.sleep(FRAME_MS / 1000)
    if done_speaking_at is None:
        done_speaking_at = time.monotonic()
        listening = True

    deadline = time.monotonic() + args.listen
    while time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        # Stop once the reply has been quiet for 2s.
        if first_audio_at and time.monotonic() - last_agent_audio_at > 2.0:
            break

    if collector:
        collector.cancel()
    await room.disconnect()

    print("\n--- result ---")
    if first_audio_at and done_speaking_at:
        print(f"stopped speaking -> first agent audio : {first_audio_at - done_speaking_at:.2f}s")
    else:
        print("no agent audio received after speaking")

    if agent_audio:
        combined = rtc.combine_audio_frames(agent_audio)
        with open(args.out, "wb") as f:
            f.write(combined.to_wav_bytes())
        print(f"agent reply audio                    : {combined.duration:.2f}s -> {args.out}")

    for role, text in transcript:
        print(f"{role:9} : {text}")

    merged: dict = {}
    for m in metrics:
        merged.update(m)
    if merged:
        print("metrics reported by the agent:")
        for k in sorted(merged):
            print(f"  {k:24} {merged[k]}")

    return 0 if first_audio_at else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
