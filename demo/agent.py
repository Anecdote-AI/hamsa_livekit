"""Demo LiveKit agent for the Hamsa plugin.

Run the token server in one terminal and this in another:

    python serve.py          # http://localhost:8080
    python agent.py dev

Every measurement the session produces is forwarded to the browser as a data
message, so the page can show where the time between "the user stopped talking"
and "the agent starts talking" actually goes.

Set ``MODE=baseline`` to run the same agent *without* the Hamsa speedups --
LiveKit's stock VAD turn detection and one TTS request for the whole reply -- so
the two can be compared back to back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

import livekithamsa
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, metrics
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("hamsa-demo")

# "hamsa"    -> EOS turn detection (HTTP-only) + chunked, prefetched TTS
# "vad"      -> VAD turn detection, same fast TTS. No EOS, so either transport
#               may be used for STT -- this is the mode for benchmarking the
#               websocket end to end.
# "baseline" -> stock VAD turn detection + one TTS request per reply
MODE = os.environ.get("MODE", "hamsa").lower()

SPEAKER = os.environ.get("HAMSA_SPEAKER", "Ali")
DIALECT = os.environ.get("HAMSA_DIALECT", "pls")
LANGUAGE = os.environ.get("HAMSA_LANGUAGE", "ar")
DATA_TOPIC = "hamsa-demo"

# Transport per component. The socket is ~29% faster per request, but it does not
# implement the [THINKING] end-of-speech signal, so HamsaEOS cannot use it -- STT
# is pinned to HTTP whenever turn detection is on.
TTS_TRANSPORT = os.environ.get("TTS_TRANSPORT", "websocket").lower()
STT_TRANSPORT = os.environ.get("STT_TRANSPORT", "http").lower()

INSTRUCTIONS = """
أنت مساعد صوتي ودود يتحدث العربية. أجب بإيجاز شديد — جملة أو جملتين على الأكثر،
لأن كلامك يُنطق بصوت مسموع. لا تستخدم قوائم أو رموزًا أو تنسيقًا.
""".strip()


class HamsaAssistant(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)


@dataclass
class Settings:
    """One session's configuration: environment defaults, overridden by whatever
    the page sent as dispatch metadata."""

    mode: str = MODE
    stt_transport: str = STT_TRANSPORT
    tts_transport: str = TTS_TRANSPORT
    speaker: str = SPEAKER
    dialect: str = DIALECT
    language: str = LANGUAGE
    instructions: str = INSTRUCTIONS
    expressiveness: int = 1
    vad_min_silence: float | None = None
    vad_threshold: float = 0.5

    @classmethod
    def from_job(cls, raw: str | None) -> Settings:
        self = cls()
        if not raw:
            return self
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("job metadata was not JSON; using defaults")
            return self
        if not isinstance(data, dict):
            return self

        def text(key: str, current: str) -> str:
            value = data.get(key)
            return value.strip() if isinstance(value, str) and value.strip() else current

        def number(key: str, current, cast, lo, hi):
            if key not in data:
                return current
            try:
                value = cast(data[key])
            except (TypeError, ValueError):
                logger.warning("ignoring %s=%r: not a number", key, data[key])
                return current
            return max(lo, min(hi, value))

        for key in ("stt_transport", "tts_transport"):
            value = data.get(key)
            if value in ("http", "websocket"):
                setattr(self, key, value)
            elif value is not None:
                logger.warning("ignoring %s=%r", key, value)

        self.speaker = text("speaker", self.speaker)
        self.dialect = text("dialect", self.dialect)
        self.instructions = text("instructions", self.instructions)
        self.expressiveness = number("expressiveness", self.expressiveness, int, 0, 2)
        self.vad_min_silence = number("vad_min_silence", self.vad_min_silence, float, 0.05, 3.0)
        self.vad_threshold = number("vad_threshold", self.vad_threshold, float, 0.05, 0.95)
        return self

    @property
    def effective_mode(self) -> str:
        """The transport choice wins over the mode name.

        Putting STT on the socket and using EOS turn detection are mutually
        exclusive: the socket does not return ``[THINKING]``. Rather than silently
        overriding the transport that was asked for, drop the feature that cannot
        work over it and fall back to VAD.
        """
        if self.mode == "hamsa" and self.stt_transport == "websocket":
            return "vad"
        return self.mode

    @property
    def silence(self) -> float:
        """VAD silence threshold. Under EOS the VAD only has to notice a pause,
        so it can be short; without EOS it *is* the turn detector."""
        if self.vad_min_silence is not None:
            return self.vad_min_silence
        env = os.environ.get("VAD_SILENCE")
        if env:
            return float(env)
        return 0.2 if self.effective_mode == "hamsa" else 0.55


def build_session(cfg: Settings) -> AgentSession:
    llm = openai.LLM(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    mode = cfg.effective_mode
    if mode != cfg.mode:
        logger.warning(
            "stt_transport=websocket: disabling EOS turn detection, using VAD. "
            "The socket does not return the [THINKING] end-of-speech signal."
        )

    if mode == "baseline":
        # A long silence threshold is what stock VAD turn detection needs in
        # order not to cut people off mid-sentence -- and every turn waits it out.
        vad = silero.VAD.load(
            min_silence_duration=cfg.silence, activation_threshold=cfg.vad_threshold
        )
        return AgentSession(
            stt=livekithamsa.STT(language=cfg.language, transport=cfg.stt_transport),
            llm=llm,
            tts=livekithamsa.TTS(
                speaker=cfg.speaker,
                dialect=cfg.dialect,
                transport=cfg.tts_transport,
                expressiveness=cfg.expressiveness,
                # Never splits: one request for the whole reply, sent only once
                # the LLM has finished.
                first_chunk_words=10_000,
                first_chunk_max_words=10_000,
                chunk_words=10_000,
                max_chunk_words=10_000,
                prefetch=0,
            ),
            vad=vad,
            turn_detection="vad",
        )

    if mode == "vad":
        # No EOS: the VAD silence threshold *is* the turn detector, so it has to
        # be long enough not to clip people.
        vad = silero.VAD.load(
            min_silence_duration=cfg.silence, activation_threshold=cfg.vad_threshold
        )
        return AgentSession(
            stt=livekithamsa.STT(language=cfg.language, transport=cfg.stt_transport),
            llm=llm,
            tts=livekithamsa.TTS(
                speaker=cfg.speaker,
                dialect=cfg.dialect,
                transport=cfg.tts_transport,
                expressiveness=cfg.expressiveness,
            ),
            vad=vad,
            turn_detection="vad",
        )

    # Here the VAD only has to notice a pause and trigger a probe, so the
    # threshold is short; Hamsa's end-of-speech model makes the actual decision.
    vad = silero.VAD.load(
        min_silence_duration=cfg.silence, activation_threshold=cfg.vad_threshold
    )
    return AgentSession(
        stt=livekithamsa.HamsaEOS(
            stt=livekithamsa.STT(language=cfg.language, transport=cfg.stt_transport),
            vad=vad,
            max_turn_silence=2.0,
        ),
        llm=llm,
        tts=livekithamsa.TTS(
            speaker=cfg.speaker,
            dialect=cfg.dialect,
            transport=cfg.tts_transport,
            expressiveness=cfg.expressiveness,
        ),
        vad=vad,
        turn_detection="stt",
    )


def metric_payload(m: object) -> dict | None:
    """Flatten the metrics the HUD cares about. Returns None for the rest."""
    if isinstance(m, metrics.EOUMetrics):
        # end_of_utterance_delay: how long after the user actually stopped before
        # the turn was committed. This is the number EOS detection shrinks.
        return {
            "eou_delay_ms": round(m.end_of_utterance_delay * 1000),
            "transcription_delay_ms": round(m.transcription_delay * 1000),
        }
    if isinstance(m, metrics.LLMMetrics):
        return {"llm_ttft_ms": round(m.ttft * 1000)}
    if isinstance(m, metrics.TTSMetrics):
        return {
            "tts_ttfb_ms": round(m.ttfb * 1000),
            "tts_audio_ms": round(m.audio_duration * 1000),
        }
    if isinstance(m, metrics.STTMetrics):
        return {
            "stt_request_ms": round(m.duration * 1000),
            "stt_audio_ms": round(m.audio_duration * 1000),
        }
    return None


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    cfg = Settings.from_job(getattr(ctx.job, "metadata", None))
    logger.info(
        "session: mode=%s stt=%s tts=%s voice=%s/%s silence=%.2fs threshold=%.2f",
        cfg.effective_mode, cfg.stt_transport, cfg.tts_transport,
        cfg.speaker, cfg.dialect, cfg.silence, cfg.vad_threshold,
    )
    session = build_session(cfg)
    usage = metrics.UsageCollector()

    async def publish(kind: str, payload: dict) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({"kind": kind, **payload}), topic=DATA_TOPIC
            )
        except Exception:
            logger.debug("could not publish %s", kind, exc_info=True)

    def publish_soon(kind: str, payload: dict) -> None:
        """Event handlers are sync; publishing is not."""
        asyncio.create_task(publish(kind, payload))

    @session.on("metrics_collected")
    def _on_metrics(ev: agents.MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)
        payload = metric_payload(ev.metrics)
        if payload is not None:
            publish_soon("metrics", {**payload, "mode": cfg.effective_mode})

    @session.on("user_input_transcribed")
    def _on_transcript(ev: agents.UserInputTranscribedEvent) -> None:
        if ev.is_final and ev.transcript.strip():
            publish_soon("transcript", {"role": "user", "text": ev.transcript})

    @session.on("conversation_item_added")
    def _on_item(ev: agents.ConversationItemAddedEvent) -> None:
        item = ev.item
        role = getattr(item, "role", None)
        if role != "assistant":
            return  # user lines already arrive via user_input_transcribed
        text = getattr(item, "text_content", None) or ""
        if text.strip():
            publish_soon("transcript", {"role": "assistant", "text": text})

    async def log_usage() -> None:
        logger.info("usage: %s", usage.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=HamsaAssistant(cfg.instructions),
        room_input_options=RoomInputOptions(),
    )

    # Open any websocket before the first turn needs it, so turn 1 does not pay
    # the handshake.
    for component in (session.tts, getattr(session.stt, "wrapped_stt", session.stt)):
        warmup = getattr(component, "warmup", None)
        if warmup is not None:
            try:
                await warmup()
            except Exception:
                logger.warning("warmup failed for %s", type(component).__name__, exc_info=True)

    await publish(
        "ready",
        {
            "mode": cfg.effective_mode,
            "requested_mode": cfg.mode,
            "speaker": cfg.speaker,
            "dialect": cfg.dialect,
            "stt_transport": cfg.stt_transport,
            "tts_transport": cfg.tts_transport,
            "eos": cfg.effective_mode == "hamsa",
            "vad_min_silence": round(cfg.silence, 3),
            "vad_threshold": cfg.vad_threshold,
            "expressiveness": cfg.expressiveness,
        },
    )
    await session.generate_reply(
        instructions="رحّب بالمستخدم بجملة قصيرة جدًا واسأله كيف يمكنك مساعدته."
    )


if __name__ == "__main__":
    opts = agents.WorkerOptions(entrypoint_fnc=entrypoint)

    # Without an agent name a worker gets *automatic* dispatch: LiveKit hands it
    # a job for every new room on the project. That is convenient on a scratch
    # project and dangerous on a shared one, where this demo would start
    # answering real traffic. Setting AGENT_NAME switches the worker to explicit
    # dispatch, so it only runs jobs addressed to that name.
    if agent_name := os.environ.get("AGENT_NAME"):
        opts.agent_name = agent_name
        logger.info("worker requires explicit dispatch as %r", agent_name)

    agents.cli.run_app(opts)
