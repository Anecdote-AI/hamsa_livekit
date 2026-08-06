"""End-of-speech driven turn detection for Hamsa STT.

The default way to drive a non-streaming STT from LiveKit is
``stt.StreamAdapter``: wait for the VAD to observe ``min_silence_duration`` of
quiet, then transcribe the turn once. The silence threshold has to be set
pessimistically -- long enough that a mid-sentence breath is not mistaken for the
end of a turn -- and every single turn pays that wait before the LLM is even
asked for a reply.

Hamsa's STT ships a turn-detection model behind the ``isEosEnabled`` flag, and it
answers the question the timer is only guessing at: *has this person finished
talking?* This module runs the VAD with a deliberately short silence threshold
and asks the server at each pause:

* real transcript back -> the turn is over. Commit immediately, no extra wait.
* ``[THINKING]`` back -> the speaker is mid-utterance. Keep the audio, stay
  listening, and let a longer timer act as the backstop.

So the common case drops the fixed silence wait entirely, and the timer only
matters when the model is unsure. The backstop is a single eos-off request, which
always returns a transcript.

Usage::

    stt = livekithamsa.HamsaEOS(
        stt=livekithamsa.STT(language="ar"),
        vad=silero.VAD.load(min_silence_duration=0.2),
    )

Pair it with ``turn_detection="stt"`` on the ``AgentSession`` so the session
commits on this plugin's ``FINAL_TRANSCRIPT`` rather than on its own timer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from livekit import rtc
from livekit.agents import APIConnectOptions, utils, vad

# Aliased because `stt` is also the name of this class's main constructor
# argument, and the module reference is needed inside those methods.
from livekit.agents import stt as agents_stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._audio import STT_SAMPLE_RATE
from .log import logger
from .stt import STT as HamsaSTT

__all__ = ["HamsaEOS"]

# Transcripts that are never worth committing as a user turn.
DEFAULT_IGNORED_TEXTS: tuple[str, ...] = ("", "no-text", "[thinking]")


class HamsaEOS(agents_stt.STT):
    """Streaming STT that uses Hamsa's end-of-speech model for turn detection.

    Args:
        stt: the :class:`~livekithamsa.stt.STT` to transcribe with. Its
            ``eos_enabled`` setting is ignored; this class drives the flag per
            request.
        vad: voice activity detector used to find pauses. Configure it with a
            **short** ``min_silence_duration`` (0.15-0.25 s): its job here is
            only to say "there was a pause, go ask the server", not to decide
            the turn is over. A long threshold puts the old wait right back.
        max_turn_silence: backstop. If the model keeps answering "still going"
            and no further speech arrives within this many seconds of the last
            pause, transcribe once with eos off and commit that.
        min_speech_duration: pauses with less than this much accumulated audio
            are not sent to the server at all. Avoids a round trip per cough,
            and the endpoint is unreliable on very short clips.
        eos_threshold: trailing-silence threshold handed to the server's
            end-of-speech model.
        ignored_texts: transcripts to drop instead of committing, compared
            case-insensitively after stripping.
        conn_options: connection options for the wrapped requests.
    """

    def __init__(
        self,
        *,
        stt: HamsaSTT,
        vad: vad.VAD,
        max_turn_silence: float = 2.0,
        min_speech_duration: float = 0.2,
        eos_threshold: float = 0.3,
        ignored_texts: Sequence[str] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            capabilities=agents_stt.STTCapabilities(
                streaming=True, interim_results=True
            )
        )
        self._vad = vad
        self._conn_options = conn_options
        self._max_turn_silence = max_turn_silence
        self._min_speech_duration = min_speech_duration
        self._ignored = {
            t.strip().lower() for t in (ignored_texts or DEFAULT_IGNORED_TEXTS)
        }

        # The whole mechanism is the `[THINKING]` sentinel, and the websocket
        # endpoint does not produce it -- measured 0/15 vs 9/15 over HTTP on
        # identical audio. On a socket every probe would look like a finished
        # turn, so the agent would commit on the first fragment it heard and cut
        # people off mid-sentence. Fail loudly rather than degrade silently.
        if getattr(stt, "transport", "http") != "http":
            raise ValueError(
                "HamsaEOS requires the HTTP transport: the websocket endpoint "
                "does not return the [THINKING] end-of-speech signal that turn "
                "detection is built on. Pass STT(transport='http') here, and use "
                "transport='websocket' for TTS or for plain STT instead."
            )

        # Two instances, one per value of the eos flag, so a probe and a
        # finalization can never race over the same mutable option. See
        # STT._sibling.
        self._probe_stt = stt._sibling(eos_enabled=True, eos_threshold=eos_threshold)
        self._final_stt = stt._sibling(eos_enabled=False)

        for inner in (self._probe_stt, self._final_stt):
            inner.on("metrics_collected", self._forward_metrics)

    def _forward_metrics(self, *args: object, **kwargs: object) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    @property
    def wrapped_stt(self) -> HamsaSTT:
        return self._final_stt

    def update_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> None:
        if is_given(language):
            self._probe_stt.update_options(language=language)
            self._final_stt.update_options(language=language)

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> agents_stt.SpeechEvent:
        return await self._final_stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _EOSStream:
        return _EOSStream(
            eos=self,
            language=language,
            conn_options=conn_options if is_given(conn_options) else self._conn_options,
        )

    async def aclose(self) -> None:
        await self._probe_stt.aclose()
        await self._final_stt.aclose()
        await super().aclose()


class _EOSStream(agents_stt.RecognizeStream):
    """Drives the VAD, probes the server at each pause, and emits turn events."""

    def __init__(
        self,
        *,
        eos: HamsaEOS,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        self._eos = eos
        self._language = language
        self._vad_stream = eos._vad.stream()

        self._frames: list[rtc.AudioFrame] = []
        self._in_speech = False
        # Bumped whenever new speech starts. A pending backstop timer or an
        # in-flight probe compares against it and drops its result if the user
        # has resumed since it was scheduled.
        self._turn = 0
        self._probe_task: asyncio.Task[None] | None = None
        self._timer_task: asyncio.Task[None] | None = None

        # LAST, deliberately: `RecognizeStream.__init__` schedules `_main_task`,
        # which runs `_run` and -- on every exit path -- `_cancel_pending`.
        # Everything that touches must already exist, or a stream torn down early
        # raises AttributeError inside its own cleanup and buries the real error.
        # Reachable today: `eos._vad.stream()` above is the first thing that can
        # throw after the task exists.
        #
        # `sample_rate` resamples once here, on ingest, rather than per request:
        # the turn buffer is re-sent on every pause, so resampling at request time
        # would redo the same work for the same audio on each probe.
        super().__init__(
            stt=eos, conn_options=conn_options, sample_rate=STT_SAMPLE_RATE
        )

    # -- helpers ---------------------------------------------------------

    def _buffered_duration(self) -> float:
        return sum(f.samples_per_channel / f.sample_rate for f in self._frames)

    def _reset_turn(self) -> None:
        self._frames = []
        self._in_speech = False

    def _cancel(self, task: asyncio.Task[None] | None) -> None:
        if task is not None and not task.done():
            task.cancel()

    def _cancel_pending(self) -> None:
        self._cancel(self._probe_task)
        self._probe_task = None
        self._cancel(self._timer_task)
        self._timer_task = None

    def _is_ignored(self, text: str) -> bool:
        return text.strip().lower() in self._eos._ignored

    def _emit(
        self,
        event_type: agents_stt.SpeechEventType,
        alternatives: list[agents_stt.SpeechData] | None = None,
    ) -> bool:
        """Send an event, refusing to emit a transcript event with nothing in it.

        This is an invariant, not a nicety. A streaming STT writes straight into
        the event channel with none of the filtering ``stt.StreamAdapter`` does
        for the non-streaming path, and LiveKit's consumer reads
        ``ev.alternatives[0]`` unguarded (``on_interim_transcript`` ->
        ``ev.alternatives[0].language``). One empty transcript event raises
        IndexError inside ``_stt_consumer``, and once that task dies the session
        stops processing STT events entirely -- transcription is dead for the
        rest of the call, with only a single logged traceback to show for it.
        """
        alts = list(alternatives or [])
        carries_transcript = event_type in (
            agents_stt.SpeechEventType.INTERIM_TRANSCRIPT,
            agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
            agents_stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
        )
        if carries_transcript and not (alts and alts[0].text.strip()):
            logger.debug("Hamsa EOS: suppressing empty %s", event_type.value)
            return False

        self._event_ch.send_nowait(
            agents_stt.SpeechEvent(type=event_type, alternatives=alts)
        )
        return True

    def _commit(self, alternatives: list[agents_stt.SpeechData], *, why: str) -> None:
        text = alternatives[0].text if alternatives else ""
        if not self._emit(agents_stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives):
            # Nothing worth committing; leave the turn open rather than sending a
            # bare END_OF_SPEECH the session would pair with no transcript.
            self._reset_turn()
            return
        logger.debug("Hamsa EOS: committed turn via %s: %r", why, text)
        self._emit(agents_stt.SpeechEventType.END_OF_SPEECH, alternatives)
        self._reset_turn()

    # -- the two async paths ---------------------------------------------

    async def _probe(self, turn: int) -> None:
        """Ask the server whether the turn is over, and commit if it is."""
        buffer = list(self._frames)
        started = time.monotonic()
        try:
            event = await self._eos._probe_stt.recognize(
                buffer=buffer,
                language=self._language,
                conn_options=self._conn_options,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Don't drop the turn because one probe failed; the backstop timer is
            # still armed and will finalize with an eos-off request.
            logger.exception("Hamsa EOS: probe failed, leaving it to the backstop")
            return

        if turn != self._turn:
            logger.debug("Hamsa EOS: discarding probe, user resumed speaking")
            return

        elapsed = (time.monotonic() - started) * 1000
        alts = list(event.alternatives)

        if not alts:
            # "[THINKING]" (still talking) or silence. There is no transcript to
            # report, so nothing is emitted -- the turn simply stays open and the
            # backstop keeps its deadline.
            logger.debug("Hamsa EOS: turn not over after %.0f ms probe", elapsed)
            return

        if self._is_ignored(alts[0].text):
            logger.debug("Hamsa EOS: ignoring %r", alts[0].text)
            self._cancel(self._timer_task)
            self._timer_task = None
            self._reset_turn()
            return

        # The model says the speaker is done. This is the fast path: commit now
        # instead of waiting out a silence timer.
        self._cancel(self._timer_task)
        self._timer_task = None
        self._commit(alts, why=f"eos model ({elapsed:.0f} ms)")

    async def _backstop(self, turn: int) -> None:
        """If the model never concedes the turn is over, finalize it ourselves."""
        try:
            await asyncio.sleep(self._eos._max_turn_silence)
            # Let anything queued in this same tick (a fresh START_OF_SPEECH, a
            # probe result) run before we act on a stale view of the turn.
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

        if turn != self._turn or not self._frames:
            return

        logger.debug(
            "Hamsa EOS: backstop after %.2fs, finalizing with eos off",
            self._eos._max_turn_silence,
        )
        buffer = list(self._frames)
        try:
            event = await self._eos._final_stt.recognize(
                buffer=buffer,
                language=self._language,
                conn_options=self._conn_options,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Hamsa EOS: backstop transcription failed, dropping turn")
            self._reset_turn()
            return

        if turn != self._turn:
            return

        alts = list(event.alternatives)
        if not alts or self._is_ignored(alts[0].text):
            self._reset_turn()
            return

        self._commit(alts, why="backstop")

    # -- main loop -------------------------------------------------------

    async def _run(self) -> None:
        async def forward_audio() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._vad_stream.flush()
                    continue
                self._vad_stream.push_frame(data)
            self._vad_stream.end_input()

        async def consume_vad() -> None:
            async for event in self._vad_stream:
                if event.type == vad.VADEventType.START_OF_SPEECH:
                    # New speech invalidates any pending decision about the
                    # previous pause, but keeps the audio: this is the same turn
                    # continuing, so the buffer is not cleared.
                    self._turn += 1
                    self._cancel_pending()
                    self._in_speech = True
                    self._emit(agents_stt.SpeechEventType.START_OF_SPEECH)

                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    if not self._in_speech:
                        continue
                    self._in_speech = False
                    # Each event carries the speech run that just ended, and the
                    # VAD resets its buffer afterwards, so extending accumulates
                    # run after run without duplication. The silence between runs
                    # is dropped, which is fine (better, even) for transcription.
                    self._frames.extend(event.frames)

                    turn = self._turn
                    # Very short blips are usually a click or a breath. Skip the
                    # round trip, but keep the audio: a genuine one-word answer
                    # ("نعم") lands here too, and the backstop still commits it.
                    if self._buffered_duration() >= self._eos._min_speech_duration:
                        self._probe_task = asyncio.create_task(
                            self._probe(turn), name="hamsa-eos-probe"
                        )
                    else:
                        logger.debug(
                            "Hamsa EOS: only %.2fs buffered, skipping the probe",
                            self._buffered_duration(),
                        )
                    self._timer_task = asyncio.create_task(
                        self._backstop(turn), name="hamsa-eos-backstop"
                    )

        tasks = [
            asyncio.create_task(forward_audio(), name="hamsa-eos-forward"),
            asyncio.create_task(consume_vad(), name="hamsa-eos-vad"),
        ]
        try:
            await asyncio.gather(*tasks)
            # The audio ended cleanly. A probe or backstop may still be deciding
            # the last turn; cancelling now would throw that turn away, so give
            # the in-flight work a bounded chance to commit it.
            await self._settle()
        finally:
            self._cancel_pending()
            await utils.aio.cancel_and_wait(*tasks)
            await self._vad_stream.aclose()

    async def _settle(self) -> None:
        """Let a pending decision about the final turn finish.

        Bounded by `max_turn_silence` plus one request's worth of slack so a
        wedged connection cannot hold the stream open indefinitely.
        """
        pending = [t for t in (self._probe_task, self._timer_task) if t and not t.done()]
        if not pending:
            return
        budget = self._eos._max_turn_silence + self._conn_options.timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=budget
            )
        except asyncio.TimeoutError:
            logger.debug("Hamsa EOS: final turn did not settle within %.1fs", budget)
