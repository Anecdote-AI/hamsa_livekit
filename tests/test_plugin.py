"""Unit tests for the pure logic in the Hamsa plugin: tokenization, chunking
policy, audio conversion and response parsing. Nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from livekit import rtc
from livekit.agents import stt as agents_stt
from livekithamsa import _audio, tokenizer
from livekithamsa.stt import STT, THINKING_SENTINEL
from livekithamsa.tts import PCM_SAMPLE_RATE, TTS

# ---------------------------------------------------------------- mu-law


def test_ulaw_table_matches_stdlib_reference():
    """The 256-entry table must agree with the canonical G.711 decoder."""
    audioop = pytest.importorskip(
        "audioop", reason="audioop was removed in Python 3.13"
    )
    raw = bytes(range(256))
    assert _audio.ulaw_to_pcm16(raw) == audioop.ulaw2lin(raw, 2)


def test_ulaw_doubles_length():
    assert len(_audio.ulaw_to_pcm16(b"\x00" * 10)) == 20


# ------------------------------------------------------------- resampling


def _frame(sample_rate: int, seconds: float, channels: int = 1) -> rtc.AudioFrame:
    n = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=b"\x00\x01" * n * channels,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=n,
    )


def test_to_wav_resamples_to_16k():
    wav_bytes = _audio.to_wav([_frame(48000, 1.0)])
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getframerate() == _audio.STT_SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        # 1 second in, 1 second out (allow a few samples of filter delay)
        assert abs(w.getnframes() - 16000) < 200


def test_to_wav_passthrough_at_16k():
    wav_bytes = _audio.to_wav([_frame(16000, 0.5)])
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getframerate() == 16000
        assert w.getnframes() == 8000


def test_to_wav_downmixes_stereo():
    wav_bytes = _audio.to_wav([_frame(16000, 0.25, channels=2)])
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getnchannels() == 1


# ------------------------------------------------------------ tokenizer


def test_splits_on_arabic_question_mark():
    """The built-in LiveKit tokenizer misses U+061F, which is why this exists."""
    text = "كيف حالك اليوم؟ أنا بخير الحمد لله وأتمنى أن تكون بخير أيضا."
    sents = tokenizer.ArabicSentenceTokenizer(min_sentence_len=10).tokenize(text)
    assert len(sents) == 2
    assert sents[0].endswith("؟")


def test_builtin_tokenizer_would_not_split_this():
    from livekit.agents.tokenize import basic

    text = "كيف حالك اليوم؟ أنا بخير الحمد لله وأتمنى أن تكون بخير أيضا"
    assert len(basic.SentenceTokenizer(min_sentence_len=10).tokenize(text)) == 1
    assert len(tokenizer.ArabicSentenceTokenizer(min_sentence_len=10).tokenize(text)) == 2


def test_arabic_comma_is_a_chunk_boundary(monkeypatch):
    """The Arabic comma has to count, or a reply punctuated only with `،` runs
    to the word cap and gets cut mid-phrase."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    assert "،" in tokenizer.CLAUSE_SEPARATORS
    text = "واحد اثنان ثلاثة اربعة، خمسة ستة سبعة ثمانية، تسعة عشرة"
    tts_obj = TTS(first_chunk_words=2, first_chunk_max_words=50, chunk_words=2, max_chunk_words=50)
    chunks = asyncio.run(_collect_chunks(tts_obj, [text + " "]))
    assert any(c.rstrip().endswith("،") for c in chunks), chunks


def test_sentence_tokenizer_handles_latin():
    text = "Hello there. How are you doing today? I am quite well."
    sents = tokenizer.ArabicSentenceTokenizer(min_sentence_len=10).tokenize(text)
    assert len(sents) == 3


def test_abbreviations_do_not_end_a_sentence():
    text = "Dr. Ahmad works here and he is very good at what he does."
    sents = tokenizer.ArabicSentenceTokenizer(min_sentence_len=10).tokenize(text)
    assert len(sents) == 1


def test_decimals_do_not_end_a_sentence():
    sents = tokenizer.ArabicSentenceTokenizer(min_sentence_len=5).tokenize(
        "The value is 3.14 and that matters here."
    )
    assert len(sents) == 1


def test_short_fragments_merge_forward():
    """A stray "نعم." must not become its own one-word TTS request."""
    sents = tokenizer.ArabicSentenceTokenizer(min_sentence_len=25).tokenize(
        "نعم. أنا موافق تماما على ما قلته وأريد أن أكمل الحديث."
    )
    assert len(sents) == 1


def test_strip_diacritics():
    assert tokenizer.strip_diacritics("مَرْحَبًا") == "مرحبا"


def test_tokenize_is_lossless_on_words():
    text = "مرحبا بك في هامسا، نحن نقدم خدمات متقدمة؟ شكرا لك."
    joined = "".join(tokenizer.ArabicSentenceTokenizer().tokenize(text))
    for word in ("مرحبا", "هامسا", "خدمات", "شكرا"):
        assert word in joined


# -------------------------------------------------------- chunk policy


async def _collect_chunks(tts_obj: TTS, tokens: list[str]) -> list[str]:
    """Push `tokens` through the real word tokenizer + chunk grouper."""
    stream = tts_obj._opts.word_tokenizer.stream()
    synth = tts_obj.stream()
    try:
        for t in tokens:
            stream.push_text(t)
        stream.end_input()
        return [c async for c in synth._chunk_texts(stream)]
    finally:
        await synth.aclose()


AR_REPLY = [
    "مرحبا بك في هامسا، ",
    "نحن نقدم خدمات الذكاء الاصطناعي المتقدمة للغة العربية ",
    "بجميع لهجاتها المختلفة. ",
    "كيف يمكنني مساعدتك اليوم؟",
]


def test_first_chunk_ends_at_punctuation_not_a_word_count(monkeypatch):
    """The opening chunk must break at punctuation, not at the floor. Cutting on
    the count alone lands mid-phrase and the voice sounds truncated."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    tts_obj = TTS(first_chunk_words=3, first_chunk_max_words=10)
    chunks = asyncio.run(_collect_chunks(tts_obj, AR_REPLY))

    assert len(chunks) >= 2, chunks
    first = chunks[0]
    assert first.rstrip()[-1] in tokenizer.SENTENCE_TERMINATORS + tokenizer.CLAUSE_SEPARATORS, first
    # released past the floor but before the ceiling
    assert 3 <= len(first.split()) <= 10, first
    # the reply's first clause ends on a comma, so that is where it should cut
    assert first == "مرحبا بك في هامسا،", first
    joined = " ".join(chunks)
    for word in ("مرحبا", "هامسا،", "الاصطناعي", "مساعدتك"):
        assert word in joined


def test_first_chunk_floor_is_respected(monkeypatch):
    """Punctuation below the floor must not trigger a one-word request."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    tts_obj = TTS(first_chunk_words=4, first_chunk_max_words=12)
    chunks = asyncio.run(_collect_chunks(tts_obj, ["نعم. أنا موافق تماما على هذا الاقتراح الجيد."]))
    assert len(chunks[0].split()) >= 4, chunks


def test_first_chunk_hard_cut_at_ceiling(monkeypatch):
    """If no punctuation shows up between floor and ceiling, cut at the ceiling."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    text = " ".join(f"كلمة{i}" for i in range(1, 21))  # 20 words, no punctuation
    tts_obj = TTS(first_chunk_words=3, first_chunk_max_words=10)
    chunks = asyncio.run(_collect_chunks(tts_obj, [text + " "]))
    assert len(chunks[0].split()) == 10, chunks[0]


def test_punctuation_past_the_ceiling_does_not_delay_the_first_chunk(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    # comma only at word 14, past the ceiling of 10
    text = " ".join(f"كلمة{i}" for i in range(1, 14)) + "، " + "خمستعش ستعش"
    tts_obj = TTS(first_chunk_words=3, first_chunk_max_words=10)
    chunks = asyncio.run(_collect_chunks(tts_obj, [text]))
    assert len(chunks[0].split()) == 10, chunks[0]


def test_word_count_is_independent_of_diacritics(monkeypatch):
    """A character threshold would not be: the same four words are 18 chars
    undiacritized and 28 with tashkeel."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    plain = "مرحبا بك في هامسا، كيف حالك اليوم؟"
    tashkeel = "مَرْحَبًا بِكَ فِي هَامْسَا، كَيْفَ حَالُكَ الْيَوْمَ؟"
    assert len(plain) != len(tashkeel)

    shapes = []
    for text in (plain, tashkeel):
        tts_obj = TTS(first_chunk_words=3)
        chunks = asyncio.run(_collect_chunks(tts_obj, [text]))
        shapes.append([len(c.split()) for c in chunks])
    assert shapes[0] == shapes[1], shapes


def test_unpunctuated_text_still_gets_released(monkeypatch):
    """LLMs emit long stretches with no sentence-ending punctuation. Without the
    word cap those buffer until the reply ends, which is the stall this plugin
    exists to avoid."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    text = " ".join(f"كلمة{i}" for i in range(40))  # 40 words, zero punctuation
    tts_obj = TTS(
        first_chunk_words=3, first_chunk_max_words=8, chunk_words=5, max_chunk_words=8
    )
    chunks = asyncio.run(_collect_chunks(tts_obj, [text + " "]))

    assert len(chunks) > 2, chunks
    assert max(len(c.split()) for c in chunks) <= 8, chunks
    assert sum(len(c.split()) for c in chunks) == 40


def test_later_chunks_prefer_sentence_boundaries(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    # 6-word sentences, so chunk_words=5 is reached mid-sentence and the chunk
    # must hold until the terminator rather than cutting at exactly 5.
    text = ("واحد اثنان ثلاثة اربعة خمسة ستة. " * 4).strip()
    tts_obj = TTS(first_chunk_words=3, first_chunk_max_words=6, chunk_words=5, max_chunk_words=50)
    chunks = asyncio.run(_collect_chunks(tts_obj, [text + " "]))

    for c in chunks[1:-1]:
        assert c.rstrip()[-1] in ".؟!،", c


def test_single_short_utterance_is_one_chunk(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    tts_obj = TTS()
    chunks = asyncio.run(_collect_chunks(tts_obj, ["نعم"]))
    assert chunks == ["نعم"]


# -------------------------------------------------- STT response parsing


def _stt(monkeypatch) -> STT:
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    return STT(language="ar")


def test_thinking_sentinel_never_becomes_user_speech(monkeypatch):
    """`[THINKING]` is a control signal. Leaking it would put the literal string
    into the conversation as if the user had said it."""
    ev = _stt(monkeypatch)._to_event(
        {"success": True, "data": {"text": THINKING_SENTINEL}},
        language="ar",
        duration=1.0,
    )
    assert ev.type == agents_stt.SpeechEventType.INTERIM_TRANSCRIPT
    assert ev.alternatives == []


def test_real_transcript_is_final(monkeypatch):
    ev = _stt(monkeypatch)._to_event(
        {"success": True, "data": {"text": " مرحبا بك "}},
        language="ar",
        duration=2.5,
    )
    assert ev.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT
    assert ev.alternatives[0].text == "مرحبا بك"
    assert ev.alternatives[0].end_time == 2.5
    assert ev.alternatives[0].language == "ar"


def test_empty_transcript_has_no_alternatives(monkeypatch):
    ev = _stt(monkeypatch)._to_event(
        {"success": True, "data": {"text": ""}}, language="ar", duration=0.5
    )
    assert ev.alternatives == []


def test_malformed_response_does_not_raise(monkeypatch):
    ev = _stt(monkeypatch)._to_event({"unexpected": True}, language="ar", duration=0.0)
    assert ev.alternatives == []


# ------------------------------------------------- HamsaEOS event invariants


class _FakeVAD:
    """Minimal VAD stand-in; the EOS stream only needs .stream()."""

    def __init__(self, events):
        self._events = events
        self.capabilities = None

    def stream(self):
        return _FakeVADStream(self._events)


class _FakeVADStream:
    def __init__(self, events):
        self._events = list(events)

    def push_frame(self, frame):  # noqa: D102
        pass

    def flush(self):  # noqa: D102
        pass

    def end_input(self):  # noqa: D102
        pass

    async def aclose(self):  # noqa: D102
        pass

    def __aiter__(self):
        async def gen():
            for ev in self._events:
                yield ev

        return gen()


def _emitted_types(monkeypatch, probe_event):
    """Drive _EOSStream's emit path with a probe result and collect event types."""
    from livekithamsa import eos as eos_mod

    monkeypatch.setenv("HAMSA_API_KEY", "test")
    eos = eos_mod.HamsaEOS(stt=STT(language="ar"), vad=_FakeVAD([]))

    async def run():
        stream = eos.stream()
        try:
            sent = []
            monkeypatch.setattr(
                stream._event_ch, "send_nowait", lambda ev: sent.append(ev)
            )

            async def fake_recognize(**kwargs):
                return probe_event

            monkeypatch.setattr(eos._probe_stt, "recognize", fake_recognize)
            stream._frames = [_frame(16000, 1.0)]
            await stream._probe(stream._turn)
            return sent
        finally:
            await stream.aclose()
            await eos.aclose()

    return asyncio.run(run())


def test_thinking_probe_emits_no_empty_transcript_event(monkeypatch):
    """A transcript event with no alternatives kills LiveKit's _stt_consumer:
    on_interim_transcript reads ev.alternatives[0].language unguarded, and once
    that task dies the session stops processing STT events for the whole call."""
    empty = agents_stt.SpeechEvent(
        type=agents_stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[]
    )
    sent = _emitted_types(monkeypatch, empty)
    assert sent == [], [e.type for e in sent]


def test_real_probe_commits_final_and_end_of_speech(monkeypatch):
    good = agents_stt.SpeechEvent(
        type=agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[agents_stt.SpeechData(language="ar", text="مرحبا")],
    )
    sent = _emitted_types(monkeypatch, good)
    assert [e.type for e in sent] == [
        agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
        agents_stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert sent[0].alternatives[0].text == "مرحبا"


def test_whitespace_only_transcript_is_not_committed(monkeypatch):
    blank = agents_stt.SpeechEvent(
        type=agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[agents_stt.SpeechData(language="ar", text="   ")],
    )
    sent = _emitted_types(monkeypatch, blank)
    assert sent == [], [e.type for e in sent]


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.delenv("HAMSA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        STT()
    with pytest.raises(ValueError, match="API key"):
        TTS()


def test_sibling_shares_session_and_flips_eos(monkeypatch):
    base = _stt(monkeypatch)
    probe = base._sibling(eos_enabled=True)
    assert probe._opts.eos_enabled is True
    assert base._opts.eos_enabled is False
    assert probe._opts.api_key == base._opts.api_key
    # Both must resolve to the same pooled session, or each pays a TLS handshake.
    assert probe._ensure_session.__self__ is base


# ------------------------------------------------------- TTS audio format


def test_pcm_sample_rate_is_declared_correctly(monkeypatch):
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    assert TTS(mulaw=False).sample_rate == PCM_SAMPLE_RATE


def test_mulaw_declares_8k_not_16k(monkeypatch):
    """mulaw=True returns 8 kHz G.711; declaring 16 kHz plays it back wrong."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    assert TTS(mulaw=True).sample_rate == _audio.MULAW_SAMPLE_RATE


def test_bogus_sample_rate_is_ignored_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    tts_obj = TTS(sample_rate=24000)
    assert tts_obj.sample_rate == PCM_SAMPLE_RATE
    assert any("wrong speed" in r.getMessage() for r in caplog.records)


def test_streams_get_an_independent_options_copy(monkeypatch):
    """update_options on one stream must not retune every other live stream."""
    monkeypatch.setenv("HAMSA_API_KEY", "test")
    tts_obj = TTS(speaker="Ali")

    async def check() -> None:
        a, b = tts_obj.stream(), tts_obj.stream()
        try:
            a.update_options(speaker="Lana")
            assert a._opts.speaker == "Lana"
            assert b._opts.speaker == "Ali"
            assert tts_obj._opts.speaker == "Ali"
        finally:
            await a.aclose()
            await b.aclose()

    asyncio.run(check())
