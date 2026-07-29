"""Arabic-aware tokenization for TTS streaming.

LiveKit's built-in sentence tokenizer only terminates on ``[.!?。！？]``. Arabic
text routinely ends sentences with ``؟`` (question mark) and separates clauses
with ``،`` (comma) and ``؛`` (semicolon), none of which appear in that set. The
practical effect on a voice agent is severe: an LLM reply written in Arabic
punctuation never produces a sentence boundary, so nothing is handed to the TTS
until the LLM has finished generating and the stream is flushed. First-audio
latency becomes "time to full completion" instead of "time to first clause".

This module provides a tokenizer that understands both scripts, and the
punctuation sets the TTS uses to decide where a synthesis chunk may end.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from livekit.agents import tokenize
from livekit.agents.tokenize import token_stream

__all__ = [
    "ArabicSentenceTokenizer",
    "SENTENCE_TERMINATORS",
    "CLAUSE_SEPARATORS",
    "strip_diacritics",
    "split_sentences",
]

# End-of-sentence marks, Arabic + Latin + CJK full-width (LLMs mix them).
#   ؟ U+061F arabic question mark   ۔ U+06D4 arabic full stop
#   ؛ U+061B arabic semicolon
SENTENCE_TERMINATORS = ".!?؟۔।。！？…"

# Marks that end a clause but not a sentence. The TTS treats these as
# acceptable chunk boundaries once a chunk is already long enough, so a chunk
# that runs on never has to be cut at an arbitrary word.
CLAUSE_SEPARATORS = "،,;؛:—"

_TERM_CLASS = re.escape(SENTENCE_TERMINATORS)
_CLAUSE_CLASS = re.escape(CLAUSE_SEPARATORS)

# A sentence = anything up to a terminator, plus any trailing quotes/brackets
# and the whitespace that follows it.
_SENTENCE_RE = re.compile(
    rf"[^{_TERM_CLASS}]*[{_TERM_CLASS}]+[\"'”’)\]]*\s*|[^{_TERM_CLASS}]+$"
)


# Arabic combining diacritics (tashkeel). Hamsa's TTS handles undiacritized text
# fine and diacritics only inflate the payload, but they must not break matching
# when callers pre-process text.
_DIACRITICS_RE = re.compile(r"[ً-ٰٟۖ-ۭ࣓-ࣿ]")

# Abbreviations and numeric patterns whose "." must not end a sentence.
_PROTECTED = (
    re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Inc|Ltd|Co|vs|etc|e\.g|i\.e)\."),
    re.compile(r"\d+\.\d+"),  # 3.14
    re.compile(r"\b[A-Z]\."),  # initials: J. Smith
)

_PLACEHOLDER = "\x00"


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel from ``text``."""
    return _DIACRITICS_RE.sub("", text)


def _protect(text: str) -> str:
    """Replace dots that are not sentence ends with a placeholder byte."""
    for pattern in _PROTECTED:
        text = pattern.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    return text


def _restore(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


def _split(
    text: str, pattern: re.Pattern[str], min_len: int, retain_format: bool
) -> list[tuple[str, int, int]]:
    """Split ``text`` with ``pattern``, merging fragments shorter than
    ``min_len`` into the following one so the TTS is not handed one-word
    requests. Returns ``(token, start, end)`` offsets into the original text."""
    protected = _protect(text)

    out: list[tuple[str, int, int]] = []
    buf = ""
    buf_start = 0

    for match in pattern.finditer(protected):
        raw = match.group(0)
        if not raw.strip():
            continue
        if not buf:
            buf_start = match.start()
        buf += raw
        if len(buf.strip()) >= min_len:
            token = _restore(buf if retain_format else buf.strip())
            out.append((token, buf_start, buf_start + len(buf)))
            buf = ""

    if buf.strip():
        token = _restore(buf if retain_format else buf.strip())
        out.append((token, buf_start, buf_start + len(buf)))

    return out


def split_sentences(
    text: str, *, min_sentence_len: int = 20, retain_format: bool = False
) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences, honouring Arabic and Latin punctuation."""
    return _split(text, _SENTENCE_RE, min_sentence_len, retain_format)




@dataclass
class _Options:
    min_token_len: int
    stream_context_len: int
    retain_format: bool


class ArabicSentenceTokenizer(tokenize.SentenceTokenizer):
    """Sentence tokenizer that terminates on Arabic punctuation as well as Latin.

    Args:
        min_sentence_len: fragments shorter than this are merged forward, so the
            TTS is never asked to synthesize a stray word.
        stream_context_len: how much text the streaming tokenizer holds back
            before deciding a boundary is real.
        retain_format: keep original whitespace instead of stripping it.
    """

    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 6,
        retain_format: bool = False,
    ) -> None:
        self._opts = _Options(min_sentence_len, stream_context_len, retain_format)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            tok[0]
            for tok in split_sentences(
                text,
                min_sentence_len=self._opts.min_token_len,
                retain_format=self._opts.retain_format,
            )
        ]

    def stream(self, *, language: str | None = None) -> tokenize.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                split_sentences,
                min_sentence_len=self._opts.min_token_len,
                retain_format=self._opts.retain_format,
            ),
            min_token_len=self._opts.min_token_len,
            min_ctx_len=self._opts.stream_context_len,
        )
