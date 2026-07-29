"""Voice and dialect names accepted by the Hamsa TTS endpoint.

The catalogue itself lives in :mod:`livekithamsa.voices`, generated from Hamsa's
published speaker list. This module keeps the convenience aliases.
"""

from __future__ import annotations

from typing import Literal

from .voices import DIALECT_NAMES, VOICES, Voice, find_voice, voices_for

__all__ = [
    "Speaker",
    "Dialect",
    "DIALECTS",
    "SPEAKERS",
    "VOICES",
    "Voice",
    "VOICES_BY_DIALECT",
    "voices_for",
    "find_voice",
]

# Not exhaustive by design: any name in VOICES works, and a cloned voice is passed
# as its UUID, so this is a hint for editors rather than a constraint.
Speaker = str

Dialect = Literal[
    "pls", "leb", "jor", "syr", "ksa", "bah", "uae", "egy", "irq",
    "qat", "kuw", "oma", "msa", "en",
]

DIALECTS: dict[str, str] = dict(DIALECT_NAMES)

SPEAKERS: tuple[str, ...] = tuple(v.name for v in VOICES)

# Every voice tuned for each dialect. Replaces the hand-maintained
# "recommended voices" list, which covered 24 of the 113 voices.
VOICES_BY_DIALECT: dict[str, tuple[str, ...]] = {
    code: tuple(v.name for v in voices_for(code)) for code in DIALECTS
}
