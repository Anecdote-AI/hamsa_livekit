"""The voice catalogue, generated from Hamsa's published speaker list.

113 voices across 14 dialects. A voice name is passed as
``speaker``; a cloned voice is passed the same way using its UUID instead, so
anything not in this list is still valid.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["Voice", "VOICES", "DIALECT_NAMES", "voices_for", "find_voice"]


class Voice(NamedTuple):
    name: str
    dialect: str
    dialect_name: str
    language: str


DIALECT_NAMES: dict[str, str] = {
    "bah": "Bahraini",
    "egy": "Egyptian",
    "en": "English",
    "irq": "Iraqi",
    "jor": "Jordanian",
    "ksa": "Saudi",
    "kuw": "Kuwaiti",
    "leb": "Lebanese",
    "msa": "Modern Standard Arabic",
    "oma": "Omani",
    "pls": "Palestinian",
    "qat": "Qatari",
    "syr": "Syrian",
    "uae": "Emirati",
}

VOICES: tuple[Voice, ...] = (
    Voice("Eyad", "bah", "Bahraini", "Arabic"),
    Voice("Hafsa", "bah", "Bahraini", "Arabic"),
    Voice("Maha", "bah", "Bahraini", "Arabic"),
    Voice("Mazen", "bah", "Bahraini", "Arabic"),
    Voice("Nadya", "bah", "Bahraini", "Arabic"),
    Voice("Ruba", "bah", "Bahraini", "Arabic"),
    Voice("Sami", "bah", "Bahraini", "Arabic"),
    Voice("Ahmed", "egy", "Egyptian", "Arabic"),
    Voice("Ali", "egy", "Egyptian", "Arabic"),
    Voice("Fouad", "egy", "Egyptian", "Arabic"),
    Voice("Layla", "egy", "Egyptian", "Arabic"),
    Voice("Mariam", "egy", "Egyptian", "Arabic"),
    Voice("Nada", "egy", "Egyptian", "Arabic"),
    Voice("Nermin", "egy", "Egyptian", "Arabic"),
    Voice("Noura", "egy", "Egyptian", "Arabic"),
    Voice("Nouran", "egy", "Egyptian", "Arabic"),
    Voice("Othman", "egy", "Egyptian", "Arabic"),
    Voice("Samir", "egy", "Egyptian", "Arabic"),
    Voice("Amanda", "en", "English", "English"),
    Voice("Arjun", "en", "English", "English"),
    Voice("Barbara", "en", "English", "English"),
    Voice("Brian", "en", "English", "English"),
    Voice("David", "en", "English", "English"),
    Voice("John", "en", "English", "English"),
    Voice("Michael", "en", "English", "English"),
    Voice("Robert", "en", "English", "English"),
    Voice("Roger", "en", "English", "English"),
    Voice("Sandra", "en", "English", "English"),
    Voice("Suzan", "en", "English", "English"),
    Voice("William", "en", "English", "English"),
    Voice("Aml", "irq", "Iraqi", "Arabic"),
    Voice("Ayman", "irq", "Iraqi", "Arabic"),
    Voice("Edward", "irq", "Iraqi", "Arabic"),
    Voice("Fatma", "irq", "Iraqi", "Arabic"),
    Voice("Ghazal", "irq", "Iraqi", "Arabic"),
    Voice("Khadiga", "irq", "Iraqi", "Arabic"),
    Voice("Lyali", "irq", "Iraqi", "Arabic"),
    Voice("Rihanna", "irq", "Iraqi", "Arabic"),
    Voice("Salam", "irq", "Iraqi", "Arabic"),
    Voice("Samer", "irq", "Iraqi", "Arabic"),
    Voice("Tasneem", "irq", "Iraqi", "Arabic"),
    Voice("ASSY", "jor", "Jordanian", "Arabic"),
    Voice("Hady", "jor", "Jordanian", "Arabic"),
    Voice("Lana", "jor", "Jordanian", "Arabic"),
    Voice("Nagib", "jor", "Jordanian", "Arabic"),
    Voice("Raghad", "jor", "Jordanian", "Arabic"),
    Voice("Rania", "jor", "Jordanian", "Arabic"),
    Voice("Renat", "jor", "Jordanian", "Arabic"),
    Voice("Sarah", "jor", "Jordanian", "Arabic"),
    Voice("Shaker", "jor", "Jordanian", "Arabic"),
    Voice("Eman", "ksa", "Saudi", "Arabic"),
    Voice("Fahd", "ksa", "Saudi", "Arabic"),
    Voice("Haneen", "ksa", "Saudi", "Arabic"),
    Voice("Hiba", "ksa", "Saudi", "Arabic"),
    Voice("Jasem", "ksa", "Saudi", "Arabic"),
    Voice("Khadija", "ksa", "Saudi", "Arabic"),
    Voice("Reem", "ksa", "Saudi", "Arabic"),
    Voice("Saly", "ksa", "Saudi", "Arabic"),
    Voice("Tamer", "ksa", "Saudi", "Arabic"),
    Voice("AbdelQader", "kuw", "Kuwaiti", "Arabic"),
    Voice("Akmal", "kuw", "Kuwaiti", "Arabic"),
    Voice("Hatem", "kuw", "Kuwaiti", "Arabic"),
    Voice("Kamla", "kuw", "Kuwaiti", "Arabic"),
    Voice("Mai", "kuw", "Kuwaiti", "Arabic"),
    Voice("Zeina", "kuw", "Kuwaiti", "Arabic"),
    Voice("Carla", "leb", "Lebanese", "Arabic"),
    Voice("Fady", "leb", "Lebanese", "Arabic"),
    Voice("Gassan", "leb", "Lebanese", "Arabic"),
    Voice("Majd", "leb", "Lebanese", "Arabic"),
    Voice("Marwan", "leb", "Lebanese", "Arabic"),
    Voice("Nabil", "leb", "Lebanese", "Arabic"),
    Voice("Sawsan", "leb", "Lebanese", "Arabic"),
    Voice("Amir", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Hasan", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Safa", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Salem", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Salwa", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Sayed", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Somaya", "msa", "Modern Standard Arabic", "Arabic"),
    Voice("Jaber", "oma", "Omani", "Arabic"),
    Voice("Noah", "oma", "Omani", "Arabic"),
    Voice("Saleh", "oma", "Omani", "Arabic"),
    Voice("Amjad", "pls", "Palestinian", "Arabic"),
    Voice("Jana", "pls", "Palestinian", "Arabic"),
    Voice("Layan", "pls", "Palestinian", "Arabic"),
    Voice("Obida", "pls", "Palestinian", "Arabic"),
    Voice("Rema", "pls", "Palestinian", "Arabic"),
    Voice("Souad", "pls", "Palestinian", "Arabic"),
    Voice("Talin", "pls", "Palestinian", "Arabic"),
    Voice("Hind", "qat", "Qatari", "Arabic"),
    Voice("Magda", "qat", "Qatari", "Arabic"),
    Voice("Majid", "qat", "Qatari", "Arabic"),
    Voice("Mansour", "qat", "Qatari", "Arabic"),
    Voice("Wael", "qat", "Qatari", "Arabic"),
    Voice("Dalal", "syr", "Syrian", "Arabic"),
    Voice("Mais", "syr", "Syrian", "Arabic"),
    Voice("Ola", "syr", "Syrian", "Arabic"),
    Voice("Rami", "syr", "Syrian", "Arabic"),
    Voice("Yara", "syr", "Syrian", "Arabic"),
    Voice("Alia", "uae", "Emirati", "Arabic"),
    Voice("Amira", "uae", "Emirati", "Arabic"),
    Voice("Dima", "uae", "Emirati", "Arabic"),
    Voice("Faiza", "uae", "Emirati", "Arabic"),
    Voice("Fares", "uae", "Emirati", "Arabic"),
    Voice("Gannat", "uae", "Emirati", "Arabic"),
    Voice("Hamdan", "uae", "Emirati", "Arabic"),
    Voice("Maher", "uae", "Emirati", "Arabic"),
    Voice("Marwa", "uae", "Emirati", "Arabic"),
    Voice("Noor", "uae", "Emirati", "Arabic"),
    Voice("Razan", "uae", "Emirati", "Arabic"),
    Voice("Salim", "uae", "Emirati", "Arabic"),
    Voice("Salma", "uae", "Emirati", "Arabic"),
    Voice("Yehya", "uae", "Emirati", "Arabic"),
)


def voices_for(dialect: str) -> tuple[Voice, ...]:
    """Every voice tuned for a dialect code."""
    return tuple(v for v in VOICES if v.dialect == dialect)


def find_voice(name: str) -> Voice | None:
    """Look a voice up by name, case-insensitively."""
    lowered = name.strip().lower()
    return next((v for v in VOICES if v.name.lower() == lowered), None)
