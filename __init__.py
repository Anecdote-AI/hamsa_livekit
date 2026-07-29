"""LiveKit Agents plugin for Hamsa's Arabic speech models.

    import livekithamsa
    from livekit.plugins import silero

    session = AgentSession(
        stt=livekithamsa.HamsaEOS(                       # fast turn detection
            stt=livekithamsa.STT(language="ar"),
            vad=silero.VAD.load(min_silence_duration=0.2),
        ),
        llm=...,
        tts=livekithamsa.TTS(speaker="Lana", dialect="jor"),
        turn_detection="stt",
    )
"""

from .eos import HamsaEOS
from .models import (
    DIALECTS,
    SPEAKERS,
    VOICES,
    VOICES_BY_DIALECT,
    Dialect,
    Speaker,
    Voice,
    find_voice,
    voices_for,
)
from .stt import STT, THINKING_SENTINEL
from .tokenizer import ArabicSentenceTokenizer
from .tts import TTS
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "HamsaEOS",
    "ArabicSentenceTokenizer",
    "THINKING_SENTINEL",
    "Speaker",
    "Dialect",
    "SPEAKERS",
    "DIALECTS",
    "VOICES",
    "Voice",
    "VOICES_BY_DIALECT",
    "voices_for",
    "find_voice",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class HamsaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(HamsaPlugin())

# Hide internals from generated docs.
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
