# Hamsa LiveKit plugin

Arabic speech-to-text and text-to-speech for [LiveKit Agents](https://docs.livekit.io/agents/),
powered by [Hamsa](https://tryhamsa.com) — 24 voices across 9 Arabic dialects.

- **Arabic-first**: text is chunked on Arabic punctuation (`؟` `،` `؛`), not just Latin
- **Server-side turn detection**: uses Hamsa's end-of-speech model instead of waiting out a silence timer
- **Streaming both ways**: audio starts playing while the LLM is still writing

## Install

```bash
pip install -e .
```

Requires Python 3.9+ and `livekit-agents >= 1.5`.

```env
HAMSA_API_KEY=your_hamsa_api_key
```

## Quick start

```python
import livekithamsa
from livekit.agents import Agent, AgentSession, agents
from livekit.plugins import openai, silero


class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful Arabic voice assistant.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # The VAD only has to notice a pause; Hamsa's end-of-speech model decides
    # whether the turn is actually over, so keep the threshold short.
    vad = silero.VAD.load(min_silence_duration=0.2)

    session = AgentSession(
        stt=livekithamsa.HamsaEOS(stt=livekithamsa.STT(language="ar"), vad=vad),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=livekithamsa.TTS(speaker="Lana", dialect="jor"),
        vad=vad,
        turn_detection="stt",
    )
    await session.start(room=ctx.room, agent=Assistant())
    await session.generate_reply(instructions="Greet the user in Arabic.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

There is a runnable version of this, plus a web page that charts per-turn
latency, in [`demo/`](demo/).

## Turn detection

The STT endpoint is request/response, so LiveKit would normally drive it through
`stt.StreamAdapter`: wait for the VAD to see `min_silence_duration` of quiet,
then transcribe. That threshold has to be set pessimistically — long enough that
a mid-sentence breath is not read as the end of a turn — and **every** turn pays
it before the LLM is even asked for a reply.

Hamsa ships a turn-detection model behind the API's `isEosEnabled` flag, which
answers the question the timer is only guessing at. `HamsaEOS` runs the VAD with
a short threshold and asks the server at each pause:

- a transcript comes back → the turn is over, commit it now
- `[THINKING]` comes back → the speaker is mid-utterance, keep listening

The fixed silence wait disappears from the common case. A `max_turn_silence`
backstop covers the case where the model never concedes.

```python
livekithamsa.HamsaEOS(
    stt=livekithamsa.STT(language="ar"),
    vad=silero.VAD.load(min_silence_duration=0.2),
    max_turn_silence=2.0,       # backstop: finalize anyway after this much quiet
    min_speech_duration=0.2,    # ignore clicks and breaths
)
```

Set `turn_detection="stt"` on the session so it commits on the plugin's
`FINAL_TRANSCRIPT` rather than running its own timer.

**The trade-off:** this spends one STT request per pause instead of one per turn.
A turn where the speaker hesitates twice costs three requests, plus one more if
the backstop fires. Requests are cheap (~0.25–0.45 s on a warm connection) and
the inner metrics are forwarded so `UsageCollector` counts every one of them, but
if you are billed per request, raise `min_speech_duration` or lengthen the VAD's
`min_silence_duration` to probe less often.

Plain `STT()` still works on its own if you want stock behaviour:

```python
session = AgentSession(
    stt=livekithamsa.STT(language="ar"),   # LiveKit wraps this in a StreamAdapter
    vad=silero.VAD.load(),
    turn_detection="vad",
)
```

> `[THINKING]` is a control signal, never speech. `STT` reports it as an interim
> with no alternatives so it can never reach your LLM as if the user had said it.

## TTS

Time-to-first-byte on the endpoint is ~0.3 s and barely moves with text length,
so the only way to start speaking sooner is to stop waiting for the LLM. Text is
cut into chunks and the opening one is sent as soon as it is worth speaking.

Every chunk uses the same rule — a word floor, then release at the first
punctuation, with a hard ceiling if none turns up:

```
|<--- floor --->|<--- looking for , . ? ; : --->|<--- ceiling
 never break     break at the first one          break anyway
```

```python
livekithamsa.TTS(
    speaker="Ali",
    dialect="pls",
    first_chunk_words=3,       # floor for the opening chunk
    first_chunk_max_words=10,  # ceiling: cut here if no punctuation appeared
    chunk_words=18,            # floor for the rest
    max_chunk_words=25,        # ceiling for the rest
    prefetch=1,                # chunks kept in flight ahead of playback
    mulaw=False,               # True -> 8 kHz G.711 for telephony
)
```

The **punctuation gate is what makes the opening sound finished**. Releasing on
the word count alone lands mid-phrase and the voice audibly stops in the middle
of a breath. On a typical Arabic reply the first chunk therefore comes out as
`مرحبا بك في هامسا،` — ending where a speaker would pause — rather than
`مرحبا بك في`.

The **ceiling** stops that gate becoming an open-ended wait. It matters more than
it looks: LiveKit's buffered tokenizers only emit a token once they can see the
*next* boundary, so a stretch of text with no punctuation emits nothing at all
until the reply ends. The ceiling is what guarantees audio keeps flowing through
unpunctuated text, which LLMs do produce.

Chunks are counted in **words, not characters**, because a character count is not
stable across the text this plugin exists to handle: the same four Arabic words
are 18 characters undiacritized and 28 with tashkeel, and Arabic runs shorter per
word than Latin — one number would mean three different things.

The endpoint's output format is fixed: 16 kHz mono PCM, or 8 kHz μ-law with
`mulaw=True` (decoded to PCM for you). There is no request field for a sample
rate, so `sample_rate=` only mislabels the audio and is ignored with a warning.

### Arabic tokenization

LiveKit's built-in sentence tokenizer terminates only on `[.!?。！？]`, so `؟` and
`،` are not boundaries to it at all. The TTS chunker uses the Arabic-aware
punctuation sets from `livekithamsa.tokenizer` instead.

`ArabicSentenceTokenizer` is exported for anywhere else you need Arabic sentence
splitting in a LiveKit pipeline:

```python
from livekithamsa import ArabicSentenceTokenizer

ArabicSentenceTokenizer().tokenize("كيف حالك اليوم؟ أنا بخير الحمد لله.")
# ['كيف حالك اليوم؟', 'أنا بخير الحمد لله.']
```

To override how the TTS splits words, pass a tokenizer that keeps trailing
punctuation attached (that is how boundaries are detected):

```python
from livekit.agents import tokenize

livekithamsa.TTS(word_tokenizer=tokenize.basic.WordTokenizer(ignore_punctuation=False))
```

## Transport: HTTP or WebSocket

Both STT and TTS can run over `wss://api.tryhamsa.com/v1/realtime/ws` instead of
the HTTP endpoints. The socket carries the *same* whole-utterance exchange — it is
not incremental audio streaming — so it does not change the latency model. What it
removes is per-request HTTP overhead, and that is worth having:

| STT, same 4.1 s of audio, warm | median |
|---|---|
| HTTP | 0.39 s |
| WebSocket | **0.27 s** (−29%) |

```python
stt = livekithamsa.STT(language="ar", transport="websocket", model="s3")
tts = livekithamsa.TTS(transport="websocket", expressiveness=1)

await stt.warmup()   # optional: open the socket before the first turn
```

The connection is **opened once and reused for every turn** — a pool holds it open
for the life of the process, and closing/reopening per turn would put the
handshake back on the critical path. Verified live: one socket served six
consecutive turns.

`ws_pool_size` exists because replies carry no request id: a reply is matched to
its request by arrival order, so a socket can serve only one request at a time.
Concurrency therefore needs several sockets, each leased exclusively. The default
is 2 for STT and `prefetch + 1` for TTS.

### `HamsaEOS` requires HTTP

The socket **does not implement the `[THINKING]` end-of-speech signal** — measured
0/15 against HTTP's 9/15 on identical mid-utterance audio. Turn detection is built
entirely on that signal, so on a socket every probe would look like a finished
turn and the agent would commit on the first fragment it heard. `HamsaEOS` raises
a `ValueError` rather than degrade silently:

```python
# fast turn detection over HTTP, fast synthesis over the socket
session = AgentSession(
    stt=livekithamsa.HamsaEOS(stt=livekithamsa.STT(language="ar"), vad=vad),
    tts=livekithamsa.TTS(transport="websocket"),
    ...
)
```

### It will not fix latency spikes

If turns occasionally take much longer, the transport is not the cause. Decomposed
over 30 requests with connection tracing:

```
acquire  p50 0.000s  p95 0.000s   <- connection setup, already free
wait     p50 1.076s  p95 1.926s   <- server side
slowest 5 requests: all "wait dominates"
```

HTTP keep-alive already makes connection cost zero; the variance is the service
queueing. `python tests/diag_spikes.py` reproduces that breakdown, and
`tests/diag_network.py` separates a slow service from a slow uplink.

Other limits worth knowing: the API allows **100 requests per 60 s per key**, the
server pings every 30 s, and idle sockets are closed after 60 minutes (the pool
reconnects transparently).

## Voices and dialects

```python
from livekithamsa import DIALECTS, RECOMMENDED_VOICES, SPEAKERS
```

| Dialect | Code | Voices tuned for it |
|---|---|---|
| Palestinian | `pls` | Amjad, Layan, Talin, Rema, Ali |
| Lebanese | `leb` | Carla, Majd |
| Jordanian | `jor` | Lana, Jasem, Nada |
| Syrian | `syr` | Dalal, Mais |
| Saudi | `ksa` | Hiba, Khadija, Fahd, Jasem |
| Bahraini | `bah` | Mazen |
| Emirati | `uae` | Salma, Dima |
| Egyptian | `egy` | Mariam, Samir, Nada, Ali, Ahmed |
| Iraqi | `irq` | Lyali, Fatma |

Any voice works with any dialect; these pairings sound the most natural.

## Tests

```bash
pytest tests/test_plugin.py     # offline: tokenization, chunking, audio, parsing

export HAMSA_API_KEY=...
python tests/smoke_live.py      # end to end against the real API
python tests/benchmark_api.py   # reproduces the measurements behind the design
```

## Support

- Docs: [docs.tryhamsa.com](https://docs.tryhamsa.com)
- API reference: [docs.tryhamsa.com/api-reference/introduction](https://docs.tryhamsa.com/api-reference/introduction)
- Email: support@tryhamsa.com

---

*Hamsa — AI models that master Arabic dialects*
