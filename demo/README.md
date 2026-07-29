# Hamsa × LiveKit — latency bench

A minimal, buildless harness for trying the plugin before publishing it: a Python
agent, a token server, and one HTML page. The page charts where each turn's
latency goes, and a `MODE` switch lets you run the same agent with and without
the plugin's speedups.

## Setup

```bash
cd ..                       # repository root
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[demo]"

cd demo
cp .env.example .env        # then fill it in
```

`.env` needs:

| variable | where it comes from |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | [LiveKit Cloud](https://cloud.livekit.io) or your own server |
| `HAMSA_API_KEY` | [tryhamsa.com](https://tryhamsa.com) |
| `OPENAI_API_KEY` | the LLM the demo agent talks to |

### Sharing a project with production

By default a worker gets *automatic* dispatch: LiveKit hands it a job for every
new room on the project. On a shared project that means this demo would start
answering real traffic. Setting `AGENT_NAME` switches the worker to explicit
dispatch, so it only runs jobs addressed to that name:

```env
AGENT_NAME=hamsa-demo-local
```

`serve.py` then dispatches that agent when it mints a token. Both halves have to
agree — a named worker with nothing dispatching it leaves the page waiting for an
agent that never joins.

> Killing the worker with `pkill -f agent.py` is not always enough: `dev` mode
> runs jobs in `multiprocessing` subprocesses whose command line does not contain
> `agent.py`, and an orphan can keep serving a room. If you see the wrong `MODE`
> reported after a restart, check for strays:
> `ps -eo pid,ppid,command | grep multiprocessing`.

## Run

Two terminals:

```bash
python serve.py             # http://localhost:8080
python agent.py dev
```

Open <http://localhost:8080>, press **connect** (or the space bar), and speak
Arabic.

## What the page shows

Each turn is broken into the three waits between you going quiet and the agent
making a sound:

| band | metric | what it measures |
|---|---|---|
| amber | `STTMetrics.duration` | the Hamsa API round trip that produced the committed transcript |
| hatched amber | `transcription_delay` − the above | the VAD's `min_silence_duration`: dead time before any request can be issued |
| rust | `end_of_utterance_delay` − `transcription_delay` | deciding the speaker had finished |
| indigo | `LLMMetrics.ttft` | prompt sent → first token back |
| cyan | `TTSMetrics.ttfb` | text sent → first audio byte back |

The first three bands together are LiveKit's `end_of_utterance_delay`, split so
you can tell a slow transcription apart from a slow endpointing decision. Measured on
this setup, a turn is roughly 390 ms of API call, 200 ms of VAD silence, and
single-digit ms of endpointing. The middle band tracks the VAD threshold almost
exactly — raising `VAD_SILENCE` from 0.2 to 0.5 moved it from ~208 ms to ~510 ms
while the API call stayed flat — which is why the fast path keeps that threshold
short and lets the server's end-of-speech model make the actual decision.

Note that `HamsaEOS` issues **one STT call per pause**, so a turn can make
several. Only the last is on the critical path (earlier probes overlap with the
user still speaking); the page lists them all under the chart.

Medians for each band accumulate in the left column.

## Comparing against stock LiveKit

`MODE` selects how the agent is wired, and nothing else changes:

```bash
MODE=hamsa    python agent.py dev     # EOS turn detection + clause-chunked TTS
MODE=baseline python agent.py dev     # stock VAD turn detection + one TTS request per reply
```

`baseline` is the configuration the plugin shipped with previously: a 0.55 s VAD
silence threshold deciding turns, and the whole reply sent to the TTS in one
request once the LLM has finished. Run a few turns in each and compare the
medians — the amber and cyan bands are where the difference shows up.

The mode badge in the top right reflects what the agent reports, so you can tell
at a glance which one you are talking to.

## Voice

```bash
HAMSA_SPEAKER=Lana HAMSA_DIALECT=jor python agent.py dev
```

See the speaker/dialect table in the [root README](../README.md#voices-and-dialects).

## Troubleshooting

**"missing environment variables"** — `demo/.env` is not filled in. The token
endpoint reports which ones.

**Page connects but nothing happens** — the agent is not running, or it joined a
different LiveKit project. The status strip shows when a participant joins;
`agent.py` logs the room it picked up.

**No audio from the agent** — browsers block autoplay until a gesture. Pressing
**connect** counts, but if you connected some other way, click the page once.

**No transcript lines** — check the agent's log for STT errors. `HAMSA_API_KEY`
being wrong shows up as a 403 there.
