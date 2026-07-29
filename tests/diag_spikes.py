"""Attribute STT latency spikes to connection setup or server processing.

aiohttp's trace hooks let each request be split into:

  acquire  - getting a connection: queued behind the pool limit, DNS, TCP, TLS
  send     - writing the request body
  wait     - request fully sent -> first response byte  (server side)
  read     - draining the response

A spike in `acquire` is a connection problem and a persistent socket would fix
it. A spike in `wait` is the service queueing or the model being slow, and no
transport change will help.

    python tests/diag_spikes.py [--n 40]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STT_URL = "https://api.tryhamsa.com/v1/realtime/stt"


def key() -> str:
    k = os.environ.get("HAMSA_API_KEY")
    if not k:
        raise SystemExit("HAMSA_API_KEY is not set")
    return k


def speech_wav(seconds: float, rate: int = 16000) -> bytes:
    """A tone rather than silence, so the model does real work."""
    import array
    import math

    n = int(rate * seconds)
    a = array.array(
        "h", (int(6000 * math.sin(2 * math.pi * 180 * i / rate)) for i in range(n))
    )
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(a.tobytes())
    return b.getvalue()


def make_trace() -> tuple[aiohttp.TraceConfig, dict]:
    marks: dict = {}

    async def on_request_start(session, ctx, params):
        marks.clear()
        marks["start"] = time.monotonic()
        marks["reused"] = None

    async def on_connection_queued_start(session, ctx, params):
        marks["queued"] = time.monotonic()

    async def on_connection_create_start(session, ctx, params):
        marks["create_start"] = time.monotonic()

    async def on_connection_create_end(session, ctx, params):
        marks["create_end"] = time.monotonic()
        marks["reused"] = False

    async def on_connection_reuseconn(session, ctx, params):
        marks["reuse"] = time.monotonic()
        marks["reused"] = True

    async def on_request_chunk_sent(session, ctx, params):
        marks["sent"] = time.monotonic()

    async def on_response_chunk_received(session, ctx, params):
        marks.setdefault("first_byte", time.monotonic())

    async def on_request_end(session, ctx, params):
        marks["end"] = time.monotonic()

    tc = aiohttp.TraceConfig()
    tc.on_request_start.append(on_request_start)
    tc.on_connection_queued_start.append(on_connection_queued_start)
    tc.on_connection_create_start.append(on_connection_create_start)
    tc.on_connection_create_end.append(on_connection_create_end)
    tc.on_connection_reuseconn.append(on_connection_reuseconn)
    tc.on_request_chunk_sent.append(on_request_chunk_sent)
    tc.on_response_chunk_received.append(on_response_chunk_received)
    tc.on_request_end.append(on_request_end)
    return tc, marks


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--audio", type=float, default=2.0, help="seconds of audio")
    args = ap.parse_args()

    body = json.dumps(
        {
            "audioBase64": base64.b64encode(speech_wav(args.audio)).decode(),
            "language": "ar",
            "isEosEnabled": False,
            "eosThreshold": 0.3,
        }
    ).encode()
    headers = {"Authorization": f"Token {key()}", "Content-Type": "application/json"}

    tc, marks = make_trace()
    samples: list[dict] = []

    print(f"{args.n} sequential requests, {args.audio}s audio, {len(body) / 1024:.0f} KB body")
    print(f"{'#':>3} {'total':>8} {'acquire':>8} {'wait':>8} {'read':>7}  conn")

    async with aiohttp.ClientSession(trace_configs=[tc]) as s:
        for i in range(args.n):
            async with s.post(STT_URL, data=body, headers=headers) as r:
                await r.read()
            m = dict(marks)
            start = m.get("start", 0.0)
            end = m.get("end", start)
            # connection acquisition: everything before the body starts going out
            if m.get("reused"):
                acquire = m.get("reuse", start) - start
            else:
                acquire = m.get("create_end", start) - start
            first_byte = m.get("first_byte", end)
            sent = m.get("sent", start + acquire)
            wait = max(0.0, first_byte - sent)
            read = max(0.0, end - first_byte)
            total = end - start
            row = {
                "total": total,
                "acquire": acquire,
                "wait": wait,
                "read": read,
                "reused": bool(m.get("reused")),
                "status": r.status,
            }
            samples.append(row)
            flag = "reuse" if row["reused"] else "NEW"
            print(
                f"{i:3} {total:7.3f}s {acquire:7.3f}s {wait:7.3f}s {read:6.3f}s  {flag}"
            )

    def stats(field: str) -> str:
        xs = sorted(x[field] for x in samples)
        p50 = statistics.median(xs)
        p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
        return f"p50 {p50:.3f}s  p95 {p95:.3f}s  max {xs[-1]:.3f}s"

    print(f"\ntotal    {stats('total')}")
    print(f"acquire  {stats('acquire')}")
    print(f"wait     {stats('wait')}   <- server side")
    print(f"read     {stats('read')}")

    news = [x for x in samples if not x["reused"]]
    print(f"\nnew connections: {len(news)}/{len(samples)}")

    # Attribute the worst requests.
    worst = sorted(samples, key=lambda x: -x["total"])[:5]
    print("\nslowest 5 requests, by where the time went:")
    for x in worst:
        dominant = max(
            ("acquire", x["acquire"]), ("wait", x["wait"]), ("read", x["read"]),
            key=lambda kv: kv[1],
        )
        print(
            f"  total {x['total']:.3f}s  acquire {x['acquire']:.3f}s  "
            f"wait {x['wait']:.3f}s  read {x['read']:.3f}s  -> {dominant[0]} dominates"
        )

    med_total = statistics.median(x["total"] for x in samples)
    spikes = [x for x in samples if x["total"] > 2 * med_total]
    print(f"\nspikes (>2x median of {med_total:.3f}s): {len(spikes)}")
    if spikes:
        by_acquire = sum(1 for x in spikes if x["acquire"] > 0.5 * x["total"])
        by_wait = sum(1 for x in spikes if x["wait"] > 0.5 * x["total"])
        print(f"  dominated by connection setup: {by_acquire}")
        print(f"  dominated by server wait:      {by_wait}")
        print(
            "\n  -> a persistent websocket would help the first group and do\n"
            "     nothing for the second."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
