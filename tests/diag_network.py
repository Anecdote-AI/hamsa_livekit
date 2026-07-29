"""Is a slow turn the Hamsa API or the link to it?

Separates the two by timing the same STT request at several payload sizes and
comparing effective upload throughput against a control host. If throughput has
collapsed, latency scales with payload and the endpoint is not at fault.

    python tests/diag_network.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STT_URL = "https://api.tryhamsa.com/v1/realtime/stt"
CONTROL_URL = "https://api.openai.com/v1/chat/completions"


def key() -> str:
    k = os.environ.get("HAMSA_API_KEY")
    if not k:
        raise SystemExit("HAMSA_API_KEY is not set")
    return k


def silence_wav(seconds: float, rate: int = 16000) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return b.getvalue()


async def timed_post(session, url, body: bytes, headers: dict) -> tuple[float, int, int]:
    t0 = time.monotonic()
    async with session.post(url, data=body, headers=headers) as r:
        await r.read()
    return time.monotonic() - t0, r.status, len(body)


async def main() -> int:
    k = key()
    hamsa_headers = {"Authorization": f"Token {k}", "Content-Type": "application/json"}

    print("STT latency vs payload size (warm keep-alive, silence so compute is trivial)")
    print(f"{'audio':>7} {'payload':>10} {'median':>9} {'upload rate':>14}")
    rows = []
    async with aiohttp.ClientSession() as s:
        # warm the pool so TLS setup is not in the numbers
        import json

        warm = json.dumps(
            {
                "audioBase64": base64.b64encode(silence_wav(0.2)).decode(),
                "language": "ar",
                "isEosEnabled": False,
                "eosThreshold": 0.3,
            }
        ).encode()
        await timed_post(s, STT_URL, warm, hamsa_headers)

        for secs in (0.5, 1.0, 2.0, 4.0):
            body = json.dumps(
                {
                    "audioBase64": base64.b64encode(silence_wav(secs)).decode(),
                    "language": "ar",
                    "isEosEnabled": False,
                    "eosThreshold": 0.3,
                }
            ).encode()
            runs = [await timed_post(s, STT_URL, body, hamsa_headers) for _ in range(3)]
            med = statistics.median(r[0] for r in runs)
            kb = len(body) / 1024
            rows.append((kb, med))
            print(f"{secs:6.1f}s {kb:9.0f}K {med:8.2f}s {kb / med:12.0f} KB/s")

        # Slope of latency against payload isolates transfer from fixed cost.
        if len(rows) >= 2:
            (kb0, t0), (kb1, t1) = rows[0], rows[-1]
            extra_kb, extra_s = kb1 - kb0, t1 - t0
            print()
            if extra_s > 0.01:
                print(
                    f"  marginal transfer rate: {extra_kb / extra_s:.0f} KB/s "
                    f"({extra_kb:.0f}K extra cost {extra_s:.2f}s)"
                )
                slope = extra_kb / extra_s
                fixed = max(0.0, t0 - kb0 / slope)
                print(f"  fixed cost (extrapolated): {fixed:.2f}s")
            else:
                print("  latency barely moves with payload -> transfer is not the bottleneck")

        print("\nControl: same-size upload to an unrelated host")
        head = b'{"model":"gpt-4o-mini","messages":[{"role":"user","content":"'
        big = head + b"x" * 300_000 + b'"}]}'
        ctrl_headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk-invalid')}",
            "Content-Type": "application/json",
        }
        try:
            dt, status, size = await timed_post(s, CONTROL_URL, big, ctrl_headers)
            rate = size / 1024 / dt
            print(f"  api.openai.com   {size / 1024:.0f}K in {dt:.2f}s"
                  f" -> {rate:.0f} KB/s (HTTP {status})")
        except Exception as e:  # noqa: BLE001
            print(f"  control upload failed: {e}")

        try:
            dt, status, size = await timed_post(s, STT_URL, big, hamsa_headers)
            rate = size / 1024 / dt
            print(f"  api.tryhamsa.com {size / 1024:.0f}K in {dt:.2f}s"
                  f" -> {rate:.0f} KB/s (HTTP {status})")
        except Exception as e:  # noqa: BLE001
            print(f"  hamsa upload failed: {e}")

    print(
        "\nReading it: if both hosts show a similar low rate, the uplink is the\n"
        "problem. If only Hamsa is slow, it is the service."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
