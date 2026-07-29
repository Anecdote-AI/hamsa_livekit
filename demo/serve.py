"""Token endpoint + static host for the demo page.

    python serve.py            # http://localhost:8080

Stdlib HTTP server plus livekit-api for signing, so there is nothing to install
beyond what the agent already needs and nothing to build.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from livekit import api

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")

STATIC = {"/": "index.html", "/index.html": "index.html"}

# Settings the page may set per session. Anything else is ignored, so a stale
# page cannot smuggle arbitrary keys into the agent.
ALLOWED_SETTINGS = {
    "stt_transport",
    "tts_transport",
    "vad_min_silence",
    "vad_threshold",
    "speaker",
    "dialect",
    "instructions",
    "expressiveness",
}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}


def _load_catalogue() -> tuple[dict | None, str | None]:
    """Build the voice payload once, at startup.

    Importing inside the request handler made a failure look intermittent -- a
    request that happened to hit a bad import returned an error object while the
    next one succeeded. Doing it here means a broken install is reported when the
    server boots, and every request is served from the same in-memory copy.
    """
    try:
        from livekithamsa.voices import DIALECT_NAMES, VOICES
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return {
        "dialects": [
            {"code": code, "name": name} for code, name in sorted(DIALECT_NAMES.items())
        ],
        "voices": [
            {
                "name": v.name,
                "dialect": v.dialect,
                "dialect_name": v.dialect_name,
                "language": v.language,
            }
            for v in VOICES
        ],
    }, None


VOICE_CATALOGUE, CATALOGUE_ERROR = _load_catalogue()


class Handler(BaseHTTPRequestHandler):
    server_version = "hamsa-demo"

    def __init__(self, *args, config: dict, **kwargs) -> None:
        self._config = config
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than the default, which logs every asset fetch. args[0] is not
        # always a string -- log_error passes an exception -- so coerce first.
        first = str(args[0]) if args else ""
        if "/token" in first or "failed" in fmt:
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/token":
            self._handle_token(parse_qs(parsed.query))
            return

        if parsed.path == "/voices":
            self._handle_voices()
            return

        name = STATIC.get(parsed.path)
        if name is None:
            self._json(404, {"error": "not found"})
            return

        path = HERE / name
        if not path.exists():
            self._json(404, {"error": f"{name} is missing"})
            return
        self._send(
            200,
            path.read_bytes(),
            CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
        )

    def _handle_voices(self) -> None:
        """The voice catalogue, so the page does not carry its own copy."""
        if VOICE_CATALOGUE is None:
            self._json(500, {"error": f"voice catalogue unavailable: {CATALOGUE_ERROR}"})
            return
        self._json(200, VOICE_CATALOGUE)

    def _collect_settings(self, query: dict[str, list[str]]) -> dict:
        """Pull the recognised settings out of the query string."""
        out: dict = {}
        for key in ALLOWED_SETTINGS:
            if key in query and query[key][0] != "":
                out[key] = query[key][0]
        return out

    def _handle_token(self, query: dict[str, list[str]]) -> None:
        key = self._config["api_key"]
        secret = self._config["api_secret"]
        url = self._config["url"]

        missing = [
            n
            for n, v in (
                ("LIVEKIT_API_KEY", key),
                ("LIVEKIT_API_SECRET", secret),
                ("LIVEKIT_URL", url),
            )
            if not v
        ]
        if missing:
            self._json(
                500,
                {
                    "error": f"missing environment variables: {', '.join(missing)}",
                    "hint": "copy demo/.env.example to demo/.env and fill it in",
                },
            )
            return

        room = (query.get("room") or [f"hamsa-demo-{secrets.token_hex(3)}"])[0]
        identity = (query.get("identity") or [f"user-{secrets.token_hex(3)}"])[0]

        token = (
            api.AccessToken(key, secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )

        # A worker started with AGENT_NAME set only runs jobs addressed to that
        # name, so joining the room is not enough to summon it -- something has
        # to dispatch it. Do that here, or the page sits on "waiting for the
        # agent to join" forever. Without AGENT_NAME the worker gets automatic
        # dispatch and this is a no-op.
        # Settings ride along as dispatch metadata: the agent reads them when the
        # job starts, which is before the session is built. That matters because
        # the VAD threshold and the transports cannot be changed afterwards.
        settings = self._collect_settings(query)

        dispatch_error = None
        agent_name = self._config.get("agent_name")
        if agent_name:
            try:
                asyncio.run(_dispatch(url, key, secret, room, agent_name, settings))
            except Exception as e:  # noqa: BLE001 - surfaced to the page
                dispatch_error = str(e)
                self.log_message('agent dispatch failed: %s', e)

        payload = {"url": url, "token": token, "room": room, "identity": identity}
        if agent_name:
            payload["agent_name"] = agent_name
        if settings:
            payload["settings"] = settings
        if settings and not agent_name:
            payload["settings_warning"] = (
                "settings are delivered via agent dispatch; set AGENT_NAME so the "
                "token server can dispatch and pass them"
            )
        if dispatch_error:
            payload["dispatch_error"] = dispatch_error
        self._json(200, payload)


async def _dispatch(
    url: str, key: str, secret: str, room: str, agent_name: str, settings: dict | None = None
) -> None:
    http_url = url.replace("wss://", "https://").replace("ws://", "http://")
    lkapi = api.LiveKitAPI(http_url, key, secret)
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room,
                agent_name=agent_name,
                metadata=json.dumps(settings or {}),
            )
        )
    finally:
        await lkapi.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    config = {
        "api_key": os.environ.get("LIVEKIT_API_KEY"),
        "api_secret": os.environ.get("LIVEKIT_API_SECRET"),
        "url": os.environ.get("LIVEKIT_URL"),
        "agent_name": os.environ.get("AGENT_NAME"),
    }

    handler = partial(Handler, config=config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"demo page:  http://{args.host}:{args.port}")
    print(f"livekit:    {config['url'] or '(LIVEKIT_URL not set)'}")
    if config["agent_name"]:
        print(f"dispatch:   explicit, to agent {config['agent_name']!r}")
    else:
        print("dispatch:   automatic (any worker on this project may pick up the room)")
    if VOICE_CATALOGUE is None:
        print(f"voices:     UNAVAILABLE -- {CATALOGUE_ERROR}")
    else:
        print(
            f"voices:     {len(VOICE_CATALOGUE['voices'])} across "
            f"{len(VOICE_CATALOGUE['dialects'])} dialects"
        )
    print("start the agent in another terminal:  python agent.py dev")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
