"""Start, stop and check the demo's two processes.

    python services.py start     # detached, survives the shell that launched them
    python services.py status
    python services.py stop
    python services.py logs

Both are started with ``start_new_session=True``, which puts them in their own
process group and session. That matters when the launcher is a tool or CI step
whose cleanup kills its whole process group on exit -- without it the agent and
the token server go down with the launcher, moments after coming up.

``stop`` also sweeps LiveKit job subprocesses. ``pkill -f agent.py`` misses them:
in dev mode a job runs in a ``multiprocessing`` child whose command line does not
mention agent.py, and an orphan keeps serving its room.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / ".run"
PAGE = "http://127.0.0.1:8080"

SERVICES = {
    "serve": ["serve.py"],
    "agent": ["agent.py", "dev"],
}


def pid_file(name: str) -> Path:
    return RUN / f"{name}.pid"


def log_file(name: str) -> Path:
    return RUN / f"{name}.log"


def python() -> str:
    """The interpreter to launch with: this one, or the repo venv."""
    venv = HERE.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def read_pid(name: str) -> int | None:
    try:
        pid = int(pid_file(name).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


def start(name: str, argv: list[str]) -> None:
    existing = read_pid(name)
    if existing:
        print(f"  {name:6} already running (pid {existing})")
        return
    RUN.mkdir(exist_ok=True)
    log = log_file(name).open("ab")
    proc = subprocess.Popen(
        [python(), *argv],
        cwd=HERE,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # Its own session, so a process-group kill aimed at the launcher does not
        # take these with it.
        start_new_session=True,
    )
    pid_file(name).write_text(str(proc.pid))
    print(f"  {name:6} started (pid {proc.pid}) -> {log_file(name)}")


def stop_all() -> None:
    for name in SERVICES:
        pid = read_pid(name)
        if pid is None:
            print(f"  {name:6} not running")
        else:
            try:
                # The whole group: the agent's dev-mode reloader forks children.
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(OSError, ValueError):
                    os.kill(pid, signal.SIGTERM)
            print(f"  {name:6} stopped (pid {pid})")
        pid_file(name).unlink(missing_ok=True)

    time.sleep(1.5)
    strays = subprocess.run(
        ["pgrep", "-f", "multiprocessing.spawn"], capture_output=True, text=True
    ).stdout.split()
    for s in strays:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(s), signal.SIGKILL)
    if strays:
        print(f"  swept {len(strays)} livekit job subprocess(es)")



def http_get(path: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(f"{PAGE}{path}", timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return None, b""


def status() -> int:
    print("processes")
    running = 0
    for name in SERVICES:
        pid = read_pid(name)
        print(f"  {name:6} {'running (pid ' + str(pid) + ')' if pid else 'stopped'}")
        running += 1 if pid else 0

    print("endpoints")
    code, _ = http_get("/")
    print(f"  page   {('HTTP ' + str(code)) if code else 'not responding'}")
    code, body = http_get("/voices")
    if code == 200:
        try:
            data = json.loads(body)
            print(f"  voices {len(data['voices'])} across {len(data['dialects'])} dialects")
        except Exception as e:  # noqa: BLE001
            print(f"  voices unparseable: {e}")
    elif code:
        print(f"  voices HTTP {code}: {body[:120].decode(errors='replace')}")

    agent_log = log_file("agent")
    if agent_log.exists():
        text = agent_log.read_text(errors="replace")
        registered = "registered worker" in text
        print(f"  worker {'registered' if registered else 'NOT registered'}")
        for line in reversed(text.splitlines()):
            if "session: mode=" in line:
                print(f"  last   {line.strip()[-90:]}")
                break
    return 0 if running == len(SERVICES) else 1


def logs(lines: int = 25) -> None:
    for name in SERVICES:
        path = log_file(name)
        print(f"=== {name} ===")
        if not path.exists():
            print("  (no log)")
            continue
        for line in path.read_text(errors="replace").splitlines()[-lines:]:
            print(f"  {line}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "restart", "status", "logs"])
    args = ap.parse_args()

    if args.action in ("stop", "restart"):
        stop_all()
    if args.action in ("start", "restart"):
        RUN.mkdir(exist_ok=True)
        for name, argv in SERVICES.items():
            start(name, argv)
        # Wait for readiness, not just for pids: the worker takes ~10s to reach
        # LiveKit, and reporting before then reads as a failure.
        print("\nwaiting for them to come up...")
        for _ in range(40):
            time.sleep(1)
            code, _ = http_get("/", timeout=2)
            log = log_file("agent")
            registered = log.exists() and "registered worker" in log.read_text(errors="replace")
            if code == 200 and registered and read_pid("agent"):
                break
        print()
        return status()
    if args.action == "status":
        return status()
    if args.action == "logs":
        logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
