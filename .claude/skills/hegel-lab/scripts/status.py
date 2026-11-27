"""
Show the runs started with timed_run.py: which are still going, how long
they have run, the last lines of each log, and free memory.

Usage (from hegel/):
    python status.py [lines]      # lines of each log to show, default 8
"""

import os
import sys
import time
from pathlib import Path
from typing import List

RUNS = Path("runs")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _memory_gb(pid: int) -> float:
    """Resident memory of one process in GB, 0 if it cannot be read."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _free_gb() -> str:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":")
        fields[key] = int(value.split()[0]) / 1e6
    return (
        f"memory: {fields['MemAvailable']:.1f} GB available of "
        f"{fields['MemTotal']:.1f}; swap free {fields['SwapFree']:.1f} GB"
    )


def main(argv: List[str]) -> int:
    lines = int(argv[1]) if len(argv) > 1 else 8
    logs = sorted(RUNS.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("No runs yet (runs/ is empty).")
    for log in logs:
        pid_path = log.with_suffix(".pid")
        state = "finished"
        if pid_path.exists():
            pid = int(pid_path.read_text())
            if _alive(pid):
                age = (time.time() - pid_path.stat().st_mtime) / 60
                state = f"RUNNING {age:.1f} min, pid {pid}, {_memory_gb(pid):.1f} GB"
            else:
                state = "died without finishing (stale pid file)"
        print(f"== {log.stem}: {state}")
        text = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in text[-lines:]:
            print(f"   {line}")
    print(_free_gb())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
