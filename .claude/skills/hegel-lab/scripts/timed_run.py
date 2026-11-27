"""
Run a command, copy its output to runs/<name>.log line by line, and record
how long it took and its peak memory.

Usage (from hegel/):
    python timed_run.py <name> <command> [args...]

Python children run unbuffered, so the log fills as the run goes. While the
command runs, runs/<name>.pid holds its process id (status.py reads it).
"""

import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import List

RUNS = Path("runs")


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    name, command = argv[1], argv[2:]
    RUNS.mkdir(exist_ok=True)
    log_path, pid_path = RUNS / f"{name}.log", RUNS / f"{name}.pid"
    started = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log:

        def emit(line: str) -> None:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

        emit(f"# {' '.join(command)}\n# started {stamp}\n")
        child = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        pid_path.write_text(str(child.pid))
        try:
            assert child.stdout is not None
            for line in child.stdout:
                emit(line)
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            code = child.wait()
        finally:
            pid_path.unlink(missing_ok=True)
        elapsed = time.time() - started
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
        emit(
            f"# exit {code}, {elapsed:.1f} s ({elapsed / 60:.1f} min), "
            f"peak memory {peak:.2f} GB (largest single process)\n"
        )
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
