#!/usr/bin/env python3
"""Sample one Linux rank without retaining its PMI pipes or following PID reuse."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

FIELDS = {
    "VmRSS",
    "VmHWM",
    "VmSwap",
    "RssAnon",
    "RssFile",
    "MemAvailable",
    "SwapFree",
}


def identity(pid):
    try:
        # comm may contain spaces or parentheses. The suffix starts at field 3.
        fields = (
            (Path("/proc") / str(pid) / "stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()
        )
        if fields[0] not in ("Z", "X"):
            return fields[19]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def sample(pid, birth, report):
    with report.open("w", buffering=1) as stream:
        while identity(pid) == birth:
            values = {"pid": pid, "time": time.time()}
            try:
                for path in (
                    Path(f"/proc/{pid}/status"),
                    Path("/proc/meminfo"),
                ):
                    for line in path.read_text().splitlines():
                        key, _, value = line.partition(":")
                        if key in FIELDS:
                            values[key] = int(value.split()[0])
            except (FileNotFoundError, ProcessLookupError):
                break
            if identity(pid) == birth:
                print(json.dumps(values), file=stream)
            time.sleep(0.5)


def launch(pid, prefix):
    birth = identity(pid)
    if birth is None:
        raise RuntimeError("rank exited before sampling")
    monitor = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "sample",
            str(pid),
            birth,
            str(prefix) + ".jsonl",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    record = prefix.with_suffix(".pids")
    record.write_text(
        json.dumps([(pid, birth), (monitor.pid, identity(monitor.pid))])
    )


def cleanup(directory, label):
    # Pin each process with a pidfd before checking its recorded start time.
    for record in directory.glob(label + "-*.pids"):
        for pid, birth in json.loads(record.read_text()):
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if birth and identity(pid) == birth:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "launch":
        launch(int(args[0]), Path(args[1]))
    elif action == "sample":
        sample(int(args[0]), args[1], Path(args[2]))
    elif action == "cleanup":
        cleanup(Path(args[0]), args[1])
    else:
        raise ValueError(action)
