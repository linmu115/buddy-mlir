import contextlib
import os
import signal
import subprocess
import sys

TIMEOUT = 900


def expired(*_):
    raise TimeoutError("900 second round deadline")


@contextlib.contextmanager
def owned_round(root, out, label, peer):
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        clean = [
            sys.executable,
            str(root / "rank-sample.py"),
            "cleanup",
            str(out),
            label,
        ]
        subprocess.run(clean, check=True, timeout=10)
        subprocess.run(
            [
                "ssh",
                "-n",
                "-oBatchMode=yes",
                "-oConnectTimeout=5",
                peer,
                *clean,
            ],
            check=True,
            timeout=15,
        )


def terminate_group(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            break
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        # Recheck the group after its leader has exited.
    p.wait(timeout=10)
