import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rank_sample", Path(__file__).with_name("rank-sample.py")
)
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


class SamplerTest(unittest.TestCase):
    def test_zombie_releases_inherited_pipe(self):
        reader, writer = os.pipe()
        rank = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(20)"]
        )
        self.addCleanup(rank.wait)
        self.addCleanup(rank.kill)
        self.addCleanup(os.close, reader)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "case-rank"
            try:
                subprocess.run(
                    [
                        sys.executable,
                        sampler.__file__,
                        "launch",
                        str(rank.pid),
                        str(prefix),
                    ],
                    pass_fds=(writer,),
                    check=True,
                    timeout=5,
                )
            finally:
                os.close(writer)
            os.set_blocking(reader, False)
            self.assertEqual(os.read(reader, 1), b"")
            (_, _), (monitor, _) = json.loads(
                prefix.with_suffix(".pids").read_text()
            )
            self.assertEqual(os.getsid(monitor), monitor)
            rank.kill()
            os.waitid(os.P_PID, rank.pid, os.WEXITED | os.WNOWAIT)
            deadline = time.monotonic() + 3
            while sampler.identity(monitor) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNone(sampler.identity(rank.pid))
            self.assertIsNone(sampler.identity(monitor))

    def test_cleanup_targets_recorded_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with subprocess.Popen(["sleep", "10"]) as rank:
                record = root / "case-rank.pids"
                record.write_text(json.dumps([(rank.pid, "different-start")]))
                sampler.cleanup(root, "case")
                self.assertIsNone(rank.poll())
                record.write_text(
                    json.dumps([(rank.pid, sampler.identity(rank.pid))])
                )
                sampler.cleanup(root, "case")
                self.assertEqual(rank.wait(timeout=2), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
