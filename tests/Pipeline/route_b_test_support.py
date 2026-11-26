"""JSONL protocol driver shared by the Route B service tests."""

import json
import queue
import subprocess
import tempfile
import threading


class Resident:
    def __init__(self, cmd, env, event_timeout=15):
        self.event_timeout = event_timeout
        self.log = []
        self.events = queue.Queue()
        self.stderr = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        for line in self.proc.stdout:
            self.events.put(line)
        self.events.put(None)

    def send(self, **event):
        self.proc.stdin.write(json.dumps(event) + "\n")
        self.proc.stdin.flush()

    def submit(self, key, prompt, **limit):
        self.send(cmd="submit", req_id=key, prompt=prompt, **limit)

    def until(self, predicate):
        while True:
            line = self.events.get(timeout=self.event_timeout)
            if line is None:
                self.stderr.seek(0)
                raise AssertionError(self.stderr.read())
            event = json.loads(line)
            self.log.append(event)
            if predicate(event):
                return event

    def done(self, key):
        return self.until(lambda e: e["event"] == "done" and e["req_id"] == key)

    def sequences(self, refs, keys):
        for key in keys:
            tokens = [
                e
                for e in self.log
                if e["event"] == "token" and e["req_id"] == key
            ]
            assert [e["token_id"] for e in tokens] == refs[key], (key, tokens)
            assert [e["step"] for e in tokens] == list(range(len(tokens)))
            assert [e["is_prefill"] for e in tokens] == [True] + [False] * (
                len(tokens) - 1
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                self.send(cmd="shutdown")
                assert self.proc.wait(timeout=15) == 0
            else:
                self.proc.kill()
                self.proc.wait()
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.reader.join(timeout=5)
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.stderr.close()
