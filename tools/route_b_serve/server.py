#!/usr/bin/env python3
"""OpenAI-compatible HTTP/SSE gateway for the resident Route B MPI engine."""

import argparse
import codecs
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def byte_alphabet():
    direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
    mapped = [b for b in range(256) if b not in direct]
    return dict(
        [(chr(b), b) for b in direct]
        + [(chr(256 + i), b) for i, b in enumerate(mapped)]
    )


BYTE_ALPHABET = byte_alphabet()


class TokenDecoder:
    def __init__(self, vocab, stop):
        self.vocab = vocab
        self.stop = stop
        # An incomplete UTF-8 suffix at the generation limit cannot be
        # displayed. Keep it buffered throughout generation, then omit it.
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    def feed(self, token_id):
        if token_id in self.stop:
            return ""
        return self.utf8.decode(self.vocab[token_id])

    def finish(self):
        return self.utf8.decode(b"", final=True)


def read_vocab(path):
    with open(path, encoding="utf-8", newline="\n") as file:
        return [
            b"".join(
                bytes((BYTE_ALPHABET[ch],))
                if ch in BYTE_ALPHABET
                else ch.encode("utf-8")
                for ch in line.removesuffix("\n")
            )
            for line in file
        ]


def mpi_command(args):
    """Assemble the common launch prefix, also usable for cold CLI runs."""
    if not args.hosts:
        return [args.mpiexec, "-n", str(args.ranks), args.binary]
    hosts = args.hosts.split(",")
    root = str(getattr(args, "tp_bundle", None) or args.bundle.parent)
    mpi_lib = str(Path(args.mpiexec).resolve().parent.parent / "lib")
    env = {
        "FI_PROVIDER": "tcp",
        "FI_TCP_NODELAY": "1",
        "MPIR_CVAR_NETWORK_IFACE": "end0",
        "MPIR_CVAR_CH4_OFI_IFNAME": "end0",
        "FI_TCP_IFACE": "end0",
        "FI_SOCKETS_IFACE": "end0",
        "OMP_NUM_THREADS": "8",
        "BUDDY_THREAD_POOL_THREADS": "8",
        "MPIR_CVAR_ASYNC_PROGRESS": "0",
        "LD_LIBRARY_PATH": ":".join([mpi_lib, root, *args.library_dir]),
    }
    for name in ("BUDDY_PP_OBSERVE", "BUDDY_PP_STORAGE"):
        if name in os.environ:
            env[name] = os.environ[name]
    command = [
        args.mpiexec,
        "-n",
        str(len(hosts)),
        "-ppn",
        "1",
        "-hosts",
        args.hosts,
        "-iface",
        "end0",
        "-wdir",
        root,
    ]
    for name, value in env.items():
        command += ["-env", name, value]
    return [
        *command,
        "sh",
        "-c",
        'set -e; test "$(cat /sys/class/net/end0/speed)" = 1000; '
        "echo 0 > /proc/set_ai_thread; "
        'vlen=$("$1" 2>&1); printf "%s\\n" "$vlen" >&2; '
        'case "$vlen" in *"vlen=1024") ;; *) exit 1;; esac; shift; exec "$@"',
        "_",
        args.vlen_probe,
        args.binary,
    ]


class Engine:
    def __init__(self, command, args):
        self.lock = threading.Lock()
        self.requests = {}
        self.ready = threading.Event()
        self.exited = threading.Event()
        self.accepting = True
        self.info = None
        self.hosts = args.hosts.split(",") if args.hosts else []
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self.read_events, daemon=True)
        self.reader.start()

    def read_events(self):
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                    kind = event["event"]
                except (ValueError, TypeError, KeyError):
                    print(line.rstrip(), file=sys.stderr, flush=True)
                    continue
                if kind == "ready":
                    self.info = event
                    self.ready.set()
                    continue
                with self.lock:
                    request = self.requests.get(event.get("req_id"))
                    if request is None:
                        continue
                    channel, active = request
                    if kind in ("done", "error"):
                        del self.requests[event["req_id"]]
                    elif kind == "accepted" and not active:
                        self.requests[event["req_id"]] = channel, True
                    channel.put(event)
        finally:
            with self.lock:
                self.accepting = False
                for channel, _ in self.requests.values():
                    channel.put(
                        {
                            "event": "error",
                            "message": "Route B engine exited",
                            "engine_failure": True,
                        }
                    )
                self.requests.clear()
            self.exited.set()
            self.ready.set()

    def submit(self, request_id, prompt, max_tokens):
        channel = queue.Queue()
        command = {
            "cmd": "submit",
            "req_id": request_id,
            ("chat" if isinstance(prompt, dict) else "prompt"): prompt,
            "max_new_tokens": max_tokens,
        }
        with self.lock:
            if not self.accepting or self.process.poll() is not None:
                raise RuntimeError("Route B engine is unavailable")
            self.requests[request_id] = channel, False
            try:
                self.write(command)
            except OSError:
                self.requests.pop(request_id, None)
                raise
        return channel

    def write(self, command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()

    def counts(self):
        with self.lock:
            active = sum(active for _, active in self.requests.values())
            return active, len(self.requests) - active

    def close(self, timeout):
        with self.lock:
            self.accepting = False
            if not self.process.stdin.closed:
                try:
                    self.write({"cmd": "shutdown"})
                except OSError:
                    pass
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    return self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    continue
            raise

    def memory(self):
        results = []
        for rank, pid in enumerate(self.info["pids"]):
            host = self.hosts[rank] if self.hosts else None
            result = {"rank": rank, "host": host, "pid": pid}
            path = f"/proc/{int(pid)}/status"
            try:
                if rank > 0 and host:
                    status = subprocess.check_output(
                        [
                            "ssh",
                            "-n",
                            "-oBatchMode=yes",
                            "-oConnectTimeout=3",
                            host,
                            "cat",
                            path,
                        ],
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=5,
                    )
                else:
                    status = Path(path).read_text()
                for line in status.splitlines():
                    field, _, value = line.partition(":")
                    if field in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
                        result[field + "_kib"] = int(value.split()[0])
            except (OSError, subprocess.SubprocessError) as error:
                result["error"] = str(error)
            results.append(result)
        return results


class Gateway(ThreadingHTTPServer):
    def __init__(self, address, engine, manifest, vocab):
        super().__init__(address, Handler)
        self.engine = engine
        self.manifest = manifest
        self.model = manifest["model_id"]
        self.vocab = vocab
        self.stop = {manifest["eos_id"], *manifest.get("stop_token_ids", [])}
        if manifest.get("tokenizer") == "qwen3":
            self.stop.update((151643, 151645))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def error_body(message, kind="invalid_request_error"):
        return {"error": {"message": message, "type": kind}}

    def do_GET(self):
        service = self.server
        engine = service.engine
        if self.path == "/health":
            active, queued = engine.counts()
            info = engine.info
            up = not engine.exited.is_set()
            self.send_json(
                200 if up else 503,
                {
                    "status": "ready" if up else "unavailable",
                    "model": service.model,
                    "ranges": info["ranges"],
                    "wire": info["wire"],
                    "slots": info["slots"],
                    "context": info["context"],
                    "active": active,
                    "queued": queued,
                    "memory": engine.memory(),
                },
            )
        elif self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": service.model,
                            "object": "model",
                            "owned_by": "buddy",
                        }
                    ],
                },
            )
        else:
            self.send_json(404, self.error_body("Unknown endpoint"))

    def read_request(self):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 1024 * 1024:
            raise ValueError("Expected a JSON body of up to 1 MiB")
        payload = json.loads(self.rfile.read(size))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        if self.server.manifest.get("tokenizer_assets"):
            options = payload.get("chat_template_kwargs", {})
            if not isinstance(options, dict):
                raise ValueError("chat_template_kwargs must be an object")
            thinking = payload.get(
                "enable_thinking", options.get("enable_thinking", True)
            )
            if type(thinking) is not bool:
                raise ValueError("enable_thinking must be boolean")
            if not messages or not all(
                isinstance(item, dict) for item in messages
            ):
                raise ValueError("messages must be a nonempty array of objects")
            json.dumps(messages, ensure_ascii=False).encode("utf-8")
            prompt = {"messages": messages, "enable_thinking": thinking}
            for key in (
                "reasoning_effort",
                "preserve_thinking",
                "add_vision_id",
            ):
                if key in options:
                    prompt[key] = options[key]
            for key in ("reasoning_effort", "tools"):
                if key in payload:
                    prompt[key] = payload[key]
        else:
            prompt = next(
                (
                    item.get("content")
                    for item in reversed(messages)
                    if isinstance(item, dict) and item.get("role") == "user"
                ),
                None,
            )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Expected a non-empty last user message")
            # Reject JSON scalar escapes for isolated surrogates before writing
            # the request to the engine's UTF-8 JSON parser.
            prompt.encode("utf-8")
        max_tokens = payload.get("max_tokens", 0)
        if (
            type(max_tokens) is not int
            or not 0 <= max_tokens <= self.server.engine.info["context"]
        ):
            raise ValueError("max_tokens is outside the model context")
        streaming = payload.get("stream", False)
        if type(streaming) is not bool:
            raise ValueError("stream must be a boolean")
        return (
            prompt,
            max_tokens,
            streaming,
            payload.get("route_b_metrics") is True,
        )

    def sse(self, value):
        data = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False)
        )
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, self.error_body("Unknown endpoint"))
            return
        try:
            self.chat()
        except (ConnectionError, TimeoutError):
            # The engine keeps the slot until its current request ends.
            pass

    def chat(self):
        service = self.server
        try:
            prompt, max_tokens, streaming, metrics = self.read_request()
        except ValueError as error:
            self.send_json(400, self.error_body(str(error)))
            return
        request_id = "chatcmpl-" + uuid.uuid4().hex
        try:
            channel = service.engine.submit(request_id, prompt, max_tokens)
        except (OSError, RuntimeError):
            self.send_json(
                503,
                self.error_body(
                    "Route B engine is unavailable", "server_error"
                ),
            )
            return

        common = {
            "id": request_id,
            "created": int(time.time()),
            "model": service.model,
        }
        decoder = TokenDecoder(service.vocab, service.stop)
        started = False
        prompt_tokens = completion_tokens = 0
        content = []

        def chunk(delta, reason=None, event=None):
            data = dict(
                common,
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": delta, "finish_reason": reason}],
            )
            if metrics and event:
                data["route_b"] = event
            self.sse(data)

        while True:
            event = channel.get()
            kind = event["event"]
            if kind == "error":
                failure = event.get("engine_failure")
                error = self.error_body(
                    event["message"],
                    "server_error" if failure else "invalid_request_error",
                )
                if started:
                    self.sse(error)
                    self.sse("[DONE]")
                else:
                    self.send_json(503 if failure else 400, error)
                return
            if kind == "accepted":
                prompt_tokens = event["prompt_tokens"]
                if streaming:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "text/event-stream; charset=utf-8"
                    )
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    started = True
                    chunk({"role": "assistant"}, event=event)
            elif kind == "token":
                completion_tokens += 1
                text = decoder.feed(event["token_id"])
                if streaming:
                    if text or metrics:
                        chunk({"content": text} if text else {}, event=event)
                else:
                    content.append(text)
            elif kind == "done":
                tail = decoder.finish()
                reason = "stop" if event["reason"] == "eos" else "length"
                if streaming:
                    if tail:
                        chunk({"content": tail})
                    chunk({}, reason, event)
                    self.sse("[DONE]")
                else:
                    content.append(tail)
                    self.send_json(
                        200,
                        dict(
                            common,
                            object="chat.completion",
                            choices=[
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "".join(content),
                                    },
                                    "finish_reason": reason,
                                }
                            ],
                            usage={
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens
                                + completion_tokens,
                            },
                        ),
                    )
                return


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    bundle = parser.add_mutually_exclusive_group(required=True)
    bundle.add_argument("--bundle", type=Path)
    bundle.add_argument("--tp-bundle", type=Path)
    parser.add_argument("--spec-mode", choices=["mtp", "ngram", "dspark"])
    parser.add_argument("--max-draft", type=int, choices=range(1, 4))
    parser.add_argument("--dspark-tau", type=float)
    parser.add_argument("--dspark-min-p", type=float)
    parser.add_argument("--binary", default="buddy-pp-route-b-serve")
    parser.add_argument("--ranges")
    parser.add_argument(
        "--vocab-file",
        type=Path,
        help="Local vocabulary for a remote TP endpoint",
    )
    parser.add_argument("--wire", default="f16", choices=["f16", "f32"])
    parser.add_argument("--slots", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--hosts", help="Comma-separated K3 end0 addresses, in rank order"
    )
    parser.add_argument(
        "--vlen-probe", help="K3 executable that prints VLEN on each rank"
    )
    parser.add_argument("--library-dir", action="append", default=[])
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.tp_bundle:
        args.tp_bundle = args.tp_bundle.resolve()
        if args.ranges or (args.hosts and len(args.hosts.split(",")) != 2):
            parser.error(
                "TP requires exactly two ranks and owns its layer ranges"
            )
        args.ranks = 2
    else:
        args.bundle = args.bundle.resolve()
        if not args.ranges:
            parser.error("--bundle requires --ranges")
    if args.hosts and (
        not args.vlen_probe or not Path(args.mpiexec).is_absolute()
    ):
        parser.error(
            "K3 --hosts requires --vlen-probe and an absolute --mpiexec"
        )
    return args


def main(args):
    if args.tp_bundle:
        # Each board can contain just its rank manifest.
        path = next(args.tp_bundle.glob("rank[01]/route-b.json"), None)
        if path is None:
            raise FileNotFoundError(f"no TP rank manifest in {args.tp_bundle}")
        model_args = ["--tp-bundle", str(args.tp_bundle), "--launcher-stdin"]
    else:
        path = args.bundle
        model_args = ["--bundle", str(path), "--ranges", args.ranges]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    command = (
        mpi_command(args)
        + model_args
        + ["--wire", args.wire, "--serve", "--slots", str(args.slots)]
    )
    for name in ("spec_mode", "max_draft", "dspark_tau", "dspark_min_p"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    engine = Engine(command, args)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not stopping.is_set() and time.monotonic() < deadline:
            if engine.ready.wait(0.1):
                break
        if not stopping.is_set():
            if not engine.info or engine.exited.is_set():
                raise RuntimeError("Route B engine did not become ready")
            vocab_path = args.vocab_file or Path(engine.info["vocab"])
            if args.vocab_file:
                with vocab_path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != manifest["vocab_sha256"]:
                    raise ValueError(
                        "local vocabulary digest differs from the bundle"
                    )
            vocab = read_vocab(vocab_path)
            with Gateway(
                (args.bind, args.port), engine, manifest, vocab
            ) as server:
                server.timeout = 0.25
                print(
                    json.dumps(
                        dict(
                            engine.info,
                            event="http_ready",
                            port=server.server_port,
                        )
                    ),
                    flush=True,
                )
                while not stopping.is_set() and not engine.exited.is_set():
                    server.handle_request()
    finally:
        rc = engine.close(args.shutdown_timeout)
    return rc


if __name__ == "__main__":
    sys.exit(main(parse_args()))
