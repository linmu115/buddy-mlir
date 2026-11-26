#!/usr/bin/env python3
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import server
from qwen_bpe import Tokenizer
from suite_guard import owned_round, terminate_group

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exp16-resume"
OUT.mkdir(exist_ok=True)
os.chdir(ROOT)
mp = Path(
    os.environ.get("MPIEXEC") or (ROOT / "mpiexec.path").read_text().strip()
)
os.environ["PATH"] = str(mp.parent) + ":" + os.environ["PATH"]
vocab = server.read_vocab(ROOT / "vocab.txt")
stop = {248044, 248046}
enc = Tokenizer(ROOT)
main = "0:20,20:64"
hosts = os.environ.get("K3_HOSTS") or (ROOT / "k3-hosts").read_text().strip()
local_host, command_peer = hosts.split(",")
port = 18082
tech = "请用两句话解释操作系统如何利用虚拟内存隔离不同进程。"
code = "Write a Python function that returns unique items in their original order. Include a short example."
logic = "A bag has three red balls and two blue balls. Draw two without replacement. What is the probability both are red? Show your calculation."
mathq = "请心算：一只鸡和一只兔，共有几条腿？用最短思考，只用一句中文给出结果。"
prompts = {"tech": tech, "code": code, "logic": logic}


def dump(file, obj):
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(file)


def event(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def chat(text, thinking):
    return {
        "messages": [{"role": "user", "content": text}],
        "enable_thinking": thinking,
    }


def command(cut, label, cold=False):
    ns = SimpleNamespace(
        hosts=hosts,
        bundle=ROOT / "route-b.json",
        mpiexec=str(mp),
        library_dir=[],
        vlen_probe=str(ROOT / "vlen"),
        binary=str(ROOT / "rank-test"),
    )
    argv = server.mpi_command(ns)
    idx = argv.index("sh")
    argv[idx:idx] = [
        "-env",
        "EXP16_LABEL",
        label,
        "-env",
        "BUDDY_PP_TIMING",
        "1",
        "-env",
        "BUDDY_PP_OBSERVE",
        "1",
    ] + (["-env", "EXP16_COLD", "1"] if cold else [])
    return argv + ["--bundle", str(ns.bundle), "--ranges", cut, "--wire", "f32"]


def text_for(ids):
    dec = server.TokenDecoder(vocab, stop)
    return "".join(dec.feed(i) for i in ids) + dec.finish()


def req(proc, key, txt, think, n):
    start = time.monotonic()
    proc.stdin.write(
        json.dumps(
            {
                "cmd": "submit",
                "req_id": key,
                "chat": chat(txt, think),
                "max_new_tokens": n,
            }
        )
        + "\n"
    )
    proc.stdin.flush()
    es = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine closed during " + key)
        e = json.loads(line)
        es.append(dict(e, client_s=time.monotonic() - start))
        if e["event"] == "error":
            raise RuntimeError(e)
        if e["event"] == "done":
            break
    toks = [e for e in es if e["event"] == "token"]
    ids = [e["token_id"] for e in toks]
    assert es[0]["prompt_tokens"] == len(enc.chat(chat(txt, think)))
    return {
        "prompt": txt,
        "thinking": think,
        "prompt_tokens": es[0]["prompt_tokens"],
        "token_ids": ids,
        "text": text_for(ids),
        "ttft": toks[0]["client_s"],
        "engine_ttft": toks[0]["ttft_us"] / 1e6,
        "done": es[-1],
        "events": es,
    }


def launch_case(label, cut, mtp, requests, cold):
    with owned_round(ROOT, OUT, label, command_peer):
        return measure_case(label, cut, mtp, requests, cold)


def measure_case(label, cut, mtp, requests, cold):
    result = OUT / (label + ".json")
    if result.exists():
        return json.loads(result.read_text())
    argv = (
        command(cut, label, cold)
        + ["--serve", "--slots", "1"]
        + (["--mtp"] if mtp else [])
    )
    event(label)
    with (OUT / (label + ".stderr")).open("w") as log:
        begin = time.monotonic()
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        data = {}
        try:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine failed before ready: " + label)
            ready = json.loads(line)
            assert ready["event"] == "ready", ready
            data = {
                "ready": ready,
                "launch_to_ready": time.monotonic() - begin,
                "requests": {},
            }
            for key, txt, think, n in requests:
                data["requests"][key] = req(proc, key, txt, think, n)
                if key == requests[0][0]:
                    data["launch_to_first_token"] = (
                        time.monotonic()
                        - begin
                        - data["requests"][key]["events"][-1]["client_s"]
                        + data["requests"][key]["ttft"]
                    )
                event(
                    f"{label} {key} {len(data['requests'][key]['token_ids'])} tokens"
                )
            proc.stdin.write('{"cmd":"shutdown"}\n')
            proc.stdin.flush()
            proc.stdin.close()
            rc = proc.wait(timeout=900)
            if rc:
                raise RuntimeError((label, rc))
        finally:
            terminate_group(proc)
        dump(result, data)
        return data


def pair(key):
    modes = (True, False) if key == "code" else (False, True)
    results = []
    for mode in modes:
        label = key + ("-on" if mode else "-off")
        # Two decode tokens fault both the prefill and decode layouts into the process.
        data = launch_case(
            label,
            main,
            mode,
            [
                ("cold", prompts[key], False, 3),
                ("hot", prompts[key], False, 28),
            ],
            True,
        )
        ids = data["requests"]["hot"]["token_ids"]
        assert ids[:3] == data["requests"]["cold"]["token_ids"], label
        results.append(ids)
    if not 24 <= len(results[0]) <= 32:
        raise ValueError(("output length", key, len(results[0])))
    if results[0] != results[1]:
        raise ValueError(("MTP divergence", key, results))


def run_cli(label, argv):
    with (OUT / (label + ".stderr")).open("w") as log:
        p = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = p.communicate(timeout=900)
            if p.returncode:
                raise RuntimeError((label, p.returncode))
        finally:
            terminate_group(p)
    return stdout


def long_test():
    dst = OUT / "long128.json"
    if dst.exists():
        return
    txt = "根据下面的测量记录，用一句话概括总体趋势。\n"
    for i in range(1, 8):
        txt += f"第{i}次运行耗时{80 - i}秒，测量条件相同。\n"
    txt += "列出首尾差值。"
    n = len(enc.chat(chat(txt, False)))
    assert n == 128
    # Exact CLI total limit gives two generated tokens including prefill.
    argv = command(main, "long128") + [
        "--no-thinking",
        "--prompt",
        txt,
        "--total-tokens",
        str(n + 1),
    ]
    start = time.monotonic()
    stdout = run_cli("long128", argv)
    data = json.loads(stdout)
    assert data["prompt_tokens"] == n
    dump(
        dst,
        dict(
            argv=argv,
            prompt=txt,
            wall_s=time.monotonic() - start,
            **data,
            text=text_for(data["token_ids"]),
        ),
    )


def thinking_cli(enabled, resident):
    label = "cli-thinking-" + str(int(enabled))
    dest = OUT / (label + ".json")
    flag = "--enable-thinking" if enabled else "--no-thinking"
    req = chat(mathq, enabled)
    prefix = enc.template.render(**req, add_generation_prompt=True).split(
        "<|im_start|>assistant\n"
    )[-1]
    assert prefix == ("<think>\n" if enabled else "<think>\n\n</think>\n\n")
    n = len(enc.chat(req))
    ids = resident["token_ids"]
    if dest.exists():
        result = json.loads(dest.read_text())
    else:
        event(label)
        argv = command(main, label) + [
            "--mtp",
            flag,
            "--prompt",
            mathq,
            "--total-tokens",
            str(n + len(ids) - 1),
        ]
        with owned_round(ROOT, OUT, label, command_peer):
            result = json.loads(run_cli(label, argv))
        result.update(
            argv=argv,
            prompt=mathq,
            template_prefix=prefix,
            text=text_for(result["token_ids"]),
        )
        dump(dest, result)
    assert result["token_ids"] == ids
    raw = b"".join(vocab[t] for t in ids if t not in stop).decode("utf-8")
    assert raw == result["text"] and "�" not in raw
    assert (prefix + raw).count("</think>") == 1


def http_req(key, txt, n, thinking=False, started=None):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = dict(
        **chat(txt, thinking),
        model="Qwen/Qwen3.8-27B",
        max_tokens=n,
        stream=True,
        route_b_metrics=True,
    )
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    begin = time.monotonic()
    ev = []
    ids = []
    pieces = []
    end = False
    with urllib.request.urlopen(req, timeout=900) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                end = True
                break
            d = json.loads(line[6:])
            e = d.get("route_b", {})
            ev.append(dict(e, client_s=time.monotonic() - begin))
            if "token_id" in e:
                ids.append(e["token_id"])
            if started and e.get("event") == "accepted":
                started.set()
            pieces.append(d["choices"][0]["delta"].get("content", ""))
    content = "".join(pieces)
    raw = b"".join(vocab[t] for t in ids if t not in stop)
    # Compare the bytes on the wire with the actual tokens; a lossy decoder
    # could otherwise hide malformed output in a replacement-character check.
    assert end and ids and "�" not in content and raw.decode("utf-8") == content
    return {
        "key": key,
        "prompt": txt,
        "token_ids": ids,
        "text": "".join(pieces),
        "events": ev,
        "ttft": next(e["client_s"] for e in ev if e["event"] == "token"),
    }


def health():
    return json.load(
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=15)
    )


def snap():
    h = health()
    for row in h["memory"]:
        cmd = "cat /proc/meminfo"
        data = (
            Path("/proc/meminfo").read_text()
            if row["rank"] == 0
            else subprocess.check_output(
                ["ssh", "-n", "-oBatchMode=yes", row["host"], cmd],
                text=True,
                timeout=10,
            )
        )
        row["MemAvailable_kib"] = int(
            next(
                l for l in data.splitlines() if l.startswith("MemAvailable:")
            ).split()[1]
        )
        assert "VmHWM_kib" in row and "VmRSS_kib" in row
    return h


def http_test():
    dest = OUT / "http.json"
    if dest.exists():
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))
    with (OUT / "http.stderr").open("w") as log:
        begin = time.monotonic()
        p = subprocess.Popen(
            [str(ROOT / "start_27b_server.sh"), "--port", str(port), "--mtp"],
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=dict(os.environ, EXP16_BENCHMARK="1"),
        )
        first_req = "请原样输出以下内容：中文测试 OK，π≈3.14。"
        data = {}
        try:
            for i in range(240):
                if p.poll() is not None:
                    raise RuntimeError(("gateway failed", p.returncode))
                try:
                    h = snap()
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("gateway readiness")
            data.update(ready_s=time.monotonic() - begin, ready=h)
            a = http_req("first", first_req, 16)
            data["launch_to_first_token"] = (
                time.monotonic()
                - begin
                - a["events"][-1]["client_s"]
                + a["ttft"]
            )
            data.update(first=a, after_first=snap())
            b = http_req("repeat", first_req, 16)
            assert (
                a["token_ids"] == b["token_ids"]
                and a["events"][0]["slot"] == b["events"][0]["slot"]
            )
            data.update(repeat=b, after_repeat=snap())
            started = threading.Event()
            with ThreadPoolExecutor(2) as pool:
                fa = pool.submit(
                    http_req, "parallel-A", first_req, 12, False, started
                )
                try:
                    if not started.wait(180):
                        raise TimeoutError("no accepted SSE")
                    fb = pool.submit(
                        http_req, "parallel-B", "用中文写出 2 + 2 的结果。", 6
                    )
                    c, d = fa.result(), fb.result()
                except BaseException:
                    p.terminate()
                    raise
            assert c["events"][0]["slot"] != d["events"][0]["slot"], (c, d)
            data.update(parallel=[c, d], after_parallel=snap())
            for i in range(2):
                r = http_req("reuse-" + str(i), first_req, 4)
                assert r["token_ids"] == a["token_ids"][: len(r["token_ids"])]
                data.setdefault("reuse", []).append(
                    {"run": r, "health": snap()}
                )
            for flag in ("--think", "--no-think"):
                cli = subprocess.run(
                    [
                        str(Path.home() / "chat_27b.py"),
                        "--port",
                        str(port),
                        flag,
                        "--max-tokens",
                        "4",
                    ],
                    input="请回答：2+2 是多少？\n/quit\n",
                    text=True,
                    capture_output=True,
                    timeout=450,
                    check=True,
                )
                assert (
                    "请求失败" not in cli.stdout
                    and "�" not in cli.stdout
                    and cli.stdout.strip() != "你> 你>"
                )
                data.setdefault("chat_script", {})[flag] = cli.stdout
            data["final_health"] = snap()
        finally:
            p.send_signal(signal.SIGTERM)
            try:
                rc = p.wait(timeout=140)
            finally:
                terminate_group(p)
            if rc:
                raise RuntimeError(("gateway shutdown", rc))
        with socket.socket() as s:
            assert s.connect_ex(("127.0.0.1", port)) != 0
        data.update(port_closed=True, exit_code=rc)
        dump(dest, data)


def run():
    for key in prompts:
        pair(key)
    for enabled in (True, False):
        label = "thinking-" + str(int(enabled))
        data = launch_case(
            label,
            main,
            True,
            [("answer", mathq, enabled, 96 if enabled else 28)],
            False,
        )
        if enabled and "</think>" not in data["requests"]["answer"]["text"]:
            data = launch_case(
                "thinking-extended",
                main,
                True,
                [("answer", mathq, True, 240)],
                False,
            )
        thinking_cli(enabled, data["requests"]["answer"])
    with owned_round(ROOT, OUT, "long128", command_peer):
        long_test()
    with owned_round(ROOT, OUT, "http", command_peer):
        http_test()
    data = launch_case(
        "cut24", "0:24,24:64", False, [("short", tech, False, 16)], False
    )
    base = json.loads((OUT / "tech-off.json").read_text())["requests"]["hot"][
        "token_ids"
    ]
    assert data["requests"]["short"]["token_ids"] == base[:16]
    (OUT / "suite.ok").write_text("ok\n")
    event("complete")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run()
