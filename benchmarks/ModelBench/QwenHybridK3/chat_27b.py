#!/usr/bin/env python3
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser(
    description="27B 流式聊天，输入 /quit 退出，/clear 清空对话。"
)
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--max-tokens", type=int, default=256)
group = parser.add_mutually_exclusive_group()
group.add_argument("--think", dest="thinking", action="store_true")
group.add_argument("--no-think", dest="thinking", action="store_false")
parser.set_defaults(thinking=False)
args = parser.parse_args()
history = []
while True:
    try:
        prompt = input("你> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if prompt == "/quit":
        break
    if prompt == "/clear":
        history.clear()
        continue
    if not prompt:
        continue
    user = {"role": "user", "content": prompt}
    payload = {
        "model": "Qwen/Qwen3.8-27B",
        "messages": [*history, user],
        "enable_thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    pieces = []
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    history.extend(
                        [
                            user,
                            {"role": "assistant", "content": "".join(pieces)},
                        ]
                    )
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(event["error"])
                piece = event["choices"][0]["delta"].get("content", "")
                print(piece, end="", flush=True)
                pieces.append(piece)
    except (OSError, ValueError) as exc:
        print(f"请求失败：{exc}")
    print()
