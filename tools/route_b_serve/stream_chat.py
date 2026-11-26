#!/usr/bin/env python3
"""Print the Route B gateway's streaming assistant response in the terminal."""

import argparse
import json
import sys
import urllib.error
import urllib.request


def chat(url, prompt, max_tokens):
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                print(flush=True)
                return
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"]["message"])
            print(
                chunk["choices"][0]["delta"].get("content", ""),
                end="",
                flush=True,
            )
    raise RuntimeError("stream closed before [DONE]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()
    try:
        if args.prompt:
            chat(args.url, " ".join(args.prompt), args.max_tokens)
        else:
            while True:
                prompt = input("> ")
                if prompt.strip():
                    chat(args.url, prompt, args.max_tokens)
    except (EOFError, KeyboardInterrupt):
        print()
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"), file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
