#!/usr/bin/env python3
"""Print the first 32 sampled IDs from the x86 pipeline-parallel reference run."""

import argparse
import json
import subprocess
import sys

PROMPT = "Explain step by step how to add the integers from 1 to 100."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buddy-cli", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run = subprocess.run(
        [
            args.buddy_cli,
            "--model",
            args.model,
            "--prompt",
            PROMPT,
            "--temperature",
            "0",
            "--max-tokens",
            "64",
            "--stream-jsonl",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    tokens = []
    done = False
    for line in run.stdout.splitlines():
        event = json.loads(line)
        if event["event"] == "token" and event["token_id"] >= 0:
            tokens.append(event["token_id"])
        elif event["event"] == "done":
            done = True
    if not done or len(tokens) < 32:
        sys.exit(f"Incomplete baseline: {len(tokens)} sampled IDs, done={done}")
    print(json.dumps({"prompt": PROMPT, "token_ids": tokens[:32]}))


if __name__ == "__main__":
    main()
