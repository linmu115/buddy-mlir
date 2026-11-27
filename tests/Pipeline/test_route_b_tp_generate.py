"""Compare complete free-running MPI generations through the public CLI."""

import json
import subprocess
import sys


def main(launcher, executable, source, shards):
    base = [launcher, "-n", "1", executable, "--bundle", source]
    tp = [launcher, "-n", "2", executable, "--tp-bundle", shards]
    prompt = ["--prompt", "x", "--total-tokens", "68"]

    def generate(cmd):
        out = subprocess.run(
            cmd + prompt, check=True, capture_output=True, text=True, timeout=90
        )
        return json.loads(out.stdout)

    reference = generate(base)["token_ids"]
    if len(reference) < 3:
        raise AssertionError("fixture needs at least two decode steps")
    for wire in ("f32", "f16"):
        for mode in (
            [],
            ["--mtp"],
            ["--spec-mode", "mtp", "--max-draft", "2"],
            ["--spec-mode", "mtp", "--max-draft", "3"],
            ["--spec-mode", "dspark"],
        ):
            result = generate(tp + ["--wire", wire] + mode)
            assert result["token_ids"] == reference, (wire, mode, result)
            if mode:
                assert result["mtp_proposed"] > 0
            print(wire, mode or "decode", len(reference), "matched tokens")


if __name__ == "__main__":
    main(*sys.argv[1:])
