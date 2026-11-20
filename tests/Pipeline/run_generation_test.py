"""Compare consecutive tokenized requests across MPI backends and a frame oracle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    launcher, flag, ranks, binary, mode = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = []
        for backend in (("async",) if mode == "throw" else ("blocking", "async")):
            prefix = root / backend
            proc = subprocess.run(
                [launcher, flag, ranks, binary, backend, prefix, mode],
                env=dict(os.environ, BUDDY_PP_TIMING="1"),
                capture_output=True, timeout=45, text=True,
            )
            if mode == "throw":
                assert proc.returncode != 0 and "injected stage exception" in proc.stderr, proc.stderr
                print("stage exception exited", proc.returncode)
                return
            assert proc.returncode == 0, proc.stderr
            outputs = [json.loads(line) for line in proc.stdout.splitlines()]
            assert len(outputs) == 12
            for i, (chunk, prompt) in enumerate((c, p) for c in (4, 1) for p in (1, 3, 4, 5, 8, 9)):
                tokens = outputs[i]["token_ids"]
                if prompt == 8:
                    assert tokens == [3]  # Real sampled EOS, hook disabled.
                else:
                    assert len(tokens) == 3 and 3 not in tokens
                path = f"{prefix}-{chunk}-{prompt}"
                logits = Path(path + ".logits").read_bytes()
                assert len(logits) == len(tokens) * 4 * 4
                if backend == "blocking":
                    baseline.append((tokens, logits))
                else:
                    assert baseline[i] == (tokens, logits)
                frames = []
                if prompt >= chunk:
                    for pos in range(0, prompt, chunk):
                        pos = min(pos, prompt - chunk)
                        frames.append(("prefill", pos, chunk, int(pos + chunk == prompt)))
                else:
                    if prompt > 1:
                        frames.append(("prefill", 0, chunk, 0))
                    frames.append(("decode", prompt - 1, 1, 1))
                frames.extend(("decode", prompt + j, 1, 1) for j in range(len(tokens) - 1))
                for rank in range(int(ranks)):
                    lines = Path(f"{path}.rank{rank}.log").read_text().splitlines()
                    expected = [f"rank={rank} phase={phase} step={s} pos={pos} rows={rows} logits={want}"
                                for s, (phase, pos, rows, want) in enumerate(frames)]
                    assert lines == expected, (lines, expected)
            if int(ranks) > 1 and backend == "async":
                for rank in range(int(ranks)):
                    assert proc.stderr.count(f"pp_drain rank={rank} pending=0") == 12
            print(backend, ranks, "ranks, 12 requests, frames/tokens/logits checked")


if __name__ == "__main__":
    main()
