"""Stage S3. Compare resident decode events against a per-wire greedy reference."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def bench(args):
    root = args.release.resolve()
    sys.path.insert(0, str(root))
    from server import Engine, mpi_command

    launch = SimpleNamespace(
        bundle=root / "route-b.json",
        binary=str(root / "rank-27b"),
        mpiexec=(root / "mpiexec.path").read_text().strip(),
        hosts=(root / "k3-hosts").read_text().strip(),
        vlen_probe=str(root / "vlen"),
        library_dir=[],
    )
    prefix = mpi_command(launch)
    prompts = [
        "Explain why the sky is blue.",
        "Write a Python function to merge two sorted lists, with tests.",
    ]
    variants = [
        ("greedy", []),
        ("mtp1", ["--mtp"]),
        ("mtp2", ["--spec-mode", "mtp", "--max-draft", "2"]),
        ("mtp3", ["--spec-mode", "mtp", "--max-draft", "3"]),
        ("ngram", ["--spec-mode", "ngram", "--max-draft", "3"]),
        (
            "dspark",
            [
                "--spec-mode",
                "dspark",
                "--max-draft",
                "3",
                "--dspark-tau",
                "0.55",
                "--dspark-min-p",
                "0.35",
            ],
        ),
    ]
    reference = {}
    for run in range(args.repetitions):
        for name, options in variants if run % 2 == 0 else reversed(variants):
            cmd = prefix + [
                "--bundle",
                str(launch.bundle),
                "--ranges",
                "0:20,20:64",
                "--wire",
                args.wire,
                "--serve",
                "--slots",
                "1",
                *options,
            ]
            engine = Engine(cmd, launch)
            try:
                if not engine.ready.wait(900) or engine.info is None:
                    raise RuntimeError("ranks did not become ready")
                for i, prompt in enumerate(["Hello", *prompts]):
                    events = engine.submit(str(i), prompt, 8 if i == 0 else 128)
                    tokens = []
                    while True:
                        event = events.get(timeout=900)
                        if event["event"] == "error":
                            raise RuntimeError(event)
                        if event["event"] == "token":
                            tokens.append(event["token_id"])
                        if event["event"] == "done":
                            break
                    if i == 0:
                        continue
                    if tokens != reference.setdefault(i, tokens):
                        raise ValueError(
                            f"token difference: {name}, prompt {i}"
                        )
                    print(
                        json.dumps(
                            dict(
                                run=run + 1,
                                mode=name,
                                wire=args.wire,
                                prompt=prompt,
                                token_ids=tokens,
                                **event,
                            )
                        ),
                        flush=True,
                    )
            finally:
                code = engine.close(120)
            if code:
                raise RuntimeError(f"MPI exit {code}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--wire", choices=("f32", "f16"), required=True)
    p.add_argument("--repetitions", type=int, default=2)
    bench(p.parse_args())
