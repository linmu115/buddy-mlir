"""Compare speculative CLI, concurrent slots and slot reuse with greedy outputs."""

import argparse
import json
import os
import shlex
import subprocess

from route_b_test_support import Resident


def run(args):
    n = args.cut.count(",") + 1
    cmd = [
        *shlex.split(args.mpi),
        "-n",
        str(n),
        *shlex.split(args.launcher),
        args.host,
        "--bundle",
        args.bundle,
        "--ranges",
        args.cut,
    ]
    env = os.environ.copy()
    prompts = {"A": "1" * 21, "B": "2" * 12, "C": "1" * 21}
    refs = {}
    for enabled in (False, True):
        launch = cmd + (["--mtp"] if enabled else [])
        for key in ("A", "B"):
            rows = 73 if key == "A" else 64
            p = subprocess.run(
                [
                    *launch,
                    "--prompt",
                    prompts[key],
                    "--total-tokens",
                    str(rows + 12),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )
            if p.returncode:
                raise RuntimeError(p.stderr)
            j = json.loads(p.stdout)
            assert j["prompt_tokens"] == rows and len(j["token_ids"]) == 13, j
            if enabled:
                assert j["token_ids"] == refs[key], (j, refs[key])
                if args.case == "accept":
                    assert (
                        j["mtp_accepted"] == j["mtp_proposed"]
                        and j["mtp_proposed"] >= 6
                    ), j
                if args.case == "mixed":
                    assert 0 < j["mtp_accepted"] < j["mtp_proposed"], j
            else:
                refs[key] = j["token_ids"]
            print(
                json.dumps(
                    dict(
                        cut=args.cut,
                        mode=args.case,
                        mtp=enabled,
                        prompt=key,
                        **j,
                    )
                ),
                flush=True,
            )
    with Resident(
        [*cmd, "--mtp", "--serve", "--slots", "2"], env, event_timeout=180
    ) as srv:
        srv.until(lambda e: e["event"] == "ready")
        # Enqueue both at admission, then reuse the first released slot.
        srv.submit("A", prompts["A"], max_new_tokens=13)
        srv.submit("B", prompts["B"], max_new_tokens=3)
        srv.done("B")
        srv.submit("C", prompts["C"], max_new_tokens=5)
        srv.until(lambda _: sum(e["event"] == "done" for e in srv.log) == 3)
        srv.sequences(
            {"A": refs["A"], "B": refs["B"][:3], "C": refs["A"][:5]}, "ABC"
        )
        slots = {
            e["req_id"]: e["slot"] for e in srv.log if e["event"] == "accepted"
        }
        assert slots["A"] != slots["B"] and slots["C"] == slots["B"], slots
        print(
            json.dumps(
                {
                    "cut": args.cut,
                    "mode": args.case,
                    "slots": slots,
                    "serve_tokens": 21,
                    "done": [e for e in srv.log if e["event"] == "done"],
                    "bit_diff": 0,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("host", "bundle", "cut", "case"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--launcher", default="")
    p.add_argument("--mpi", default="mpiexec")
    run(p.parse_args())
