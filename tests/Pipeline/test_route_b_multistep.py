"""Long two-rank greedy parity and adaptive width coverage on the B4 mixed fixture."""

import argparse
import json
import subprocess


def matrix(args):
    for wire in ("f32", "f16"):
        reference = None
        for mode, depth in (
            (None, 0),
            ("mtp", 1),
            ("mtp", 2),
            ("mtp", 3),
            ("ngram", 3),
            ("dspark", 3),
            ("dspark-strict", 3),
        ):
            if mode == "dspark-strict" and args.scenario != "mixed":
                continue
            cmd = [
                args.mpi,
                "-n",
                "2",
                args.host,
                "--bundle",
                args.bundle,
                "--ranges",
                "0:2,2:4",
                "--prompt",
                "1" * 21,
                "--total-tokens",
                "202",
                "--wire",
                wire,
            ]
            if mode:
                cmd.extend(
                    [
                        "--spec-mode",
                        mode.split("-")[0],
                        "--max-draft",
                        str(depth),
                    ]
                )
            if mode == "dspark-strict":
                cmd.extend(["--dspark-tau", ".99999", "--dspark-min-p", "0"])
            result = json.loads(subprocess.check_output(cmd, timeout=180))
            tokens = result.pop("token_ids")
            assert len(tokens) >= 128
            if not mode:
                reference = tokens
                continue
            assert tokens == reference, (wire, mode, depth)
            hist = result["draft_k_histogram"]
            if mode == "dspark" and args.scenario == "mixed":
                assert all(hist) and 0 < result["ngram_steps"] < sum(hist), (
                    result
                )
            if mode == "dspark-strict":
                # The two-token MTP proposal occurs before the lookup cycle,
                # with room for three: this width came from the confidence gate.
                assert all(hist) and result["mtp_k_histogram"][1] > 0, result
            if mode == "mtp" and args.scenario == "accept":
                assert result["draft_depth_accepted"][depth - 1] > 0, result
            if mode == "mtp":
                assert hist[depth - 1] and not any(hist[depth:]), result
            assert sum(result["draft_depth_accepted"]) == result["mtp_accepted"]
            print(
                json.dumps(
                    dict(
                        wire=wire,
                        mode=mode,
                        max_draft=depth,
                        tokens=len(tokens),
                        token_diff=0,
                        **result,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("host", "bundle", "mpi"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--scenario", choices=("accept", "mixed", "random"), default="mixed"
    )
    matrix(p.parse_args())
