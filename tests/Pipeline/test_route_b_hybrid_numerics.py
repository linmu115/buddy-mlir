#!/usr/bin/env python3
"""Compare native or QEMU Route B DSOs to the exact locked HF fallback."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from qwen35_route_b_reference import PROMPTS, prompt_ids, reference


def measure(expected, observed):
    if not np.isfinite(observed).all():
        raise AssertionError("nonfinite output")
    diff = np.abs(observed.astype(np.float64) - expected)
    maximum = float(diff.max())
    return {
        "max_abs": maximum,
        "max_abs_over_reference_peak": maximum
        / max(float(np.abs(expected).max()), 1e-12),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "elements": int(diff.size),
    }


def run_case(args, rows):
    oracle = reference(
        args.spec, args.checkpoint, args.embed_dtype, rows, args.steps
    )
    whole = (
        reference(
            args.spec,
            args.checkpoint,
            args.embed_dtype,
            rows,
            args.steps,
            whole_prefill=True,
        )
        if rows == 73
        else {}
    )
    ids = args.work / f"ids-{rows}.i64"
    ids.write_bytes(
        np.asarray(prompt_ids(PROMPTS[rows]), dtype="<i8").tobytes()
    )
    command = [
        *shlex.split(args.launcher),
        str(args.probe),
        str(args.bundle),
        str(ids),
    ]

    def probe(name, steps, pad_id):
        dump = args.work / f"{rows}-{name}.bin"
        start = time.monotonic()
        subprocess.run(
            [*command, str(dump), str(rows), str(steps), str(pad_id)],
            check=True,
            timeout=600,
        )
        return dump, time.monotonic() - start

    trace, seconds = probe("trace", args.steps, 33)
    obs = np.memmap(trace, mode="r", dtype="<f4")
    cursor = 0
    metrics = {}
    whole_metrics = {}
    for name, tensor in oracle.items():
        if name == "tokens":
            break
        n = tensor.numel()
        metrics[name] = measure(
            tensor.numpy().reshape(-1), obs[cursor : cursor + n]
        )
        if metrics[name]["max_abs"] >= 1e-4:
            raise AssertionError(f"{rows} tokens, {name}: {metrics[name]}")
        if whole:
            if name.startswith("prefill0"):
                # Whole-prompt inference only exposes its final cache snapshot.
                expected = (
                    whole.get(name)[:64] if name.endswith(".hidden") else None
                )
            elif name.startswith("prefill1"):
                expected = whole[name.replace("prefill1", "prefill0", 1)]
                if name.endswith(".hidden"):
                    expected = expected[64:]
            else:
                expected = whole[name]
            if expected is not None:
                whole_metrics[name] = measure(
                    expected.numpy().reshape(-1), obs[cursor : cursor + n]
                )
                if whole_metrics[name]["max_abs"] >= 1e-4:
                    raise AssertionError(
                        f"whole-prompt HF, {name}: {whole_metrics[name]}"
                    )
        cursor += n
    expected_tokens = oracle["tokens"].tolist()
    tokens = obs[cursor:].view("<i8").tolist()
    if tokens != expected_tokens:
        raise AssertionError(
            f"{rows} tokens: expected {expected_tokens}, got {tokens}"
        )
    if whole and tokens != whole["tokens"].tolist():
        raise AssertionError("tokens differ from whole-prompt HF")
    padding_exact = None
    if rows == 73:
        alt, _ = probe("alternate-padding", 1, 248044)
        with alt.open("rb") as alternate, trace.open("rb") as original:
            # Probe stops after prefill in this run, so exclude its trailing token ID
            # when comparing the prefix to the full prefill + decode trace.
            compare_bytes = alt.stat().st_size - 8
            padding_exact = original.read(compare_bytes) == alternate.read(
                compare_bytes
            )
        if not padding_exact:
            raise AssertionError(
                "tail padding altered a prefill output or state"
            )

    totals = defaultdict(
        lambda: {"prefill_max_abs": 0.0, "decode_max_abs": 0.0, "elements": 0}
    )
    for key, value in metrics.items():
        phase = "prefill" if key.startswith("prefill") else "decode"
        maximum = phase + "_max_abs"
        bucket = totals[key.rsplit(".", 1)[-1]]
        bucket[maximum] = max(bucket[maximum], value["max_abs"])
        bucket["elements"] += value["elements"]
    return {
        "prompt_rows": rows,
        "steps": args.steps,
        "decode_steps": args.steps - 1,
        "probe_seconds": seconds,
        "tokens": tokens,
        "token_diff": 0,
        "alternate_padding_exact": padding_exact,
        "summary": totals,
        "detail": metrics,
        "whole_prompt_hf": whole_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("spec", "checkpoint", "probe", "bundle", "work"):
        parser.add_argument("--" + path, required=True, type=Path)
    parser.add_argument("--embed-dtype", choices=["f16", "f32"], required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 73])
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("BUDDY_TEST_THREADS", 2)))
    args.work.mkdir(parents=True, exist_ok=True)
    spec_digests = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.spec.glob("*"))
        if p.is_file()
    }
    result = {
        "bundle": str(args.bundle),
        "embed_dtype": args.embed_dtype,
        "launcher": args.launcher,
        "spec_sha256": spec_digests,
        "cases": [],
    }
    for rows in args.rows:
        case = run_case(args, rows)
        result["cases"].append(case)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("detail", "whole_prompt_hf")
                }
            ),
            flush=True,
        )
        (args.work / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
