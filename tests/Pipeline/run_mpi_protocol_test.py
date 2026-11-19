#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
"""Run BlockingMPI protocol tests (normal and bad-frame injection) under MPI."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_counter_file(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise SystemExit(f"counter file was not written: {path}")
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        values[key.strip()] = int(raw_val.strip())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpiexec",
        default=os.environ.get("BUDDY_MPIEXEC", "mpiexec"),
        help="Path to mpiexec / mpirun executable",
    )
    parser.add_argument(
        "--numproc-flag",
        default=os.environ.get("BUDDY_MPIEXEC_NUMPROC_FLAG", "-n"),
        help="Flag used by mpiexec to set the number of ranks",
    )
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--expect-fail", action="store_true")
    parser.add_argument("--expect-output", action="append", default=[])
    parser.add_argument("--counter-file", type=Path, default=None)
    parser.add_argument("--expect-compute-calls", type=int, default=None)
    parser.add_argument("--expect-prefill-calls", type=int, default=None)
    parser.add_argument("--expect-decode-calls", type=int, default=None)
    args = parser.parse_args()

    if args.counter_file is not None:
        args.counter_file.parent.mkdir(parents=True, exist_ok=True)
        if args.counter_file.exists():
            args.counter_file.unlink()

    cmd = [
        args.mpiexec,
        args.numproc_flag,
        str(args.ranks),
        args.binary,
        args.mode,
    ]
    if args.counter_file is not None:
        cmd.append(str(args.counter_file))

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        sys.stderr.write(f"TIMEOUT ({args.timeout}s) running {' '.join(cmd)}\n")
        if exc.stdout:
            sys.stderr.write(f"stdout:\n{exc.stdout}\n")
        if exc.stderr:
            sys.stderr.write(f"stderr:\n{exc.stderr}\n")
        return 1

    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if args.expect_fail:
        if proc.returncode == 0:
            raise SystemExit(
                f"mode {args.mode} unexpectedly exited 0 (expected non-zero)"
            )
    else:
        if proc.returncode != 0:
            raise SystemExit(
                f"mode {args.mode} exited {proc.returncode} (expected 0)"
            )

    for needle in args.expect_output:
        if needle not in combined:
            raise SystemExit(
                f"mode {args.mode}: expected output substring not found: "
                f"{needle!r}"
            )

    if args.counter_file is not None:
        counters = parse_counter_file(args.counter_file)
        checks = [
            ("compute_calls", args.expect_compute_calls),
            ("prefill_calls", args.expect_prefill_calls),
            ("decode_calls", args.expect_decode_calls),
        ]
        for key, expected in checks:
            if expected is not None and counters.get(key) != expected:
                raise SystemExit(
                    f"mode {args.mode}: {key}={counters.get(key)}, "
                    f"expected {expected}"
                )

    print(f"OK protocol test mode={args.mode} rc={proc.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
