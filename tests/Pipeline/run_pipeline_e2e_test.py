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
"""End-to-end pipeline-parallel tests for buddy-cli --stage-model."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(
    cmd: list[str],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        sys.stderr.write(f"TIMEOUT ({timeout}s): {' '.join(cmd)}\n")
        if exc.stdout:
            sys.stderr.write(f"stdout:\n{exc.stdout}\n")
        if exc.stderr:
            sys.stderr.write(f"stderr:\n{exc.stderr}\n")
        raise SystemExit(1) from exc


def run_logits_test(args: argparse.Namespace) -> int:
    stage_rax = [Path(p) for p in args.stage_rax]
    if not stage_rax:
        raise SystemExit("at least one --stage-rax is required")
    for p in stage_rax:
        if not p.is_file():
            raise SystemExit(f"stage .rax package not found: {p}")

    validation_dir = Path(args.validation_dir)
    if not validation_dir.is_dir():
        raise SystemExit(f"validation directory not found: {validation_dir}")

    check_result = Path(args.check_result)
    if not check_result.is_file():
        raise SystemExit(f"check_result.py not found: {check_result}")

    baselines: list[tuple[str, Path, Path]] = []
    for name in ("full-reference", "m1"):
        ref_json = validation_dir / f"{name}.json"
        ref_logits = validation_dir / f"{name}-logits.f16"
        if ref_json.is_file() and ref_logits.is_file():
            baselines.append((name, ref_json, ref_logits))
    if not baselines:
        raise SystemExit(
            f"no reference baseline pair (*.json + *-logits.f16) found in "
            f"{validation_dir}"
        )

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    payload_cache = work_dir / "payload-cache"
    if payload_cache.exists():
        shutil.rmtree(payload_cache)
    payload_cache.mkdir(parents=True, exist_ok=True)

    actual_json = work_dir / "actual.json"
    actual_logits = work_dir / "actual-logits.f16"
    for p in (actual_json, actual_logits):
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env["BUDDY_PP_DUMP_LOGITS"] = str(actual_logits)
    env["BUDDY_RAX_PAYLOAD_DIR"] = str(payload_cache)

    cmd = [args.mpiexec, args.numproc_flag, str(len(stage_rax)), args.buddy_cli]
    for p in stage_rax:
        cmd.extend(["--stage-model", str(p)])
    cmd.extend(["--prompt", args.prompt, "--max-tokens", str(args.max_tokens)])

    proc = run_cmd(cmd, timeout=args.timeout, env=env)
    if proc.returncode != 0:
        sys.stderr.write(f"buddy-cli failed with exit code {proc.returncode}\n")
        if proc.stdout:
            sys.stderr.write(f"stdout:\n{proc.stdout}\n")
        if proc.stderr:
            sys.stderr.write(f"stderr:\n{proc.stderr}\n")
        return 1

    actual_json.write_text(proc.stdout, encoding="utf-8")
    if not actual_logits.is_file():
        raise SystemExit(
            f"expected dumped logits file missing: {actual_logits}"
        )

    for name, ref_json, ref_logits in baselines:
        check_cmd = [
            sys.executable,
            str(check_result),
            str(ref_json),
            str(actual_json),
            "--logits",
            str(ref_logits),
            str(actual_logits),
            "--vocab-size",
            str(args.vocab_size),
        ]
        check_proc = run_cmd(check_cmd, timeout=60)
        if check_proc.stdout:
            print(f"[{name}] {check_proc.stdout.strip()}")
        if check_proc.returncode != 0:
            if check_proc.stderr:
                sys.stderr.write(check_proc.stderr)
            raise SystemExit(f"check_result.py failed against baseline {name}")

    if args.whole_model:
        whole_model = Path(args.whole_model)
        if not whole_model.is_file():
            raise SystemExit(f"whole-model .rax package not found: {whole_model}")
        whole_cmd = [
            args.buddy_cli,
            "--model",
            str(whole_model),
            "--prompt",
            args.prompt,
            "--temperature",
            "0",
            "--max-tokens",
            str(args.max_tokens),
            "--stream-jsonl",
        ]
        whole_proc = run_cmd(whole_cmd, timeout=args.timeout, env=env)
        if whole_proc.returncode != 0:
            sys.stderr.write(
                f"whole-model buddy-cli failed with exit code {whole_proc.returncode}\n"
            )
            if whole_proc.stdout:
                sys.stderr.write(f"stdout:\n{whole_proc.stdout}\n")
            if whole_proc.stderr:
                sys.stderr.write(f"stderr:\n{whole_proc.stderr}\n")
            return 1
        whole_tokens: list[int] = []
        done = False
        for line in whole_proc.stdout.splitlines():
            ev = json.loads(line)
            if ev.get("event") == "token" and ev.get("token_id", -1) >= 0:
                whole_tokens.append(int(ev["token_id"]))
            elif ev.get("event") == "done":
                done = True
        if not done:
            raise SystemExit("whole-model stream-jsonl output missing done event")
        stage_data = json.loads(actual_json.read_text(encoding="utf-8"))
        stage_tokens = [int(t) for t in stage_data.get("token_ids", [])]
        if whole_tokens != stage_tokens:
            raise SystemExit(
                f"whole-model vs pipeline token mismatch for --max-tokens {args.max_tokens}: "
                f"whole={len(whole_tokens)} tokens, stage={len(stage_tokens)} tokens"
            )
        print(
            f"[whole-model] {len(stage_tokens)}/{len(whole_tokens)} sampled token IDs "
            f"match with shared --max-tokens {args.max_tokens}"
        )

    print("PASS pipeline e2e 2-stage bit-exact logits")
    return 0


def run_cli_errors_test(args: argparse.Namespace) -> int:
    stage_rax = [Path(p) for p in args.stage_rax]
    if len(stage_rax) < 2:
        raise SystemExit("cli-errors requires at least 2 --stage-rax packages")
    stage0, stage1 = stage_rax[0], stage_rax[1]
    for p in (stage0, stage1):
        if not p.is_file():
            raise SystemExit(f"stage .rax package not found: {p}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    missing_rax = work_dir / "nonexistent-stage1.rax"
    if missing_rax.exists():
        missing_rax.unlink()

    cases: list[tuple[str, int, list[str], str]] = [
        (
            "too-few-stage-models",
            2,
            ["--stage-model", str(stage0), "--prompt", args.prompt],
            "does not match MPI world size",
        ),
        (
            "too-many-stage-models",
            1,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--prompt",
                args.prompt,
            ],
            "does not match MPI world size",
        ),
        (
            "reversed-stage-order",
            2,
            [
                "--stage-model",
                str(stage1),
                "--stage-model",
                str(stage0),
                "--prompt",
                args.prompt,
                "--max-tokens",
                "4",
            ],
            "startLayer invalid on first rank 0",
        ),
        (
            "missing-stage-file",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(missing_rax),
                "--prompt",
                args.prompt,
            ],
            "cannot open",
        ),
        (
            "missing-prompt",
            2,
            ["--stage-model", str(stage0), "--stage-model", str(stage1)],
            "requires an explicit non-empty --prompt",
        ),
        (
            "conflict-with-model",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--model",
                str(stage0),
                "--prompt",
                args.prompt,
            ],
            "--stage-model cannot be combined with --model",
        ),
        (
            "unsupported-interactive",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--interactive",
                "--prompt",
                args.prompt,
            ],
            "--interactive is not supported with --stage-model",
        ),
        (
            "unsupported-batch-size",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--batch-size",
                "2",
                "--prompt",
                args.prompt,
            ],
            "--stage-model only supports default or --batch-size 1",
        ),
        (
            "conflict-with-weights",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--weights",
                "dummy.bin",
                "--prompt",
                args.prompt,
            ],
            "--stage-model cannot be combined with --weights",
        ),
        (
            "negative-max-tokens",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--max-tokens",
                "-1",
                "--prompt",
                args.prompt,
            ],
            "--max-tokens must be non-negative",
        ),
        (
            "max-tokens-exceeds-context",
            2,
            [
                "--stage-model",
                str(stage0),
                "--stage-model",
                str(stage1),
                "--max-tokens",
                "4096",
                "--prompt",
                args.prompt,
            ],
            "exceeds pipeline stage context",
        ),
    ]

    env = os.environ.copy()
    payload_cache = work_dir / "cli-errors-payload-cache"
    payload_cache.mkdir(parents=True, exist_ok=True)
    env["BUDDY_RAX_PAYLOAD_DIR"] = str(payload_cache)

    for case_name, ranks, cli_args, expected_diag in cases:
        cmd = [
            args.mpiexec,
            args.numproc_flag,
            str(ranks),
            args.buddy_cli,
            *cli_args,
        ]
        proc = run_cmd(cmd, timeout=args.timeout, env=env)
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            raise SystemExit(
                f"CLI error case '{case_name}' unexpectedly exited 0\n"
                f"output:\n{combined}"
            )
        if expected_diag not in combined:
            raise SystemExit(
                f"CLI error case '{case_name}' missing expected diagnostic "
                f"{expected_diag!r}\noutput:\n{combined}"
            )
        print(f"OK cli-error {case_name} (rc={proc.returncode})")

    print(f"PASS pipeline e2e cli-errors ({len(cases)} cases)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--mpiexec",
            default=os.environ.get("BUDDY_MPIEXEC", "mpiexec"),
        )
        sub.add_argument(
            "--numproc-flag",
            default=os.environ.get("BUDDY_MPIEXEC_NUMPROC_FLAG", "-n"),
        )
        sub.add_argument(
            "--buddy-cli",
            default=os.environ.get("BUDDY_CLI", ""),
            required=not bool(os.environ.get("BUDDY_CLI")),
        )
        sub.add_argument(
            "--stage-rax",
            action="append",
            default=[],
            help="Stage .rax package path (repeat in rank order)",
        )
        sub.add_argument(
            "--prompt",
            default=os.environ.get(
                "BUDDY_PIPELINE_E2E_PROMPT",
                "Explain step by step how to add the integers from 1 to 100.",
            ),
        )
        sub.add_argument("--work-dir", required=True)
        sub.add_argument("--timeout", type=int, default=120)

    logits_p = subparsers.add_parser("logits")
    add_common(logits_p)
    logits_p.add_argument(
        "--validation-dir",
        default=os.environ.get("BUDDY_PIPELINE_VALIDATION_DIR", ""),
        required=not bool(os.environ.get("BUDDY_PIPELINE_VALIDATION_DIR")),
    )
    logits_p.add_argument(
        "--check-result",
        default=os.environ.get("BUDDY_CHECK_RESULT_PY", ""),
        required=not bool(os.environ.get("BUDDY_CHECK_RESULT_PY")),
    )
    logits_p.add_argument("--max-tokens", type=int, default=56)
    logits_p.add_argument("--vocab-size", type=int, default=151936)
    logits_p.add_argument(
        "--whole-model",
        default=os.environ.get("BUDDY_PIPELINE_E2E_WHOLE_MODEL_RAX", ""),
        help="Optional whole-model .rax to compare directly with the same --max-tokens",
    )

    err_p = subparsers.add_parser("cli-errors")
    add_common(err_p)

    args = parser.parse_args()
    if not args.stage_rax and os.environ.get("BUDDY_PIPELINE_STAGE_RAX"):
        args.stage_rax = [
            p
            for p in os.environ["BUDDY_PIPELINE_STAGE_RAX"].split(os.pathsep)
            if p
        ]

    if args.subcommand == "logits":
        return run_logits_test(args)
    if args.subcommand == "cli-errors":
        return run_cli_errors_test(args)
    raise SystemExit(f"unknown subcommand: {args.subcommand}")


if __name__ == "__main__":
    raise SystemExit(main())
