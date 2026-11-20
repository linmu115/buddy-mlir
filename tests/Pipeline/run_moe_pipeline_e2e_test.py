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
"""End-to-end bit-exact pipeline-parallel test for tiny Qwen3-MoE (w4g32)."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_DIR = REPO_ROOT / "tools" / "buddy-codegen"

SHORT_PROMPT = "Hello"
LONG_PROMPT = (
    "Explain step by step how a pipeline parallel transformer executes "
    "multiple prefill chunks across distributed MPI ranks with a mixture "
    "of experts decoder, rotary positional embeddings, grouped query "
    "attention, and deterministic greedy token sampling."
)


def run_cmd(
    cmd: list[str],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=capture,
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


def must_run(
    cmd: list[str],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    label: str = "",
) -> subprocess.CompletedProcess[str]:
    proc = run_cmd(cmd, timeout=timeout, env=env, capture=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"FAILED ({label or cmd[0]}, rc={proc.returncode}): {' '.join(cmd)}\n"
        )
        if proc.stdout:
            sys.stderr.write(f"stdout:\n{proc.stdout}\n")
        if proc.stderr:
            sys.stderr.write(f"stderr:\n{proc.stderr}\n")
        raise SystemExit(1)
    return proc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            if not dst.is_symlink() and os.path.samefile(src, dst):
                return
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def generate_tiny_qwen3_moe(
    model_dir: Path, *, head_dim: int, seed: int
) -> Path:
    marker = model_dir / ".tiny_moe_meta.json"
    expected_meta = {"head_dim": head_dim, "seed": seed}
    if (
        (model_dir / "config.json").is_file()
        and marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")) == expected_meta
    ):
        return model_dir

    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=256,
        intermediate_size=6144,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=151644,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        dtype=torch.float16,
    )
    model = Qwen3MoeForCausalLM(cfg).to(dtype=torch.float16)
    model.eval()
    model.save_pretrained(model_dir, safe_serialization=True)
    marker.write_text(json.dumps(expected_meta), encoding="utf-8")
    return model_dir


def resolve_mlir_runtime_deps(mlir_runtime_dir: Path) -> list[Path]:
    deps: list[Path] = []
    for base in (
        "libmlir_apfloat_wrappers",
        "libmlir_float16_utils",
        "libmlir_c_runner_utils",
    ):
        cand = mlir_runtime_dir / f"{base}.so"
        if not cand.exists():
            matches = sorted(mlir_runtime_dir.glob(f"{base}.so.*"))
            if matches:
                cand = matches[0]
        if cand.exists():
            deps.append(cand.resolve())
        elif base == "libmlir_c_runner_utils":
            raise SystemExit(
                f"missing required MLIR runtime library in {mlir_runtime_dir}: {cand}"
            )
    return deps


def export_compile_and_pack_stage(
    *,
    base_config_path: Path,
    start_layer: int,
    end_layer: int,
    stage_dir: Path,
    rax_dir: Path,
    rank: int,
    args: argparse.Namespace,
    mlir_deps: list[Path],
    env: dict[str, str],
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    rax_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lazy export stage MLIR + weights (reuse if already exported/compiled).
    stage_cfg_path = stage_dir / "config.json"
    k3_o = stage_dir / "k3_kernels.o"
    if not (stage_cfg_path.is_file() and k3_o.is_file()):
        must_run(
            [
                sys.executable,
                str(CODEGEN_DIR / "import_pipeline_stage.py"),
                "--config",
                str(base_config_path),
                "--layers",
                str(start_layer),
                str(end_layer),
                "--load-mode",
                "lazy",
                "--output-dir",
                str(stage_dir),
            ],
            timeout=args.timeout,
            env=env,
            label=f"import_pipeline_stage [{start_layer},{end_layer})",
        )
        must_run(
            [
                sys.executable,
                str(CODEGEN_DIR / "compile_pipeline.py"),
                "--config",
                str(stage_cfg_path),
                "--compile-all",
                "--mlir-dir",
                str(stage_dir),
                "--output-dir",
                str(stage_dir),
                "--buddy-opt",
                str(args.buddy_opt),
                "--llvm-tools-dir",
                str(args.llvm_tools_dir),
                "--llc-attrs=-mcpu=native",
                "-j",
                str(args.compile_jobs),
            ],
            timeout=args.timeout,
            env=env,
            label=f"compile_pipeline [{start_layer},{end_layer})",
        )

    # 2. Verify ABI and emit stage-config.inc.
    stage_inc = stage_dir / "stage-config.inc"
    must_run(
        [
            sys.executable,
            str(CODEGEN_DIR / "verify_pipeline_stage.py"),
            "--stage-dir",
            str(stage_dir),
            "--header",
            str(stage_inc),
        ],
        timeout=60,
        env=env,
        label=f"verify_pipeline_stage [{start_layer},{end_layer})",
    )

    # 3. Compile Stage.cpp + BuddyArena.c + BuddyThreadPool.c and link stage.so.
    stage_so = stage_dir / "stage.so"
    if not stage_so.is_file():
        arena_o = stage_dir / "BuddyArena.o"
        tp_o = stage_dir / "BuddyThreadPool.o"
        stage_o = stage_dir / "Stage.o"

        must_run(
            [
                args.c_compiler,
                "-O2",
                "-fPIC",
                "-c",
                str(REPO_ROOT / "runtime" / "arena" / "BuddyArena.c"),
                "-o",
                str(arena_o),
            ],
            timeout=60,
            env=env,
            label="compile BuddyArena.c",
        )
        must_run(
            [
                args.c_compiler,
                "-O2",
                "-fPIC",
                "-c",
                str(REPO_ROOT / "runtime" / "threadpool" / "BuddyThreadPool.c"),
                "-o",
                str(tp_o),
            ],
            timeout=60,
            env=env,
            label="compile BuddyThreadPool.c",
        )
        inc_flags = [f"-I{stage_dir}", f"-I{REPO_ROOT / 'runtime' / 'include'}"]
        for d in args.mlir_include_dirs.replace(";", os.pathsep).split(os.pathsep):
            d = d.strip()
            if d:
                inc_flags.append(f"-I{d}")
        must_run(
            [
                args.cxx_compiler,
                "-std=c++17",
                "-O2",
                "-DNDEBUG",
                "-fPIC",
                *inc_flags,
                "-c",
                str(REPO_ROOT / "models" / "deepseek_r1" / "pipeline" / "Stage.cpp"),
                "-o",
                str(stage_o),
            ],
            timeout=60,
            env=env,
            label="compile Stage.cpp",
        )
        crunner_lib = mlir_deps[-1]
        must_run(
            [
                args.cxx_compiler,
                "-shared",
                "-fPIC",
                "-Wl,-soname,stage.so",
                "-Wl,--allow-multiple-definition",
                "-Wl,-z,defs",
                "-Wl,-rpath,$ORIGIN",
                "-o",
                str(stage_so),
                str(stage_o),
                str(stage_dir / "forward_prefill.o"),
                str(stage_dir / "forward_decode.o"),
                str(stage_dir / "subgraph_prefill.o"),
                str(stage_dir / "subgraph_decode.o"),
                str(stage_dir / "k3_kernels.o"),
                str(arena_o),
                str(tp_o),
                str(crunner_lib),
                "-lpthread",
                "-ldl",
            ],
            timeout=60,
            env=env,
            label=f"link stage.so [{start_layer},{end_layer})",
        )

    # 4. Generate manifest and stage files next to stage<rank>.rax (manifest-only).
    stage_cfg = json.loads(stage_cfg_path.read_text(encoding="utf-8"))
    runner_src = Path(args.runner_lib).resolve()
    runner_name = runner_src.name
    manifest_path = rax_dir / f"stage{rank}.mlir"
    dep_args: list[str] = []
    for dep in mlir_deps:
        dep_args.extend(["--dep-shared-lib", dep.name])
    must_run(
        [
            sys.executable,
            str(CODEGEN_DIR / "gen_manifest.py"),
            "--config",
            str(stage_cfg_path),
            *dep_args,
            "--runner-library",
            runner_name,
            "-o",
            str(manifest_path),
        ],
        timeout=60,
        env=env,
        label=f"gen_manifest stage{rank}",
    )

    shutil.copy2(stage_so, rax_dir / "stage.so")
    shutil.copy2(runner_src, rax_dir / runner_name)
    for dep in mlir_deps:
        shutil.copy2(dep, rax_dir / dep.name)
    for w in stage_cfg["weights"]:
        wfile = w["file"]
        hardlink_or_copy(stage_dir / wfile, rax_dir / wfile)
    if start_layer == 0:
        vocab_name = stage_cfg["tokens"]["vocab_file"]
        shutil.copy2(Path(args.vocab_file), rax_dir / vocab_name)

    rax_out = rax_dir / f"stage{rank}.rax"
    must_run(
        [str(args.rax_pack), str(manifest_path), "-o", str(rax_out)],
        timeout=60,
        env=env,
        label=f"rax-pack stage{rank}",
    )
    return rax_out


def run_pipeline_case(
    *,
    stage_raxs: list[Path],
    prompt: str,
    max_tokens: int,
    out_json: Path,
    out_logits: Path,
    payload_cache: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[dict, str]:
    run_env = env.copy()
    run_env["BUDDY_PP_DUMP_LOGITS"] = str(out_logits)
    run_env["BUDDY_RAX_PAYLOAD_DIR"] = str(payload_cache)
    cmd = [
        args.mpiexec,
        args.numproc_flag,
        str(len(stage_raxs)),
        args.buddy_cli,
    ]
    for r in stage_raxs:
        cmd.extend(["--stage-model", str(r)])
    cmd.extend(
        [
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--max-tokens",
            str(max_tokens),
        ]
    )
    proc = must_run(
        cmd,
        timeout=args.timeout,
        env=run_env,
        label=f"mpiexec -n {len(stage_raxs)} max_tokens={max_tokens}",
    )
    out_json.write_text(proc.stdout, encoding="utf-8")
    if not out_logits.is_file():
        raise SystemExit(f"expected dumped f32 logits missing: {out_logits}")
    data = json.loads(proc.stdout)
    return data, sha256_file(out_logits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpiexec", default=os.environ.get("BUDDY_MPIEXEC", "mpiexec"))
    parser.add_argument("--numproc-flag", default="-n")
    parser.add_argument("--buddy-cli", required=True)
    parser.add_argument("--rax-pack", required=True)
    parser.add_argument("--buddy-opt", required=True)
    parser.add_argument("--llvm-tools-dir", required=True)
    parser.add_argument("--c-compiler", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--mlir-include-dirs", required=True)
    parser.add_argument("--mlir-runtime-dir", required=True)
    parser.add_argument("--runner-lib", required=True)
    parser.add_argument(
        "--vocab-file",
        default=str(REPO_ROOT / "examples" / "BuddyQwen3" / "vocab.txt"),
    )
    parser.add_argument(
        "--check-result",
        default=str(REPO_ROOT / "examples" / "BuddyDeepSeekR1" / "pipeline" / "check_result.py"),
    )
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whole-model", default="")
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    payload_cache = work_dir / "payload-cache"
    if payload_cache.exists():
        shutil.rmtree(payload_cache)
    payload_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"model dir missing config.json: {model_dir}")
    else:
        model_dir = generate_tiny_qwen3_moe(
            work_dir / "tiny-qwen3-moe", head_dim=args.head_dim, seed=args.seed
        )

    hf_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(hf_cfg["vocab_size"])

    spec_path = work_dir / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "hf_model_path": str(model_dir),
                "model_family": "qwen3_vl_moe",
                "variant": "w4g32",
                "max_token_len": 128,
                "num_threads": 4,
                "prefill_chunk": 32,
                "arena": True,
                "hugepages": False,
                "thread_pool": True,
                "prefill_ime": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    base_cfg_path = work_dir / "config.json"
    must_run(
        [
            sys.executable,
            str(CODEGEN_DIR / "gen_config.py"),
            "--spec",
            str(spec_path),
            "--hf-config",
            str(model_dir / "config.json"),
            "-o",
            str(base_cfg_path),
        ],
        timeout=60,
        env=env,
        label="gen_config",
    )

    mlir_deps = resolve_mlir_runtime_deps(Path(args.mlir_runtime_dir))
    topologies = {
        "1stage": [(0, 4)],
        "2stage": [(0, 2), (2, 4)],
        "3stage": [(0, 1), (1, 3), (3, 4)],
    }

    topology_raxs: dict[str, list[Path]] = {}
    for topo_name, intervals in topologies.items():
        raxs: list[Path] = []
        for rank, (start_l, end_l) in enumerate(intervals):
            stage_dir = work_dir / "stages" / f"{start_l}_{end_l}"
            rax_dir = work_dir / "rax" / topo_name / f"stage{rank}"
            raxs.append(
                export_compile_and_pack_stage(
                    base_config_path=base_cfg_path,
                    start_layer=start_l,
                    end_layer=end_l,
                    stage_dir=stage_dir,
                    rax_dir=rax_dir,
                    rank=rank,
                    args=args,
                    mlir_deps=mlir_deps,
                    env=env,
                )
            )
        topology_raxs[topo_name] = raxs

    prompt_cases = [
        ("short", SHORT_PROMPT, 20),
        ("long", LONG_PROMPT, 80),
    ]
    short_ref_tokens: list[int] | None = None

    for case_name, prompt, max_tokens in prompt_cases:
        case_dir = work_dir / "runs" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, tuple[Path, Path, dict, str]] = {}
        for topo_name in ("1stage", "2stage", "3stage"):
            out_json = case_dir / f"{topo_name}.json"
            out_f32 = case_dir / f"{topo_name}.f32"
            data, digest = run_pipeline_case(
                stage_raxs=topology_raxs[topo_name],
                prompt=prompt,
                max_tokens=max_tokens,
                out_json=out_json,
                out_logits=out_f32,
                payload_cache=payload_cache,
                args=args,
                env=env,
            )
            results[topo_name] = (out_json, out_f32, data, digest)

        ref_json, ref_f32, ref_data, ref_sha = results["1stage"]
        prompt_tokens = int(ref_data["prompt_tokens"])
        token_ids = [int(t) for t in ref_data["token_ids"]]
        if case_name == "short":
            if prompt_tokens >= 32:
                raise SystemExit(
                    f"short prompt expected < 32 tokens, got {prompt_tokens}"
                )
            short_ref_tokens = token_ids
        else:
            if prompt_tokens <= 32:
                raise SystemExit(
                    f"long prompt expected > 32 tokens (multi-chunk), got {prompt_tokens}"
                )

        for topo_name in ("2stage", "3stage"):
            act_json, act_f32, _, act_sha = results[topo_name]
            chk = must_run(
                [
                    sys.executable,
                    str(args.check_result),
                    str(ref_json),
                    str(act_json),
                    "--logits",
                    str(ref_f32),
                    str(act_f32),
                    "--vocab-size",
                    str(vocab_size),
                    "--dtype",
                    "f32",
                ],
                timeout=60,
                env=env,
                label=f"check_result {case_name} {topo_name}",
            )
            print(
                f"[{case_name}][{topo_name}] prompt_tokens={prompt_tokens} "
                f"gen_tokens={len(token_ids)} logits_elems={len(token_ids) * vocab_size} "
                f"sha256={act_sha} ({chk.stdout.strip().replace(chr(10), '; ')})"
            )
            if act_sha != ref_sha:
                raise SystemExit(
                    f"SHA-256 mismatch for {case_name} {topo_name}: {act_sha} != {ref_sha}"
                )

    # Verify early-stop hook on 2-stage pipeline.
    stop_env = env.copy()
    stop_env["BUDDY_PP_TEST_EOS_STEP"] = "2"
    stop_env["BUDDY_RAX_PAYLOAD_DIR"] = str(payload_cache)
    stop_cmd = [
        args.mpiexec,
        args.numproc_flag,
        "2",
        args.buddy_cli,
        "--stage-model",
        str(topology_raxs["2stage"][0]),
        "--stage-model",
        str(topology_raxs["2stage"][1]),
        "--prompt",
        SHORT_PROMPT,
        "--temperature",
        "0",
        "--max-tokens",
        "20",
        "--stream-jsonl",
    ]
    stop_proc = must_run(
        stop_cmd, timeout=args.timeout, env=stop_env, label="stop-hook"
    )
    stop_events = [
        json.loads(line)
        for line in stop_proc.stdout.splitlines()
        if line.strip()
    ]
    tok_events = [ev for ev in stop_events if ev.get("event") == "token"]
    done_events = [ev for ev in stop_events if ev.get("event") == "done"]
    if len(tok_events) != 2 or not tok_events[-1].get("stopped") or len(done_events) != 1:
        raise SystemExit(f"unexpected stop-hook events: {stop_events}")
    print("[stop-hook] 2-stage stopped cleanly at step 2 with stopped=true")

    # Optional whole-model parity check.
    if args.whole_model:
        whole_rax = Path(args.whole_model)
        if not whole_rax.is_file():
            raise SystemExit(f"whole-model .rax not found: {whole_rax}")
        whole_env = env.copy()
        whole_env["BUDDY_RAX_PAYLOAD_DIR"] = str(payload_cache)
        whole_proc = must_run(
            [
                args.buddy_cli,
                "--model",
                str(whole_rax),
                "--prompt",
                SHORT_PROMPT,
                "--temperature",
                "0",
                "--max-tokens",
                "20",
                "--stream-jsonl",
            ],
            timeout=args.timeout,
            env=whole_env,
            label="whole-model parity",
        )
        whole_tokens = [
            int(ev["token_id"])
            for line in whole_proc.stdout.splitlines()
            if line.strip()
            for ev in [json.loads(line)]
            if ev.get("event") == "token" and int(ev.get("token_id", -1)) >= 0
        ]
        if whole_tokens != short_ref_tokens:
            raise SystemExit(
                f"whole-model token mismatch: {whole_tokens} vs {short_ref_tokens}"
            )
        print(
            f"[whole-model] {len(whole_tokens)}/{len(whole_tokens)} tokens match"
        )

    print("PASS pipeline moe e2e bit-exact f32 logits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
