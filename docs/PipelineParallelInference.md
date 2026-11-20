# Pipeline-Parallel Inference

This document describes how to build, export, package, run, and deploy
pipeline-parallel LLM inference in Buddy-MLIR using MPI and per-stage `.rax`
bundles, with DeepSeek-R1-Distill-Qwen-1.5B (FP16) as the worked example.

## Overview

In pipeline-parallel inference, the decoder layers of a transformer model are
partitioned into contiguous, zero-based half-open intervals
`[start_layer, end_layer)` with one MPI rank per stage. Each rank holds only
its own stage weights (`arg0-f16.data`) and its own stage-local KV cache
(`2 * (end_layer - start_layer)` tensors). Rank 0 owns the vocabulary and token
embedding layer, middle ranks consume and emit hidden states, and the final rank
applies the final RMSNorm and LM head, samples the next token, returns the
sampled token ID to rank 0, and broadcasts the stop decision. Adjacent ranks
exchange only hidden states: a 16-byte header (`int32_t[4]`) on MPI tag 2
followed by the raw FP16 activation payload (`MPI_UINT16_T`) on MPI tag 0.

```text
  prompt ──► [ Rank 0: layers [0, L1) ]
               vocab.txt + embed_tokens + weights_0 + KV_0
                   │
                   │  tag 2: 16B header {version=1, step, pos, rows}
                   │  tag 0: FP16 hidden [1, rows, 1536]
                   ▼
             [ Rank 1: layers [L1, L2) ]
               weights_1 + KV_1  (forwards validated header verbatim)
                   │
                   │  tag 2 + tag 0 (same wire protocol)
                   ▼
                   ⋮
                   ▼
             [ Rank N-1: layers [L_{N-1}, 28) ]
               weights_{N-1} + KV_{N-1} + final norm + lm_head + sampler
                   │
                   ├──► tag 1: sampled token ID (MPI_INT) ──► Rank 0
                   ├──► MPI_Bcast: stop flag (int) ─────────► all ranks
                   └──► stdout: JSON token IDs (final rank only)
```

For the lower-level layer-partitioning background and standalone non-RAX example
runner, see [LayerPartitioning.md](LayerPartitioning.md) and
[examples/BuddyDeepSeekR1/pipeline/README.md](../examples/BuddyDeepSeekR1/pipeline/README.md).

## Build

### 1. Build MPICH 4.2.0 from Source

Install MPICH 4.2.0 from source with the `ch4:ofi` device rather than using the
distribution `apt` package:

```bash
MPI_BUILD="$HOME/opt/build-mpich"
MPI_HOME="$HOME/opt/mpich-4.2.0"
mkdir -p "$MPI_BUILD"
cd "$MPI_BUILD"

curl -fsSL -o mpich-4.2.0.tar.gz \
  https://www.mpich.org/static/downloads/4.2.0/mpich-4.2.0.tar.gz
tar xzf mpich-4.2.0.tar.gz
cd mpich-4.2.0

./configure --prefix="$MPI_HOME" --with-device=ch4:ofi --disable-fortran
make -j"$(nproc)"
make install

"$MPI_HOME/bin/mpichversion"
```

Verify that `mpichversion` reports `MPICH Version: 4.2.0` and
`MPICH Device: ch4:ofi`.

> **Why not `apt install mpich` on Ubuntu 24.04?**
> The Ubuntu 24.04 (`noble`) `mpich` package (`4.2.0-5build3`) is built with
> `--with-device=ch4:ucx --with-pmix=/usr/lib/x86_64-linux-gnu/pmix2 --with-pm=hydra`.
> In testing on x86_64 WSL (recorded in
> `reports/pp-docs/01-mpich-check.log`), launching a 2-rank job with
> `/usr/bin/mpiexec -n 2` fails to bootstrap the multi-rank communicator: every
> process initializes as an isolated singleton (`rank=0, size=1`) and aborts on
> the first point-to-point send with:
>
> ```text
> Abort(...) on node 0 (rank 0 in comm 0): Fatal error in internal_Send: Invalid rank, error stack:
> internal_Send(62919): Invalid rank has value 1 but must be nonnegative and less than 1
> ```
>
> Building MPICH 4.2.0 from source with `--with-device=ch4:ofi` avoids this bug
> and ensures consistent Hydra/MPI behavior across single-node and multi-node
> launches.

### 2. Configure Buddy-MLIR with MPI Enabled

Point CMake at your source-built MPICH installation so it does not pick up a
system `mpicxx` from `/usr/bin`:

```bash
export MPI_HOME="$HOME/opt/mpich-4.2.0"

cmake -S . -B "$BUDDY_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM_BUILD/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_BUILD/bin/clang++" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DCMAKE_PREFIX_PATH="$MPI_HOME" \
  -DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" \
  -DMPIEXEC_EXECUTABLE="$MPI_HOME/bin/mpiexec" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -DBUDDY_CLI_ENABLE_MPI=ON \
  -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON \
  -DBUDDY_DSR1_SPEC="$SPEC_JSON" \
  -DBUDDY_DSR1_HF_CONFIG="$DEEPSEEKR1_MODEL_PATH/config.json" \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DBUDDY_ENABLE_TESTS=ON
```

Notes on key CMake options:

- `BUDDY_CLI_ENABLE_MPI` defaults to `ON` on native builds and `OFF` during
  cross-compilation (`CMAKE_CROSSCOMPILING`). When `ON` and `MPI::MPI_CXX` is
  found, CMake prints `[buddy-cli] MPI found; enabling --stage-model support.`
  and builds `buddy-cli` with MPI support. When MPI is unavailable or disabled,
  whole-model `buddy-cli --model` builds normally without linking `libmpi`.
- Passing `-DDEEPSEEKR1_BUILD_DIR=<dir>` (Mode A) reuses pre-built whole-model
  objects if you only want to build `buddy-cli` and pipeline targets without
  re-importing the whole model.
- In offline environments, pass `-DBUDDY_DOWNLOAD_CLI11=ON` and
  `-DFETCHCONTENT_SOURCE_DIR_CLI11_HPP=<path-to-cli11>` and, if needed,
  `-DBUDDY_OPENMP_RUNTIME_LIBRARY=<path-to-libomp.so>`.
- If `-DBUDDY_ENABLE_TESTS=ON` is set, keep
  `-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON` so the repository's `check-tests`
  dependency graph configures cleanly.

### 3. Build `buddy-cli` and the Pipeline Runner Plugin

Build the MPI-enabled CLI and the shared pipeline runner plugin
(`models/deepseek_r1/deepseek_r1_pipeline_runner.so`):

```bash
cmake --build "$BUDDY_BUILD" \
  --target buddy-cli buddy_models_deepseek_r1_pipeline_runner -j
```

Verify the dynamic MPI dependency on `buddy-cli` and the exported v1 plugin
factory symbols on the runner:

```bash
readelf -d "$BUDDY_BUILD/bin/buddy-cli" | grep -E 'NEEDED|RUNPATH|RPATH'
nm -D --defined-only \
  "$BUDDY_BUILD/models/deepseek_r1/deepseek_r1_pipeline_runner.so" | grep buddy_
```

The `nm` check should list `buddy_create_inference_runner_v1` and
`buddy_destroy_inference_runner_v1`.

To verify the manifest parser, 16-byte activation protocol, and multi-rank stage
chain checks without model weights, run the pipeline CTest unit suite (16/16
tests passed in 0.75 s on x86_64 WSL; `reports/pp-docs/02-build-and-unit-tests.log`):

```bash
cmake --build "$BUDDY_BUILD" --target check-buddy-pipeline -j
```

## Exporting and Compiling Stages

Stage export operates on a plain FP16 DeepSeek R1 config (`variant: "f16"`).

### 1. Generate a Base FP16 Config

Start from `models/deepseek_r1/specs/f16.json` (adjusting `max_token_len` and
`num_threads` for your target host; for example, `max_token_len: 64` and
`num_threads: 8` on an 8-core development host) and generate the full config:

```bash
export DEEPSEEKR1_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
export PYTHONPATH="$BUDDY_BUILD/python_packages:$MLIR_PYTHON_PACKAGES"

python3 tools/buddy-codegen/gen_config.py \
  --spec "$SPEC_JSON" \
  --hf-config "$DEEPSEEKR1_MODEL_PATH/config.json" \
  -o "$STAGE_ROOT/base-f16-config.json"
```

### 2. Generic N-Stage Export and Compilation

Each stage directory is exported with `tools/buddy-codegen/import_pipeline_stage.py`,
checked with `tools/buddy-codegen/verify_pipeline_stage.py`, and lowered to
position-independent object files (`forward_prefill.o`, `forward_decode.o`,
`subgraph_prefill.o`, `subgraph_decode.o`) with
`tools/buddy-codegen/compile_pipeline.py`:

```bash
export_and_compile_stages() {
  local root_dir=$1
  shift
  mkdir -p "$root_dir"
  for interval in "$@"; do
    local start=${interval%-*}
    local end=${interval#*-}
    local stage_dir="$root_dir/$start-$end"

    python3 tools/buddy-codegen/import_pipeline_stage.py \
      --config "$STAGE_ROOT/base-f16-config.json" \
      --layers "$start" "$end" \
      --output-dir "$stage_dir"

    python3 tools/buddy-codegen/verify_pipeline_stage.py \
      --stage-dir "$stage_dir"

    python3 tools/buddy-codegen/compile_pipeline.py \
      --config "$stage_dir/config.json" \
      --compile-all \
      --mlir-dir "$stage_dir" \
      --output-dir "$stage_dir" \
      --buddy-opt "$BUDDY_BUILD/bin/buddy-opt" \
      --llvm-tools-dir "$LLVM_BUILD/bin" \
      --llc-attrs=-mcpu=native \
      -j 1
  done
}
```

Example — **2 stages** over 28 decoder layers (`[0, 14)` and `[14, 28)`):

```bash
export_and_compile_stages "$STAGE_ROOT/stages-2" 0-14 14-28
```

Example — **4 stages** over 28 decoder layers (`[0, 7)`, `[7, 14)`, `[14, 21)`,
`[21, 28)`):

```bash
export_and_compile_stages "$STAGE_ROOT/stages-4" 0-7 7-14 14-21 21-28
```

For any `N` stages, pass `N` contiguous half-open intervals covering
`[0, num_hidden_layers)` in rank order.

### 3. Import Memory Usage (`--load-mode lazy` vs `full`)

By default, `import_pipeline_stage.py` uses `--load-mode lazy`:

- It instantiates an empty model skeleton on `torch.device("meta")`, trims the
  model to `[start, end)`, and reads only the tensors required by that stage
  from `model.safetensors` or `model.safetensors.index.json` in per-layer
  buckets.
- After tracing and pre-transforming `prefill` and `decode`, duplicate
  transposed parameter buffers are shared and `arg0-f16.data` is streamed
  directly to disk without mapping the output file into RSS.
- For legacy PyTorch `.bin` checkpoints without `safetensors`, pass
  `--load-mode full` to load the full checkpoint before trimming.

Measured peak RSS on x86_64 WSL for DeepSeek-R1-Distill-Qwen-1.5B FP16
(context 64, single-file BF16 `model.safetensors`, 3.55 GB on disk):

| Stage interval | Load mode | Stage FP16 weights | Measured peak RSS | Source |
| --- | --- | ---: | ---: | --- |
| Whole model `[0, 28)` | legacy full | 3.310 GiB | 10.63 GiB | `reports/pp-m0-m1-m2-report.md` (`pp-baseline-logs/import.time`) |
| `[0, 7)` | `--load-mode full` | 1.045 GiB | 7.16 GiB | `reports/pp-b01-lazy-import-report.md` |
| `[7, 14)` | `--load-mode lazy` | 0.610 GiB | 1.84 GiB (1,931,368 KiB) | `reports/pp-docs/03-export-compile.log` |
| `[14, 21)` | `--load-mode lazy` | 0.610 GiB | 1.84 GiB (1,931,328 KiB) | `reports/pp-docs/03-export-compile.log` |
| `[0, 7)` | `--load-mode lazy` | 1.045 GiB | 2.28 GiB (2,386,000 KiB) | `reports/pp-docs/03-export-compile.log` |
| `[21, 28)` | `--load-mode lazy` | 1.045 GiB | 3.12 GiB (3,270,576 KiB) | `reports/pp-docs/03-export-compile.log` |
| `[0, 14)` | `--load-mode lazy` | 1.655 GiB | 3.52 GiB (3,689,348 KiB) | `reports/pp-docs/03-export-compile.log` |
| `[14, 28)` | `--load-mode lazy` | 1.655 GiB | 4.37 GiB (4,579,812 KiB) | `reports/pp-docs/03-export-compile.log` |

All lazy-exported artifacts (`config.json`, 4 MLIR files, `arg0-f16.data`) are
byte-for-byte identical to full-load exports
(`reports/pp-b01-lazy-import-report.md`).

## Packaging Stage `.rax` Bundles

The `deepseek_r1_pipeline_rax` CMake target verifies each stage's ABI, links
each stage's `stage.so`, generates its RHAL manifest (`stage.mlir`), and packs
`stage<i>.rax` via `rax-pack`.

### Stage Ordering and CMake Variables

Pass `DEEPSEEKR1_PIPELINE_STAGE_DIRS` as an ordered semicolon-separated list of
exported stage directories. **The list order is the MPI rank order** (`stage0`
is rank 0, `stage1` is rank 1, ..., `stage<N-1>` is rank `N-1`).

| CMake variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEKR1_PIPELINE_STAGE_DIRS` | `""` | Semicolon-separated stage directories in MPI rank order. Leave empty for normal whole-model builds. Setting this without MPI fails at configure time. |
| `DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD` | `ON` | `ON`: self-contained `<out>/stage<i>.rax` (`rax-pack --embed-payload`). `OFF`: lightweight manifest-only `<out>/stage<i>/stage<i>.rax` with relative `file:` assets. |
| `DEEPSEEKR1_PIPELINE_OUTPUT_DIR` | `${CMAKE_CURRENT_BINARY_DIR}/pipeline_rax` | Publication directory for `stage<i>.rax`. |
| `DEEPSEEKR1_PIPELINE_RAX_PACK` | `""` | Optional pre-built `rax-pack` binary; empty uses the in-tree `rax-pack` target. |
| `DEEPSEEKR1_PIPELINE_VOCAB_FILE` | `examples/BuddyDeepSeekR1/vocab.txt` | Fallback `vocab.txt` for stage 0 when absent from its stage directory. |

### Embedded Mode vs. Manifest-Only Mode

**Embedded mode (default, `DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=ON`)** produces one
self-contained `.rax` per rank under `<output_dir>/stage<i>.rax`. Each bundle
embeds `stage.so`, `arg0-f16.data`, `deepseek_r1_pipeline_runner.so`, the four
runtime shared libraries (`libmlir_apfloat_wrappers`, `libmlir_float16_utils`,
`libmlir_c_runner_utils`, `libomp`), and—for stage 0 only—`vocab.txt` (8 payload
entries for stage 0, 7 for later stages). On x86_64 WSL for 2-stage R1-1.5B
FP16, `stage0.rax` is 1,785,293,415 bytes (~1.66 GiB) and `stage1.rax` is
1,803,872,352 bytes (~1.68 GiB) (`reports/pp-docs/04-package-rax.log`).

```bash
cmake -S . -B "$PKG_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM_BUILD/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_BUILD/bin/clang++" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DCMAKE_PREFIX_PATH="$MPI_HOME" \
  -DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" \
  -DBUDDY_CLI_ENABLE_MPI=ON \
  -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON \
  -DBUDDY_DSR1_SPEC="$SPEC_JSON" \
  -DBUDDY_DSR1_HF_CONFIG="$DEEPSEEKR1_MODEL_PATH/config.json" \
  -DDEEPSEEKR1_PIPELINE_STAGE_DIRS="$STAGE_ROOT/stages-2/0-14;$STAGE_ROOT/stages-2/14-28"

cmake --build "$PKG_BUILD" --target deepseek_r1_pipeline_rax -j
# Outputs:
#   $PKG_BUILD/models/deepseek_r1/pipeline_rax/stage0.rax
#   $PKG_BUILD/models/deepseek_r1/pipeline_rax/stage1.rax
```

**Manifest-only mode (`-DDEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=OFF`)** produces a
~2.2 KB manifest `.rax` per rank inside per-stage subdirectories
`<output_dir>/stage<i>/stage<i>.rax` alongside its relative `file:` assets
(large weight blobs are hardlinked on the same filesystem, falling back to a
copy across filesystems). This avoids multi-GB packing and cold-start payload
extraction during development or when deploying stage directories directly:

```bash
cmake -S . -B "$PKG_BUILD_MO" -G Ninja \
  ... \
  -DDEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=OFF \
  -DDEEPSEEKR1_PIPELINE_STAGE_DIRS="$STAGE_ROOT/stages-4/0-7;$STAGE_ROOT/stages-4/7-14;$STAGE_ROOT/stages-4/14-21;$STAGE_ROOT/stages-4/21-28"

cmake --build "$PKG_BUILD_MO" --target deepseek_r1_pipeline_rax -j
# Outputs:
#   $PKG_BUILD_MO/models/deepseek_r1/pipeline_rax/stage0/stage0.rax (+ stage.so, arg0-f16.data, deps, vocab.txt)
#   $PKG_BUILD_MO/models/deepseek_r1/pipeline_rax/stage1/stage1.rax (+ stage.so, arg0-f16.data, deps)
#   ...
```

You can also invoke the same target through `tools/buddy-codegen/build_model.py`
(verified in `reports/pp-docs/04b-build-model-py.log`):

```bash
CMAKE_GENERATOR=Ninja python3 tools/buddy-codegen/build_model.py \
  --spec "$SPEC_JSON" \
  --build-dir "$PKG_BUILD_PY" \
  --target deepseek_r1_pipeline_rax \
  --hf-config "$DEEPSEEKR1_MODEL_PATH/config.json" \
  --cmake-args=-DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  --cmake-args=-DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  --cmake-args=-DCMAKE_PREFIX_PATH="$MPI_HOME" \
  --cmake-args=-DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" \
  --cmake-args=-DDEEPSEEKR1_PIPELINE_STAGE_DIRS="$STAGE_ROOT/stages-2/0-14;$STAGE_ROOT/stages-2/14-28" \
  --jobs 4
```

### Incremental Rebuild Behavior

Each stage builds in an isolated working directory
`models/deepseek_r1/pipeline_stages/stage<i>/` with its own CMake targets
(`deepseek_r1_pipeline_stage_<i>_so` and `deepseek_r1_pipeline_stage_<i>_rax`),
while `deepseek_r1_pipeline_runner.so` is compiled once and shared across all
stages. Updating one stage's compiled object file rebuilds only that stage's
shared library and `.rax` bundle. For example, touching only
`stages-2/14-28/forward_decode.o` and re-running
`cmake --build "$PKG_BUILD" --target deepseek_r1_pipeline_rax -j` runs only 2
Ninja steps (`reports/pp-docs/04-package-rax.log`):

```text
[1/2] Linking CXX shared library models/deepseek_r1/pipeline_stages/stage1/stage.so
[2/2] [deepseek_r1_pipeline] stage1: packing .../pipeline_rax/stage1.rax
```

Neither `stage0.rax` nor `deepseek_r1_pipeline_runner.so` is re-linked or
re-packed.

## Single-Node Multi-Process Execution

Launch `N` MPI processes with `mpiexec -n N` and repeat `--stage-model` `N`
times in rank order. Each rank selects and opens only `stageModels[rank]`.
Only the final rank writes the generated JSON output to `stdout`.

### 2-Stage Embedded Example

```bash
export MPI_HOME="$HOME/opt/mpich-4.2.0"
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export BUDDY_RAX_PAYLOAD_DIR="/tmp/buddy-rax-cache"

PROMPT='Explain step by step how to add the integers from 1 to 100.'

"$MPI_HOME/bin/mpiexec" -n 2 "$BUDDY_BUILD/bin/buddy-cli" \
  --stage-model "$PKG_DIR/stage0.rax" \
  --stage-model "$PKG_DIR/stage1.rax" \
  --prompt "$PROMPT" \
  --max-tokens 56
```

Observed `stdout` on x86_64 WSL (`reports/pp-docs/05-run-and-verify.log`):

```json
{"prompt_tokens":25,"token_ids":[5338,11,358,1184,311,912,678,279,25780,504,220,16,311,220,16,15,15,13,358,3278,1191,553,48083,429,419,374,458,34784,4013,1380,1817,4647]}
```

For manifest-only bundles, pass the per-stage manifest paths instead (for
example, `--stage-model "$PKG_MO/stage0/stage0.rax" --stage-model "$PKG_MO/stage1/stage1.rax"`
for 2 ranks, or 4 paths with `-n 4` for 4 ranks).

In both whole-model (`--model`) and pipeline stage (`--stage-model`) modes,
`--max-tokens N` specifies the **total sequence length** (prompt + decode
steps): prefill always emits 1 token (unless EOS), followed by
`max(0, N - prompt_tokens)` decode steps, producing up to
`max(1, N - prompt_tokens + 1)` generated tokens for `1 <= N <= context`. For
the 25-token reference prompt above, `--max-tokens 56` produces 32 generated
tokens (`56 - 25 + 1 = 32`). Passing `--max-tokens 0` removes the user cap up
to the stage context limit (`context - prompt_tokens + 1`). Values
`N > context` are rejected in pipeline mode because pipeline stages use a
fixed-size KV cache without overflow eviction (whereas whole-model mode performs
sliding-window KV eviction via `handleKVCacheOverflow`).

Measured single-node x86_64 WSL results (`reports/pp-docs/05-run-and-verify.log`,
8-core host, context 64, 32 generated tokens, greedy):

| Run | Mode | Wall clock | GNU time peak RSS (per process) | Match vs. baseline |
| --- | --- | ---: | ---: | --- |
| 2 ranks (`0-14`, `14-28`) | Embedded (cold payload cache) | 72.16 s | 1.73 GiB (1,813,472 KiB) | 32/32 IDs, 4,861,952/4,861,952 FP16 logits bit-for-bit |
| 2 ranks (`0-14`, `14-28`) | Manifest-only | 27.09 s | 1.73 GiB (1,816,932 KiB) | 32/32 IDs, 4,861,952/4,861,952 FP16 logits bit-for-bit |
| 4 ranks (`0-7`..`21-28`) | Manifest-only | 59.73 s | 1.12 GiB (1,175,648 KiB) | 32/32 IDs, 4,861,952/4,861,952 FP16 logits bit-for-bit |

For comparison, the single-process whole-model reference on the same host took
13.29 s wall clock and 3.40 GiB peak RSS (`reports/pp-m0-m1-m2-report.md`).

## Multi-Node Deployment

This section covers deploying across multiple hosts connected over Ethernet
(for example, multiple SpacemiT K3 boards over 10GbE).

### 1. Node Consistency Requirements

Every participating node must match on:

- **MPICH version, ABI, and installation path**: install the same source-built
  MPICH 4.2.0 (`ch4:ofi`) at the same prefix (e.g., `/opt/mpich-4.2.0` or
  `$HOME/opt/mpich-4.2.0`) so Hydra can launch `hydra_pmi_proxy` and resolve
  `libmpi.so.12` identically on all hosts.
- **`buddy-cli` binary**: identical architecture and build path across all
  nodes.
- **Runner plugin and stage packages**: because `mpiexec` broadcasts the same
  command line (`--stage-model s0.rax --stage-model s1.rax ...`) to every rank
  and rank `r` opens `stageModels[r]`, the path at index `r` must exist on the
  host assigned to rank `r` (the simplest convention is mirroring the same
  directory layout on every node, or storing embedded `stage<r>.rax` on node
  `r` at the identical path). For manifest-only bundles, copy each node's full
  `stage<r>/` directory (including `stage.so`, `arg0-f16.data`,
  `deepseek_r1_pipeline_runner.so`, runtime `.so` dependencies, and `vocab.txt`
  on rank 0).
- **Local scratch space for embedded bundles**: set `BUDDY_RAX_PAYLOAD_DIR` to a
  local fast directory with enough free space for that node's unpacked stage
  weights (~1.8–2.0 GB per rank for 2-stage R1-1.5B FP16).

### 2. Passwordless SSH

MPICH Hydra launches remote ranks over SSH by default. From the launch node,
configure key-based passwordless SSH to every worker host (including itself if
listed in the hostfile):

```bash
ssh-keygen -t ed25519
for host in k3-node0 k3-node1; do
  ssh-copy-id "$host"
  ssh -o BatchMode=yes "$host" "hostname && $MPI_HOME/bin/mpichversion | head -n 4"
done
```

Ensure non-interactive SSH shells can find `$MPI_HOME/bin` and `$MPI_HOME/lib`
(either via `~/.bashrc` / `/etc/profile.d` or by forwarding variables with
`mpiexec -genv`).

### 3. Hostfile and Interface Binding

Create a hostfile listing one host per line with `:1` (one rank slot per node)
in desired rank order:

```text
# hosts — rank 0 -> k3-node0, rank 1 -> k3-node1
k3-node0:1
k3-node1:1
```

Launch with `-f hosts` and bind MPICH Hydra to the dedicated 10GbE interface
using `-iface <ifname>` (replace `eth10g` with the actual interface name from
`ip -br addr`, such as `enp1s0` or `eth1`). Under `ch4:ofi`, also export
`FI_TCP_IFACE` (or `MPIR_CVAR_CH4_OFI_IFNAME`) so libfabric routes activation
traffic over the same 10GbE interface rather than a management NIC:

```bash
"$MPI_HOME/bin/mpiexec" -f hosts -n 2 -iface eth10g \
  -genv PATH "$MPI_HOME/bin:$PATH" \
  -genv LD_LIBRARY_PATH "$MPI_HOME/lib" \
  -genv FI_TCP_IFACE eth10g \
  -genv OMP_NUM_THREADS 8 \
  -genv OPENBLAS_NUM_THREADS 1 \
  -genv BUDDY_RAX_PAYLOAD_DIR /var/tmp/buddy-rax-cache \
  /opt/buddy/bin/buddy-cli \
    --stage-model /opt/buddy/stages/stage0.rax \
    --stage-model /opt/buddy/stages/stage1.rax \
    --prompt 'Explain step by step how to add the integers from 1 to 100.' \
    --max-tokens 56
```

For `N` nodes (e.g., 4 boards), list all `N` hosts in `hosts`, pass `-n N`, and
pass all `N` `--stage-model` paths in rank order.

**Single-node hostfile syntax check (verified on x86_64 WSL):** Before moving to
multiple hosts, you can validate the hostfile and `-iface` command line locally
by pointing `hosts` to `localhost` twice with `-iface lo` (verified in
`reports/pp-docs/05-run-and-verify.log`, 32/32 token IDs matched):

```bash
cat <<'EOF' > hosts
localhost:1
localhost:1
EOF

"$MPI_HOME/bin/mpiexec" -f hosts -n 2 -iface lo "$BUDDY_BUILD/bin/buddy-cli" \
  --stage-model "$PKG_MO/stage0/stage0.rax" \
  --stage-model "$PKG_MO/stage1/stage1.rax" \
  --prompt 'Explain step by step how to add the integers from 1 to 100.' \
  --max-tokens 56
```

### 4. Rank-to-Host Placement Guidelines

- **Keep adjacent ranks topologically close**: rank `r` sends activations only to
  rank `r + 1` (plus rank `N - 1` sends a 4-byte token ID back to rank 0 and
  broadcasts a 4-byte stop flag). Map consecutive ranks (`0, 1, ..., N - 1`) onto
  nodes attached to the same L2 switch so every inter-stage transfer stays
  within a single switch hop.
- **Exploit shared memory when co-locating ranks**: if multiple ranks share a
  physical machine, assign consecutive ranks to the same host (e.g.,
  `node0:2` then `node1:2`) so intra-node hops use MPICH shared memory instead
  of crossing the NIC.
- **Target board execution**: on SpacemiT K3 boards, run one rank per board
  (`host:1`) and enable AI-core thread placement (`/proc/set_ai_thread`, see
  [K3DeepSeekR1.md](K3DeepSeekR1.md)) when hardware bring-up begins.

### 5. Target Hardware Metrics (SpacemiT K3 + 10GbE)

Target K3 boards and 10GbE fabric have not yet been benchmarked. All K3 / 10GbE
numbers are intentionally left blank until measured on hardware:

| Metric (SpacemiT K3, 10GbE, DeepSeek-R1-1.5B FP16) | 2 nodes | 4 nodes |
| --- | --- | --- |
| Prefill wall time / throughput | TBD (pending hardware) | TBD (pending hardware) |
| Decode throughput (tok/s) | TBD (pending hardware) | TBD (pending hardware) |
| Inter-stage 10GbE transfer latency (prefill / decode) | TBD (pending hardware) | TBD (pending hardware) |
| Peak RSS per board | TBD (pending hardware) | TBD (pending hardware) |
| Recommended tuned layer split | TBD (pending hardware) | TBD (pending hardware) |

For reference, the activation wire payload computed from the FP16 R1-1.5B config
(`width=1536`, `context=64`) is **196,608 B** (192 KiB) per hop during prefill
plus the 16 B header, and **3,072 B** (3 KiB) per hop per decode step plus the
16 B header (calculated from config; MPI transport overhead on 10GbE is
**TBD (pending hardware)**).

## Partitioning Guidelines

1. **Start with uniform layer counts**: divide `num_hidden_layers` evenly across
   `N` ranks (for DeepSeek-R1-Distill-Qwen-1.5B with 28 layers: `14 / 14` for 2
   stages, `7 / 7 / 7 / 7` for 4 stages).
2. **Account for heavier boundary stages**:
   - **Stage 0** additionally stores `embed_tokens`
     (`151,936 × 1,536` FP16 = 233,373,696 elements ≈ **0.47 GB (estimate)**,
     0.435 GiB).
   - **The final stage** additionally stores `lm_head`
     (`151,936 × 1,536` FP16 = 233,373,696 elements ≈ **0.47 GB (estimate)**,
     0.435 GiB) plus the final RMSNorm (`1,536` elements), and computes the
     vocabulary projection on every step.
   - Each Qwen2-1.5B decoder layer contributes ~46.80M FP16 parameters
     (~93.6 MB / ~89.3 MiB, derived from the measured 7-layer middle stage
     weight file below).

Measured FP16 weight sizes across uniform 2-stage and 4-stage splits of
DeepSeek-R1-Distill-Qwen-1.5B (`reports/pp-docs/03-export-compile.log`):

| Split | Stage interval | Extra non-layer modules | FP16 weight elements | `arg0-f16.data` size |
| --- | --- | --- | ---: | ---: |
| 2-stage | `[0, 14)` | `embed_tokens` | 888,543,296 | 1.655 GiB (1,777,086,592 B) |
| 2-stage | `[14, 28)` | final norm + `lm_head` | 888,544,832 | 1.655 GiB (1,777,089,664 B) |
| 4-stage | `[0, 7)` | `embed_tokens` | 560,958,528 | 1.045 GiB (1,121,917,056 B) |
| 4-stage | `[7, 14)` | none | 327,584,832 | 0.610 GiB (655,169,664 B) |
| 4-stage | `[14, 21)` | none | 327,584,832 | 0.610 GiB (655,169,664 B) |
| 4-stage | `[21, 28)` | final norm + `lm_head` | 560,960,064 | 1.045 GiB (1,121,920,128 B) |

3. **Tune from measured memory and latency**:
   - In a 4-stage uniform split, stages 0 and 3 each hold ~71% more weight bytes
     than middle stages 1 and 2 (1.045 GiB vs. 0.610 GiB). If board RAM is tight
     or the final rank dominates step time due to `lm_head`, shift 1–2 decoder
     layers from the boundary stages into the middle stages (for example,
     testing `6 / 8 / 8 / 6` or `5 / 9 / 9 / 5` on 28 layers as a starting
     **estimate**—each shifted layer moves ~89.3 MiB of FP16 weights and 2 KV
     tensors).
   - Always re-measure per-rank RSS and per-stage compute time on the actual
     target boards before finalizing a non-uniform split (optimal K3 split:
     **TBD (pending hardware)**).

## Verification and Troubleshooting

### Verifying Output Against a Whole-Model Baseline

Capture a greedy baseline from the whole-model `.rax` package:

```bash
python3 examples/BuddyDeepSeekR1/capture-greedy-baseline.py \
  --buddy-cli "$BUDDY_BUILD/bin/buddy-cli" \
  --model "$NATIVE_BUILD/models/deepseek_r1/deepseek_r1.rax" \
  > baseline.json
```

Run the pipeline with `BUDDY_PP_DUMP_LOGITS` set to record raw FP16 logits at
every sampled position on the final rank, then compare both token IDs and
bit-for-bit logits using `examples/BuddyDeepSeekR1/pipeline/check_result.py`:

```bash
BUDDY_PP_DUMP_LOGITS=stage-logits.f16 \
  "$MPI_HOME/bin/mpiexec" -n 2 "$BUDDY_BUILD/bin/buddy-cli" \
    --stage-model "$PKG_DIR/stage0.rax" \
    --stage-model "$PKG_DIR/stage1.rax" \
    --prompt 'Explain step by step how to add the integers from 1 to 100.' \
    --max-tokens 56 \
  > stage-out.json

# Token ID comparison:
python3 examples/BuddyDeepSeekR1/pipeline/check_result.py \
  baseline.json stage-out.json

# Optional bit-for-bit FP16 logits comparison against a full-reference dump:
python3 examples/BuddyDeepSeekR1/pipeline/check_result.py \
  baseline.json stage-out.json \
  --logits full-reference-logits.f16 stage-logits.f16 \
  --vocab-size 151936
```

Expected output (`reports/pp-docs/05-run-and-verify.log`):

```text
32/32 sampled token IDs match
4,861,952 sampled FP16 logit elements match bit for bit
```

You can also run the repository's integrated CTest suites:

- Unit tests (no model weights required): `cmake --build "$BUDDY_BUILD" --target check-buddy-pipeline -j`
- Optional end-to-end tests: configure with
  `-DBUDDY_ENABLE_PIPELINE_E2E_TESTS=ON`,
  `-DBUDDY_PIPELINE_STAGE_ARTIFACTS_DIR=<stages-dir>`, and
  `-DBUDDY_PIPELINE_VALIDATION_DIR=<validation-dir>`, then build
  `check-buddy-pipeline-e2e`.

### Common Errors and Diagnostics

All stage-mode failures log the rank, package path, and conflicting field to
`stderr` and terminate the job via `MPI_Abort(MPI_COMM_WORLD, 1)` so peer ranks
do not hang.

1. **Number of `--stage-model` arguments does not match `-n` (`MPI_Comm_size`)**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log`):

   ```text
   buddy-cli: rank 0 [package=.../stage0.rax]: number of --stage-model packages (1) does not match MPI world size (2)
   buddy-cli: rank 1 [package=<unset>]: number of --stage-model packages (1) does not match MPI world size (2)
   ```

   *Fix*: pass exactly `N` `--stage-model` arguments when launching `mpiexec -n N`.

2. **Overlapping layer intervals**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log`, e.g., `[0, 14)` followed by `[7, 14)`):

   ```text
   rank 1 [package=.../stage1.rax, model=deepseek_r1_f16]: field startLayer/endLayer overlap: rank 1 startLayer=7 overlaps rank 0 endLayer=14
   ```

   *Fix*: ensure `stage[r].end_layer == stage[r+1].start_layer` for all adjacent ranks.

3. **Missing layers (gap between adjacent stages)**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log`, e.g., `[0, 7)` followed by `[14, 28)`):

   ```text
   rank 0 [package=.../stage0.rax, model=deepseek_r1_f16]: field startLayer/endLayer gap (missing layers): rank 0 endLayer=7 < rank 1 startLayer=14
   ```

   *Fix*: include all intermediate stages covering `[0, total_layers)`.

4. **Stages passed out of order**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log`, e.g., `[14, 28)` before `[0, 14)`):

   ```text
   rank 0 [package=.../stage1.rax, model=deepseek_r1_f16]: field startLayer invalid on first rank 0: expected 0, got 14
   ```

   *Fix*: list `--stage-model` arguments (and `DEEPSEEKR1_PIPELINE_STAGE_DIRS`)
   strictly from layer `0` to `total_layers`.

5. **Activation protocol error (bad header step/pos or truncated payload)**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log` and
   `reports/pp-stage-rax-step3-report.md`):

   ```text
   rank 1 protocol error (source=0, expected_source=0): header step mismatch; expected={version=1, step=2, pos=6, rows=1}; received={version=1, step=1, pos=5, rows=1}
   rank 1 protocol error (source=0, expected_source=0): payload element count mismatch (got 1535, expected 1536); expected={version=1, step=1, pos=5, rows=1}; received={version=1, step=1, pos=5, rows=1}
   ```

   *Fix*: check that all nodes run the exact same `buddy-cli`,
   `deepseek_r1_pipeline_runner.so`, and MPICH build, and that no mismatched
   stage binary or stale runner plugin is being loaded.

6. **Missing runner plugin `.so` or missing `runner_library` manifest attribute**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log` and
   `reports/pp-stage-rax-step6-report.md`):

   ```text
   buddy-cli: rank 1 [package=.../stage1.rax]: buddy-cli: dlopen runner failed: .../deepseek_r1_pipeline_runner.so: cannot open shared object file: No such file or directory
   buddy-cli: rank 1 [package=.../stage1.rax]: stage package is missing required runner_library attribute
   ```

   *Fix*: for manifest-only bundles, copy `deepseek_r1_pipeline_runner.so` (and
   the other runtime `.so` files) next to `stage<i>.rax` on every node, or use
   embedded bundles (`DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=ON`). Re-pack via
   `deepseek_r1_pipeline_rax` if `runner_library` is absent.

7. **Whole-model `.rax` passed to `--stage-model`**
   (observed in `reports/pp-docs/06-troubleshoot-checks.log`):

   ```text
   buddy-cli: rank 0 [package=.../deepseek_r1.rax]: missing stage field pp_stage_version
   ```

   *Fix*: use `--model` for whole-model `.rax` packages, or build stage packages
   via `deepseek_r1_pipeline_rax` for `--stage-model`.

8. **`buddy-cli` built without MPI**
   (observed in `reports/pp-stage-rax-step6-report.md`, exit code 2):

   ```text
   [Error] --stage-model requires an MPI-enabled build of buddy-cli. Rebuild with MPI available and -DBUDDY_CLI_ENABLE_MPI=ON.
   ```

   *Fix*: reconfigure with `-DCMAKE_PREFIX_PATH="$MPI_HOME" -DMPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx" -DBUDDY_CLI_ENABLE_MPI=ON` and rebuild `buddy-cli`.

## Current Limitations

- **Precision and model family**: stage export currently supports only plain
  FP16 `Qwen2ForCausalLM` configurations (`full_attention` layers). Quantized
  variants (such as `w4g32`, INT4, INT8), `decode_pack`, chunked prefill,
  tiered KV cache, and custom memory arenas are not supported yet.
- **CPU execution only**: stage kernels execute on CPU via LLVM/MLIR + OpenMP.
- **Batch size 1 only**: `--stage-model` accepts only default or
  `--batch-size 1`.
- **Blocking point-to-point communication**: adjacent stages communicate with
  blocking `MPI_Send` / `MPI_Recv` without compute-communication overlap.
- **No single-request latency speedup with more stages**: for a single batch-1
  request, only one rank computes at any instant while every stage boundary adds
  serialization and network latency. Adding stages reduces per-node weight and
  KV memory footprint rather than accelerating single-request decoding.
- **Unsupported CLI flags in stage mode**: `--stage-model` requires an explicit
  non-empty `--prompt` and cannot be combined with `--model`, `--model-so`,
  `--weights`, `--vocab`, `--runner-so`, `--interactive`, `--prompt-file`,
  `--prompt-length`, `--audio`, `--image`, `--chat-template`,
  `--perplexity`/`--ppl-*`, `--defer-decode-token-readback`, or
  `--print-all-batch`.
