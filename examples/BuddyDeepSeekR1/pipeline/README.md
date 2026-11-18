# DeepSeek R1 pipeline on x86

This example runs DeepSeek-R1-Distill-Qwen-1.5B in FP16 with one decoder layer
interval per MPI rank. It uses static full-context prefill and one-token decode.
Every rank keeps its own KV cache. Rank zero tokenizes the prompt and embeds
tokens. The last rank samples greedily and sends each sampled ID to rank zero.

## Build

Run from the Buddy-MLIR repository root in the Python environment used to
import DeepSeek. Build the native FP16 whole model first with context 64 so the
fixed reference prompt and 32 generated IDs fit. Set these variables to the
build paths in your environment:

- `BUDDY_BUILD` points to the Buddy build with `bin/buddy-opt`, `bin/buddy-cli`
  and `models/deepseek_r1`. The model directory must contain
  `generated/config.json`, `vocab.txt`, `libomp.so` and `deepseek_r1.rax`.
- `LLVM_BUILD` points to the LLVM build with Clang, `llc` and MLIR runner
  libraries. `MLIR_INCLUDE` points to the MLIR include directory containing
  `mlir/ExecutionEngine/CRunnerUtils.h`.
- `MPI_HOME` points to the MPICH installation prefix. The example was tested
  with MPICH 4.2.0.
- `DEEPSEEKR1_MODEL_PATH` points to the local Hugging Face checkpoint.
  `PYTHONPATH` includes the Buddy compiler and MLIR Python bindings from the
  native model build environment.

```bash
SRC=examples/BuddyDeepSeekR1/pipeline
MODEL="$BUDDY_BUILD/models/deepseek_r1"
OUT="$PWD/build-pipeline"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$MODEL:$LLVM_BUILD/lib:${LD_LIBRARY_PATH:-}"

build_stage() {
  local start=$1 end=$2 stage="$OUT/$1-$2"
  python tools/buddy-codegen/import_pipeline_stage.py \
    --config "$MODEL/generated/config.json" --layers "$start" "$end" \
    --output-dir "$stage" || return
  python tools/buddy-codegen/verify_pipeline_stage.py \
    --stage-dir "$stage" --header "$stage/stage-config.inc" || return
  python tools/buddy-codegen/compile_pipeline.py \
    --config "$stage/config.json" --compile-all --mlir-dir "$stage" \
    --output-dir "$stage" --buddy-opt "$BUDDY_BUILD/bin/buddy-opt" \
    --llvm-tools-dir "$LLVM_BUILD/bin" --llc-attrs=-mcpu=native -j 1 || return
  "$LLVM_BUILD/bin/clang++" -O2 -std=c++17 -fPIC -shared \
    -I"$stage" -I"$MLIR_INCLUDE" "$SRC/Stage.cpp" \
    "$stage"/{forward_prefill,forward_decode,subgraph_prefill,subgraph_decode}.o \
    -L"$LLVM_BUILD/lib" -lmlir_c_runner_utils "$MODEL/libomp.so" \
    -Wl,-z,defs -o "$stage/stage.so"
}

"$MPI_HOME/bin/mpicxx" -O2 -std=c++17 -Ifrontend/Interfaces -Iruntime/include \
  "$SRC/pipeline.cpp" -ldl -o "$OUT/pipeline"

build_stage 0 14 &&
build_stage 14 28
```

## Run and compare

```bash
PROMPT='Explain step by step how to add the integers from 1 to 100.'
python examples/BuddyDeepSeekR1/capture-greedy-baseline.py \
  --buddy-cli "$BUDDY_BUILD/bin/buddy-cli" --model "$MODEL/deepseek_r1.rax" \
  > "$OUT/baseline.json"

"$MPI_HOME/bin/mpiexec" -n 2 "$OUT/pipeline" "$MODEL/vocab.txt" 32 "$PROMPT" \
  "$OUT/two-logits.f16" "$OUT/0-14" "$OUT/14-28" > "$OUT/two.json"
python "$SRC/check_result.py" "$OUT/baseline.json" "$OUT/two.json"

build_stage 0 7 &&
build_stage 7 14 &&
build_stage 14 21 &&
build_stage 21 28
"$MPI_HOME/bin/mpiexec" -n 4 "$OUT/pipeline" "$MODEL/vocab.txt" 32 "$PROMPT" \
  "$OUT/four-logits.f16" "$OUT/0-7" "$OUT/7-14" "$OUT/14-21" "$OUT/21-28" \
  > "$OUT/four.json"
python "$SRC/check_result.py" "$OUT/baseline.json" "$OUT/four.json"
```

The rank order follows the stage directory order. The receiver obtains the
activation row count through `MPI_Probe` and `MPI_Get_count`, then validates it
against the compiled prefill or decode shape. The fourth argument to the
runner is a path for raw FP16 logits at the sampled positions. Pass `-` to
disable the dump.
