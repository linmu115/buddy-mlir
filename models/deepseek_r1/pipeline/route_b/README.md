# Qwen2 and Qwen3-MoE K3 component kernels

Export from the original full-model configuration. Use the same native Buddy
Python packages, MLIR binaries and K3 cross toolchain as a working Route A build.

```sh
python3 tools/buddy-codegen/import_pipeline_stage.py --config "$CONFIG" \
  --route-b --load-mode lazy --output-dir "$BUNDLE"
cmake -G Ninja -S models/deepseek_r1/pipeline/route_b -B "$BUILD" \
  -DCMAKE_C_COMPILER="$RISCV_CC" -DCMAKE_CXX_COMPILER="$RISCV_CXX" \
  -DBJ47_ROUTE_B_BUNDLE="$BUNDLE" -DBJ47_BUDDY_OPT="$BUDDY_OPT" \
  -DBJ47_LLVM_TOOLS="$LLVM_BIN" -DBJ47_CRUNNER="$RISCV_CRUNNER"
cmake --build "$BUILD" --target bj47-route-b-kernels
```

The export directory must be new. Publication occurs after every layer graph and weight layout has been checked. Only the
first decoder goes through fusion and MLIR emission. `components.json` indexes
content hashes and global layer IDs. `slots.json` uses layer 0 names as
template-local names. For another global layer, substitute its index in the
source keys. RoPE has its own file and independent offset space. Use the
ordered slots filtered by bucket when mapping weights.

Read the info descriptor on each DSO's own local handle. The decoder uses
caller-owned contiguous K, V, output and prepared cos/sin. Prepare once per
position per rank. Use distinct aligned input and output buffers. The arena
and worker pool are shared across the three libraries. Place
`libbj47_kernel_runtime.so` next to the kernels. Entries serialize scratch
arena use and the host retains ownership of results across calls.

The head reports kernel ABI 2 using the V1 info structure. Call
`lm_head_prefill` on the final hidden row of a chunk and `lm_head_decode` on
a decoded row. `lm_head` aliases decode. Route A normalizes before selecting
the chunk's last row and fuses decode normalization into the quantized
projection. These paths have different reduction orders.

Each `*.kernel.json` identifies a linked artifact and its exports. The build
directory contains the generated C++ signatures, MLIR before and after copy
elimination, LLVM IR and RISC-V objects. The runtime can construct
`TemplateStage` from these contracts in the next phase.

## Verification

```sh
python3 tests/Pipeline/test_route_b_counts_and_roles.py
python3 tests/Pipeline/test_route_b_pack_layout.py --self-test
python3 tests/Pipeline/test_route_b_pack_layout.py "$BUNDLE" \
  "$A/0-14" "$A/14-28" --receipt "$BUILD/slices.json"
"$RISCV_CC" -O2 -Iruntime/include tests/Pipeline/route_b_k3_harness.c \
  -lm -ldl -o "$BUILD/route-b-harness"
```

For the board fixture, supply the bundle and three Route A reference
directories. The first two contain the normal 0:14 and 14:28 `stage.so`
and both constant files. The last contains the normal 7:8 export's FP32
bucket and a `ref.so` exposing its two MLIR ciface entries and arena reset.
The harness uses layer 7's component Q4 file for this probe. Compare it
against the 7:8 Q4 export with `cmp` during fixture preparation.
Set the K3 runtime library path, select the AI core in the same shell and
exec the harness with these four arguments. It compares intermediate
hidden, logits and complete KV slots and verifies untouched cache spans.

The optional target `bj47_checked_heap_runtime` builds a verification backend in
`$BUILD/checked-heap`. Put that directory first in `LD_LIBRARY_PATH` and rerun
the same executable and component DSOs. It allocates scratch from the system
heap, rejects invalid frees, and clears and frees outstanding scratch on return.
The loaded runtime path is printed by the harness.

## Publish and run the MPI host

Place the matching tokenizer vocabulary alongside the stage-1 component
files. Publishing checks every weight and kernel against its receipt and
binds their hashes, tokenizer, runtime and model settings into the bundle ID.

```sh
python3 tools/buddy-codegen/gen_manifest.py --format route-b-json \
  --config "$BUNDLE/components.json" -o "$BUNDLE/route-b.json"
cmake -S models/deepseek_r1/pipeline/route_b/host -B "$HOST_BUILD" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOST_BUILD" --target buddy-pp-route-b
```

The host requires a C++17 compiler, MPI, nlohmann-json and OpenSSL Crypto.
Use the K3 toolchain and target OpenSSL libraries for a cross build. Set
`RISCV_MPI_DIR` to the target MPICH prefix. A kernel build can also enable
`BJ47_ROUTE_B_HOST` to expose the executable target in the same build tree.
Native host builds with testing enabled include three mock-kernel tests.
Build the default target before running `ctest --test-dir "$HOST_BUILD"`.

Deploy the index, vocabulary, kernels, shared runtime and weights to the
same relative paths on all participating hosts. Verify deployed file hashes
against the published index. Launch with the board's MPI fabric settings,
runtime library paths and AI-core selection. The host validates the index,
range chain and live kernel descriptors before generation. Each process
opens only the weight blobs belonging to its current range plus RoPE.

```sh
mpiexec -n 2 buddy-pp-route-b --bundle "$BUNDLE/route-b.json" \
  --ranges 0:14,14:28 --prompt "What is the capital of France?" --total-tokens 128
```

For a new request, restart with `--ranges 0:16,16:28`. Each process starts
with empty persistent KV for its new range. In MPMD jobs supply a single
`--stage-range A:B` per rank. A single rank defaults to the whole model.
`--total-tokens` follows the Route A total sequence length convention.
The baseline uses blocking FP32 transport and greedy sampling. Set
`BUDDY_PP_TIMING=1` for existing per-frame compute/communication intervals
and per-stage logical mapping/KV byte counts. Set `BUDDY_PP_OBSERVE=1`
to also emit READY, actual prompt IDs, rank-0 first-frame-to-token time and
per-rank Linux VmHWM for either route. READY precedes tokenization.

Kernel files are small enough to check their complete SHA-256 before
dlopen on every launch. To recheck the contents of every local weight
mapping, add `--verify-weights`. This touches all its pages and is excluded
from hot-restart and residency measurements.
