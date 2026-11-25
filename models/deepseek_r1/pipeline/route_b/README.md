# Qwen2 K3 component kernels

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

The export directory must be new. Publication occurs after all 28 component
graphs and weight layouts have been checked for the 1.5B checkpoint. Only the
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
