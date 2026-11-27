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
Native host builds with testing enabled include mock-kernel and wire-protocol tests.
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

The runtime defaults to `--wire f32 --transport blocking`. Select
`--wire f16` to round hidden activations to binary16 during MPI transfer.
Each receiver expands them to f32 before running the next layer. Generated
tokens may change. All ranks must agree on both flags. With
`--transport async`, MPI sends own their buffers. On two ranks, set
`BUDDY_PP_COMPLETE_PREFILL_SEND=1` to complete each multi-row send before
the next chunk when using software MPI progress.

For a new request, restart with `--ranges 0:16,16:28`. Each process starts
with empty persistent KV for its new range. In MPMD jobs supply a single
`--stage-range A:B` per rank. A single rank defaults to the whole model.
`--total-tokens` follows the Route A total sequence length convention.
The baseline uses blocking FP32 transport and greedy sampling. Set
`BUDDY_PP_TIMING=1` for existing per-frame compute/communication intervals
and per-stage logical mapping/KV byte counts. Set `BUDDY_PP_OBSERVE=1`
to also emit READY, actual prompt IDs, rank-0 first-frame-to-token time and
per-rank Linux VmHWM for either route. READY precedes tokenization.

### Multi-stream decode

Repeat `--prompt` to run two to eight independent requests in one MPI job.
Every rank must use the same prompt count, ranges, wire precision and
transport. Rank 0 tokenizes the prompts and resolves `--total-tokens`
against each prompt length separately.

```sh
mpiexec -n 2 buddy-pp-route-b --bundle "$BUNDLE/route-b.json" \
  --ranges 0:16,16:28 --wire f16 --total-tokens 128 \
  --prompt "Explain why ice floats." --prompt "Explain why the sky looks blue."
```

Prefill runs in prompt order and initializes a separate, zeroed KV slot for
each request. All slots share the loaded templates, mapped weights and
operator scratch storage. Decode uses one nonblocking activation message per
stage hop and one nonblocking sample reply from the last rank to rank 0.
A slot gets its next step after its own reply. EOS and the token limit retire
each slot independently. In multi-stream mode `--transport` selects the
Prefill transport. Decode uses the same nonblocking protocol in either case.

Rank 0 writes one JSON object with aggregate Decode wall time, token count,
and throughput, plus each stream's prompt length, token IDs, TTFT and Decode
rate. Decode counts exclude the first token produced by Prefill. TTFT starts
at each stream's first Prefill frame. Per-stream Decode rate spans batch
Decode start through that stream's last reply. Set `BUDDY_PP_STORAGE=1` to
emit mapped weight and total KV byte counts without per-frame logging.
`BUDDY_PP_OBSERVE=1` reports each rank's Linux VmHWM.

Kernel files are small enough to check their complete SHA-256 before
dlopen on every launch. To recheck the contents of every local weight
mapping, add `--verify-weights`. This touches all its pages and is excluded
from hot-restart and residency measurements.

### Resident MPI and HTTP serving

Build the host target against the existing bundle. Each rank loads its
assigned templates and weights once. Rank 0 checks the vocabulary digest
when present and keeps the tokenizer vocabulary loaded between requests. Each
rank checks its loaded template DSOs and the bundle
runtime digest when present.

```sh
cmake --build "$HOST_BUILD" --target buddy-pp-route-b-serve
mpiexec -n 2 buddy-pp-route-b-serve --bundle "$BUNDLE/route-b.json" \
  --ranges 0:16,16:28 --wire f16 --serve --slots 2
```

Rank 0 reads JSONL on stdin. Wait for `ready`, then send one command per line.
The default is two KV slots, with one to eight supported. A submit with
`total_tokens` uses the same limit as the offline CLI. A submit with
`max_new_tokens` counts generated tokens, including the first prefill sample.
Omit the limit or pass zero to use the remaining context.

```jsonl
{"cmd":"submit","req_id":"a","prompt":"Why does ice float?","max_new_tokens":64}
{"cmd":"submit","req_id":"b","prompt":"为什么天空是蓝色的？","total_tokens":128}
{"cmd":"shutdown"}
```

Each request emits `accepted`, one or more `token` events, then `done`.
Invalid input emits `error`. Token step zero is the prefill sample. TTFT in
each token event starts at receipt of the complete command and includes queue
time. Engine event timestamps use the rank-0 monotonic clock. When a request
can be admitted, rank 0 drains already sent Decode frames before forwarding
PrefillStart. Each rank clears the selected KV slot before forwarding prompt
frames. Surviving requests resume Decode alongside the admitted request.
Shutdown and EOF finish the accepted and queued work, then close all ranks.

The standard-library gateway owns the MPI process and takes its launch
arguments directly. Build or install the host first, then use
`python3 tools/route_b_serve/server.py --help` for launcher flags. K3 mode
takes an absolute MPICH launcher, rank-ordered 1G `end0` addresses and the
board's VLEN probe. It selects the AI cores in each rank process and uses
TCP with NODELAY and eight compute threads. The gateway binds to loopback
by default. `GET /health` includes rank PIDs, VmRSS and VmHWM. It reads local
rank memory from `/proc` and uses the same SSH host addresses for other K3
ranks.

Use `GET /v1/models` to discover the served model and
`POST /v1/chat/completions` with the last user message, optional `max_tokens`
and `stream`. SSE deltas preserve UTF-8 across token boundaries. An
incomplete codepoint at the output limit is omitted. Set
`"route_b_metrics": true` to include a Route B event with each step, even
when the UTF-8 decoder is waiting for a complete character. For a streaming
terminal session, run `python3 tools/route_b_serve/stream_chat.py`.

## Qwen3.5 text hybrid bundles

The hybrid manifest carries one layer type per layer. The host binds the
corresponding `gdn_layer.so` or `full_attn_layer.so` at each local layer and
allocates the state sizes advertised by that template for each slot. Hybrid
templates extend the descriptor returned by `bj47_kernel_info_v1` and preserve
the six-argument layer entry points. `BJ47FrameRowsV2` supplies the logical row
count independently of the fixed wire capacity. Hybrid prefill appends each
prompt position once and uses the last valid row for the head.

Build a bundle from an indexed safetensors checkpoint and a line-indexed BPE
vocabulary (one token per ID). The f32 validation path accepts an f16 or f32
embedding table and uses f32 activations, projections and state. The exporter
reads only the text tensors required by the manifest. It retains lowered
MLIR, LLVM IR, the benchmark-independent build timings and the DSO hashes
under `--work`. Choose a new output directory for each build.

```sh
python3 tools/buddy-codegen/build_route_b_hybrid.py \
  --checkpoint "$CHECKPOINT" --vocab "$VOCAB" --model-id tiny-qwen3_8 \
  --embed-dtype f16 --output "$BUNDLE" --work "$WORK" \
  --buddy-opt "$BUDDY_OPT" --llvm-bin "$LLVM_BIN"
buddy-pp-route-b --bundle "$BUNDLE/route-b.json" --ranges 0:4 \
  --prompt 'Hello' --total-tokens 24
```

For K3, append the Clang target, sysroot, GCC toolchain and
`--cxx-flag=-march=rv64gcv_zfh_zvfh`. The f32 path uses RVV and an in-place,
per-token recurrence in prefill and decode. Projections, embedding lookup and
the GDN state update are destination-passing MLIR. The C++ template supplies
frame handling, convolution, normalization, attention and residual ordering.

The B1 scripts in `tests/Pipeline` generate the 4-layer fixture using the
locked checkpoint config and index, run the locked HF source as a CPU
oracle, and compare tensor traces from `RouteBHybridProbe.cpp`. The service
test switches all documented stage cuts against the same DSO hashes and
checks a staggered two-slot run with slot reuse. The hybrid tokenizer checks
the Qwen3.5 special token IDs and rejects an unencodable prompt.

### Native single-layer MTP

The hybrid exporter detects the 15 tensors of the one-layer predictor. Use
`--no-mtp` to export the backbone alone. Packed hybrid version 2 interleaves
gate and up RVV tiles and stores the two IME block ranges separately. Version
1 bundles still load with their original DSOs.

Pass `--mtp` to the CLI or resident server to enable greedy speculation.
Rank zero gathers shifted embeddings directly from its table in the stored
dtype. The last rank owns the predictor, its per-slot KV cache and the shared
head. It applies the backbone final norm before verification and feeds the
predictor's normalized output directly to the linear head. The first prompt
sample supplies the final shifted embedding.

A two-row verification saves GDN state after the current token. On rejection,
each rank restores that state and advances the slot by one position. The next
attention write overwrites the rejected KV row. Acceptance advances the slot
by two positions. The service interleaves active slots at step boundaries and
emits a token event for each committed token. CLI results and resident done
events include proposal and acceptance counts, their ratio and generated
decode tokens per verification step. The first prefill token is excluded.

### Qwen3.8-27B on K3

The fixed 1024-bit VLEN head streams four weight banks per group for both
decode and two-row verification. Verification uses `lm_head_linear` with
`rows=2`. Both rows visit an 18 KiB weight window before advancing.
The packed layer templates fuse both residual additions into the projection
stores. For the 24-query/4-KV attention geometry, decode scores two queries
per group through visible length 384. Longer contexts split each KV head's
six queries into groups of four and two.

With `--prefill-ime`, the 64-row dense projections consume the existing
144-byte IME blocks using the original `smt.vmadot.hp` arithmetic. The
two-row 27B GLU uses the full-width RVV tile for each gate and up projection.
The weight offsets, pack version and exported kernel signatures are stable.
A binary upgrade also needs the matching template sizes, digests, runtime
digest and bundle ID in `route-b.json`.

GDN assigns each key head and its associated value heads to the same worker.
This workshare includes the paired f32 gate projections, depthwise convolution,
normalization and state update. The two-row entry snapshots each worker's state
slices immediately after the first row. Padded rows leave state untouched.
The A/B kernel retains the MLIR projection's eight K accumulators and rounds
products and additions separately.

The verifier schedules both rows in one workshare. Each row selects its own
group size and causal endpoint, including when it crosses visible length 384. Prefill appends valid
KV rows, assigns query heads to workers and scores only through each row's
position. Its tiled softmax recomputes an earlier prefix when the running
maximum rises. This preserves the ordered denominator and rounds probabilities
before the value product, including at the input boundary to the W4 mixer.
