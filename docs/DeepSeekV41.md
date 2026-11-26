# DeepSeek V4.1 text inference on Route B

The batch-one tiny backbone runs six layers with 128 hidden channels and four
residual streams. Layers 1 and 3 publish compressed KV and index keys. Layer 3
publishes candidate blocks, layer 5 reindexes, and layers 1 and 4 use read-only
mapped Engram tables. The exporter initializes a deterministic small checkpoint
with real FP4 and FP8 storage. It uses a 128-token synthetic vocabulary, seed
4106, a 64-position cache and YaRN starting at position 32.

## Run the model and its differential tests

Use a Python environment with PyTorch CPU supporting E2M1x2 and E8M0, NumPy,
SymPy, tokenizers and safetensors. Set `SPEC` to the directory containing the
upstream `inference_model.py` and `inference_engram.py`. Set `BUILD` to a
configured Buddy build and `OUT` to a scratch directory.

```sh
D=tests/Models/Inputs/deepseek_v41
cmake -S . -B "$BUILD" -DBUDDY_DEEPSEEK_V41_SPEC_DIR="$SPEC"
cmake --build "$BUILD" --target tiny-deepseek-v41 buddy-v41-differential
ctest --test-dir "$BUILD" -R buddy-v41 --output-on-failure
python "$D/export_v41_fixture.py" "$SPEC" "$OUT/fixture"
"$BUILD/bin/tiny-deepseek-v41" "$OUT/fixture" 16 5 18 31
```

The executable emits the prefill prediction and one additional prediction per
decode step. It reads only the weight manifest. The differential driver also
reads expected tensors. Each stage owns its window and compressor tail. At an
interior cut the receiver retains its replica and appends new source rows.
The tests check every hidden stream, mix, valid cache prefix, index list,
candidate mask and final logit after 37 prefill tokens and 16 decode steps.
The C++ driver compares both partitionings byte for byte. Its isolated operator
tests replay the upstream module inputs.

The CPU kernel module keeps quantization groups, fp32 accumulators, stored
scales, bf16 casts, 64-entry sparse attention tiles and the online denominator.
Engram uses the normalized token mapping and a distinct DEAD sentinel. The
compressed pad ID fills unavailable history. A literal text pad ID remains a
normal hash input as in the upstream implementation.

## Stream a projection into K3 tiles

With the Buddy Python bindings on PYTHONPATH, pass a shard and tensor key for
both weights and scales. Repeat `--matrix` for gate and up, including when the
two live in separate shards. The default layout interleaves GLU tiles. With
`--ime`, all gate blocks precede the up blocks. Slabs bound decoded memory.

```sh
python tools/buddy-codegen/deepseek_v4_quant.py "$OUT/gate.tiles" \
  --format fp4 --matrix "$SHARD" "$WEIGHT_KEY" "$SCALE_SHARD" "$SCALE_KEY"
python "$D/deepseek_v4_quant_test.py" "$SPEC" "$OUT/quant"
python "$D/lower_deepseek_w4.py" "$OUT" "$BUILD/bin/buddy-opt" "$LLVM_BIN"
```

E2M1 uses the low nibble for the first K element. FP4 weights have one
scale per row and 32 K values. FP8 linear weights share scales across 32×32
or 128×128 blocks. Engram FP8 scales are row-local. K3 w4g32 uses signed
integer nibbles, fp16 group scales and dynamically quantized int8 activations.
The import test compares actual tile/IME bytes and the MLIR kernel's output.
For the V4 expert path apply the asymmetric clamp before SwiGLU and multiply
the route coefficient before the down projection.

## Measure and plan

Build `DeepSeekV4Bench.cpp` for RVV with OpenMP and
`-ffp-contract=off`. From its board directory run `run_when_idle.sh` with
one of `hc`, `moe`, `swa`, `full1`, `full2`, `reindex`, `reuse` or `head`.
For `engram`, also supply a new file name on NVMe. The benchmark populates
the pages sampled by its direct reads, prefaults every warm sample, and
removes the sparse file at exit. Capture each successful invocation in
`board-MODE.jsonl`. The board driver returns 75 on contention.

```sh
python tools/buddy-codegen/plan_deepseek_v4_cluster.py --spec "$SPEC" \
  --v4-reference "$V4_FLASH_MODEL_PY" --measurements "$OUT" > "$OUT/plans.json"
```

The V4 reference is the release's own `inference/model.py`. The planner
instantiates layer prototypes on the PyTorch meta device and counts tensor
bytes after repacking all experts. It keeps the BF16 checkpoint head in
memory and accumulates logits in fp32. All Engram table weights and scale bytes
are mapped. It emits memory components, layer bounds, source-cache replicas,
per-link bytes and predicted aggregate throughput for each stream count.
The default budget is 32 GiB per board with 1 GiB reserved for the OS,
working buffers, allocator, and file pages. Use `--board-gib` for a measured
capacity. Wire elements default to fp32 with four additional fp32 mix values.
The nominal link rate is scaled by `--goodput`.

The optimizer minimizes the larger of bottleneck service time and loop
latency divided by the number of independent streams. It evaluates contiguous
stages under the memory limit. Every V4.1 KV or index source starts a stage in
the strict plan. If there are too few boards it also solves for cuts that
transfer live source state. Throughputs are steady-state predictions at
context 4096. V4 uses shape-scaled V4.1 timing, and the output head uses
a measured 4096-row projection extrapolated to the vocabulary size.
