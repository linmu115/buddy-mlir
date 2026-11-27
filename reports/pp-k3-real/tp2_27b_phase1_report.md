# Route B dual-K3 tensor parallelism — Phase 1

Use the symmetric 2:2 KV split, one KV slot, rank 1 as the endpoint, and f16 reductions for the first silicon run. Both MPI ranks execute all 64 layers. Each reads its own body weights. The 1:3 split provides a smaller rank-0 working set at the cost of a longer critical path on rank 1. All results below were obtained in WSL on x86, using the locally exported release. Silicon throughput, process RSS and swap remain Phase 2 measurements.

## Release and memory budget

The source manifest SHA-256 identity is `baa97ccb8498897b600090e244c9cd2820d085a21c4632295dda878200d19daa`. The checkpoint and manifest agree on H=5120, MLP=17408, vocabulary=248320, 4096 context positions and prefill capacity 64. The 48 GDN layers use 16 key heads and 48 value heads, K=V=128 and convolution length 4. The other 16 layers use 24 gated query heads, four KV heads and head dimension 256. There is one host-resident MTP predictor.

* One GDN layer occupies **432,743,296 B**. The decode pack and IME pack each occupy 215,285,760 B. Norms, A/B projections, convolution and other FP32 fields occupy 2,171,776 B. Each full-attention layer occupies **418,818,048 B**, comprising two 209,387,520 B packs and 43,008 B of FP32 fields. W4G32 uses 16 bytes of nibbles and a 2-byte scale per 32 weights in each layout.
* Symmetric ranks each get 12 query heads, two KV heads, eight GDN key heads, 24 GDN value heads and 8704 MLP channels. The local body is **13,737,722,880 B (12.7943 GiB)** per rank. Replicated norms account for the difference from exactly half the source. Each rank maps another 128 B for RoPE. Rank 1 additionally owns embedding 2,542,796,800 B, final norm/head 715,182,080 B and MTP 477,861,888 B. Weight mappings with MTP enabled total **12.7943 / 16.2735 GiB** against **15.6 / 31.3 GiB** physical memory.
* Decode touches **6,895,764,480 B (6.4222 GiB)** of body weights per symmetric rank. With one slot, local full-attention KV is 268,435,456 B and GDN conv/SSM state is 79,429,632 B. MTP verification adds a 79,429,632 B GDN snapshot on each rank. The host predictor has another 33,554,432 B of KV. Retaining both body layouts plus verification state would consume 13.1922 GiB on rank 0 before scratch. With phase-specific paging, rank 0 has **6.8201 GiB** of decode body plus persistent state with MTP. Rank 1 has **10.3307 GiB** on the same basis even when charging all endpoint weights, including the entire embedding table, plus the predictor KV. Activations, allocator scratch, library code, thread stacks, MPI and file-page rounding add to this. Every extra slot adds another 0.39795 GiB on rank 0 with MTP. The Phase 2 acceptance limit is 13,631,488 KiB peak RSS and zero swap.
* With KV split 1:3, body mappings are **6.3984 / 19.1901 GiB**, and body decode working sets are **3.2123 / 9.6320 GiB**. Endpoint weights add 3.4793 GiB on rank 1. Its 75% body shard limits ideal body bandwidth to 32 GB/s at 24 GB/s per board. The symmetric body ceiling is 48 GB/s. Integer KV-group partitioning also gives four equal shards for a future TP=4 build. The current transport and index format implement two ranks.
* Linux paging check on real symmetric shards, measured in WSL using private read-only mappings and the compiled projection offsets. GDN layer 0 mapped 215,285,760 B of Q4. Its RSS in KiB went **0 → 105,152 → 0 → 105,152** for cold map, IME touch, `MADV_DONTNEED`, decode touch. Full-attention layer 3 mapped 209,387,520 B and followed **0 → 102,336 → 0 → 102,336**. The runtime discards body-Q4 mappings on prefill/decode transitions and streams optional integrity checks separately. These are per-mapping measurements. Linux can retain file data in its page cache after removing process mappings.

## Collectives and throughput estimate

The first sum follows the local mixer output projection and precedes residual addition and normalization. The second follows the MLP down projection and precedes its residual. Column partitions preserve the three GDN Q/K/V sections and per-query gate pairing. Recurrence, convolution, RoPE, attention, KV and verification snapshots are local. The shared row-capacity interface selects 1, 2 or 64 compiled rows and reduces only valid rows. M=4 would additionally require four-row kernels, state snapshots and generation policy.

For one decode token, **128 reductions × 5120 × 2 B = 1,310,720 B (1.25 MiB) sent and the same received per rank**. The two directions together carry 2.5 MiB. Each reduction is a single simultaneous peer exchange, followed by an FP32 sum of both rounded inputs. FP32 reference mode uses MPI_SUM on 20,480 B of FP32 partials per reduction. Verify-2 doubles the payload per call and a full prefill chunk multiplies it by 64. Both still require 128 reductions per backbone invocation. The host also sends a 20 B forward command and 20,480 B of FP32 embedding per decode token. Token selection and the MTP predictor execute on the endpoint. TCP, MPI framing and request-level control are extra.

An x86 MPICH 4.2.0 loopback test forced the OFI TCP provider and disabled local-process transport. Eight timed batches of 128 reductions of 5120 elements took a median **4.015 ms in f16** and **2.753 ms in f32**, including buffer initialization and the local sum. Loopback TX increased by 71,856,300 B across the warmup and timed runs. The W4G32 logits and greedy counts over TCP matched the native MPI results below. The K3 build uses the RVV Zvfh conversion path.

At the supplied STREAM read rate, the symmetric body plus serial head has a weight-read floor of **317 ms/token**, or **3.15 tok/s** before synchronization, KV, compute and scheduling. At 1 Gbit/s full duplex, f16 reductions cost at least 10.49 ms in serialization. A suggested sensitivity range of 50–200 µs additional latency per exchange adds 6.4–25.6 ms/token. Applying the measured EXP-16 PP decode baseline of 1.296 tok/s to the ratio of critical-path bytes gives **about 2.27–2.37 tok/s** for symmetric TP without MTP. This estimate assumes the smaller kernels sustain the earlier effective rate. MTP throughput also depends on verification cost and proposal acceptance.

## Verification

* Two native MPI processes compared every logit at matching prefixes for seven prefill rows, decode, both verification rows, a replacement token after rewind and four predictor projections. The independent reference loaded the original body and host DSOs. Each mode matched **15/15** greedy choices.
* FP32 fixture, equal head counts: maximum absolute differences **0.00000476837158203125** (f32 wire) and **0.00084018707275390625** (f16 wire). Tile-aligned W4G32 fixture, symmetric: **0.00000095367431640625** and **0.078772753477096558**. W4G32 fixture, 1:3 KV split with host moved to rank 0: **0.00000095367431640625** and **0.007039189338684082**. Row-parallel partial sums change accumulation order even with FP32 communication.
* Public CLI free generation matched all 16 reference IDs on both native fixtures in each of four configurations (f32/f16 × ordinary/MTP). Resident MPI smoke tests completed both warm and measured requests using the host-rank input routing below. All eight original Route B CTests passed (six PP host cases and two pack/count cases). Four additional CTest entries cover both wire precisions, public CLI generation and sharding. The full check-buddy run passed 238 example tests and 495 regression tests, with six marked unsupported by the build configuration. The packed-layout unit test compared actual bytes with independent float slicing and repacking for fused GLU, segmented QKV and row-parallel input groups, including IME and 1:3 partitions.
* Both K3 host executables linked with transitive dependencies resolved. The guest loader under QEMU bound the runtime immediately and reached CLI argument handling. All eight body DSO instances for the 2:2 and 1:3 plans loaded with RTLD_NOW. The descriptors reported ABI 2, 184 bytes, H=5120 and the expected local KV head count. Symbols included `layer_tp_rows`, `layer_prefill`, `layer_decode` and `layer_verify_2`. The original global endpoint DSOs are retained. The K3 IME execution gate is the Phase 2 board run.

## Sharding and deployment

The sharder copies complete quantization groups. Decode GLU interleaves tiles, while IME groups all gate tiles before the up tiles. No checkpoint tensor is dequantized. The offline compiler prepares rank-local DSOs and byte ranges. The deployment step needs Python 3.11+ and the existing release pack. The audit writes and reads back a body shard at a time, hashes all output payloads, then discards the scratch files. Allow at least 30 GiB of disk to retain a complete symmetric bundle. The WSL worktree retained the plan, rank manifests, index and binaries and materialized the real body shards sequentially. Two full scratch-write/readback runs took 4 min 42 s and 9 min 56 s under different host loads, peaking at 277,468 and 277,324 KiB process RSS. All 128 body files per rank were sliced and checked. The final two rank manifest hashes agree with the independent hash-only pass. Local logs are under `build-tp-host/`, including `k3-final/readback.log`, `residency.log`, `q4-parity-fixed.log` and `final-elf.log`.

From the worktree, set `SOURCE` to the release directory, `CONFIG` to the checkpoint config, `LLVM` to the matching LLVM bin directory, `BUDDY` to buddy-opt, `SDK` to the RV64 GCC toolchain with its sysroot, and `PY` to the Buddy/MLIR exporter Python. Use the matching checkout's `k3_w4` Python package with two-row kernel support.

```bash
out=build-tp-host/k3-final
mkdir -p "$out"
# A fresh destination is required for each publication.
"$PY" tools/buddy-codegen/prepare_route_b_tp.py \
  --config "$CONFIG" --source "$SOURCE" --output "$out/plan" \
  --kv-heads 2 2 --host-rank 1 --llvm-bin "$LLVM" --buddy-opt "$BUDDY" \
  --cxx-flag=--target=riscv64-unknown-linux-gnu \
  --cxx-flag=--sysroot="$SDK/sysroot" --cxx-flag=--gcc-toolchain="$SDK" \
  --cxx-flag=-march=rv64gcv_zfh_zvfh
python3 tools/buddy-codegen/shard_route_b_tp.py --source "$SOURCE" \
  --prepared "$out/plan" --audit-only --output "$out/audit"
```

For Phase 2, stage the artifacts after acquiring the board lock. Set `SMALL_HOST` and `LARGE_HOST` to the MPI/SSH names in rank order. The commands below install into a dedicated directory on each board. They assume the unchanged source release at `$HOME/route-b-qwen38-27b-release` on both boards. The two output directories together constitute the bundle. Each board gets only its own plan and writes only its own weights and shared index. Existing `bundle` outputs can be reused when their `tp.json` matches the audited index.

```bash
out=build-tp-host/k3-final
hosts=("$SMALL_HOST" "$LARGE_HOST")
for rank in 0 1; do
  host=${hosts[$rank]}
  ssh "$host" "mkdir -p route-b-27b-tp2/plan/rank$rank"
  rsync -a "$out/plan/rank$rank/" "$host:route-b-27b-tp2/plan/rank$rank/"
  rsync -a "$out/audit/audit.json" tools/buddy-codegen/shard_route_b_tp.py \
    build-tp-host/k3/buddy-pp-route-b-serve "$host:route-b-27b-tp2/"
  ssh "$host" "cd route-b-27b-tp2 && python3 shard_route_b_tp.py \
    --source ../route-b-qwen38-27b-release --prepared plan --rank $rank \
    --index audit.json --output bundle"
done
```

## Phase 2 benchmark

Run on the MPI launch host with the same board aliases, MPICH installation on PATH on both boards, and one process per board. The launch loop uses the same executable, prompt, slot count and warmup for the original PP path (EXP-16 split 20/44) and TP. Each setting runs PP, TP, TP, PP. The default deployment maps logical rank 1 to the larger board. Hydra normally directs launcher stdin to rank 0, so the shell below provides the request stream directly to the endpoint process. This input path was exercised on native MPI. Use a fresh log directory for each repeat.

```bash
mkdir -p tp-results
export LD_LIBRARY_PATH="$(cd "$(dirname "$(command -v mpiexec)")/../lib" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FI_PROVIDER=tcp FI_TCP_NODELAY=1 FI_TCP_IFACE=end0 FI_SOCKETS_IFACE=end0
export MPIR_CVAR_NETWORK_IFACE=end0 MPIR_CVAR_CH4_OFI_IFNAME=end0 MPIR_CVAR_ASYNC_PROGRESS=0
export OMP_NUM_THREADS=8 BUDDY_THREAD_POOL_THREADS=8 BUDDY_ARENA_PREFAULT_MB=0
export BUDDY_PP_OBSERVE=1 BUDDY_PP_STORAGE=1
for TP_WIRE in f32 f16; do
  for TP_MTP in off on; do
    export TP_WIRE TP_MTP
    iteration=0
    for MODE in pp tp tp pp; do
      export MODE
      iteration=$((iteration+1))
      mpiexec -genvall -ppn 1 -n 2 -hosts "$SMALL_HOST,$LARGE_HOST" -iface end0 sh -ec '
      test "$(cat /sys/class/net/end0/speed)" = 1000
      echo 0 > /proc/set_ai_thread
      cd "$HOME/route-b-27b-tp2"
      if [ "$MODE" = tp ]; then
        input_rank=1
        set -- ./buddy-pp-route-b-serve --tp-bundle bundle
      else
        input_rank=0
        set -- ./buddy-pp-route-b-serve --bundle ../route-b-qwen38-27b-release/route-b.json --ranges 0:20,20:64
      fi
      set -- "$@" --wire "$TP_WIRE" --slots 1 --serve
      if [ "$TP_MTP" = on ]; then set -- "$@" --mtp; fi
      if [ "$PMI_RANK" = "$input_rank" ]; then
        exec "$@" <<REQUESTS
{"cmd":"submit","req_id":"warm","prompt":"Explain how a KV cache works.","max_new_tokens":3}
{"cmd":"submit","req_id":"sample","prompt":"Explain how a KV cache works.","max_new_tokens":32}
{"cmd":"shutdown"}
REQUESTS
      fi
      exec "$@"
    ' >"tp-results/$TP_WIRE-$TP_MTP-$iteration-$MODE.jsonl" 2>"tp-results/$TP_WIRE-$TP_MTP-$iteration-$MODE.stderr"
    done
  done
done
python3 - <<'PY'
import json
from pathlib import Path
def sample(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tokens = [e for e in rows if e.get("req_id") == "sample" and e["event"] == "token"]
    done = next(e for e in rows if e.get("req_id") == "sample" and e["event"] == "done")
    return [e["token_id"] for e in tokens], tokens[0]["step_us"], done
logs = Path("tp-results")
reference, _, _ = sample(logs / "f32-off-1-pp.jsonl")
for path in sorted(logs.glob("*.jsonl")):
    ids, prefill_us, done = sample(path)
    print(path.name, "ids_match", ids == reference, "prefill_step_us", prefill_us,
          "decode_tok_s", done["decode_tok_s"], "steps", done["decode_steps"],
          "acceptance", done.get("mtp_accept_rate"))
PY
grep 'tp_memory' tp-results/*.stderr
```

Record per-rank VmHWM, VmRSS and VmSwap. Sample both processes during execution if board swap is enabled. Compare generated IDs and first-token latency at matching prompts, then compare warm decode throughput. For the small board, require peak VmHWM ≤13,631,488 KiB and VmSwap=0. Measure MTP acceptance and tokens per verification step separately.
