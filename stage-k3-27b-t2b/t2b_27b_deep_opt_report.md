# T2b-clean · Qwen3.8-27B Stage S2b handoff

Kernel source `17e56c2c` is based on T2b `70e04c4f`. Its comparison release is EXP-16/18 `4832e11f`, which EXP-19 restored on both ranks. The staged bundle ID is `01945ad448dda685d6774c3cfb7789843a66c3053f9cd8eba04b4588a667f73a`. All timings below come from EXP-19. Candidate latency and the two long-prompt greedy sequences await the Stage S2b board run.

## Changes driven by silicon measurements

- **Dense P64** now selects the original half-precision IME step in GDN, FullAttn and MTP. Each linked DSO contains eight `smt.vmadot.hp` instructions and zero integer `smt.vmadot` instructions. The same 144-byte blocks feed the restored kernel. EXP-19 measured the GDN GLU at 13.173 ms with HP and 41.610 ms with T2 integer dots. That experiment also observed higher three-chunk TTFT and token divergence on the 164- and 384-token prompts.
- **Two-row attention** uses the same grouped kernel as decode in both the backbone verifier and the predictor. T2 dispatched it only for M=1. T2b already grouped both rows, but a worker processed them serially at one group size. The new workshare schedules independent (row, KV head, query group) items. At prior KV 383, row zero uses three pairs at visible length 384 and row one uses four plus two at 385. Each reads only its causal prefix and uses its own rotary row. The linked two- and four-query kernels contain RVV instructions in both DSOs. The allocator audit found zero vector spill/reload sites in the 21 inspected mixer functions and workers. T2b's P64 attention, GDN gate projections, convolution, norms and recurrence remain active. EXP-19 measured M=1 at KV 1024 at 25.546 → 11.512 ms, and M=2 at 41.680 → 43.411 ms.
- **Two-row GLU** restores the baseline full-width accumulator for the 5120 → 17408 gate and up projections. Each traverses its packed tile once for both rows. The one-row helper, streamed M=1/2 head, final-workshare barrier removal and fused residual stores are preserved. EXP-19's eight M=2 samples ranged from 4.443–4.519 ms on baseline and 4.404–4.607 ms on T2. Their medians were 4.475 and 4.525 ms. The overlap makes the silicon remeasurement important. The M=1 GLU medians were 4.277 and 4.138 ms.

## Offline acceptance

The production reference DSOs match the EXP-19 baseline SHA-256 inventory. Native reference templates were compiled from `git archive 4832e11f` using the same tiny fixture and toolchain as the candidate. Candidate and baseline used the same packed weights. All 20 trace pairs match byte for byte across **467,630,624 compared bytes**. [Per-case byte counts and SHA-256](checks/parity.json) · [Linked ELF and descriptor checks](checks/elf-audit.json)

- **x86 tiny hybrid.** A 73-token backbone trace exercises full and partial P64 followed by M=1, logging hidden rows, GDN state, KV, logits and token IDs. The MTP trace runs P64 → M=2 → M=1. A separate verifier rejects the second row, restores the row-zero GDN snapshot, overwrites the rejected KV position and compares the replacement output and complete state against single-row decode. Mixed-speculation service tests cover accepted and rejected drafts, two slots and slot reuse.
- **Production QEMU, RVV VLEN 1024.** GDN layer 0 at M=1/2/64, FullAttn layer 3 at every combination of M=1/2/64 and prior KV=64/256/1024, plus M=2 at prior KV 383. The M=2/KV=1024 trace covers 33,595,392 bytes including both output rows and the complete KV arrays. MTP starts at prior KV 1024 and advances through P64 → M=2 → M=1, comparing 102,035,456 bytes of output and KV. Head decode and two-row linear produce 2,979,840 bytes of logits. Embedding and the isolated M=2 GLU are checked separately. All 34,816 GLU output floats match.
- **Scalar oracles.** The two-row grouped kernels in both production attention DSOs match all six query heads byte for byte at prior KV 64/256/383/1024. Tests include normalization, partial RoPE, softmax, the V reduction and output gate. P64 attention and GDN A/B tests also pass. [Probe source](checks/operators.cpp)
- **Payload.** Five template DSOs, the runtime and the host were cross-compiled. All 134 original `qwen3_5-w4g32/2` weight records match, including on-disk SHA-256 for all 31,208,607,872 bytes. The rebuilt manifest changes only template receipts, runtime digest and bundle ID. Local rehearsal exercised repeated prepare/activate, rollback, and recovery from a mixed installation. The restored full receipt SHA-256 is `80a11acef3bd1fad9d1743a119e2953eb292cb2eccdb2c568e4bdebbfa9548cf`.
- **Regression suites.** CTest passed all six host/MPI tests and both component tests. Lit discovered 500 tests, passed all 496 supported cases and marked four unsupported. The hybrid generator selection regression also passed its focused run. The final GLU board probe completed its two warmups and eight timed calls for each row count under QEMU.

The worktree retains the build and replay inputs under `build-t2b/`. From its root, the local rebuild and comparison commands are:

```bash
source build-t2b/env
"$P" "$T/compile.py" clean k3
"$P" "$T/compile.py" clean tiny
ninja -j6 -C "$T/cross" buddy-pp-route-b
"$T/clean/replay" base "$B"
"$T/clean/replay" clean "$PWD/stage-k3-27b-t2b"
QEMU_RUN="$T/qrun" PROBE="$T/clean/operators" GDN_F32="$B/gdn-0.f32" \
  stage-k3-27b-t2b/checks/run-operators
"$P" "$T/clean/parity.py"
```

The replay command regenerates the production GDN, FullAttn and MTP traces. The comparison also reads the retained head, embedding, isolated GLU and native fixtures. The Git-tracked receipts identify the staged ELFs and exact comparison data. The seven executables and DSOs are local build artifacts under the stage directory.

## Stage S2b dual-K3 acceptance

Run from the staged directory in WSL **after S3 #160 releases the board lock**, with both ranks drained and the EXP-16/18 release installed. The coordinator uses the EXP-19 quality and clean-grid harnesses, the 28/36 split, f32 wire and eight workers. The MTP-off and MTP-on quality runs use the same four prompts. The grid uses the original shorter budgets to measure warm TTFT and decode throughput.

```bash
set -euo pipefail
source checks/peers.env
sha256sum --strict -c SHA256SUMS
sha256sum --strict -c checks/probes.sha256
tag="s2b-clean-$(date -u +%Y%m%dT%H%M%SZ)"
out="../build-t2b/acceptance/$tag"
mkdir -p "$out"

checks/bench-dual prepare
trap 'checks/bench-dual rollback' EXIT
checks/bench-dual run "$tag-A1"
checks/bench-dual activate
checks/bench-dual run "$tag-B1"
checks/bench-dual rollback
checks/bench-dual run "$tag-A2"
checks/bench-dual activate
checks/bench-dual run "$tag-B2"
checks/bench-dual rollback
trap - EXIT
for node in "$K3_R0" "$K3_R1"; do
  ssh "$node" 'cd ~/route-b-qwen38-27b-release && sha256sum --strict -c ~/t2b-incoming/baseline.sha256 && sha256sum SHA256SUMS' \
    | tee "$out/$node-rollback.log"
done

for phase in A1 B1 A2 B2; do
  label="$tag-$phase"
  for mode in off on; do
    for workload in quality grid; do
      scp "$K3_R0:exp19-s2/28-$mode-f32-$workload-$label.json" "$out/"
    done
  done
  scp "$K3_R1:t2b-incoming/$label-operators.log" "$out/"
done
python3 checks/compare-runs.py "$out" "$tag"
```

The checker requires identical input lengths, quality-run input IDs, every output ID through EOS or the generation cap, stop reason, and MTP accepted/proposed counts across A1/B1/A2/B2. It prints TTFT and decode throughput in that order. Local replay against the EXP-19 records rejects T2 at the 164-token prompt.

Compare the eight raw samples per operator after two warmups. Use paired medians and ranges to assess P64 GLU and three-chunk TTFT recovery, M=2 FullAttn at KV 1024, boundary behavior at prior KV 383, and MTP verify. Inspect M=1 FullAttn, MTP, head and GLU alongside them to check the existing decode gains. The operator logs also include the M=2 head and isolated M=2 GLU. Record both rank receipt checks after the final rollback. Keep quality failures and timing deltas separate in the acceptance record.
