# T2b · Qwen hybrid 27B on K3

**The seven RISC-V binaries are staged.** The final 27B GDN and attention layer outputs and caller-owned state matched the Wave 1–3 baseline byte for byte in QEMU. Silicon timing awaits the dual-board lock. The candidate source is `00dc59b016df`. The source baseline is `804a3fd0`. EXP-18 measured `4832e11f`, an earlier deployment.

## Changes and numerical checks

- **GDN A/B.** One eight-thread workshare owns the 16 key-head groups. Each worker handles six value heads. For each head a four-row tile loads each A/B weight once per K slice and uses one input load for both products. Two-row and one-row tails cover every valid length. The eight-lane reduction order matches MLIR. The final ELF matched the independent oracle on the actual layer-0 f32 gates for **all 48 heads at M=1, 2, 3, 4, 64** with maximum absolute difference zero. The predictor uses full attention and has no A/B gates.
- **GDN state.** The same worker shifts its convolution channels, normalizes Q/K, evaluates the gates, updates S and writes its snapshot slices. Four-tap RVV segment loads/stores preserve the caller's channel-major history. The recurrent update and gated norm use 32 independent f32 lanes at VLEN 1024. P64 retains the chunk delta-rule algorithm. The workshare has one team join. Final real-weight layer-0 QEMU traces matched over **832,512 / 1,665,024 / 1,155,072 floats** at M=1/2/64, including the two-row rollback image.
- **Full attention.** Prefill installs valid KV rows first. Each worker handles three query heads and respects each query's causal endpoint. RVV scores 256-wide Q·K, reduces the softmax denominator in order, divides probabilities and accumulates P·V in 32-position tiles. If a later tile raises the maximum, the kernel replays earlier exponents at most once. The libm exponential and division before the value product preserve baseline rounding. A discarded single-pass approximation passed the isolated threshold but reached **0.01155** final-layer error at prior KV 64. On the final build, **all 8,716,288 floats** of layer-3 P64 output and KV matched at **each** prior KV of 64, 256 and 1024. Two-row verification also matched at 1024. Independent core tests exercised every causal row with ordinary and sharply peaked scores. Maximum absolute difference was zero.
- **Native and ELF checks.** Tiny packed x86 backbone and MTP traces matched the baseline over 4,415,512 and 3,800,576 bytes. Partial M=3/4, verify-row snapshots, accepted/rejected proposals and two-slot reuse passed. The real-weight 27B MTP predictor matched all 102,035,456 bytes of hidden output and KV through P64→M2→M1 from prior KV 1024. The staged embedding and head also matched on real rows and logits. See [trace sizes and SHA-256](checks/parity.json). The staged RISC-V host produced the baseline's 13 tokens and 1/11 acceptance on the tiny MTP fixture under QEMU. CTest passed 6 host/MPI and 2 component tests. Lit passed all 496 supported tests, with 4 unsupported. Objdump and allocator assembly cover 17 changed mixer functions and workers across the three DSOs. They show RVV instructions and **zero vector-register spill/reload sites** in this scope. Instruction histograms and per-function PCs are in [the ELF audit](checks/elf-audit.json).

## Estimated time saved

The planning estimate assumes **six effective workers out of eight** for formerly serial work. The model credits no further speedup to the already-parallel recurrent core, weight reuse, or earlier waves. Figures below are reductions in summed layer time against EXP-18. Pipeline overlap, cache behavior and the actual Wave 1–3 latency affect the wall-clock result.

| Target | Case | Saved per layer (ms) | Saved across affected layers (ms) |
| --- | --- | ---: | ---: |
| A/B projections | M=1 / 2 / 64 | 1.033 / 2.076 / 67.565 | 49.56 / 99.64 / 3,243.12 |
| GDN preparation and recurrence | M=1 / 2 / 64 | 0.758 / 1.515 / 45.679 | 36.40 / 72.72 / 2,192.60 |
| Full attention P64 | Prior KV 64 / 256 / 1024 | 80.467 / 243.580 / 880.964 | 1,287.47 / 3,897.29 / 14,095.43 |

A/B uses five sixths of the isolated 1.239 / 2.491 / 81.078 ms. Its two matrices total 1,966,080 bytes, giving a cold one-read floor of 0.08192 ms at 24 GB/s. At M=64 there are also 31,457,280 multiply-adds. The modeled candidate A/B times are 0.207 / 0.415 / 13.513 ms. The sub-0.10 / sub-2 ms targets need silicon measurements.

GDN preparation uses five sixths of EXP-18's pre-core gap after removing A/B (0.910 / 1.818 / 54.815 ms). For attention the KV64→1024 secant is 1.000622 ms per extra cached position per P64 layer. Applying that slope to the 96.5 mean visible positions at KV64 gives an estimated 96.560 ms attention component. For longer contexts add the measured whole-layer increase above KV64, then multiply by five sixths. The modeled reduction in the **KV64→1024 growth alone** is 12,807.96 ms across 16 layers.

## Payload and dual-K3 acceptance

`SHA256SUMS` covers five template DSOs, the kernel runtime, the host and `route-b.json`. The old and new descriptors and dynamic exports match. The manifest changed only template receipts and the derived bundle ID. All **134 original weight files, totaling 31,208,607,872 bytes**, passed SHA-256 verification, with the original `qwen3_5-w4g32/2` receipts. The new bundle ID is `6ab906cb94da83083a9d35e24ba6af19f9bcc76b216b29ef92395632712f18d4`.

After S2/S3 release the lock, drain both ranks and start with Wave 1–3 installed. Run the following in WSL from this staged directory. The packaged `peers.env` sets the rank-0 and rank-1 SSH aliases. The coordinator uses the installed EXP-18 harness and prompts. The operator probe calls public layer entries, so it works on either ELF without projection-symbol offsets.

```bash
set -euo pipefail
source checks/peers.env
sha256sum --strict -c SHA256SUMS
tag="t2b-$(date -u +%Y%m%dT%H%M%SZ)"
checks/bench-dual prepare
checks/bench-dual run "$tag-A1"
checks/bench-dual activate
checks/bench-dual run "$tag-B1"
checks/bench-dual rollback
checks/bench-dual run "$tag-A2"
checks/bench-dual activate
checks/bench-dual run "$tag-B2"
checks/bench-dual rollback
```

Each pass records eight operator samples after two warmups for M=1/2/64 and the three attention contexts. Rank 1 writes `~/t2b-incoming/<label>-operators.log`. Rank 0 writes `~/exp18-s1/28-<off|on>-f32-<trace|grid>-<label>.json` for the 28/36 split. Compare medians, individual samples, token sequences, acceptance and TTFT across both A/B pairs. The final rollback returns the ranks to Wave 1–3. Local rehearsal covered repeated prepare/activate, interrupted activation followed by rollback, and byte-exact restoration of the complete receipt file.
