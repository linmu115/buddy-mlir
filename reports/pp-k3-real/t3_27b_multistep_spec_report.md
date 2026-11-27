# T3+ multi-step speculative decoding for Qwen3.8-27B

**The x86 lossless gate passed.** Two pipeline ranks produced the same greedy token IDs as scalar decode in 32 comparisons across three hybrid fixtures and both wire precisions. Each comparison covered 130 generated tokens. A separate bytewise check covered 481 successive verification frames across four traces. The staged K3 binaries passed the projection spill audit. Dual-board throughput is pending Stage S3.

The implementation commits are `b3ef3889` (kernels and pack-compatible template build) and `f572305f` (runtime, scheduler and tests) on `feat/route-b-27b-spec`, based on `4832e11f`.

## Losslessness and scheduling

- The mixed fixture used three GDN layers, one full-attention layer, one MTP layer, W4 projections, an f16 embedding table, a 248,320-way head, and a two-rank cut at layer 2. Each wire precision had its own scalar reference. A separate replay against the frozen B4 executable and original DSOs also matched all 130 DSpark tokens on each wire. The mixed matrix covered MTP depths 1, 2 and 3, prompt lookup, default DSpark and a stricter cumulative threshold. The accept and random fixtures each covered the five default configurations per wire.
- On the accept fixture, chained MTP at depth 3 accepted all 96 proposals in 32 full-width batches on each wire. On the random fixture, DSpark selected depth 1 on 128 frames per wire and finished with a scalar tail. Pure lookup on that fixture exercised scalar fallback. Each mode reproduced the entire target sequence.
- Four seeded prompts forced genuine three-token lookup proposals with accepted lengths 0, 1, 2 and 3. Each trace continued for at least 100 verification steps with varying widths and forced partial rejections. After every commit, a second slot replayed the same prefix one row at a time. All token IDs, every GDN convolution and recurrent-state byte, and the complete full-attention K/V arrays including their unused tails matched.
- The Route B host suite passed 8/8, including all six pre-existing tests. Resident tests also checked simultaneous requests and slot reuse with single-step MTP and DSpark. The component counts/roles and seeded pack-layout checks passed.
- The independent PyTorch check used 64- and 73-token prompts. When chaining the predicted token embedding and post-normalization MTP hidden, the largest absolute differences at depths 2 and 3 were below 0.000001 for hidden states, 0.000002 for predictor K/V, and 0.000016 for logits. These values compare FP32 implementations. The greedy and backbone cache checks above used exact equality. The one-layer reuse and normalization order follow the upstream implementation. [vLLM Qwen3.5 MTP source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)

### DSpark telemetry on the mixed fixture

At `--dspark-tau 0.55 --dspark-min-p 0.35`, both f32 and f16 wire runs chose K=1 on 3 steps, K=2 on 1, and K=3 on 32. The 36 verification steps proposed 101 tokens, accepted 93, and emitted 129 decode tokens plus the initial prefill token.

- Mean proposed depth was **2.806**. Mean accepted drafts per verification step was **2.583**. Including corrections and bonus tokens, the mean output was **3.583 tokens per verification step**.
- Prefix acceptance by depth was **32/36 (88.89%)**, **31/33 (93.94%)**, and **30/32 (93.75%)**. Each denominator counts every proposal at that depth. The accepted-prefix histogram for lengths 0 through 3 was **[4, 1, 1, 30]**.
- Lookup supplied 31 draft batches; MTP supplied the other five batches. With the cumulative threshold raised to 0.99999 and the local threshold set to zero, the MTP-only K histogram changed from **[3, 0, 2]** to **[4, 1, 0]**. This exercised a confidence-selected two-token MTP proposal with room for three. Greedy outputs still matched.

These are synthetic-fixture results. The 27B acceptance rate and tokens per second for the new kernels require the board run.

## Runtime and kernel behavior

The verifier processes the current token plus the selected drafts. M=1 uses the decode path. M=2, 3 and 4 have separate projection shapes, including the shared output head. The packed layout is unchanged. Each strip of a packed tile serves every active row before advancing. FP32 decay and beta gates retain their original storage and share their weight loads across short verify rows. Embedding lookup gathers each candidate ID directly from the existing table.

GDN records the convolution window and the recurrent matrix after each non-final prefix. Each rank restores the selected prefix on rejection, clears discarded full-attention K/V rows, and advances the slot by the committed length. Verification scratch is shared across slots because complete verification and commit are synchronous. A PP hop carries exactly M activation rows. The target reply carries two integers, the accepted prefix length and one correction or bonus token.

Pure lookup loads no predictor weights. DSpark uses exact suffix matches of length 5 through 3, buffers the committed predictor inputs on hits, and replays them on the next miss. MTP computes a max-shifted softmax denominator without materializing probabilities. The first uncertain proposal is verified at K=1. A failed deeper candidate is discarded.

## K3 artifacts

The staged directory is `stage-k3-27b-spec/` in this worktree. All executable and shared-library ELF headers identify 64-bit little-endian RISC-V. The host resolves the existing MPI and OpenSSL SONAMEs and contains no build-machine RUNPATH. The staging manifest has **134 identical weight receipts**, totaling **31,208,607,872 bytes**. Only the template entries and bundle ID changed in the EXP-16 index. The rebuilt runtime also matched its original SHA-256. An independent clean template build reproduced all five DSOs, runtime, and index byte for byte.

The objdump and LLVM audit covered **34 M=3/4 packed tile symbols** across GDN (10), full attention (10), MTP (12) and the head (2). It checked the linked disassembly for direct vector stack accesses, reproduced the verification objects byte for byte, and inspected compiler annotations for indirect spills and reloads throughout those objects. The final count was **0**. Four-row fused GLU uses 32-column strips because the 64-column trial spilled a vector. The other new projection shapes use 64 columns.

The full SHA-256 manifest follows. The same list is included as `SHA256SUMS.t3`.

```text
8ed3041aff234dd39ebf18fd519ce0be52c09be4f02961dd8670be72fc06875f  bench_multistep.py
c1502bb63923fe9523e96eef38eec5e7bf3eabcd7608f26945fa7194d52fe3bd  buddy-pp-route-b
88527cca8dfdf4920c5604b3521060c4fe44e36373ae1f8e009bff88f01f9f4c  embed.so
b70ef593734046afa7997c753e6ec32d294165f273e6914913edb6cd0bd3cc77  full_attn_layer.so
e080a95389bb3a81c5735b488a024451d86d9dd1545bb26760893534df115f09  gdn_layer.so
ca05f312af3e8042295b5253263ed247f0350f841edd5911a61d1707031e198f  libbj47_kernel_runtime.so
544bfdeba78f123352256f54634daeb661fbae629b68608639b61617dcd5b2cc  lm_head.so
12ad8fb71d0701f35e57f05372df5eebe37dec5a811addcf8ddf11204cf24706  mtp_layer.so
80b25fb8be46555dc522cf4ce70a6677a99164acd4130dec095dc5cc7e68dea9  route-b.json
```

## Recheck the x86 gate and the RISC-V artifacts

From the worktree root, the prepared host build has the mixed fixture registered in CTest. The spill checker accepts the same Buddy and LLVM tools used to compile the templates.

```sh
ctest --test-dir build-t3/host --output-on-failure
build-t3/host/route_b_spec_state build-t3/mixed/route-b.json
(cd stage-k3-27b-spec && sha256sum -c SHA256SUMS.t3)

source build-t3/env
"$P" tests/Pipeline/audit_route_b_verify_rvv.py \
  --work "$PWD/build-t3/k3-work" --bundle "$PWD/stage-k3-27b-spec" \
  --llvm-bin "$llvm" --buddy-opt "$HOME/bj47/pp-native-build/bin/buddy-opt"
```

The host CTest output, numerical-oracle output, register audit and per-wire accept/random run records are retained under `build-t3/`. The host gate used a matched MPICH library and loopback launcher. The distribution launcher initially started two independent singleton ranks, which failed the stage-range check. The matched launcher ran the complete suite.

## Stage S3 dual-K3 benchmark commands

Run after release of the board lock. Install the staged files, including the updated index and checksum manifest, into the offline EXP-16 release on both ranks. Keep the existing weight files, tokenizer, rank launcher, server module, host list, MPI path and VLEN probe. From the release directory on **each** rank, verify the package.

```sh
cd "$HOME/route-b-qwen38-27b-release"
sha256sum -c SHA256SUMS.t3
```

Then run these two commands **only on the launch rank**. Each starts both ranks using the deployed MPI host list.

```sh
cd "$HOME/route-b-qwen38-27b-release"
set -e
python3 bench_multistep.py --release "$PWD" --wire f32 --repetitions 2 >s3-f32.jsonl
python3 bench_multistep.py --release "$PWD" --wire f16 --repetitions 2 >s3-f16.jsonl
```

The driver uses the existing 1 Gb/s end0 and VLEN-1024 checks, eight compute threads, the `0:20,20:64` split and one resident slot. Each mode warms up before measuring prose and Python-code prompts with a 128-new-token limit. The second pass reverses the order of the six configurations. Every measured token sequence is checked against the same-wire greedy reference before its JSONL record is written. Compare `decode_tok_s` by prompt, wire and repetition. DSpark records also include the chosen depths and acceptance counts.
