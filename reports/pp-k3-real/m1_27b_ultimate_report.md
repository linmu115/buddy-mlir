# Qwen3.8-27B Route B M1

M1 merges T2b-clean (`4117425d`), multi-step MTP and DSpark (`37172fa1`), and TP=2 (`8b764cb2`). The same `buddy-pp-route-b` and `server.py` launch PP and TP. TP rank 1 owns embedding, the LM head, MTP and the confidence scheduler. Rank 0 receives the verifier IDs and committed prefix. Each TP body layer reduces `M × 5120` float partials after the mixer and after the down projection. The f16 path exchanges 20, 30 or 40 KiB per rank per reduction at M=2, 3 or 4. Each rank adds the residual after the reduction.

GDN A/B uses the fused four-, two- and one-row RVV projection workshare. Convolution and recurrent S take a snapshot after each tentative prefix, and attention clears the rejected KV suffix. The grouped RVV attention path handles all four verifier widths and the 384/385 visible-length boundary. MTP uses the same grouped attention and preserves its causal replay. PP retains residual fusion. K3 GLU and non-head M=2 projections use narrower RVV tiles to fit the register file. The P64 GLU narrows its SiLU epilogue while retaining HP IME dots and the original accumulation order. The M=2 streamed head keeps the four-bank schedule and a fixed accumulator buffer for each row.

## Offline receipts

- **Native PP.** Two MPI ranks matched all 130 greedy IDs in each of 12 comparisons. Both f32 and f16 covered K=1, 2 and 3, prompt lookup, DSpark and the stricter confidence gate. Each rank on each wire also checked 481 committed frames against scalar replay, including accepted prefixes 0, 1, 2 and 3. GDN convolution, recurrent S and full KV buffers matched byte for byte.
- **Native TP 2:2.** The public CLI matched all 16 IDs in each of 10 comparisons across f32 and f16, scalar, K=1, K=2, K=3 and DSpark. On a longer request, DSpark matched all 63 IDs on each wire and selected K=1, 2 and 3 on 6, 1 and 13 frames. Its accepted prefix histogram was `[5, 1, 1, 13]` on each wire. A separate probe matched 61/61 greedy choices per wire across prefill, decode, verification, rollback and predictor projections. The maximum absolute logit error was `9.5367431640625e-7` for f32 and `0.078772753477096558` for f16. Across M=2, 3 and 4, all 36 rank/wire/prefix checks matched every local GDN and KV cache byte.
- **HTTP.** PP and TP both returned HTTP 200, reported both physical ranks, and emitted the same 13-token completion. The TP test used Hydra stdin delivery to rank 0 and the relay to rank 1.
- **K3 ELFs under QEMU.** Immediate binding reached CLI validation and loaded all six PP DSOs and eight TP body instances. Kernel descriptors reported ABI 2, 184 bytes and H=5120. The TP members reported local KV-head counts 2/2 and 1/3. The symbol audit checked all declared entry points. The linked `objdump` audit and compiler annotations found **zero vector spill or reload sites**. All 11 body/MTP DSOs carried eight `smt.vmadot.hp` instructions and zero integer `smt.vmadot` instructions. A second audit covered 34 PP M=3/4 projection tiles with zero spill sites. The staged files matched the audited SHA-256s.
- **K3 numerical probes.** Twelve production-width GDN, FullAttn and MTP cases at M=1 through 4 and prior position 383 matched T2b outputs and complete cache bytes. The 248K head M=1/2 produced the identical streamed digest `88c10e21eaeeeef311036f3a0699ac61af1536d18e6d957eabced545c6d29a42`. The P64 instruction audit completed. The optional whole-layer P64 emulator run stopped after seven minutes while testing the GDN case. Dual-board P64 latency remains part of silicon acceptance.
- **Publication.** Both checksum inventories passed strict validation. Their SHA-256s are `f781d93e8b91ad5ec065da6f08731d24b696178bbf58c6abe37ccab64c2b4767` for PP and `68c044d773580cc670f1bf633957b49a21948b88df3046c85927db7525acae28` for TP. The PP updater passed repeated prepare, activate and rollback. The final rollback restored the original files and checksum inventory.

- **Regression suites.** Route B CTest passed **15/15** (12 MPI host and 3 pack tests). The final host suite passed in 155.46 s. An earlier concurrent run reproduced one transient resident MPI launch failure. The isolated resident check and final suite both passed. Lit passed **734** supported cases (238 examples and 496 regressions), with six configuration-marked unsupported cases and zero failures.

PP bundle ID `0df5a80c4de921aee37e00e061c4a7e74df448e2b254fb9fc929f16c61cfe303`. The EXP-16 source bundle ID is `baa97ccb8498897b600090e244c9cd2820d085a21c4632295dda878200d19daa`. All **134** `qwen3_5-w4g32/2` weight receipts and their on-disk SHA-256s matched across **31,208,607,872 bytes**. The sharder copied complete groups from those bytes, wrote at most one body shard at a time, and hashed each shard again from disk before discarding it. Each layout checked **128 body shards per rank**. The 2:2 and 1:3 runs took 401.03 s and 625.04 s. The final DSO relink changed only binary receipts and source identity. The byte-range recipes, layouts and source weight receipts matched exactly, so the final index reuses the verified payload digests.

- Symmetric 2:2 rank IDs `4a619c43c6f9c434bf26ac4b85168537ac5e929a2c70efce305cc7f0c2c47776` and `1bf5b71c7f4f4dbeb3119df0732c24e3b408f68605510da815e38734916eceac`.
- Asymmetric 1:3 rank IDs `3899d89717b0f04b30784da6216a6f26b23f6493c1f7cd71453e3ff5267cffed` and `588d5ce2ef5924a152097fa323b7d2b70cda718d3eb6ed523429da27cb837634`.
- `stage-k3-pp-ultimate/` contains the seven binaries, PP manifest, checksum inventory and transactional updater. `stage-k3-tp2-ultimate/` contains the shared runtime and endpoint DSOs, both ranks for both splits, the audited index for each split, the same executable, the sharder and the gateway.
- Local logs and reproducibility inputs are under `build-m1/`. Load `build-m1/env`, add `$MPI/lib` to `LD_LIBRARY_PATH`, and set `BUDDY_THREAD_POOL_THREADS=2` for the native tests. Use `ctest --test-dir build-m1/matched-host --output-on-failure`, `ctest --test-dir build-m1/pack-tests --output-on-failure` and `ninja -j6 -C build-m1/lit check-buddy`. The linked audit is `build-m1/elf-audit-final.json`.

## Dual-K3 acceptance after Stage S2b #163 releases the lock

Set `K3_R0` and `K3_R1` to the authorized K3 end0 SSH/MPI addresses in rank order. Both boards need the EXP-16 release at `$HOME/route-b-qwen38-27b-release`, the same absolute MPICH and VLEN-probe paths, and enough disk for the selected rank's shards. Drain the ranks before installing. The PP copy below uses hard links for existing weights. The updater installs each file with an atomic rename and retains a rollback copy.

```bash
set -euo pipefail
: "${K3_R0:?}" "${K3_R1:?}"
hosts=("$K3_R0" "$K3_R1")
(cd stage-k3-pp-ultimate && sha256sum --strict -c SHA256SUMS)
(cd stage-k3-tp2-ultimate && sha256sum --strict -c SHA256SUMS)
for rank in 0 1; do
  node=${hosts[rank]}
  ssh "$node" 'mkdir -p "$HOME/route-b-m1/pp-kit" "$HOME/route-b-m1/tp-kit"'
  rsync -a stage-k3-pp-ultimate/ "$node:route-b-m1/pp-kit/"
  rsync -a stage-k3-tp2-ultimate/ "$node:route-b-m1/tp-kit/"
  ssh "$node" 'bash -se' <<'REMOTE'
set -euo pipefail
cd "$HOME/route-b-m1"
(cd pp-kit && sha256sum --strict -c SHA256SUMS)
(cd tp-kit && sha256sum --strict -c SHA256SUMS)
if [ ! -d pp ]; then
  cp -al "$HOME/route-b-qwen38-27b-release" pp
fi
pp-kit/m1-update prepare "$PWD/pp"
REMOTE
done
for rank in 0 1; do
  ssh "${hosts[rank]}" "cd ~/route-b-m1 && pp-kit/m1-update activate \$PWD/pp"
done
# Materialize a selected TP layout on both ranks. Repeat with layout=13 for 1:3.
layout=22
for rank in 0 1; do
  ssh "${hosts[rank]}" bash -se -- "$layout" "$rank" <<'REMOTE'
set -euo pipefail
cd "$HOME/route-b-m1"
if [ ! -d "tp-$1" ]; then
  python3 tp-kit/shard_route_b_tp.py --source pp --prepared "tp-kit/$1" \
    --rank "$2" --index "tp-kit/$1/audit.json" --output "tp-$1"
fi
cmp "tp-$1/tp.json" "tp-kit/$1/audit.json"
REMOTE
done
```

Run the benchmark from the MPI launch board. Export `K3_R0`, `K3_R1`, `K3_MPI` and `K3_VLEN`. The last two are absolute paths to MPICH and the VLEN probe on both boards. Run each configuration twice in alternating order. The output reports TTFT, warm decode throughput, emitted IDs and speculation counts. Compare peak rank memory from stderr and require zero swap during the run.

```bash
set -euo pipefail
export BUDDY_PP_OBSERVE=1 BUDDY_PP_STORAGE=1
export M1_ROOT="$HOME/route-b-m1"
export M1_TP_LAYOUT="${M1_TP_LAYOUT:-22}"
cd "$M1_ROOT/tp-kit"
python3 - <<'PY' | tee "$M1_ROOT/acceptance.jsonl"
import json, os
from pathlib import Path
from types import SimpleNamespace
from server import Engine, mpi_command

env = os.environ
root = Path(env["M1_ROOT"])
pp = root / "pp" / "route-b.json"
layout = root / ("tp-" + env["M1_TP_LAYOUT"])
prompts = ["Explain how a KV cache works.", "1" * 21]
options = {
    "greedy": [],
    "mtp1": ["--spec-mode", "mtp", "--max-draft", "1"],
    "mtp2": ["--spec-mode", "mtp", "--max-draft", "2"],
    "mtp3": ["--spec-mode", "mtp", "--max-draft", "3"],
    "dspark": ["--spec-mode", "dspark", "--max-draft", "3",
               "--dspark-tau", "0.55", "--dspark-min-p", "0.35"],
}
references = {}
settings = [(tp, spec) for tp in (False, True) for spec in options]
for wire in ("f32", "f16"):
    for repeat in (1, 2):
        for tp, spec in settings if repeat == 1 else reversed(settings):
            launch = SimpleNamespace(
                bundle=pp, tp_bundle=layout if tp else None,
                hosts=f'{env["K3_R0"]},{env["K3_R1"]}',
                mpiexec=env["K3_MPI"], vlen_probe=env["K3_VLEN"],
                binary=str(root / "tp-kit" / "buddy-pp-route-b"), library_dir=[],
            )
            source = ["--tp-bundle", str(layout), "--launcher-stdin"] if tp else [
                "--bundle", str(pp), "--ranges", "0:20,20:64"
            ]
            cmd = mpi_command(launch) + source + [
                "--wire", wire, "--serve", "--slots", "1", *options[spec]
            ]
            engine = Engine(cmd, launch)
            try:
                assert engine.ready.wait(900) and engine.info, cmd
                for i, prompt in enumerate(["Hello", *prompts]):
                    events = engine.submit(str(i), prompt, 8 if i == 0 else 128)
                    tokens = []
                    while True:
                        row = events.get(timeout=900)
                        if row["event"] == "error":
                            raise RuntimeError(row)
                        if row["event"] == "token":
                            tokens.append(row["token_id"])
                        if row["event"] == "done":
                            break
                    if i:
                        key = (wire, i)
                        assert tokens == references.setdefault(key, tokens), (wire, tp, spec, i)
                        print(json.dumps(dict(repeat=repeat, tp=tp, spec=spec, wire=wire,
                                              prompt=prompt, token_ids=tokens, **row)), flush=True)
            finally:
                code = engine.close(120)
            if code:
                raise RuntimeError(f"MPI exited {code}")
PY
```

For HTTP, use the same staged server and binary. Run on the gateway board with the same environment and a completed symmetric shard publication.

```bash
python3 "$M1_ROOT/tp-kit/server.py" --binary "$M1_ROOT/tp-kit/buddy-pp-route-b" \
  --tp-bundle "$M1_ROOT/tp-22" --vocab-file "$M1_ROOT/pp/vocab.txt" \
  --mpiexec "$K3_MPI" --vlen-probe "$K3_VLEN" --hosts "$K3_R0,$K3_R1" \
  --wire f16 --slots 1 --spec-mode dspark --max-draft 3
```

For PP HTTP, use `--bundle "$M1_ROOT/pp/route-b.json" --ranges 0:20,20:64` in place of the two TP options. To roll back the PP copy after draining both ranks, run `pp-kit/m1-update rollback "$PWD/pp"` in `route-b-m1` on each rank.
