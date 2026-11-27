"""Audit the linked multi-row kernels and LLVM's annotated spill/reload sites."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
)
from build_route_b import ATTRS, elide_final_workshare_barriers
from compile_pipeline import build_stages, run_pipeline


def check(args):
    total = 0
    for role in ("gdn_layer", "full_attn_layer", "mtp_layer", "lm_head"):
        work = args.work / role
        dso = args.bundle / (role + ".so")
        assert dso.read_bytes() == (work / dso.name).read_bytes()
        asm = subprocess.check_output(
            [args.llvm_bin / "llvm-objdump", "-d", "--no-show-raw-insn", dso],
            text=True,
        )
        functions = re.findall(
            r"^[a-f\d]+ <([^>]+)>:\n(.*?)(?=^[a-f\d]+ <|\Z)", asm, re.M | re.S
        )
        tiles = set()
        for name, body in functions:
            if not re.search(r"(?:k3_q4_\w+_m[34]_|project_[34]_)", name):
                continue
            assert not re.search(r"\bv[ls]\w+\.v\s+[^\n]+\(sp\)", body), name
            if name.startswith("k3_q4_") and name.endswith("__tile"):
                tiles.add(name)
        assert {int(re.search(r"_m(\d)_", n)[1]) for n in tiles} == {3, 4}
        # Check compiler annotations too: stack addresses may use a temporary
        # scalar register. Regenerate the object first to tie the inspection
        # to the exact build flags and contents of the staged object.
        stages = build_stages(
            "kernels_a100", args.threads, ATTRS, "w4g32", arena=True
        )
        stages[-1][1].append("-combiner-store-merging=false")
        with tempfile.TemporaryDirectory(dir=args.work) as tmp:
            obj = Path(tmp) / "verify.o"
            source = Path(tmp) / "verify-lowered.mlir"
            run_pipeline(
                stages[:1],
                work / "w4-verify.mlir",
                source,
                args.buddy_opt,
                args.llvm_bin,
            )
            elide_final_workshare_barriers(source)
            stages = stages[1:]
            run_pipeline(stages, source, obj, args.buddy_opt, args.llvm_bin)
            assert obj.read_bytes() == (work / "w4-verify.o").read_bytes()
            stages[-1][1][stages[-1][1].index("-filetype=obj")] = (
                "-filetype=asm"
            )
            annotated = obj.with_suffix(".s")
            run_pipeline(
                stages, source, annotated, args.buddy_opt, args.llvm_bin
            )
            text = annotated.read_text()
            assert all(t + ":" in text for t in tiles)
            spills = re.findall(
                r"^\s*v[ls]\w+\.v[^\n]*#.*(?:Spill|Reload).*", text, re.M
            )
            assert not spills, (role, spills)
        total += len(tiles)
        print(role, "tiles", len(tiles), "vector spills/reloads 0")
    print("verified tile symbols", total)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("llvm-bin", "work", "bundle", "buddy-opt"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    check(p.parse_args())
