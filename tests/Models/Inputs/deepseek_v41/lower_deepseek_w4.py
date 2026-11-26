"""Lower the real Route B tile consumer for the import differential test."""

import argparse
import sys
from pathlib import Path

from buddy.compiler.graph.transform import k3_w4

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "tools/buddy-codegen")
)
from build_route_b import ATTRS
from compile_pipeline import build_stages, run_pipeline

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("buddy_opt", type=Path)
    p.add_argument("llvm_bin", type=Path)
    args = p.parse_args()
    source = args.output / "v4-w4.mlir"
    spec = k3_w4.kernel_spec("plain", 1, 256, [384], False, 1)
    source.write_text(k3_w4.gen_kernels([spec]))
    for target, name, attrs in (
        ("native", "kernels", ""),
        ("rv", "kernels_a100", ATTRS),
    ):
        run_pipeline(
            build_stages(name, 1, attrs, "w4g32"),
            source,
            args.output / f"v4-w4-{target}.o",
            args.buddy_opt,
            args.llvm_bin,
        )
