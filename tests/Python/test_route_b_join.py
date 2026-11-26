# RUN: %PYTHON %s | FileCheck %s

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
from build_route_b import elide_final_workshare_barriers  # noqa: E402

workshare = """
omp.wsloop {
  omp.loop_nest (%i) : i64 = (%lo) to (%hi) step (%step) {
    omp.yield
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "workshares.mlir"
    path.write_text(
        "module { func.func @test(%lo: i64, %hi: i64, %step: i64) {"
        "omp.parallel {" + workshare * 2 + "omp.terminator } return } }"
    )
    elide_final_workshare_barriers(path)
    result = path.read_text()
    assert result.count("omp.wsloop nowait") == 1
    assert result.index("omp.wsloop {") < result.index("omp.wsloop nowait")
    print("kept the producer barrier and elided the final barrier")
# CHECK: kept the producer barrier and elided the final barrier
