"""Check production-width Engram byte offsets using two local sparse files."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == "__main__":
    # The first queried row begins at byte 2**32. Only the last 24 rows of
    # either array have backing pages. Remove the files when the test exits.
    first = 1 << 24
    with TemporaryDirectory(dir=sys.argv[2]) as temporary:
        root = Path(temporary)
        with (root / "large.txt").open("w") as index:
            for suffix, width, kind in (
                ("weight", 256, "fp8"),
                ("scale", 8, "scale"),
            ):
                file = root / (suffix + ".rows")
                with file.open("wb") as f:
                    f.truncate((first + 24) * width)
                    f.seek(first * width)
                    for i in range(24):
                        f.write(
                            bytes([0x38 + i % 2]) * width
                            if kind == "fp8"
                            else bytes(range(124, 132))
                        )
                print(
                    "embed." + suffix,
                    file.name,
                    kind,
                    2,
                    first + 24,
                    width,
                    file=index,
                )
        subprocess.run([sys.argv[1], "--large-table", str(root)], check=True)
