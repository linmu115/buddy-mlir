#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
"""Check generated token IDs and optionally the sampled rows of FP16/FP32 logits."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument(
        "--logits", nargs=2, type=Path, metavar=("REF", "ACTUAL")
    )
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument(
        "--dtype",
        choices=("f16", "f32"),
        default="f16",
        help="Element dtype of dumped logits (default: f16)",
    )
    args = parser.parse_args()

    expected = json.loads(args.reference.read_text())["token_ids"]
    actual = json.loads(args.actual.read_text())["token_ids"]
    if not expected or len(expected) != len(actual):
        raise SystemExit(
            f"token count differs: {len(expected)} vs {len(actual)}"
        )
    for step, (left, right) in enumerate(zip(expected, actual), start=1):
        if left != right:
            raise SystemExit(f"token {step} differs: {left} vs {right}")
    print(f"{len(expected)}/{len(expected)} sampled token IDs match")

    if args.logits:
        if not args.vocab_size or args.vocab_size < 1:
            parser.error("--logits needs a positive --vocab-size")
        if args.dtype == "f16":
            elem_bytes, fmt, hex_width, label = 2, "H", 4, "FP16"
        else:
            elem_bytes, fmt, hex_width, label = 4, "I", 8, "FP32"
        total = len(expected) * args.vocab_size
        reference, actual = (path.read_bytes() for path in args.logits)
        expected_bytes = elem_bytes * total
        if len(reference) != expected_bytes or len(actual) != expected_bytes:
            raise SystemExit(
                f"expected {expected_bytes} logit bytes in each file"
            )
        if reference != actual:
            for i, (left, right) in enumerate(
                zip(
                    memoryview(reference).cast(fmt),
                    memoryview(actual).cast(fmt),
                )
            ):
                if left != right:
                    step, token_id = divmod(i, args.vocab_size)
                    raise SystemExit(
                        f"logit differs at step {step + 1}, ID {token_id}: "
                        f"0x{left:0{hex_width}x} vs 0x{right:0{hex_width}x}"
                    )
        print(f"{total:,} sampled {label} logit elements match bit for bit")


if __name__ == "__main__":
    main()
