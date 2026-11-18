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
"""Check generated token IDs and optionally the sampled rows of FP16 logits."""

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
        total = len(expected) * args.vocab_size
        reference, actual = (path.read_bytes() for path in args.logits)
        if len(reference) != 2 * total or len(actual) != 2 * total:
            raise SystemExit(f"expected {2 * total} logit bytes in each file")
        if reference != actual:
            for i, (left, right) in enumerate(
                zip(
                    memoryview(reference).cast("H"),
                    memoryview(actual).cast("H"),
                )
            ):
                if left != right:
                    step, token_id = divmod(i, args.vocab_size)
                    raise SystemExit(
                        f"logit differs at step {step + 1}, ID {token_id}: "
                        f"0x{left:04x} vs 0x{right:04x}"
                    )
        print(f"{total:,} sampled FP16 logit elements match bit for bit")


if __name__ == "__main__":
    main()
