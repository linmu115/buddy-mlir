"""Compare byte-copy recipes with directly packing the local float matrices."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
)
from buddy.compiler.graph.transform import k3_w4
from prepare_route_b_tp import records
from shard_route_b_tp import digest, shard, split


def encode(pack, arrays):
    return b"".join(part.tobytes() for part in pack(*arrays))


def local_matrix(name, w, begin, end):
    def part(n):
        return slice(n * begin // 4, n * end // 4)

    if name == "mixerOut":
        return w[part(w.shape[0])]
    if name == "inProjQKV":
        return np.concatenate(
            [c[:, part(c.shape[1])] for c in np.split(w, [512, 1024], axis=1)],
            axis=1,
        )
    return w[:, part(w.shape[1])]


class Tiles(unittest.TestCase):
    def test_independent_rank_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = bytes(range(32))
            (root / "body.f32").write_bytes(body)
            (root / "vocab").write_bytes(b"words")
            entry = {
                "owner": "gdn",
                "bucket": "f32",
                "path": "body.f32",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            base = {
                "weights": [entry],
                "templates": [],
                "vocab_file": "vocab",
                "vocab_sha256": hashlib.sha256(b"words").hexdigest(),
            }
            source = digest(base)
            (root / "route-b.json").write_text(
                json.dumps(dict(base, bundle_id=source))
            )
            for rank in (0, 1):
                folder = root / f"rank{rank}"
                folder.mkdir()
                manifest = dict(
                    base,
                    layouts={"gdn": "local"},
                    tp={"rank": rank, "host": 1, "source": source},
                )
                field = {
                    "records": [[rank * 16, 1, 16, 16]],
                    "source_bytes": 32,
                }
                (folder / "recipe.json").write_text(
                    json.dumps(
                        {"manifest": manifest, "fields": {"gdn.f32": field}}
                    )
                )
            audit = root / "audit"
            audit.mkdir()
            shard(root, root, audit, True)
            index = json.loads((audit / "tp.json").read_text())
            for rank in (0, 1):
                output = root / f"board{rank}"
                output.mkdir()
                shard(root, root, output, False, rank, index)
                member = output / f"rank{rank}"
                j = json.loads((member / "route-b.json").read_text())
                self.assertEqual(j.pop("bundle_id"), index["ranks"][rank])
                self.assertEqual(digest(j), index["ranks"][rank])
                self.assertEqual(
                    (member / "body.f32").read_bytes(),
                    body[rank * 16 : (rank + 1) * 16],
                )
                self.assertEqual((member / "vocab").exists(), rank == 1)
                self.assertFalse((output / f"rank{1 - rank}").exists())

    def test_decode_and_ime(self):
        d = {"kvHeads": 4, "keyWidth": 512, "valueWidth": 1024}
        rng = np.random.default_rng(31)
        for name, n, k in [
            ("glu", 512, 256),
            ("inProjQKV", 2048, 256),
            ("mixerOut", 256, 512),
        ]:
            weights = [
                rng.normal(size=(k, n)).astype("f4")
                for _ in range(2 if name == "glu" else 1)
            ]
            pack = k3_w4.pack_glu_both if name == "glu" else k3_w4.pack_q4_both

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "original.q4"
                path.write_bytes(encode(pack, weights))
                for begin, end in ((0, 2), (2, 4), (0, 1), (1, 4)):
                    with self.subTest(name=name, begin=begin, end=end):
                        rec, size = records(
                            {name: (None, (n, k))}, d, begin, end, True, True
                        )
                        out = io.BytesIO()
                        split(path, out, {"records": rec, "source_bytes": size})
                        self.assertEqual(
                            out.getvalue(),
                            encode(
                                pack,
                                [
                                    local_matrix(name, w, begin, end)
                                    for w in weights
                                ],
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
