"""Losslessly split Route B packed weights with a prepared recipe (stdlib only)."""

import argparse
import hashlib
import json
import mmap
import shutil
import tempfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def check(path, wanted):
    with path.open("rb") as f:
        result = hashlib.file_digest(f, "sha256").hexdigest()
    if result != wanted:
        raise ValueError(f"invalid digest for {path}")


def split(source, target, field):
    size = source.stat().st_size
    if size != field["source_bytes"]:
        raise ValueError(f"invalid source size for {source}")
    sha = hashlib.sha256()
    count = 0
    with (
        source.open("rb") as f,
        mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        for offset, rows, stride, length in field["records"]:
            if (
                min(offset, rows - 1, stride - length, length - 1) < 0
                or offset + (rows - 1) * stride + length > size
            ):
                raise ValueError("invalid shard range")
            if stride == length:
                length *= rows
                rows = 1
            for i in range(rows):
                start = offset + i * stride
                for j in range(0, length, 4 << 20):
                    block = data[start + j : start + min(j + (4 << 20), length)]
                    sha.update(block)
                    target.write(block)
            count += rows * length
    return count, sha.hexdigest()


def shard(source, prepared, dest, dry, selected=None, index=None):
    src = json.loads((source / "route-b.json").read_text())
    source_id = src.pop("bundle_id")
    if digest(src) != source_id:
        raise ValueError("invalid source manifest")
    verified = set()
    ids = [None, None] if index is None else list(index["ranks"])
    if len(ids) != 2:
        raise ValueError("invalid expected TP index")
    for rank in range(2):
        if selected is not None and rank != selected:
            continue
        folder = f"rank{rank}"
        inputs = prepared / folder
        out = dest / folder
        out.mkdir()
        plan = json.loads((inputs / "recipe.json").read_text())
        j = plan["manifest"]
        tp = j["tp"]
        if tp["source"] != source_id or tp["rank"] != rank:
            raise ValueError("recipe belongs to another source or rank")

        def copy(path, expected, directory=out):
            if path not in verified:
                check(path, expected)
                verified.add(path)
            if not dry:
                shutil.copyfile(path, directory / path.name)

        for w in j["weights"]:
            owner, bucket = w["owner"], w["bucket"]
            path = source / w["path"]
            if owner in ("gdn", "full_attn"):
                field = plan["fields"][owner + "." + bucket]
                if path not in verified:
                    check(path, w["sha256"])
                    verified.add(path)
                with (out / path.name).open("wb") as f:
                    size, sha = split(path, f, field)
                if dry:
                    check(out / path.name, sha)
                    (out / path.name).unlink()
                w.update(
                    nbytes=size,
                    elements=size // (4 if bucket == "f32" else 1),
                    sha256=sha,
                    layout_id=j["layouts"][owner],
                )
            elif owner == "rope" or rank == tp["host"]:
                copy(path, w["sha256"])
        for k in j["templates"]:
            if k["role"] in ("gdn_layer", "full_attn_layer"):
                copy(inputs / k["path"], k["sha256"])
            elif rank == tp["host"]:
                copy(source / k["path"], k["sha256"])
        if "runtime" in j:
            copy(source / j["runtime"], j["runtime_sha256"])
        if rank == tp["host"]:
            for name, sha in (
                j.get("tokenizer_assets", {})
                | {j["vocab_file"]: j["vocab_sha256"]}
            ).items():
                copy(source / name, sha)
        local_id = digest(j)
        if ids[rank] is not None and ids[rank] != local_id:
            raise ValueError("rank differs from expected TP index")
        ids[rank] = local_id
        j["bundle_id"] = local_id
        (out / "route-b.json").write_text(json.dumps(j, indent=2))
    (dest / "tp.json").write_text(
        json.dumps({"format": "bj47.tp/1", "ranks": ids})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for attr in ("source", "prepared", "output"):
        p.add_argument("--" + attr, type=Path, required=True)
    p.add_argument(
        "--audit-only",
        action="store_true",
        help="write, hash and discard one body shard at a time",
    )
    p.add_argument(
        "--rank",
        type=int,
        choices=(0, 1),
        help="publish just this rank on its deployment host",
    )
    p.add_argument(
        "--index",
        type=Path,
        help="expected two-rank index from a prior audit or full split",
    )
    a = p.parse_args()
    if a.rank is not None and a.index is None:
        p.error("--rank requires --index")
    index = json.loads(a.index.read_text()) if a.index else None
    if a.output.exists():
        raise FileExistsError(a.output)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        dest = Path(tmp) / "bundle"
        dest.mkdir()
        shard(a.source, a.prepared, dest, a.audit_only, a.rank, index)
        if a.audit_only:
            (dest / "tp.json").rename(dest / "audit.json")
        dest.rename(a.output)
