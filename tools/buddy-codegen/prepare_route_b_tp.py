"""Compile local body templates and a byte-copy recipe for an existing hybrid pack."""

import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path

import build_route_b_hybrid as hybrid
import route_b_hybrid_w4 as w4


def partition(d, begin, end):
    local = dict(d)
    for key in (
        "kvHeads",
        "attentionHeads",
        "keyHeads",
        "valueHeads",
        "intermediate",
    ):
        total = d[key]
        if total * begin % d["kvHeads"] or total * end % d["kvHeads"]:
            raise ValueError(f"split cuts a {key} group")
        local[key] = total * (end - begin) // d["kvHeads"]
    local["keyWidth"] = local["keyHeads"] * d["keyDim"]
    local["valueWidth"] = local["valueHeads"] * d["valueDim"]
    local["convDim"] = 2 * local["keyWidth"] + local["valueWidth"]
    return local


def ranges(name, shape, d, begin, end):
    # Output rows and input columns in checkpoint [N,K] order.
    def cut(n):
        return (n * begin // d["kvHeads"], n * end // d["kvHeads"])

    n, k = shape[0], math.prod(shape[1:])
    if name in ("inProjQKV", "convWeight"):
        widths = [d["keyWidth"], d["keyWidth"], d["valueWidth"]]
        offset = 0
        rows = []
        for width in widths:
            a, b = cut(width)
            rows.append((offset + a, offset + b))
            offset += width
        return rows, (0, k)
    if name in ("downProj", "mixerOut"):
        return [(0, n)], cut(k)
    if name in ("inputNorm", "postNorm", "qNorm", "kNorm", "gatedNorm"):
        return [(0, n)], (0, k)
    if name not in (
        "glu",
        "gateProj",
        "upProj",
        "inProjZ",
        "inProjA",
        "inProjB",
        "aLog",
        "dtBias",
        "qProj",
        "kProj",
        "vProj",
    ):
        raise ValueError(f"unknown shard layout for {name}")
    return [cut(n)], (0, k)


def records(schema, dims, begin, end, packed, dual):
    """(first byte, repetitions, source stride, bytes per repetition)."""
    result = []
    offset = 0
    for name, (_, shape) in schema.items():
        rows, (lo, hi) = ranges(name, shape, dims, begin, end)
        n, k = shape[0], math.prod(shape[1:])
        if not packed:
            for a, b in rows:
                result.append(
                    (offset + 4 * (a * k + lo), b - a, 4 * k, 4 * (hi - lo))
                )
            offset += 4 * n * k
            continue
        fused = name == "glu"
        for ime in range(1 + dual):
            tile = 8 if ime else 128
            group = 18 * tile
            stride = k // 32 * group
            # Decode interleaves gate/up tiles. IME concatenates both matrices.
            interleave = 2 if fused and not ime else 1
            for _ in range(2 if fused and ime else 1):
                for a, b in rows:
                    if a % tile or b % tile or lo % 32 or hi % 32:
                        raise ValueError(f"unaligned Q4 shard for {name}")
                    first = (
                        offset
                        + (a // tile * interleave) * stride
                        + lo // 32 * group
                    )
                    result.append(
                        (
                            first,
                            (b - a) // tile * interleave,
                            stride,
                            (hi - lo) // 32 * group,
                        )
                    )
                offset += n // tile * interleave * stride
    return result, offset


def prepare(args, dest):
    manifest = json.loads((args.source / "route-b.json").read_text())
    source_id = manifest.pop("bundle_id")
    if hybrid.digest(manifest) != source_id or "tp" in manifest:
        raise ValueError("invalid source manifest")
    quant = manifest["variant"] == "w4g32"
    if manifest["pack_version"] != (
        "qwen3_5-w4g32/2" if quant else "qwen3_5-f32/1"
    ):
        raise ValueError("unsupported hybrid pack")
    c, dims, types = hybrid.read_config(
        args.config, manifest["context"], manifest["prefill_chunk"]
    )
    if (
        types != manifest["layer_types"]
        or hybrid.digest(
            {
                "dims": dims,
                "epsilon": c["rms_norm_eps"],
                "rope": c["rope_parameters"],
            }
        )
        != manifest["shape_id"]
    ):
        raise ValueError("source config and pack disagree")
    heads = args.kv_heads or [dims["kvHeads"] // 2] * 2
    if len(heads) != 2 or min(heads) < 1 or sum(heads) != dims["kvHeads"]:
        raise ValueError(
            "supply two positive KV head counts summing to kvHeads"
        )
    args.chunk, args.threads = dims["chunk"], manifest.get("threads", 1)
    dual = manifest["prefill_ime"]
    cache = {}
    for rank in range(2):
        begin = sum(heads[:rank])
        end = begin + heads[rank]
        local = partition(dims, begin, end)
        args.output = dest / f"rank{rank}"
        args.output.mkdir()
        args.work = dest / f"compile{rank}"
        if quant:
            (args.output / "libbj47_kernel_runtime.so").symlink_to(
                (args.source / "libbj47_kernel_runtime.so").resolve()
            )
        j = copy.deepcopy(manifest)
        shape = hybrid.digest(
            {"tp_body": local, "base": manifest["shape_id"], "abi": 1}
        )
        j["tp"] = {
            "source": source_id,
            "rank": rank,
            "host": args.host_rank,
            "kv_heads": heads,
            "body_shape": shape,
        }
        recipe = {}
        for kernel in j["templates"]:
            role = kernel["role"]
            owner = hybrid.OWNERS[role]
            if owner not in ("gdn", "full_attn"):
                continue
            schemas = [hybrid.schemas(d)[owner] for d in (dims, local)]
            for i, schema in enumerate(schemas):
                quants = w4.quantized_fields(schema) if quant else {}
                pos, size = w4.offsets(quants, dual) if quant else ({}, 0)
                layout = hybrid.digest(
                    {
                        "schema": list(schema.items()),
                        "q4": pos,
                        "variant": "w4g32",
                    }
                    if quant
                    else {"schema": list(schema.items()), "dtype": "f32"}
                )
                floats = {
                    k: v
                    for k, v in schema.items()
                    if k not in quants
                    and (not quant or k not in w4.Q4_PROJECTIONS)
                }
                offsets, count = {}, 0
                for name, (_, field_shape) in floats.items():
                    offsets[name] = count
                    count += math.prod(field_shape)
                if i == 0:
                    if layout != j["layouts"][owner]:
                        raise ValueError(f"unknown source layout: {owner}")
                    for bucket, fields in [("f32", floats)] + (
                        [("q4", quants)] if quant else []
                    ):
                        rec, total = records(
                            fields, dims, begin, end, bucket == "q4", dual
                        )
                        recipe[owner + "." + bucket] = {
                            "records": rec,
                            "source_bytes": total,
                        }
            for bucket, expected in [("f32", 4 * count)] + (
                [("q4", size)] if quant else []
            ):
                actual = sum(
                    n * length
                    for _, n, _, length in recipe[owner + "." + bucket][
                        "records"
                    ]
                )
                if actual != expected:
                    raise ValueError(
                        f"shard size disagrees with {owner} {bucket} kernel"
                    )
            j["layouts"][owner] = layout
            key = (role, shape)
            if key in cache:
                shutil.copyfile(cache[key][0], args.output / kernel["path"])
                binary = dict(cache[key][1])
            else:
                header, mlir, symbols = hybrid.generate(
                    role,
                    local,
                    c,
                    floats,
                    offsets,
                    count,
                    "f32",
                    layout,
                    shape,
                    size,
                )
                specs = []
                if quant:
                    code, specs = w4.projection_code(
                        args, owner, quants, pos, fuse_residual=False
                    )
                    header += (
                        f"\n#define HYBRID_THREADS {args.threads}\n#include <algorithm>\n"
                        + code
                    )
                header += "\n#define HYBRID_TENSOR_PARALLEL 1\n"
                symbols.append("layer_tp_rows")
                binary = hybrid.compile_template(
                    args, role, header, mlir, symbols, specs
                )
                binary.pop("compile_seconds")
                cache[key] = (args.output / kernel["path"], dict(binary))
            kernel.update(
                binary,
                layout_id=layout,
                shape_id=shape,
                f32_elements=count,
                i8_elements=size,
            )
        if quant:
            (args.output / "libbj47_kernel_runtime.so").unlink()
        (args.output / "recipe.json").write_text(
            json.dumps({"manifest": j, "fields": recipe}, indent=2)
        )
    # Only the two rank directories are needed on the machine doing the copy.


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for attr in ("config", "source", "output", "buddy-opt", "llvm-bin"):
        p.add_argument("--" + attr, type=Path, required=True)
    p.add_argument("--kv-heads", type=int, nargs=2)
    p.add_argument("--host-rank", choices=(0, 1), type=int, default=1)
    p.add_argument("--cxx-flag", action="append", default=[])
    args = p.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "prepared"
        staged.mkdir()
        prepare(args, staged)
        staged.rename(dest)
