"""Capacity-constrained contiguous pipeline planner for batch-one decode streams.

Use --help for inputs. Byte counts come from upstream meta tensors. Stage
service times use the supplied board medians, and links use payload goodput.
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

GIB = 1 << 30


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(loader)
    sys.modules[name] = result
    loader.loader.exec_module(result)
    return result


def inventory(spec, v4_source, context):
    import deepseek_v4_cpu as kernel
    import torch

    cfg41 = json.loads((spec / "inference_config.json").read_text())
    cfg4 = json.loads((spec / "v4flash_config.json").read_text())
    vision = ModuleType("vision")
    vision.ViT = vision.Aligner = None
    image = ModuleType("image_processor")
    image.IMAGE, image.IMAGE_NEW_LINE, image.IMAGE_END, image.IMAGE_START = (
        range(4)
    )
    eg = module("deepseek_plan_engram", spec / "inference_engram.py")
    with patch.dict(
        sys.modules,
        kernel=kernel,
        vision=vision,
        image_processor=image,
        engram=eg,
    ):
        v41 = module("plan_v41", spec / "inference_model.py")
        v4 = module("plan_v4", v4_source)
    c = v4.ModelArgs(
        dim=cfg4["hidden_size"],
        n_layers=cfg4["num_hidden_layers"],
        n_mtp_layers=0,
        moe_inter_dim=cfg4["moe_intermediate_size"],
        n_hash_layers=cfg4["num_hash_layers"],
        n_heads=cfg4["num_attention_heads"],
        n_routed_experts=cfg4["n_routed_experts"],
        n_activated_experts=cfg4["num_experts_per_tok"],
        n_shared_experts=cfg4["n_shared_experts"],
        expert_dtype="fp4",
    )
    for k in (
        "head_dim",
        "o_groups",
        "o_lora_rank",
        "q_lora_rank",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "vocab_size",
        "compress_ratios",
        "hc_mult",
    ):
        setattr(c, k, cfg4[k])
    c.rope_head_dim = cfg4["qk_rope_head_dim"]
    v4.scale_dtype = torch.float8_e8m0fnu
    v4.default_dtype = torch.float8_e4m3fn
    configs = {"V4": (v4, c), "V4.1": (v41, v41.ModelArgs(**cfg41))}
    out = {}
    with v41.set_dtype(torch.bfloat16), torch.device("meta"):
        for model, (impl, c) in configs.items():
            c.max_batch_size = 1
            c.max_seq_len = context
            newer = model == "V4.1"
            layout = eg.EngramLayout.from_args(c) if newer else None
            result = []
            kinds = {}
            for i in range(c.n_layers):
                ratio = c.compress_ratios[i]
                kv = newer and i in c.kv_source_layers
                index = newer and i in c.index_source_layers
                engram = newer and i in c.engram_layer_ids
                tag = (
                    ratio,
                    kv,
                    index,
                    i if engram else -1,
                    not newer and i < c.n_hash_layers,
                )
                if tag not in kinds:
                    blk = (
                        impl.Block(i, c, layout) if newer else impl.Block(i, c)
                    )
                    weights = mapped = 0
                    for k, p in blk.named_parameters():
                        num = p.numel()
                        if k.startswith("engram.embed."):
                            mapped += num * p.element_size()
                        elif k.startswith(
                            ("ffn.experts.", "ffn.shared_experts.")
                        ):
                            if k.endswith(".weight"):
                                n, kdim = p.shape
                                if p.dtype == torch.float4_e2m1fn_x2:
                                    kdim *= 2
                                weights += n * (kdim // 32) * 18
                        else:
                            weights += num * p.element_size()
                    cache = sum(
                        b.numel() * b.element_size()
                        for key, b in blk.named_buffers()
                        if not key.endswith("freqs_cis")
                    )
                    kinds[tag] = {
                        "weights": weights,
                        "mmap": mapped,
                        "cache": cache,
                    }
                result.append(
                    dict(
                        layer=i,
                        ratio=ratio,
                        kv=kv,
                        index=index,
                        engram=engram,
                        **kinds[tag],
                    )
                )
            h = c.dim
            out[model] = {
                "h": h,
                "d": c.head_dim,
                "di": c.index_head_dim,
                "topk": c.index_topk,
                "vocab": c.vocab_size,
                "layers": result,
                # V4 has an additional learned hc_head, absent in V4.1.
                "input_bytes": 2 * h * c.vocab_size,
                "output_bytes": 2 * h * c.vocab_size
                + (2 * h if newer else 4 * h + 4 * (4 * 4 * h + 4 + 1)),
                "rope_bytes": context * c.rope_head_dim * 4,
                "eng_vocab_bytes": 8 * c.vocab_size if newer else 0,
                "candidate_source": c.candidate_source_layer if newer else -1,
            }
    return out


def boundaries(model, config, ctx, wire):
    layers = config["layers"]
    h, d, di = config["h"], config["d"], config["di"]
    result = {}
    for i in range(1, len(layers)):
        payload = 4 * h * wire + 16
        replica = 0
        if model == "V4.1":
            ratio = layers[i]["ratio"]
            if ratio and not layers[i]["kv"]:
                # A rank mirrors the source cache once, then receives append-only deltas.
                payload += 2 * (d + di) // ratio
                replica = (ctx // ratio) * 2 * (d + di)
                if not layers[i]["index"]:
                    payload += config["topk"] * 4
            if (
                config["candidate_source"]
                < i
                <= max(l["layer"] for l in layers if l["index"])
            ):
                payload += (ctx + 7) // 8
        result[i] = (payload, replica)
    return result


def compute(model, c, med):
    h = c["h"]
    head = med[f"bf16head4096_{h}"] * (c["vocab"] / 4096)
    times = []
    for layer in c["layers"]:
        r = layer["ratio"]
        if model == "V4.1":
            mode = (
                "swa"
                if not r
                else f"full{r}"
                if layer["kv"]
                else "reindex"
                if layer["index"]
                else "reuse"
            )
            t = 2 * med["mhc_1"] + med["moe"] + med["csa_" + mode]
            if layer["engram"]:
                t += med["engram_warm"]
        else:
            # V4 uses per-layer 4/128 compressors and a ratio-4 indexer.
            # At 4096, ratio-four sees half as many keys with twice as
            # many index heads as V4.1 Full2. Use that observed increment.
            # Ratio-128 has no indexer; its incremental cost is uncalibrated.
            base = med["csa_swa"]
            extra = max(0, med["csa_full2"] - base) if r == 4 else 0

            # Q/O/kv projection MACs for the two actual model shapes.
            def attn_macs(dim, qrank):
                return (
                    dim * qrank
                    + qrank * 32768
                    + dim * 512
                    + 8 * 4096 * 1024
                    + 8192 * dim
                )

            t = (base + extra) * attn_macs(4096, 1024) / attn_macs(5120, 1280)
            t += (
                med["moe"] * (4096 * 2048) / (5120 * 2304)
                + 2 * med["mhc_1"] * 4096 / 5120
            )
        times.append(t)
    times[-1] += head
    return times


def solve(
    model,
    c,
    board_count,
    streams,
    gbps,
    ctx,
    reserve,
    wire,
    times,
    goodput=1.0,
    source_aligned=True,
    board_gib=32,
):
    layers = c["layers"]
    edges = boundaries(model, c, ctx, wire)
    n = len(layers)
    source_starts = {
        l["layer"] for l in layers if source_aligned and (l["kv"] or l["index"])
    }

    def stage(a, b):
        part = layers[a:b]
        weights = (
            sum(l["weights"] for l in part)
            + (a == 0) * c["input_bytes"]
            + (b == n) * c["output_bytes"]
        )
        if any(l["engram"] for l in part):
            weights += c["eng_vocab_bytes"]
        cache = (
            sum(l["cache"] for l in part) + edges.get(a, (0, 0))[1]
        ) * streams
        rope = c["rope_bytes"] * len({bool(l["ratio"]) for l in part})
        hc = streams * (2 * 4 * c["h"] * 4 + 3 * 4 * 4 + 4 * 4 * 4)
        link = edges.get(b, (0, 0))[0]
        ms = link * 8 / (gbps * 1e6 * goodput)
        return {
            "start": a,
            "end": b,
            "weight_bytes": weights,
            "cache_bytes": cache,
            "hc_bytes": hc,
            "rope_bytes": rope,
            "ram_bytes": weights + cache + hc + rope,
            "mapped_bytes": sum(l["mmap"] for l in part),
            "wire_bytes": link,
            "compute_ms": sum(times[a:b]),
            "link_ms": ms,
        }

    # Pareto-optimal labels in (sum service, maximum stage service).
    # Closed-stream throughput is min(S / sum, 1 / max).
    states = {0: [(0.0, 0.0, ())]}
    for j in range(1, board_count + 1):
        nxt = {}
        for b in range(j, n - (board_count - j) + 1):
            choices = []
            for a, labels in states.items():
                if a >= b or any(a < i < b for i in source_starts):
                    continue
                row = stage(a, b)
                if row["ram_bytes"] > board_gib * GIB - reserve:
                    continue
                t = sum(round(ms * 1e6) for ms in times[a:b]) + round(
                    row["link_ms"] * 1e6
                )
                choices.extend(
                    (total + t, max(peak, t), path + (b,))
                    for total, peak, path in labels
                )
            best = math.inf
            front = []
            for label in sorted(choices):
                if label[1] < best:
                    best = label[1]
                    front.append(label)
            if front:
                nxt[b] = front
        states = nxt
    if n not in states:
        # Exact smallest board count for this capacity and mandatory cuts.
        minimum = {0: (0, ())}
        for b in range(1, n + 1):
            options = [
                (count + 1, path + (b,))
                for a, (count, path) in minimum.items()
                if not any(a < i < b for i in source_starts)
                and stage(a, b)["ram_bytes"] <= board_gib * GIB - reserve
            ]
            if options:
                minimum[b] = min(options)
        return {
            "feasible": False,
            "minimum_boards": minimum.get(n, (None,))[0],
            "mandatory_starts": sorted(source_starts),
        }
    total, peak, path = min(
        states[n], key=lambda l: (max(l[0] / streams, l[1]), l[1], l[0])
    )
    total /= 1e6
    peak /= 1e6
    rows = [stage(a, b) for a, b in zip((0,) + path, path)]
    return {
        "feasible": True,
        "stages": rows,
        "latency_ms": total,
        "bottleneck_ms": peak,
        "tokens_s": 1000 / max(total / streams, peak),
        "wire_percent": 100 * sum(r["link_ms"] for r in rows) / total,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument(
        "--v4-reference",
        required=True,
        type=Path,
        help="V4-Flash inference/model.py",
    )
    p.add_argument(
        "--measurements",
        required=True,
        type=Path,
        help="directory with board-*.jsonl",
    )
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--board-gib", type=float, default=32)
    p.add_argument("--reserve-gib", type=float, default=1)
    p.add_argument("--wire-bytes", type=int, choices=(2, 4), default=4)
    p.add_argument("--goodput", type=float, default=1.0)
    args = p.parse_args()
    if args.context != 4096:
        p.error("latency measurements are calibrated at context 4096")
    if not 0 < args.goodput <= 1 or not 0 <= args.reserve_gib < args.board_gib:
        p.error("invalid capacity or goodput")
    samples = [
        json.loads(line)
        for file in sorted(args.measurements.glob("board-*.jsonl"))
        for line in file.read_text().splitlines()
    ]
    med = {s["name"]: statistics.median(s["ms"]) for s in samples}
    configs = inventory(args.spec, args.v4_reference, args.context)
    output = {
        "context": args.context,
        "reserve_gib": args.reserve_gib,
        "board_gib": args.board_gib,
        "wire_bytes": args.wire_bytes,
        "goodput": args.goodput,
        "models": configs,
        "plans": [],
    }
    for model, boards in (("V4", (8, 16)), ("V4.1", (10, 12, 16))):
        c = configs[model]
        times = compute(model, c, med)
        for count in boards:
            for streams in (1, 8, 16):
                for link in (1, 10):
                    result = solve(
                        model,
                        c,
                        count,
                        streams,
                        link,
                        args.context,
                        args.reserve_gib * GIB,
                        args.wire_bytes,
                        times,
                        args.goodput,
                        board_gib=args.board_gib,
                    )
                    if model == "V4.1" and not result["feasible"]:
                        result["with_cache_transfer"] = solve(
                            model,
                            c,
                            count,
                            streams,
                            link,
                            args.context,
                            args.reserve_gib * GIB,
                            args.wire_bytes,
                            times,
                            args.goodput,
                            False,
                            board_gib=args.board_gib,
                        )
                    output["plans"].append(
                        dict(
                            model=model,
                            boards=count,
                            streams=streams,
                            gbps=link,
                            **result,
                        )
                    )
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
