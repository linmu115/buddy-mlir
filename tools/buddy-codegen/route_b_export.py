"""Component capture, K3 rewrite and atomic publication for Route B."""

import copy
import gc
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from gen_config import w4g32_component_param_counts
from import_pipeline_stage import (
    check_parameter_layout,
    load_stage_lazy,
    resolve_model_path,
    stage_text_config,
    validate_stage_interval,
)
from transformers import AutoConfig


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Endpoint(torch.nn.Module):
    def __init__(self, model, role):
        super().__init__()
        self.config = model.config
        self.op = (
            model.model.embed_tokens
            if role == "embed"
            else torch.nn.Sequential(model.model.norm, model.lm_head)
        )

    def forward(self, x):
        return self.op(x)


def trace_endpoint(model, role, width, rows, phase):
    from import_model import DynamoCompiler, inductor_decomp, tosa

    module = Endpoint(model, role)
    x = (
        torch.zeros((1, rows), dtype=torch.int64)
        if role == "embed"
        else torch.zeros((1, rows, width))
    )
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_" + phase,
    )
    with torch.no_grad():
        (graph,) = compiler.importer(module, x)
    return graph


def provenance(graph, model):
    names = {
        v.data_ptr(): k
        for k, v in (*model.named_parameters(), *model.named_buffers())
    }
    return {
        p.name: (names[t.data_ptr()], list(t.shape))
        for p, t in zip(graph.params, graph._params_ref, strict=True)
    }


def slots(graph, original):
    offsets = {"f32": 0, "q4": 0, "rope": 0}
    entries = []
    source_shapes = dict(original.values())
    for p, t in zip(graph.params, graph._params_ref, strict=True):
        key, shape = original[p.name.removesuffix("_ime")]
        sources = [key]
        transform = "identity"
        if "q_proj." in key:
            sources = [key.replace("q_proj", x + "_proj") for x in "qkv"]
            transform = "concat-qkv"
        if "gate_proj." in key:
            sources.append(key.replace("gate_proj", "up_proj"))
            transform = "glu"
        if t.dtype == torch.int8:
            bucket = "q4"
            transform += "/transpose-q4g32/" + (
                "ime" if p.name.endswith("_ime") else "tiles"
            )
            if key.endswith(("experts.gate_up_proj", "experts.down_proj")):
                if p.name.endswith("_ime"):
                    raise ValueError("MoE experts must have one tile layout")
                transform = (
                    "expert-wise/"
                    + ("glu" if key.endswith("gate_up_proj") else "down")
                    + "/q4g32"
                )
        elif t.dtype == torch.float32:
            if key.endswith("mlp.gate.weight"):
                transform = "transpose-router"
            bucket = "rope" if key.endswith("inv_freq") else "f32"
        else:
            raise ValueError(f"unexpected dtype for {key}: {t.dtype}")
        entries.append(
            {
                "bucket": bucket,
                "semantic_key": key
                + ("_ime" if p.name.endswith("_ime") else ""),
                "source_keys": sources,
                "source_shapes": [source_shapes[k] for k in sources],
                "transform_id": transform,
                "logical_shape": shape,
                "packed_shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": str(t.dtype),
                "element_offset": offsets[bucket],
                "elements": t.numel(),
            }
        )
        offsets[bucket] += t.numel()
    return entries


def write_blobs(root, owner, layer, graph, layout, expected, layout_id):
    records = []
    for bucket in ("f32", "q4", "rope"):
        selected = [
            (slot, ref)
            for slot, ref in zip(layout, graph._params_ref, strict=True)
            if slot["bucket"] == bucket
        ]
        if not selected or (bucket == "rope" and layer != 0):
            continue
        filename = (
            "rope.f32.data" if bucket == "rope" else f"{owner}.{bucket}.data"
        )
        sha = hashlib.sha256()
        count = 0
        with (root / filename).open("xb") as stream:
            for slot, ref in selected:
                data = memoryview(ref.detach().contiguous().numpy()).cast("B")
                stream.write(data)
                sha.update(data)
                count += ref.numel()
        tag = "i8_elements" if bucket == "q4" else "f32_elements"
        size = (root / filename).stat().st_size
        role = (
            "rope"
            if bucket == "rope"
            else "decoder"
            if layer is not None
            else "head"
            if owner == "lm_head"
            else owner
        )
        if count != expected[role][tag] or size != count * (
            1 if bucket == "q4" else 4
        ):
            raise ValueError(
                f"{owner}/{bucket}: unexpected counts {count}, {size}"
            )
        records.append(
            dict(
                owner=role,
                **({"global_layer_id": layer} if role == "decoder" else {}),
                bucket="f32" if bucket == "rope" else bucket,
                path=filename,
                elements=count,
                dtype="i8" if bucket == "q4" else "f32",
                nbytes=size,
                sha256=sha.hexdigest(),
                layout_id=layout_id
                if role != "rope"
                else digest(selected[0][0]),
            )
        )
    return records


def graph_signature(graph):
    """Fingerprint the post-rewrite topology, constants and kernel targets, excluding tensor contents."""
    nodes = [
        (
            type(n).__name__,
            n.name,
            str(n.args),
            str(n.kwargs),
            str(n.tensor_meta),
            getattr(n, "call_func_name", None),
        )
        for n in graph.body
    ]
    return digest(nodes)


def export_one(config, root, role, index, expected, layouts, signatures):
    from buddy.compiler.graph.transform.k3_w4 import (
        NB,
        _Rewriter,
        gen_kernels,
        k3_w4_rewrite,
        pack_q4,
        packed_bytes,
    )
    from import_model import (
        apply_fusion,
        compile_chunk_graphs,
        create_drivers,
        eliminate_matmul_transpose_reshape,
        eliminate_transpose,
        export_mlir,
    )

    class HeadRewrite(_Rewriter):
        def pack_q4(self, w):
            # Output tiles are independent along N. Bound quantization scratch for the vocabulary matrix.
            import numpy as np

            k, n = w.shape
            tiles = np.empty(packed_bytes(k, n), dtype=np.int8)
            for start in range(0, n, 8 * NB):
                part = pack_q4(w[:, start : start + 8 * NB])
                offset = packed_bytes(k, start)
                tiles[offset : offset + part.size] = part
            return tiles, None

    class PrefillHeadRewrite(HeadRewrite):
        def fuse_rmsnorm(self):
            # Route A normalizes the chunk before selecting its last row.
            pass

    model, _, width = load_stage_lazy(config, index, index + 1, role)
    owner = (
        f"layer_{index:02}"
        if role == "decoder"
        else "lm_head"
        if role == "head"
        else "embed"
    )
    template = "decoder_layer" if role == "decoder" else owner
    emit = role != "decoder" or index == 0
    target = root / "mlir" / template
    if emit:
        target.mkdir(parents=True)
    if role == "decoder":
        (prefill,), (decode,), _ = compile_chunk_graphs(
            model, config, hidden_input=True, logits_to_keep=None
        )
    else:
        prefill = trace_endpoint(
            model,
            role,
            width,
            config["prefill_chunk"] if role == "embed" else 1,
            "prefill",
        )
        decode = trace_endpoint(model, role, width, 1, "decode")
    graphs = [prefill, decode]
    names = [provenance(graph, model) for graph in graphs]
    del model
    specs, ime_weights = {}, set()
    # The two graphs refer to the same layer. Pack its expert bank once.
    expert_cache = {}
    for graph, rows in zip(
        graphs,
        (config["prefill_chunk"] if role != "head" else 1, 1),
        strict=True,
    ):
        graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
        if role != "embed":
            pos = graph.inputs[1].name if role == "decoder" else ""
            for spec in k3_w4_rewrite(
                graph,
                rows,
                pos,
                config["compilation"]["num_threads"],
                role == "decoder" and rows > 1,
                ime_weights,
                cache=expert_cache,
                rewriter=(
                    PrefillHeadRewrite if graph is prefill else HeadRewrite
                )
                if role == "head"
                else _Rewriter,
            ):
                specs[spec["name"]] = spec
    layout = slots(prefill, names[0])
    check_parameter_layout(prefill, decode)
    if slots(decode, names[1]) != layout:
        raise ValueError(f"{owner}: prefill/decode slot mismatch")
    sig = {
        phase: graph_signature(g)
        for phase, g in zip(("prefill", "decode"), graphs, strict=True)
    }
    if signatures.setdefault(template, sig) != sig:
        raise ValueError(
            f"{owner}: post-rewrite graph differs from representative"
        )
    if template in layouts and layouts[template] != layout:
        raise ValueError(f"{owner}: layout differs from representative")
    layouts[template] = layout
    for bucket, tag in (("f32", "f32_elements"), ("q4", "i8_elements")):
        if (
            sum(s["elements"] for s in layout if s["bucket"] == bucket)
            != expected[role][tag]
        ):
            raise ValueError(
                f"{owner}: missing or incorrect {bucket} slots: {layout}"
            )
    weights = write_blobs(
        root,
        owner,
        index if role == "decoder" else None,
        prefill,
        layout,
        expected,
        digest([s for s in layout if s["bucket"] != "rope"]),
    )
    if emit:
        if specs:
            (target / "k3_kernels-w4g32.mlir").write_text(
                gen_kernels(list(specs.values()))
            )
            if role == "decoder":
                (target / "k3_kernels_ime-w4g32.mlir").write_text(
                    gen_kernels(list(specs.values()), part="ime")
                )
        apply_fusion(prefill, decode, chunked_prefill=True)
        export_mlir(*create_drivers(prefill, decode), config, target)
    return weights


def export_template_bundle(config, out, load_mode):
    if (
        load_mode != "lazy"
        or config["architecture"]
        not in ("Qwen2ForCausalLM", "Qwen3MoeForCausalLM")
        or config["variant"] != "w4g32"
        or config["kernels"] != "k3"
        or config.get("prefill_chunk") != 64
        or not config.get("prefill_ime")
        or config["precision"]["activation_type"] != "f32"
        or config["shape"]["max_token_len"] < 64
        or not config.get("arena")
        or not config.get("thread_pool")
        or any(
            config.get(k, {}).get("enabled")
            for k in ("tiered_kv_cache", "decode_pack")
        )
    ):
        raise ValueError(
            "Route B v1 requires lazy Qwen2 or Qwen3-MoE K3 w4g32 f32, IME chunk=64 with static KV, arena and thread pool"
        )
    if out.exists():
        raise FileExistsError(out)
    text, _ = stage_text_config(
        AutoConfig.from_pretrained(resolve_model_path(config))
    )
    validate_stage_interval(text, 0, text.num_hidden_layers)
    hf = text.to_dict()
    if config["architecture"] != hf["architectures"][0]:
        raise ValueError(
            "checkpoint architecture differs from component config"
        )
    total = hf["num_hidden_layers"]
    shape = config["shape"]
    dims = {
        "num_hidden_layers": total,
        "vocab_size": hf["vocab_size"],
        "head_num": hf["num_key_value_heads"],
        "hidden_size": hf.get("head_dim")
        or hf["hidden_size"] // hf["num_attention_heads"],
    }
    for field, value in dims.items():
        if shape[field] != value:
            raise ValueError(
                f"config shape.{field}={shape[field]} vs checkpoint {value}"
            )
    counts = w4g32_component_param_counts(hf, True, prefill_chunk=64)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=out.name + ".partial-", dir=out.parent
    ) as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir()
        layouts, signatures, weights = {}, {}, []
        for role, index in [
            ("embed", 0),
            *(("decoder", i) for i in range(total)),
            ("head", total - 1),
        ]:
            weights.extend(
                export_one(
                    copy.deepcopy(config),
                    root,
                    role,
                    index,
                    counts,
                    layouts,
                    signatures,
                )
            )
            torch._dynamo.reset()
            gc.collect()
        (root / "slots.json").write_text(json.dumps(layouts, indent=2) + "\n")
        manifest = {
            "format": "bj47.components/1",
            "total_layers": total,
            "activation_width": hf["hidden_size"],
            "config": config,
            "counts": counts,
            "weights": weights,
            "slot_table": "slots.json",
            "pack_version": "k3-w4g32/1",
            "graph_signatures": signatures,
            "template_mlir": {
                p.relative_to(root).as_posix(): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted((root / "mlir").rglob("*.mlir"))
            },
            "shape_id": digest({"config": config, "hf": hf}),
            "templates": list(layouts),
        }
        (root / "components.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        root.rename(out)
