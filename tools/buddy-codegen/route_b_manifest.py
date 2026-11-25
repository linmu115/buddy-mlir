"""Publish the linked Route B components as a single content-addressed index."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def publish(config_path, output, token_attrs):
    source = Path(config_path)
    root = source.parent
    components = json.loads(source.read_text())
    cfg = components["config"]
    shape, tokens = cfg["shape"], cfg["tokens"]
    attrs = token_attrs(tokens)
    if components["format"] != "bj47.components/1":
        raise ValueError("invalid component format")

    def check_file(entry):
        name = entry["path"]
        if Path(name).name != name or not name:
            raise ValueError(f"expected a bundle filename: {name}")
        path = root / name
        if (
            path.stat().st_size != entry["nbytes"]
            or digest(path) != entry["sha256"]
        ):
            raise ValueError(f"{path}: content differs from receipt")

    templates = []
    roles = {"embed": "embed", "decoder_layer": "decoder", "lm_head": "head"}
    layouts = {}
    for role, owner in roles.items():
        entry = json.loads((root / f"{role}.kernel.json").read_text())
        if (
            entry["role"] != role
            or entry["shape_id"] != components["shape_id"]
            or entry["abi_version"] != (2 if owner == "head" else 1)
        ):
            raise ValueError(f"{role}: incompatible kernel receipt")
        check_file(entry)
        layouts[owner] = entry["layout_id"]
        templates.append(entry)
    counts = components["counts"]
    layers = components["total_layers"]
    expected = {
        (owner, layer, bucket)
        for owner, buckets in counts.items()
        for layer in (range(layers) if owner == "decoder" else [None])
        for bucket, field in [("f32", "f32_elements"), ("q4", "i8_elements")]
        if buckets[field]
    }
    names = set()
    for w in components["weights"]:
        owner, bucket = w["owner"], w["bucket"]
        key = (owner, w.get("global_layer_id"), bucket)
        if key not in expected or w["path"] in names:
            raise ValueError(f"unexpected weight {key}")
        expected.remove(key)
        names.add(w["path"])
        f32 = bucket == "f32"
        if (
            w["elements"]
            != counts[owner]["f32_elements" if f32 else "i8_elements"]
            or w["dtype"] != ("f32" if f32 else "i8")
            or w["nbytes"] != w["elements"] * (4 if f32 else 1)
            or w["layout_id"] != layouts.setdefault(owner, w["layout_id"])
        ):
            raise ValueError(f"{key}: incompatible weight layout")
        check_file(w)
    if expected:
        raise ValueError(f"missing weights {expected}")
    vocab = tokens["vocab_file"]
    index = {
        "format": "bj47.route-b/1",
        "model_id": cfg["model_id"],
        "checkpoint_id": cfg["hf_model_path"],
        "architecture": cfg["architecture"],
        "variant": cfg["variant"],
        "kernels": cfg["kernels"],
        "prefill_ime": cfg["prefill_ime"],
        "compile_options": cfg["compilation"],
        "rope_theta": tokens["rope_theta"],
        "activation_type": cfg["precision"]["activation_type"],
        "total_layers": layers,
        "activation_width": components["activation_width"],
        "context": shape["max_token_len"],
        "prefill_chunk": cfg["prefill_chunk"],
        "kv_heads": shape["head_num"],
        "head_dim": shape["hidden_size"],
        "vocab_size": shape["vocab_size"],
        "eos_id": tokens["eos_id"],
        "tokenizer": attrs.get("pp_tokenizer", "deepseek_r1"),
        "stop_token_ids": [
            int(x) for x in attrs.get("pp_stop_token_ids", "").split(",") if x
        ],
        "vocab_file": vocab,
        "vocab_sha256": digest(root / vocab),
        "runtime_sha256": digest(root / "libbj47_kernel_runtime.so"),
        "pack_version": components["pack_version"],
        "shape_id": components["shape_id"],
        "layouts": layouts,
        "templates": templates,
        "weights": components["weights"],
    }
    if cfg.get("arena_prefault_mb") is not None:
        index["arena_prefault_mb"] = cfg["arena_prefault_mb"]
    canonical = json.dumps(
        index, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    index["bundle_id"] = hashlib.sha256(canonical.encode()).hexdigest()
    text = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        with tempfile.NamedTemporaryFile(
            "w", dir=Path(output).parent, delete=False
        ) as tmp:
            tmp.write(text)
        os.replace(tmp.name, output)
