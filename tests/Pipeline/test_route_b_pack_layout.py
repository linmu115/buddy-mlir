"""Compare every real component byte with both Route A driver slice tables."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def compare(bundle, stages):
    manifest = json.loads((bundle / "components.json").read_text())
    layouts = json.loads((bundle / "slots.json").read_text())
    files = {
        (w["owner"], w.get("global_layer_id"), w["bucket"]): w
        for w in manifest["weights"]
    }
    result = []
    seen = set()
    for stage in stages:
        cfg = json.loads((stage / "config.json").read_text())
        part = cfg["pipeline_stage"]
        start, end = part["start_layer"], part["end_layer"]
        slots = []

        def add(role, index, layout, slots=slots):
            for s in layout:
                slots.append((role, index, s))

        if start == 0:
            add("embed", None, layouts["embed"])
        for i in range(start, end):
            add(
                "decoder",
                i,
                [
                    s
                    for s in layouts["decoder_layer"]
                    if s["bucket"] != "rope"
                    and not s["semantic_key"].endswith("_ime")
                ],
            )
        if end == manifest["total_layers"]:
            add("head", None, layouts["lm_head"])
        add(
            "rope",
            None,
            [s for s in layouts["decoder_layer"] if s["bucket"] == "rope"],
        )
        for i in range(start, end):
            add(
                "decoder",
                i,
                [
                    s
                    for s in layouts["decoder_layer"]
                    if s["semantic_key"].endswith("_ime")
                ],
            )
        for arg, bucket, unit in [(0, "f32", 4), (1, "q4", 1)]:
            views = defaultdict(list)
            for phase in ("prefill", "decode"):
                mlir = (stage / f"forward_{phase}-w4g32.mlir").read_text()
                pairs = [
                    (int(o), int(n))
                    for o, n in re.findall(
                        rf"memref.subview %arg{arg}\[(\d+)\] \[(\d+)\] \[1\]",
                        mlir,
                    )
                ]
                if phase == "prefill":
                    baseline = pairs
                elif pairs != baseline:
                    raise ValueError("Route A phase layouts differ")
            # Equal-shaped post-rewrite placeholders retain global layer order within each rewrite pass.
            for off, n in pairs:
                views[n].append(off)
            source = stage / cfg["weights"][arg]["file"]
            if sum(n for _, n in pairs) * unit != source.stat().st_size:
                raise ValueError(f"incomplete Route A slice coverage: {source}")
            with source.open("rb") as a:
                for role, i, s in slots:
                    if ("f32" if role == "rope" else s["bucket"]) != bucket:
                        continue
                    item = files[role, i, bucket]
                    n = s["elements"]
                    off = views[n].pop(0) * unit
                    b_off = s["element_offset"] * unit
                    size = unit * n
                    a.seek(off)
                    with (bundle / item["path"]).open("rb") as b:
                        b.seek(b_off)
                        for j in range(0, size, 4 << 20):
                            length = min(4 << 20, size - j)
                            x, y = a.read(length), b.read(length)
                            if len(y) != length or x != y:
                                raise ValueError(
                                    f"{item['path']}/{s['semantic_key']} differs at chunk offset {j}"
                                )
                    seen.add(item["path"])
                    result.append(
                        {
                            "stage": stage.name,
                            "source": source.name,
                            "src_offset": off,
                            "path": item["path"],
                            "dst_offset": b_off,
                            "nbytes": size,
                            "slot": s["semantic_key"],
                        }
                    )
            if any(views.values()):
                raise ValueError(f"unmatched {stage}/{bucket} slices")
    if seen != {w["path"] for w in files.values()}:
        raise ValueError("unverified component files")
    for w in files.values():
        p = bundle / w["path"]
        with p.open("rb") as f:
            if (
                p.stat().st_size != w["nbytes"]
                or hashlib.file_digest(f, "sha256").hexdigest() != w["sha256"]
            ):
                raise ValueError(f"{p}: component hash or length mismatch")
    return result


def test_route_b_pack_layout():
    """Seeded checkpoint, real A/B exporters, wrong-layer and equal-length slot mutations."""
    import copy
    import os
    import sys
    import tempfile
    from unittest.mock import patch

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools/buddy-codegen"))
    from gen_config import gen_config
    from import_pipeline_stage import export_stage
    from route_b_export import export_template_bundle

    with (
        tempfile.TemporaryDirectory() as d,
        patch.dict(os.environ, {"DEEPSEEKR1_MODEL_PATH": d}),
    ):
        root = Path(d)
        torch.manual_seed(4747)
        hf = Qwen2Config(
            num_hidden_layers=3,
            hidden_size=256,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=640,
            vocab_size=768,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(hf)
        # Both norm slots have the same shape. Give them and each layer different contents.
        with torch.no_grad():
            for i, layer in enumerate(model.model.layers):
                layer.input_layernorm.weight.fill_(1 + i * 0.125)
                layer.post_attention_layernorm.weight.fill_(1.5 + i * 0.125)
                layer.self_attn.k_proj.weight.mul_(3.0)
        model.save_pretrained(root)
        del model
        spec = json.loads(
            (repo / "models/deepseek_r1/specs/k3_w4g32.json").read_text()
        )
        spec.update(hf_model_path=d, max_token_len=128)
        cfg = gen_config(spec, str(root / "config.json"))
        refs = []
        for start, end in ((0, 2), (2, 3)):
            ref = root / f"{start}-{end}"
            export_stage(copy.deepcopy(cfg), start, end, ref)
            refs.append(ref)
        bundle = root / "bundle"
        export_template_bundle(cfg, bundle, "lazy")
        compare(bundle, refs)

        def expect_mismatch():
            try:
                compare(bundle, refs)
            except ValueError as e:
                assert "layer_01" in str(e) and "differs" in str(e), e
            else:
                raise AssertionError("corruption went undetected")

        path = bundle / "layer_01.q4.data"
        original = path.read_bytes()
        path.write_bytes((bundle / "layer_02.q4.data").read_bytes())
        expect_mismatch()
        path.write_bytes(original)
        path = bundle / "layer_01.f32.data"
        original = bytearray(path.read_bytes())
        norms = [
            s
            for s in json.loads((bundle / "slots.json").read_text())[
                "decoder_layer"
            ]
            if "layernorm.weight" in s["semantic_key"]
        ]
        left, right = (
            slice(
                s["element_offset"] * 4,
                (s["element_offset"] + s["elements"]) * 4,
            )
            for s in norms
        )
        original[left], original[right] = original[right], original[left]
        path.write_bytes(original)
        expect_mismatch()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        test_route_b_pack_layout()
        print("PASS seeded A/B layout, layer swap and same-shape slot swap")
        sys.exit(0)
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("stages", type=Path, nargs="+")
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    rows = compare(a.bundle, a.stages)
    a.receipt.write_text(json.dumps(rows, indent=2))
    print(
        len(rows), "slices;", sum(x["nbytes"] for x in rows), "bytes identical"
    )
