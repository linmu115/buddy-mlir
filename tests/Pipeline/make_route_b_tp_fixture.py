"""Make a deterministic tile-aligned hybrid checkpoint for two or four KV groups."""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
)
from build_route_b_hybrid import PREFIX, read_config, schemas


def make(dest):
    dest.mkdir()
    config = {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "hidden_act": "silu",
        "attn_output_gate": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "eos_token_id": 248044,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1e7,
        },
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
    }
    cfg = dest / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3_8", "text_config": config}))
    _, dims, _ = read_config(cfg, 128, 64)
    gen = torch.Generator().manual_seed(4218)
    index = {}
    for owner, fields in schemas(dims).items():
        prefix = (
            PREFIX + "layers." + str(["gdn", "full_attn"].index(owner)) + "."
            if owner in ("gdn", "full_attn")
            else ""
        )
        tensors = {}
        for key, (suffix, shape) in fields.items():
            t = torch.randn(shape, generator=gen) * 0.02
            if key == "gatedNorm":
                t.fill_(1)
            if key == "lmHead":
                t.mul_(0.01)
                for i in range(4):
                    t[100 + i, i] = 1
            if key == "table":
                for i in range(4):
                    t[100 + i, (i + 1) % 4] = 1
            tensors[prefix + suffix] = t.to(torch.float16)
        filename = owner + ".safetensors"
        save_file(tensors, dest / filename)
        index.update(dict.fromkeys(tensors, filename))
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    make(p.parse_args().output)
