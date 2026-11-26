"""Add a one-layer predictor to an indexed tiny checkpoint for B4 testing."""

import argparse
import json
import os
from pathlib import Path

import torch
from qwen35_route_b_reference import PREFIX, read_config, schemas
from safetensors.torch import load_file, save_file


def build(base, dest, scenario, seed):
    dest.mkdir()
    index_name = "model.safetensors.index.json"
    index = json.loads((base / index_name).read_text())["weight_map"]
    for p in base.iterdir():
        if p.name != index_name:
            os.link(p.resolve(), dest / p.name)
    _, dims, _ = read_config(base / "config.json", 256, 64)
    gen = torch.Generator().manual_seed(seed)
    weights = {
        key: torch.randn(shape, generator=gen) * 0.05
        for key, shape in schemas(dims)["mtp"].values()
    }
    if scenario != "random":
        # Four possible winners rotate. Strong one-hot embeddings keep the
        # original, nonzero GDN and attention layers active on every step.
        h = dims["width"]
        for key in (PREFIX + "embed_tokens.weight", "lm_head.weight"):
            path = dest / index[key]
            shard = load_file(path)
            t = shard[key].clone()
            if key == "lm_head.weight":
                t.zero_()
            for i in range(4):
                t[100 + i].zero_()
                t[100 + i, (i + 1) % 4 if key.startswith(PREFIX) else i] = 32
            shard[key] = t
            path.unlink()
            save_file(shard, path)
        if scenario == "accept":
            weights["mtp.fc.weight"].zero_()
            weights["mtp.fc.weight"][:, :h] = torch.eye(h)
    save_file(weights, dest / "mtp.safetensors")
    index.update(dict.fromkeys(weights, "mtp.safetensors"))
    (dest / index_name).write_text(json.dumps({"weight_map": index}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument(
        "--scenario", choices=("random", "accept", "mixed"), default="random"
    )
    p.add_argument("--seed", type=int, default=4747)
    args = p.parse_args()
    build(args.base, args.dest, args.scenario, args.seed)
