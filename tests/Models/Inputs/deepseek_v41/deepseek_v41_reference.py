"""Run the supplied upstream text backbone on CPU with deterministic packed weights."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "tools/buddy-codegen")
)
import deepseek_v4_cpu as kernels


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream(spec):
    engram = load_module("v41_spec_engram", spec / "inference_engram.py")
    # Upstream imports vision at module scope even for a text-only backbone.
    vision = ModuleType("vision")
    vision.Aligner = vision.ViT = None
    image = ModuleType("image_processor")
    image.IMAGE, image.IMAGE_END, image.IMAGE_NEW_LINE, image.IMAGE_START = (
        range(4)
    )
    with patch.dict(
        sys.modules,
        kernel=kernels,
        engram=engram,
        vision=vision,
        image_processor=image,
    ):
        return load_module(
            "v41_spec_model", spec / "inference_model.py"
        ), engram


class TinyTokenizer:
    def __init__(self, n):
        self.backend_tokenizer = Tokenizer(
            models.WordLevel({f"tok{i}": i for i in range(n)})
        )

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()


def tiny(spec):
    model, engram = upstream(spec)
    args = model.ModelArgs(
        max_seq_len=64,
        max_batch_size=1,
        vocab_size=128,
        temperature=0,
        dim=128,
        n_layers=6,
        n_mtp_layers=0,
        hc_mult=4,
        n_heads=8,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=64,
        o_groups=4,
        o_lora_rank=32,
        window_size=16,
        compress_ratios=(0, 2, 2, 1, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 5),
        candidate_source_layer=3,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=4,
        moe_inter_dim=128,
        swiglu_limit=10.0,
        engram_layer_ids=(1, 4),
        engram_max_ngram_size=4,
        engram_n_heads=4,
        engram_head_dim=32,
        engram_vocab_size=37,
        engram_compressed_vocab_size=128,
        route_scale=1.5,
        original_seq_len=32,
        rope_factor=16,
        compress_rope_theta=160000,
    )
    layout = engram.EngramLayout.from_args(args)
    args.engram_num_embeddings = tuple(
        sum(map(sum, layer)) for layer in layout.primes
    )
    net = model.Transformer(args, TinyTokenizer(args.vocab_size))
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.dtype == kernels.E8:
                p.view(torch.uint8).random_(119, 122)
            elif p.dtype == kernels.F4:
                p.view(torch.uint8).random_(0, 256)
            elif p.dtype == kernels.E4:
                p.copy_(torch.randn(p.shape, dtype=torch.float32) * 2)
            elif (
                "norm.weight" in name
                or "q_weight" in name
                or "k_weight" in name
            ):
                p.fill_(1)
            else:
                p.copy_(torch.randn(p.shape, dtype=torch.float32) * 0.03)
    return model, net


@torch.inference_mode()
def execute(model, net, cuts):
    """A stage owns its caches; copy only live shared state at each handoff."""
    ids = ((torch.arange(37) * 13 + 5) % 128)[None]
    states = [model.SharedAttentionRuntime() for _ in range(len(cuts) + 1)]
    traces, tokens = {}, []
    for start in [0, *range(37, 53)]:
        hashes = net.engram_hash(ids, start)
        h = net.embed(ids).unsqueeze(2).repeat(1, 1, 4, 1)
        pre = model.make_identity_pre_mix(h, 4)
        stage = 0
        for i, layer in enumerate(net.layers):
            if i in cuts:
                previous = states[stage]
                stage += 1
                h, pre = h.clone(), pre.clone()
                if layer.attn.compress_ratio and not layer.attn.is_kv_source:
                    for key in (
                        "index_k",
                        "compress_kv",
                        "topk_idxs",
                        "candidates",
                    ):
                        value = getattr(previous, key)
                        setattr(
                            states[stage],
                            key,
                            None if value is None else value.clone(),
                        )
            model.shared_attn = states[stage]
            if layer.engram:
                h = layer.engram(
                    h, hashes[:, :, layer.engram.layer_hash_index, :]
                )
            h, pre = layer(h, start, pre, None)
            prefix = f"{start}/{i}/"
            traces[prefix + "h"] = h.clone()
            traces[prefix + "pre_mix"] = pre.clone()
            if layer.attn.compress_ratio:
                for key in (
                    "index_k",
                    "compress_kv",
                    "topk_idxs",
                    "candidates",
                ):
                    value = getattr(model.shared_attn, key)
                    if value is not None and (key != "candidates" or i >= 3):
                        if key in ("index_k", "compress_kv"):
                            value = value[
                                :,
                                : (start + ids.shape[1])
                                // layer.attn.compress_ratio,
                            ]
                        traces[prefix + key] = value.clone()
        logits = net.head(net.norm(layer.hc_pre(h, pre)))
        traces[f"{start}/logits"] = logits
        ids = logits.argmax(-1, keepdim=True)
        tokens.append(ids.item())
    return traces, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    torch.manual_seed(4106)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    model, net = tiny(args.spec)
    base, tokens = execute(model, net, ())
    if not all(t.isfinite().all() for t in base.values()):
        raise ValueError("nonfinite reference trace")
    if args.trace:
        torch.save(
            {"traces": base, "tokens": tokens, "weights": net.state_dict()},
            args.trace,
        )
    for cuts in ((1, 3), (2,)):
        current, ids = execute(model, net, cuts)
        assert current.keys() == base.keys() and ids == tokens
        for k, t in current.items():
            if not torch.equal(
                t.contiguous().view(torch.uint8),
                base[k].contiguous().view(torch.uint8),
            ):
                raise AssertionError(k)
    # Independently call the upstream Transformer entry point for every step.
    ids = ((torch.arange(37) * 13 + 5) % 128)[None]
    for j, pos in enumerate([0, *range(37, 53)]):
        token, logits, _ = net(ids, pos)
        assert (
            torch.equal(logits, base[f"{pos}/logits"])
            and token.item() == tokens[j]
        )
        ids = token[:, None]
    print(
        json.dumps(
            {
                "seed": 4106,
                "prefill": 37,
                "decode_steps": 16,
                "tokens": tokens,
                "tensor_checks_per_partition": len(base),
                "partition_max_abs": [0, 0],
                "upstream_logits_max_abs": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
