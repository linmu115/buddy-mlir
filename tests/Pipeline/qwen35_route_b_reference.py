"""Build the deterministic B1 checkpoint and run the locked HF CPU fallback."""

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
)
from build_route_b_hybrid import PREFIX, read_config, schemas  # noqa: E402

SPECIAL = {
    248044: "<|endoftext|>",
    248045: "<|im_start|>",
    248046: "<|im_end|>",
    248068: "<think>",
    248069: "</think>",
}
PROMPTS = {64: "a" * 56, 73: "b" * 65, 32: "c" * 24}
VOCAB_IDS = {"user": 256, "assistant": 257, "Ċ": 258}


def prompt_ids(prompt):
    return [
        248045,
        256,
        258,
        *prompt.encode("ascii"),
        248046,
        258,
        248045,
        257,
        258,
    ]


def fixture(spec, path):
    config = json.loads((spec / "qwen38_27b_config.json").read_text())
    c = config["text_config"]
    c.update(
        hidden_size=256,
        intermediate_size=512,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=4,
        layer_types=c["layer_types"][:4],
        max_position_embeddings=256,
        dtype="float32",
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _, dims, types = read_config(path / "config.json", 256, 64)
    schema = schemas(dims)
    real = json.loads((spec / "qwen38_27b_index.json").read_text())[
        "weight_map"
    ]
    generator = torch.Generator().manual_seed(474738)

    def make(owner, layer=None):
        prefix = "" if layer is None else f"{PREFIX}layers.{layer}."
        tensors = {}
        for name, shape in schema[owner].values():
            key = prefix + name
            if key not in real:
                raise ValueError(
                    f"test tensor absent from B0 checkpoint index: {key}"
                )
            scale = 0.05 if len(shape) == 1 else 0.035
            if ".conv1d." in name:
                scale = 0.35
            shift = 1.0 if name == "linear_attn.norm.weight" else 0.0
            tensors[key] = (
                torch.randn(shape, generator=generator) * scale + shift
            )
        return tensors

    # Use the same indexed reader path as the real 18-shard checkpoint.
    weights = {}
    for owner, layer in [
        ("embed", None),
        ("head", None),
        *((t, i) for i, t in enumerate(types)),
    ]:
        tensors = make(owner, layer)
        shard = f"{owner}-{layer}.safetensors"
        save_file(tensors, path / shard)
        weights.update(dict.fromkeys(tensors, shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}, indent=2) + "\n"
    )
    token_config = json.loads(
        (spec / "qwen38_27b_tokenizer_config.json").read_text()
    )
    for token_id, token in SPECIAL.items():
        assert (
            token_config["added_tokens_decoder"][str(token_id)]["content"]
            == token
        )
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    byte_tokens = dict(
        zip(sorted(set(range(256)) - visible), map(chr, range(256, 512)))
    )
    vocab = [chr(i) if i in visible else byte_tokens[i] for i in range(256)]
    vocab += [""] * (c["vocab_size"] - len(vocab))
    for token, token_id in VOCAB_IDS.items():
        vocab[token_id] = token
    for token_id, token in SPECIAL.items():
        vocab[token_id] = token
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")


NAMES = {
    "Qwen3_5TextRotaryEmbedding",
    "Qwen3_5RMSNormGated",
    "apply_mask_to_padding_states",
    "causal_conv1d_update",
    "causal_conv1d_fn",
    "causal_conv1d_forward",
    "l2norm",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "Qwen3_5GatedDeltaNet",
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "eager_attention_forward",
    "Qwen3_5Attention",
    "Qwen3_5MLP",
    "Qwen3_5RMSNorm",
    "Qwen3_5DecoderLayer",
}


def load_torch_fallback(path):
    """Run the pinned bodies directly, dropping optional integration decorators."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in NAMES]
    if {node.name for node in nodes} != NAMES:
        raise ValueError("B0 model source is missing required CPU fallbacks")
    for node in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            node.decorator_list = [
                deco
                for deco in node.decorator_list
                if isinstance(deco, ast.Name) and deco.id == "staticmethod"
            ]
    future = ast.parse("from __future__ import annotations").body[0]
    code = compile(
        ast.Module(body=[future, *nodes], type_ignores=[]), str(path), "exec"
    )
    scope = {
        "nn": nn,
        "F": F,
        "torch": torch,
        "ACT2FN": {"silu": F.silu},
        "GradientCheckpointingLayer": nn.Module,
        "is_torchdynamo_exporting": lambda: False,
        "ALL_ATTENTION_FUNCTIONS": SimpleNamespace(
            get_interface=lambda _, fallback: fallback
        ),
    }
    exec(
        code, scope
    )  # The supplied B0 HF source is the authority for the oracle.
    return scope


class TextCache:
    """Storage adapter for the HF fallback, with full K/V snapshots for the probe."""

    def __init__(self):
        self.conv = {}
        self.ssm = {}
        self.kv = {}

    def get_recurrent_state(self, layer, state_idx):
        return self.ssm.get(layer)

    def update_recurrent_state(self, state, layer):
        self.ssm[layer] = state

    def get_conv_state(self, layer, state_idx):
        return self.conv.get(layer)

    def is_recording_past(self, layer):
        return False

    def update_conv_state(self, x, layer, state_idx, conv_kernel_size):
        if layer in self.conv:
            x = torch.cat((self.conv[layer], x), dim=-1)
        pad = max(0, conv_kernel_size - x.shape[-1])
        self.conv[layer] = (
            F.pad(x, (pad, 0))[:, :, -conv_kernel_size:].contiguous().clone()
        )
        return x

    def update(self, key, value, layer, **kwargs):
        if layer in self.kv:
            old_k, old_v = self.kv[layer]
            key, value = (
                torch.cat((old_k, key), dim=-2),
                torch.cat((old_v, value), dim=-2),
            )
        self.kv[layer] = (key, value)
        return key, value


@torch.no_grad()
def reference(spec, checkpoint, dtype, rows, steps, whole_prefill=False):
    operators = load_torch_fallback(spec / "modeling_qwen3_5.py")
    config, _, _ = read_config(checkpoint / "config.json", 256, 64)
    d = SimpleNamespace(**config, _attn_implementation="eager")
    weights = {}

    for shard in checkpoint.glob("*.safetensors"):
        weights.update(load_file(shard))
    embed = weights[PREFIX + "embed_tokens.weight"]
    final_norm = operators["Qwen3_5RMSNorm"](d.hidden_size, d.rms_norm_eps)
    final_norm.load_state_dict({"weight": weights[PREFIX + "norm.weight"]})
    rope = operators["Qwen3_5TextRotaryEmbedding"](d)
    layers = [
        operators["Qwen3_5DecoderLayer"](d, i).eval()
        for i in range(d.num_hidden_layers)
    ]
    for i, layer in enumerate(layers):
        prefix = f"{PREFIX}layers.{i}."
        layer.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in weights.items()
                if key.startswith(prefix)
            }
        )

    cache = TextCache()
    token_ids = prompt_ids(PROMPTS[rows])
    assert len(token_ids) == rows
    pos = 0
    snapshots, tokens = {}, []

    def forward(ids, phase, sample):
        nonlocal pos
        x = embed[ids]
        if dtype == "f16":
            x = x.half().float()
        x = x.unsqueeze(0)
        positions = torch.arange(pos, pos + len(ids)).unsqueeze(0)
        cos_sin = rope(x, positions)
        key = torch.arange(pos + len(ids)).unsqueeze(0)
        mask = torch.zeros((1, 1, len(ids), pos + len(ids))).masked_fill(
            key > positions.T, float("-inf")
        )
        for i, layer in enumerate(layers):
            is_gdn = d.layer_types[i] == "linear_attention"
            x = layer(x, cos_sin, None if is_gdn else mask, cache=cache)
            snapshots[f"{phase}.layer{i}.hidden"] = x.squeeze(0).clone()
            if is_gdn:
                snapshots[f"{phase}.layer{i}.conv_state"] = cache.conv[
                    i
                ].clone()
                snapshots[f"{phase}.layer{i}.ssm_state"] = cache.ssm[i].clone()
            else:
                for name, data in zip(("K", "V"), cache.kv[i]):
                    snapshots[f"{phase}.layer{i}.{name}"] = F.pad(
                        data, (0, 0, 0, 256 - data.shape[-2])
                    )
        pos += len(ids)
        if sample:
            logits = F.linear(final_norm(x[0, -1]), weights["lm_head.weight"])
            snapshots[f"{phase}.logits"] = logits
            tokens.append(int(logits.argmax()))

    block = rows if whole_prefill else 64
    for p in range(0, rows, block):
        forward(
            token_ids[p : p + block], f"prefill{p // block}", p + block >= rows
        )
    for i in range(steps - 1):
        forward(tokens[-1:], f"decode{i}", True)
    snapshots["tokens"] = torch.tensor(tokens, dtype=torch.int64)
    return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.fixture:
        fixture(args.spec, args.checkpoint)
