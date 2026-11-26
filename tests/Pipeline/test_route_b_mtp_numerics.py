"""Check native f32 MTP hidden rows, cache and draft logits against pinned HF ops."""

import argparse
import ctypes as ct
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from qwen35_route_b_reference import (
    PREFIX,
    PROMPTS,
    TextCache,
    load_torch_fallback,
    prompt_ids,
    reference,
)
from safetensors.torch import load_file


class Weights(ct.Structure):
    _fields_ = [("f32", ct.c_void_p), ("q4", ct.c_void_p)]


class Frame(ct.Structure):
    _fields_ = [
        ("position", ct.c_int64),
        ("cos", ct.c_void_p),
        ("sin", ct.c_void_p),
        ("aux", ct.c_void_p),
    ]


def mapped(path):
    return torch.from_file(str(path), size=path.stat().st_size // 4)


@torch.no_grad()
def run(checkpoint, bundle, spec, length):
    c = json.loads((checkpoint / "config.json").read_text())["text_config"]
    config = SimpleNamespace(**c, _attn_implementation="eager")
    ops = load_torch_fallback(spec / "modeling_qwen3_5.py")
    layer = ops["Qwen3_5DecoderLayer"](config, 3)
    rope = ops["Qwen3_5TextRotaryEmbedding"](config)
    wt = load_file(checkpoint / "mtp.safetensors")
    prefix = "mtp.layers.0."
    layer.load_state_dict(
        {k[len(prefix) :]: v for k, v in wt.items() if k.startswith(prefix)}
    )

    def norm(x, w):
        return (
            x
            * torch.rsqrt(
                x.square().mean(-1, keepdim=True) + config.rms_norm_eps
            )
            * (1 + w)
        )

    ck = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    head = load_file(checkpoint / ck["lm_head.weight"])
    emb = load_file(checkpoint / ck[PREFIX + "embed_tokens.weight"])[
        PREFIX + "embed_tokens.weight"
    ].half()
    h = config.hidden_size
    trunk = reference(spec, checkpoint, "f16", length, 5)
    ids = prompt_ids(PROMPTS[length]) + trunk["tokens"].tolist()
    torch_cache = TextCache()
    kernels = [
        ct.CDLL(str(bundle / name)) for name in ("mtp_layer.so", "lm_head.so")
    ]
    predict = kernels[0].mtp_forward
    predict.argtypes = [ct.c_void_p] * 7 + [ct.c_int]
    linear, normalize = kernels[1].lm_head_linear, kernels[1].lm_head_normalize
    linear.argtypes = normalize.argtypes = [ct.c_void_p] * 3 + [ct.c_int]
    weights = [mapped(bundle / file) for file in ("mtp.f32", "head.f32")]
    mw, hw = [Weights(t.data_ptr(), None) for t in weights]

    def kv():
        return torch.zeros(config.num_key_value_heads, 256, config.head_dim)

    states = [(kv(), kv()), (kv(), kv())]
    peaks = {}

    def compare(key, x, y):
        err = (x - y).abs().max().item()
        assert torch.isfinite(x).all() and err < 1e-4, (key, err)
        peaks[key] = max(peaks.get(key, 0), err)

    pos = 0
    for name, valid, rows in [
        *(
            (f"prefill{i // 64}", min(64, length - i), 64)
            for i in range(0, length, 64)
        ),
        ("decode0", 1, 1),
        ("decode1", 2, 2),
        ("decode3", 1, 1),
    ]:
        raw = trunk[name + ".layer3.hidden"].contiguous()
        if rows == 2:
            raw = torch.cat((raw, trunk["decode2.layer3.hidden"]))
        hidden = norm(raw, head[PREFIX + "norm.weight"])
        e = emb[ids[pos + 1 : pos + 1 + valid]].float()
        x = torch.nn.functional.linear(
            torch.cat(
                (
                    norm(e, wt["mtp.pre_fc_norm_embedding.weight"]),
                    norm(hidden, wt["mtp.pre_fc_norm_hidden.weight"]),
                ),
                dim=-1,
            ),
            wt["mtp.fc.weight"],
        )[None]
        positions = torch.arange(pos, pos + valid)[None]
        cs = rope(x, positions)
        mask = torch.zeros(1, 1, valid, pos + valid).masked_fill(
            torch.arange(pos + valid) > positions.T, -torch.inf
        )
        expected = norm(
            layer(x, cs, mask, cache=torch_cache)[0], wt["mtp.norm.weight"]
        )
        valid_ptr = ct.c_int64(valid)
        co, si = [v.contiguous() for v in cs]
        frame = Frame(
            pos, co.data_ptr(), si.data_ptr(), ct.addressof(valid_ptr)
        )
        observed = []
        for idx, (k, v) in enumerate(states):
            ep = torch.full((rows, h), float(idx * 29))
            hp = ep.clone()
            ep[:valid] = e
            assert (
                normalize(ct.byref(hw), raw.data_ptr(), hp.data_ptr(), valid)
                == 0
            )
            compare("trunk_final_norm", hp[:valid], hidden)
            out = torch.empty_like(ep)
            assert (
                predict(
                    ct.byref(mw),
                    ep.data_ptr(),
                    hp.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    ct.byref(frame),
                    out.data_ptr(),
                    rows,
                )
                == 0
            )
            compare("h_mtp", out[:valid], expected)
            for label, state, ref in zip(("K", "V"), (k, v), torch_cache.kv[3]):
                compare(label, state[:, : pos + valid], ref[0])
                assert not state[:, pos + valid :].count_nonzero()
            logits = torch.empty(config.vocab_size)
            assert (
                linear(
                    ct.byref(hw),
                    out[valid - 1].data_ptr(),
                    logits.data_ptr(),
                    1,
                )
                == 0
            )
            compare(
                "draft_logits", logits, head["lm_head.weight"] @ expected[-1]
            )
            if rows == 2:
                pair = torch.empty(2, config.vocab_size)
                assert (
                    linear(ct.byref(hw), out.data_ptr(), pair.data_ptr(), 2)
                    == 0
                )
                compare(
                    "two_row_logits", pair, expected @ head["lm_head.weight"].T
                )
                assert torch.equal(
                    pair[1].view(torch.int32), logits.view(torch.int32)
                )
            observed.append((out, k, v, logits))
        for a, b in zip(*observed):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        pos += valid
    return {"prompt": length, "max_abs": peaks, "padding_bit_diff": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("checkpoint", "bundle", "spec"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps([run(**vars(args), length=n) for n in (64, 73)]))
