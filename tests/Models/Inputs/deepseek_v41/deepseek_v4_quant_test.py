"""Independent decoder and actual K3 packing checks across slab boundaries."""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.graph.transform import k3_w4
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "tools/buddy-codegen")
)
import deepseek_v4_quant as quant
from deepseek_v41_reference import load_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4106)
    levels = (
        load_module("ds_convert", args.spec / "inference_convert.py")
        .FP4_TABLE.numpy()
        .copy()
    )
    levels[8] = -0.0
    scales = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .float()
        .numpy()
    )
    with np.errstate(over="ignore", invalid="ignore"):
        (levels[None, :] * scales[:, None]).tofile(args.out / "e2m1_all")
    all8 = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .numpy()
    )
    for block in (32, 128):
        (
            all8 * scales[np.arange(124, 124 + 256 // block)].repeat(block)
        ).tofile(args.out / f"fp8_{block}")
    n, k = 384, 256
    for fmt, block in (("fp4", 32), ("fp8", 32), ("fp8", 128)):
        raw = rng.integers(
            0, 256, (n, k // 2 if fmt == "fp4" else k), dtype=np.uint8
        )
        # Exercise both E4M3 NaN codes separately; packed weights must be finite.
        if fmt == "fp8":
            allc = torch.arange(256, dtype=torch.uint8)
            np.testing.assert_equal(
                quant.E4M3, allc.view(torch.float8_e4m3fn).float().numpy()
            )
            raw[(raw & 127) == 127] = 0
        sc = rng.integers(
            123,
            129,
            (n if fmt == "fp4" else n // block, k // block),
            dtype=np.uint8,
        )
        vals = (
            levels[np.stack((raw & 15, raw >> 4), -1)].reshape(n, k)
            if fmt == "fp4"
            else torch.from_numpy(raw).view(torch.float8_e4m3fn).float().numpy()
        )
        scale = (
            scales[sc].repeat(1 if fmt == "fp4" else block, 0).repeat(block, 1)
        )
        expected = vals * scale
        path = args.out / "slice.safetensors"
        save_file(
            {"w": raw, "s": sc, "u": raw[::-1].copy(), "us": sc[::-1].copy()},
            path,
        )
        with safe_open(path, framework="numpy") as f:
            pair = (f.get_slice("w"), f.get_slice("s"))
            np.testing.assert_array_equal(
                quant.decode_rows(*pair, fmt, block, 71, 309), expected[71:309]
            )
            for fused in (False, True):
                inputs = [pair]
                w = [expected.T]
                if fused:
                    inputs.append((f.get_slice("u"), f.get_slice("us")))
                    w.append(expected[::-1].T)
                layouts = (
                    k3_w4.pack_glu_both if fused else k3_w4.pack_q4_both
                )(*w)
                for ime in (False, True):
                    out = io.BytesIO()
                    quant.repack(out, inputs, fmt, block, 128, ime)
                    assert out.getvalue() == layouts[ime].tobytes()
                    if fmt == "fp4" and not fused and not ime:
                        (args.out / "tiles").write_bytes(out.getvalue())
        if fmt == "fp4":
            raw.tofile(args.out / "codes")
            sc.tofile(args.out / "scales")
            expected.tofile(args.out / "decoded")
            rng.normal(size=k).astype(np.float32).tofile(args.out / "x")
    # A second oracle exercises the complete routed + shared FFN through
    # f32 and K3 Q8-activation / w4g32-weight projections.
    h, n_expert = 128, 4
    x = torch.from_numpy(rng.normal(size=h).astype(np.float32))
    router = torch.from_numpy(
        rng.normal(0, 0.1, (n_expert, h)).astype(np.float32)
    )
    bias = torch.tensor([2.0, -1.0, -0.5, 1.0])
    score = torch.nn.functional.softplus(router @ x).sqrt()
    ids = (score + bias).topk(2).indices
    coeff = score[ids] / score[ids].sum() * 1.5
    arrays = rng.normal(0, 0.025, (n_expert + 1, 3, h, h)).astype(np.float32)
    arrays[:, 0, 0, 0] = -50
    arrays[:, 1, 0, 0] = 50
    arrays.tofile(args.out / "expert.f32")
    x.numpy().tofile(args.out / "expert.x")
    router.numpy().tofile(args.out / "router.f32")
    with (args.out / "expert.w4").open("wb") as out:
        packed = {}
        for i, expert in enumerate(arrays):
            for j, m in enumerate(expert):
                q, s = k3_w4.quantize_q4(m.T)
                packed[i, j] = torch.from_numpy(
                    k3_w4.dequantize_q4(q, s).T.copy()
                )
                out.write(k3_w4.pack_tiles(q, s))
    for mode in ("f32", "w4"):

        def proj(i, j, v, mode=mode):
            if mode == "f32":
                return torch.from_numpy(arrays[i, j]) @ v
            groups = v.view(-1, 32)
            peak = groups.abs().amax(-1, keepdim=True)
            q = (groups * (127 / peak).nan_to_num()).round().clamp(-127, 127)
            v = (q * (peak / 127)).flatten()
            return packed[i, j] @ v

        y = torch.zeros(h)
        for i in sorted(ids.tolist()) + [n_expert]:
            g = proj(i, 0, x).clamp_max(10.0)
            u = proj(i, 1, x).clamp(-10.0, 10.0)
            v = torch.nn.functional.silu(g) * u
            if i < n_expert:
                v *= coeff[(ids == i).nonzero()[0, 0]]
            y += proj(i, 2, v)
        y.numpy().tofile(args.out / ("ffn." + mode))
    print(
        json.dumps(
            {
                "slab_max_abs": 0,
                "layout_comparisons": 12,
                "layout_byte_diff": 0,
                "seed": 4106,
            }
        )
    )


if __name__ == "__main__":
    main()
