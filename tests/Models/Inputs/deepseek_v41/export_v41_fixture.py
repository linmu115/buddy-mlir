"""Export named tensors and all reference checkpoints for the C++ differential runner."""

import argparse
from pathlib import Path

import torch
from deepseek_v41_reference import execute, kernels, tiny


def tensor_file(root, name, tensor, index):
    t = tensor.detach().contiguous()
    dims = list(t.shape)
    if t.dtype in (kernels.F4, kernels.E4, kernels.E8):
        kind = {kernels.F4: "fp4", kernels.E4: "fp8", kernels.E8: "scale"}[
            t.dtype
        ]
        data = t.view(torch.uint8)
    elif t.is_floating_point():
        kind, data = "float", t.float()
    else:
        kind, data = "int", t.long()
    filename = name.replace("/", "_") + ".bin"
    data.numpy().tofile(root / filename)
    print(name, filename, kind, len(dims), *dims, file=index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(4106)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    mod, net = tiny(args.spec)
    with (args.out / "weights.txt").open("w") as f:
        for k, v in net.state_dict().items():
            tensor_file(args.out, k, v, f)
        for k, v in net.engram_hash.named_buffers():
            if k != "cache":
                tensor_file(args.out, "hash." + k, v, f)
        for layer in net.layers:
            freq = layer.attn.freqs_cis
            tensor_file(
                args.out, f"freq.{layer.layer_id}", torch.view_as_real(freq), f
            )
    # Capture isolated inputs at the official module boundaries. The C++
    # replay consumes each module's reference input, so earlier numerical
    # differences cannot compensate for a later operator error.
    samples = {}
    hooks = []
    pos = 0

    def save(label):
        def hook(module, args, out):
            base = f"op/{pos}/{label}"
            samples[base + "/in"] = args[0].clone()
            samples[base + "/out"] = out.clone()

        return hook

    for i, layer in enumerate(net.layers):
        for label in ("attn", "ffn", "engram"):
            op = getattr(layer, label)
            if op is not None:
                hooks.append(op.register_forward_hook(save(f"{i}/{label}")))
    ids = ((torch.arange(37) * 13 + 5) % 128)[None]
    for pos in (0, 37, 38):
        samples[f"ids/{pos}"] = ids.clone()
        token, _, _ = net(ids, pos)
        ids = token[:, None]
    for hook in hooks:
        hook.remove()
    stream = torch.randn(1, 1, 4, 128, dtype=torch.float32) * 1e-18
    block = net.layers[0]
    samples["mhc/in"] = stream
    pre, post, comb = block.hc_mixes(
        stream, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    for k, v in [("pre", pre), ("post", post), ("comb", comb)]:
        samples["mhc/" + k] = v
    with (args.out / "operators.txt").open("w") as f:
        for k, v in samples.items():
            tensor_file(args.out, k, v, f)
    traces, _ = execute(mod, net, ())
    with (args.out / "expected.txt").open("w") as f:
        for k, v in traces.items():
            tensor_file(args.out, k, v, f)


if __name__ == "__main__":
    main()
