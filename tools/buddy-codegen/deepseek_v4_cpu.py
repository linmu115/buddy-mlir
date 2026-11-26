"""CPU implementations of the six text-inference kernel entry points."""

import torch

E4 = torch.float8_e4m3fn
E8 = torch.float8_e8m0fnu
F4 = torch.float4_e2m1fn_x2
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pow2ceil(x):
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), exponent - (mantissa == 0.5).int())


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):
    z = x.float().unflatten(-1, (-1, block_size))
    s = z.abs().amax(-1, keepdim=True).clamp_min(1e-4) * (1 / 448)
    if scale_fmt is not None:
        s = pow2ceil(s)
    q = (z / s).clamp(-448, 448).to(E4)
    if inplace:
        return x.copy_((q.float() * s).flatten(-2))
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_values(q):
    raw = q.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    mag = torch.tensor(LEVELS, dtype=torch.float32)[(codes & 7).long()]
    return torch.where(codes >= 8, -mag, mag)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=E8):
    z = x.float().unflatten(-1, (-1, block_size))
    peak = z.abs().amax(-1, keepdim=True)
    if scale_dtype == E4:
        s = (peak.clamp_min(6 * 2**-9) / 6).to(E4).float()
    else:
        s = pow2ceil(peak.clamp_min(6 * 2**-126) * (1 / 6))
    val = (z / s).clamp(-6, 6)
    # Round halfway values to the even nibble (not always the lower one).
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    mids = (levels[1:] + levels[:-1]) * 0.5
    idx = torch.bucketize(val.abs().contiguous(), mids)
    idx += (
        (idx % 2 == 1) & (val.abs() == mids[idx.clamp_max(6)]) & (idx < 7)
    ).long()
    if inplace:
        return x.copy_((torch.copysign(levels[idx], val) * s).flatten(-2))
    codes = (idx.byte() | (val.signbit().byte() << 3)).flatten(-2)
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).view(F4), s.squeeze(
        -1
    ).to(scale_dtype)


def _gemm(a, sa, w, sw, group, act_group, row_group):
    out = torch.zeros((*a.shape[:-1], w.shape[0]), dtype=torch.float32)
    rows = torch.arange(w.shape[0]) // row_group
    for i, (x, y) in enumerate(
        zip(a.float().split(group, -1), w.split(group, -1))
    ):
        prod = x @ y.T
        out += prod * sa[..., i * group // act_group, None] * sw[rows, i]
    return out.to(torch.get_default_dtype())


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    return _gemm(
        a,
        a_s.float(),
        b.float(),
        b_s.float(),
        block_size,
        block_size,
        block_size,
    )


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    return _gemm(
        a, a_s.float(), fp4_values(b), b_s.float(), 32, act_block_size, 1
    )


def hc_split_sinkhorn(
    mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    pre, post, comb = mixes.split((hc_mult, hc_mult, hc_mult**2), -1)
    a, b, c = hc_base.split((hc_mult, hc_mult, hc_mult**2))
    pre = (pre * hc_scale[0] + a).sigmoid() + eps
    post = 2 * (post * hc_scale[1] + b).sigmoid()
    comb = (comb * hc_scale[2] + c).unflatten(-1, (hc_mult, hc_mult))
    # TileLang materializes exp, then divides by the row sum before adding eps.
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    maximum = torch.full(q.shape[:-1] + (1,), -1e30, dtype=torch.float32)
    denom = torch.zeros_like(maximum)
    out = torch.zeros_like(q, dtype=torch.float32)
    batch = torch.arange(q.size(0))[:, None, None]
    for ids in topk_idxs.split(64, -1):
        if not ids.size(-1):
            continue
        vals = kv[batch, ids.clamp_min(0).long()].float()
        vals.masked_fill_((ids < 0).unsqueeze(-1), 0)
        scores = (q.float() @ vals.transpose(-1, -2)) * softmax_scale
        scores.masked_fill_((ids < 0).unsqueeze(-2), -torch.inf)
        current = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        adjust = (maximum - current).exp()
        weights = (scores - current).exp()
        denom = denom * adjust + weights.sum(-1, keepdim=True)
        out = out * adjust + weights.bfloat16().float() @ vals
        maximum = current
    denom += (attn_sink[:, None] - maximum).exp()
    return (out / denom).to(q.dtype)
