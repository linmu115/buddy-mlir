"""Decode checkpoint row slabs and emit Route B K3 projection layouts."""

import json
import os
from pathlib import Path

import numpy as np

MAG = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
E2M1 = np.concatenate((MAG, -MAG))


def raw_bytes(x):
    if hasattr(x, "detach"):
        import torch

        return x.detach().view(torch.uint8).numpy()
    x = np.asarray(x)
    if x.dtype != np.uint8:
        raise ValueError("expected raw uint8 storage")
    return x


def e4m3_lut():
    code = np.arange(256, dtype=np.int32)
    exp, frac = (code >> 3) & 15, code & 7
    val = np.ldexp(
        np.where(exp == 0, frac, 8 + frac).astype(np.float32),
        np.maximum(exp, 1) - 10,
    )
    val = np.where(code & 128, -val, val)
    val[(code & 127) == 127] = np.nan
    return val


E4M3 = e4m3_lut()


def decode_rows(data, scales, fmt, block, start, stop):
    """Decode only [start, stop) of a sliceable tensor or numpy memmap."""
    n, width = data.get_shape() if hasattr(data, "get_shape") else data.shape
    if not (0 <= start <= stop <= n):
        raise ValueError("invalid row interval")
    if (fmt, block) not in {("fp4", 32), ("fp8", 32), ("fp8", 128)}:
        raise ValueError("unsupported weight encoding")
    k = width * (2 if fmt == "fp4" else 1)
    rows_per_scale = 1 if fmt == "fp4" else block
    expected = (
        (n + rows_per_scale - 1) // rows_per_scale,
        (k + block - 1) // block,
    )
    shape = scales.get_shape() if hasattr(scales, "get_shape") else scales.shape
    if tuple(shape) != expected:
        raise ValueError(f"expected scale shape {expected}, got {shape}")
    q = raw_bytes(data[start:stop])
    if fmt == "fp4":
        val = E2M1[np.stack((q & 15, q >> 4), axis=-1)].reshape(stop - start, k)
    else:
        val = E4M3[q]
    first = start // rows_per_scale
    sc = raw_bytes(
        scales[first : (stop + rows_per_scale - 1) // rows_per_scale]
    )
    sc = sc[np.arange(start, stop) // rows_per_scale - first]
    factors = np.ldexp(np.ones(sc.shape, np.float32), sc.astype(np.int32) - 127)
    factors[sc == 255] = np.nan
    return val * factors[:, np.arange(k) // block]


def repack(out, tensors, fmt, block=32, slab=128, ime=False):
    """Write one projection or interleaved gate/up, optionally in IME format.

    tensors contains (data, scale) pairs, e.g. safetensors get_slice handles.
    Output starts at the file's current position.
    """
    from buddy.compiler.graph.transform import k3_w4 as k3

    if len(tensors) not in (1, 2) or slab <= 0 or slab % k3.NB:
        raise ValueError(
            "expected one or two projections and tile-aligned slabs"
        )

    def shape(t):
        return tuple(t.get_shape() if hasattr(t, "get_shape") else t.shape)

    n, width = shape(tensors[0][0])
    k = width * (2 if fmt == "fp4" else 1)
    if k % k3.G or n % k3.NB or any(shape(t) != (n, width) for t, _ in tensors):
        raise ValueError("incompatible K3 projection dimensions")
    base = out.tell()
    length = k3.packed_bytes(k, n)
    for i in range(0, n, slab):
        matrices = [
            decode_rows(t, s, fmt, block, i, min(n, i + slab)).T
            for t, s in tensors
        ]
        if any(not np.isfinite(m).all() for m in matrices):
            raise ValueError("nonfinite projection cannot be packed")
        if not ime:
            out.write(
                (k3.pack_glu if len(matrices) == 2 else k3.pack_q4)(*matrices)
            )
        else:
            for j, m in enumerate(matrices):
                out.seek(base + j * length + k3.packed_bytes(k, i))
                out.write(k3.pack_ime(*k3.quantize_q4(m)))
    out.seek(base + len(tensors) * length)


class CheckpointShard:
    """Map encoded bytes using safetensors offsets without creating a full torch tensor."""

    def __init__(self, file):
        self.file = file
        self.size = os.fstat(file.fileno()).st_size
        length = int.from_bytes(file.read(8), "little")
        if not 0 < length <= min(self.size - 8, 100_000_000):
            raise ValueError("invalid safetensors header length")
        self.header = json.loads(file.read(length))
        self.base = 8 + length

    def matrix(self, key, encoding):
        item = self.header[key]
        dtype = item["dtype"]
        allowed = {
            "fp4": {"F4", "I8", "U8"},
            "fp8": {"F8_E4M3", "I8", "U8"},
            "scale": {"F8_E8M0", "U8"},
        }
        if dtype not in allowed[encoding]:
            raise ValueError(f"{key}: incompatible {dtype} for {encoding}")
        rows, width = item["shape"]
        if rows <= 0 or width <= 0 or (dtype == "F4" and width % 2):
            raise ValueError(f"{key}: invalid matrix shape")
        # Native F4 headers carry logical K. Byte-carrier I8/U8 headers
        # already carry K/2. Both expose the same packed row layout here.
        if dtype == "F4":
            width //= 2
        begin, end = item["data_offsets"]
        if (
            begin < 0
            or end - begin != rows * width
            or self.base + end > self.size
        ):
            raise ValueError(f"{key}: inconsistent tensor bounds")
        return np.memmap(
            self.file,
            mode="r",
            dtype=np.uint8,
            offset=self.base + begin,
            shape=(rows, width),
        )


def main():
    import argparse
    from contextlib import ExitStack
    from tempfile import NamedTemporaryFile

    parser = argparse.ArgumentParser(
        description="Stream one K3 projection or fused gate/up pair from safetensors."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--matrix",
        nargs=4,
        action="append",
        required=True,
        metavar=("SHARD", "WEIGHT", "SCALE_SHARD", "SCALE"),
    )
    parser.add_argument("--format", choices=("fp4", "fp8"), required=True)
    parser.add_argument("--block", type=int, choices=(32, 128), default=32)
    parser.add_argument("--slab", type=int, default=128)
    parser.add_argument("--ime", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        shards = {}

        def get(path, key, encoding):
            path = Path(path).resolve()
            if path == args.output.resolve():
                raise ValueError("output would replace an input shard")
            if path not in shards:
                shards[path] = CheckpointShard(
                    stack.enter_context(path.open("rb"))
                )
            return shards[path].matrix(key, encoding)

        tensors = [
            (get(p, k, args.format), get(s, v, "scale"))
            for p, k, s, v in args.matrix
        ]
        with NamedTemporaryFile(dir=args.output.parent, delete=False) as out:
            tmp = Path(out.name)
            stack.callback(tmp.unlink, missing_ok=True)
            repack(out, tensors, args.format, args.block, args.slab, args.ime)
        tmp.replace(args.output)


if __name__ == "__main__":
    main()
