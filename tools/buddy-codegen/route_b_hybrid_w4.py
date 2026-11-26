"""Packed projection planning, streaming and lowering for Route B hybrid layers."""

import re
import subprocess

import torch
from build_route_b import ATTRS
from compile_pipeline import build_stages, run_pipeline
from safetensors import safe_open

# Keep the decay and beta gates in fp32 even on shapes with tile-aligned
# head counts. New schema fields require an explicit quantization decision.
Q4_PROJECTIONS = frozenset(
    {
        "inProjQKV",
        "inProjZ",
        "mixerOut",
        "qProj",
        "kProj",
        "vProj",
        "gateProj",
        "upProj",
        "downProj",
        "lmHead",
    }
)


def quantized_fields(schema):
    return {
        name: field for name, field in schema.items() if name in Q4_PROJECTIONS
    }


def offsets(schema, dual):
    from buddy.compiler.graph.transform import k3_w4

    result, size = {}, 0
    for name, (_, (n, k)) in schema.items():
        if k % k3_w4.G or n % k3_w4.NB:
            raise ValueError(f"{name} [{n}, {k}] violates K3 tile dimensions")
        length = k3_w4.packed_bytes(k, n)
        result[name] = (size, size + length if dual else size, length)
        size += length * (2 if dual else 1)
    return result, size


def pack(path, checkpoint, index, prefix, schema, positions):
    from buddy.compiler.graph.transform import k3_w4

    with path.open("wb") as out:
        for name, (field, (n, k)) in schema.items():
            base, ime, _ = positions[name]
            key = prefix + field
            with safe_open(checkpoint / index[key], framework="pt") as shard:
                source = shard.get_slice(key)
                if tuple(source.get_shape()) != (n, k):
                    raise ValueError(f"{key}: expected {(n, k)}")
                # The tile index is outermost in both layouts. Export bounded
                # output-column slabs, including for the 248K-row head.
                slab = max(128, (1 << 22) // (k * 128) * 128)
                for i in range(0, n, slab):
                    weights = source[i : i + slab].to(torch.float32).numpy().T
                    if base == ime:
                        tiles = k3_w4.pack_q4(weights)
                    else:
                        tiles, matrix = k3_w4.pack_q4_both(weights)
                        out.seek(ime + k3_w4.packed_bytes(k, i))
                        out.write(matrix)
                    out.seek(base + k3_w4.packed_bytes(k, i))
                    out.write(tiles)


def projection_code(args, owner, schema, positions):
    from buddy.compiler.graph.transform import k3_w4

    specs = {}
    code = []
    for name, (_, (n, k)) in schema.items():
        tiles, ime, size = positions[name]
        branches = []
        for rows in [1] if owner == "head" else [1, args.chunk]:
            matrix = tiles != ime and rows == args.chunk
            spec = k3_w4.kernel_spec(
                "plain", rows, k, [n], False, args.threads, matrix
            )
            symbol = spec["name"]
            if symbol not in specs:
                code.append(
                    f'extern "C" void _mlir_ciface_{symbol}(Descriptor<2>*, Descriptor<2>*, Descriptor<1>*);'
                )
            specs[symbol] = spec
            branches.append(f"""
    if (rows == {rows}) {{
      KernelCall call;
      Descriptor<2> result(nullptr, {{rows, {n}}}), input(x, {{rows, {k}}});
      Descriptor<1> packed(w.q4 + {ime if matrix else tiles}, {{{size}}});
      _mlir_ciface_{symbol}(&result, &input, &packed);
      std::copy_n(static_cast<const float*>(result.aligned) + result.offset,
                  rows * {n}, out);
    }}""")
        code.append(
            f"inline void proj_{name}(const BJ47WeightsV1 &w, const float *x, float *out, int rows) {{"
            + "".join(branches)
            + "\n}"
        )
    return "\n".join(code), list(specs.values())


def compile_kernels(args, dest, specs):
    from buddy.compiler.graph.transform import k3_w4

    cross = any("target=riscv64" in flag for flag in args.cxx_flag)
    objects = []
    for part in ["main"] + (["ime"] if any(s["ime"] for s in specs) else []):
        pipeline = (
            "kernels_ime"
            if part == "ime"
            else "kernels_a100"
            if cross
            else "kernels"
        )
        if part == "ime" and not cross:
            raise ValueError("IME requires a RISC-V target")
        stages = build_stages(
            pipeline, args.threads, ATTRS if cross else "", "w4g32", arena=True
        )
        path = dest / f"w4-{part}.mlir"
        path.write_text(k3_w4.gen_kernels(specs, part=part))
        obj = path.with_suffix(".o")
        if not cross and any(
            "fsanitize=address" in flag for flag in args.cxx_flag
        ):
            # llc alone does not add redzone checks to generated memory accesses.
            llvm = path.with_suffix(".ll")
            run_pipeline(stages[:2], path, llvm, args.buddy_opt, args.llvm_bin)
            llvm.write_text(
                re.sub(
                    r"^(define .*) \{$",
                    r"\1 sanitize_address {",
                    llvm.read_text(),
                    flags=re.M,
                )
            )
            subprocess.run(
                [
                    args.llvm_bin / "clang",
                    *args.cxx_flag,
                    "-O2",
                    "-fPIC",
                    "-c",
                    llvm,
                    "-o",
                    obj,
                ],
                check=True,
            )
        else:
            run_pipeline(stages, path, obj, args.buddy_opt, args.llvm_bin)
        objects.append(obj)
    return objects
