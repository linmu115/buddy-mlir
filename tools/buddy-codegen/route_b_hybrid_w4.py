"""Packed projection planning, streaming and lowering for Route B hybrid layers."""

import re
import subprocess
from contextlib import ExitStack

from build_route_b import ATTRS, elide_final_workshare_barriers
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
        "mtpFC",
    }
)


def quantized_fields(schema):
    fields = {
        name: field for name, field in schema.items() if name in Q4_PROJECTIONS
    }
    if "gateProj" in fields:
        gate, shape = fields.pop("gateProj")
        up, up_shape = fields.pop("upProj")
        if shape != up_shape:
            raise ValueError("incompatible GLU projections")
        fields["glu"] = ((gate, up), shape)
    return fields


def offsets(schema, dual):
    from buddy.compiler.graph.transform import k3_w4

    result, size = {}, 0
    for name, (_, (n, k)) in schema.items():
        if k % k3_w4.G or n % k3_w4.NB:
            raise ValueError(f"{name} [{n}, {k}] violates K3 tile dimensions")
        length = k3_w4.packed_bytes(k, n) * (2 if name == "glu" else 1)
        result[name] = (size, size + length if dual else size, length)
        size += length * (2 if dual else 1)
    return result, size


def pack(path, checkpoint, index, prefix, schema, positions):
    from buddy.compiler.graph.transform import k3_w4

    with path.open("wb") as out:
        for name, (field, (n, k)) in schema.items():
            base, ime, _ = positions[name]
            fused = name == "glu"
            with ExitStack() as files:
                slices = []
                for suffix in field if fused else (field,):
                    key = prefix + suffix
                    shard = files.enter_context(
                        safe_open(checkpoint / index[key], framework="pt")
                    )
                    source = shard.get_slice(key)
                    if tuple(source.get_shape()) != (n, k):
                        raise ValueError(f"{key}: expected {(n, k)}")
                    slices.append(source)
                slab = max(128, (1 << 22) // (k * 128 * len(slices)) * 128)
                for i in range(0, n, slab):
                    arrays = [
                        src[i : i + slab].float().numpy().T for src in slices
                    ]
                    offset = k3_w4.packed_bytes(k, i)
                    if base == ime:
                        tiles = (k3_w4.pack_glu if fused else k3_w4.pack_q4)(
                            *arrays
                        )
                    else:
                        tiles, matrix = (
                            k3_w4.pack_glu_both if fused else k3_w4.pack_q4_both
                        )(*arrays)
                        # IME keeps all gate blocks before all up blocks, even
                        # when the exporter reads them from different shards.
                        out.seek(ime + offset)
                        if fused:
                            half = len(matrix) // 2
                            out.write(matrix[:half])
                            out.seek(ime + k3_w4.packed_bytes(k, n) + offset)
                            matrix = matrix[half:]
                        out.write(matrix)
                    out.seek(base + len(slices) * offset)
                    out.write(tiles)


def projection_code(args, owner, schema, positions, *, fuse_residual=True):
    from buddy.compiler.graph.transform import k3_w4

    cross = any("target=riscv64" in flag for flag in args.cxx_flag)
    specs = {}
    code = []
    for name, (_, (n, k)) in schema.items():
        tiles, ime, size = positions[name]
        branches = []
        for rows in (
            [1, 2, 3, 4] if owner == "head" else [1, 2, 3, 4, args.chunk]
        ):
            matrix = tiles != ime and rows == args.chunk
            spec = k3_w4.kernel_spec(
                "glu" if name == "glu" else "plain",
                rows,
                k,
                [n, n] if name == "glu" else [n],
                False,
                args.threads,
                matrix,
            )
            residual = fuse_residual and name in ("mixerOut", "downProj")
            if residual:
                spec.update(residual=True, residual_rhs=True)
                spec["name"] += "_res_rhs"
            if (
                cross
                and not matrix
                and name != "lmHead"
                and (rows == 2 or (rows == 1 and name == "glu"))
            ):
                spec["strip_lanes"] = 64
            if matrix and name == "glu":
                spec["narrow_ime_epilogue"] = True
            if rows in (3, 4):
                spec["verify"] = True
            symbol = spec["name"]
            extra_arg = ", Descriptor<2>*" if residual else ""
            res_desc = (
                f"Descriptor<2> rhs(residual, {{rows, {n}}});"
                if residual
                else ""
            )
            if symbol not in specs:
                code.append(
                    f'extern "C" void _mlir_ciface_{symbol}(Descriptor<2>*, Descriptor<2>*, Descriptor<1>*{extra_arg});'
                )
            specs[symbol] = spec
            branches.append(f"""
    if (rows == {rows}) {{
      KernelCall call;
      Descriptor<2> result(nullptr, {{rows, {n}}}), input(x, {{rows, {k}}});
      Descriptor<1> packed(w.q4 + {ime if matrix else tiles}, {{{size}}});
      {res_desc}
      _mlir_ciface_{symbol}(&result, &input, &packed{", &rhs" if residual else ""});
      std::copy_n(static_cast<const float*>(result.aligned) + result.offset,
                  rows * {n}, out);
    }}""")
        code.append(
            f"inline void proj_{name}(const BJ47WeightsV1 &w, const float *x, float *out, int rows{', const float *residual' if residual else ''}) {{"
            + "".join(branches)
            + "\n}"
        )
    if fuse_residual and "glu" in schema:
        code.insert(0, "#define HYBRID_FUSED_RESIDUAL 1")
    if "glu" in schema:
        code.insert(0, "#define HYBRID_FUSED_GLU 1")
    return "\n".join(code), list(specs.values())


def compile_kernels(args, dest, specs):
    from buddy.compiler.graph.transform import k3_w4

    cross = any("target=riscv64" in flag for flag in args.cxx_flag)
    objects = []
    parts = [("main", specs)]
    if cross:
        parts = [
            ("main", [s for s in specs if not s.get("verify")]),
            ("verify", [s for s in specs if s.get("verify")]),
        ]
    if any(s["ime"] for s in specs):
        parts.append(("ime", specs))
    for part, selected in parts:
        if not selected:
            continue
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
        if cross and part != "ime":
            # Otherwise LLVM combines adjacent scalar descriptor fields into
            # an RVV register and spills it across the arena allocation call.
            stages[-1][1].append("-combiner-store-merging=false")
        path = dest / f"w4-{part}.mlir"
        path.write_text(
            k3_w4.gen_kernels(
                selected,
                part="ime" if part == "ime" else "main",
                fixed_vlen=cross,
            )
        )
        obj = path.with_suffix(".o")
        if part in ("main", "verify"):
            lowered = dest / f"w4-{part}-workshares.mlir"
            run_pipeline(
                stages[:1], path, lowered, args.buddy_opt, args.llvm_bin
            )
            elide_final_workshare_barriers(lowered)
            path, stages = lowered, stages[1:]
        if not cross and any(
            "fsanitize=address" in flag for flag in args.cxx_flag
        ):
            # llc alone does not add redzone checks to generated memory accesses.
            llvm = path.with_suffix(".ll")
            run_pipeline(stages[:1], path, llvm, args.buddy_opt, args.llvm_bin)
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
