"""Lower a captured component using existing Buddy/LLVM binaries and a cross C++ compiler."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from compile_pipeline import _resolve_tool, build_stages
from route_b_lower import (
    final_projection,
    prepare_ir,
    prepared_attention,
    subgraph,
)

ATTRS = "-march=riscv64 -mattr=+m,+d,+v,+zfh,+zvfh,+zvl1024b -mtriple=riscv64-unknown-linux-gnu -riscv-v-vector-bits-max=1024"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models/deepseek_r1/pipeline/route_b"


def compile_ir(a, name, source, pipeline, cfg):
    path = a.work / (name + ".mlir")
    path.write_text(source)
    stages = build_stages(
        pipeline,
        cfg["compilation"]["num_threads"],
        ATTRS,
        "w4g32",
        arena=True,
        chunked_prefill=True,
    )
    for i, (tool, flags) in enumerate(stages):
        if "-one-shot-bufferize=bufferize-function-boundaries" in flags:
            # Keep the actual bufferized IR for ownership and byte-copy auditing.
            split = flags.index("-expand-strided-metadata")
            snap = a.work / (name + ".bufferized.mlir")
            subprocess.run(
                [
                    a.buddy_opt,
                    path,
                    *flags[:split],
                    "-canonicalize",
                    "-o",
                    snap,
                ],
                check=True,
            )
            last = flags.index("-eliminate-memref-copy") + 1
            post = a.work / (name + ".after-copy-elimination.mlir")
            subprocess.run(
                [a.buddy_opt, path, *flags[:last], "-canonicalize", "-o", post],
                check=True,
            )
            if "memref.copy" in post.read_text():
                raise ValueError(
                    f"unexpected residual copy after destination binding: {post}"
                )
        suffix = (
            ".o"
            if tool == "llc"
            else ".ll"
            if tool.endswith("translate")
            else f".{i}.mlir"
        )
        nxt = a.work / (name + suffix)
        subprocess.run(
            [_resolve_tool(tool, a.buddy_opt, a.llvm), path, *flags, "-o", nxt],
            check=True,
        )
        path = nxt
    return path


def thunk(a, name, symbol, args, signature, checks, prologue=""):
    desc, calls, types = [], [], []
    for i, (ptr, shape) in enumerate(args):
        dims = shape.split("x")
        ty = f"Descriptor<{len(dims)}>"
        types.append(ty + " *")
        desc.append(f"  {ty} d{i}({ptr}, {{{', '.join(dims)}}});")
        calls.append(f"&d{i}")
    llvm = (a.work / (symbol + ".ll")).read_text()
    found = re.findall(
        r"^define void @_mlir_ciface_" + symbol + r"\(([^)]*)\)", llvm, re.M
    )
    if (
        len(found) != 1
        or len(found[0].split(",")) != len(args)
        or not all("ptr " in x for x in found[0].split(","))
    ):
        raise ValueError(f"unexpected lowered ABI for {symbol}: {found}")
    return (
        f'extern "C" void _mlir_ciface_{symbol}({", ".join(types)});\n'
        + f'extern "C" int32_t {name}({signature}) {{\n  if ({checks}) return -1;\n{prologue}\n  KernelCall call;\n'
        + "\n".join(desc)
        + f"\n  _mlir_ciface_{symbol}({', '.join(calls)});\n  return 0;\n}}\n"
    )


def build(a):
    a.work.mkdir(parents=True, exist_ok=True)
    m = json.loads((a.bundle / "components.json").read_text())
    cfg = m["config"]
    shape = cfg["shape"]
    chunk, ctx, dim = (
        cfg["prefill_chunk"],
        shape["max_token_len"],
        shape["hidden_size"],
    )
    heads, vocab, width = (
        shape["head_num"],
        shape["vocab_size"],
        m["activation_width"],
    )
    role = a.role
    src = a.bundle / "mlir" / role
    owner = {"embed": "embed", "decoder_layer": "decoder", "lm_head": "head"}[
        role
    ]
    expected = {
        key: value
        for key, value in m["template_mlir"].items()
        if Path(key).parent == src.relative_to(a.bundle)
    }
    observed = {
        p.relative_to(a.bundle).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in src.glob("*.mlir")
    }
    if not expected or observed != expected:
        raise ValueError(
            f"{role}: template MLIR set/hash differs from components.json"
        )
    count = m["counts"][owner]
    layout = next(w["layout_id"] for w in m["weights"] if w["owner"] == owner)

    def octets(h):
        return "{" + ",".join(str(x) for x in bytes.fromhex(h)) + "}"

    abi = 2 if role == "lm_head" else 1
    fields = [
        abi,
        "sizeof(BJ47KernelInfoV1)",
        ("embed", "decoder_layer", "lm_head").index(role) + 1,
        chunk,
        ctx,
        width,
        heads,
        dim,
        vocab,
        count["f32_elements"],
        count["i8_elements"],
        dim // 2 if role == "decoder_layer" else 0,
        0,
        0,
        64,
        octets(layout),
        octets(m["shape_id"]),
    ]
    cpp = (
        '#include "Shim.h"\n'
        + "static const BJ47KernelInfoV1 info = {"
        + ",".join(map(str, fields))
        + "};\n"
    )
    cpp += 'extern "C" const BJ47KernelInfoV1 *bj47_kernel_info_v1() {return &info;}\n'
    objs, symbols = [], ["bj47_kernel_info_v1"]
    kernels = {p.stem: p.read_text() for p in src.glob("k3_kernels*.mlir")}
    if role == "decoder_layer":
        kernels["k3_kernels-w4g32"] = prepared_attention(
            kernels["k3_kernels-w4g32"], chunk, dim
        )
    for phase in ("prefill", "decode"):
        rows = chunk if phase == "prefill" else 1
        captured = "subgraph0_" + phase
        symbol = f"rb_{role}_{phase}"
        raw = (src / (captured + "-w4g32.mlir")).read_text()
        driver = (src / ("forward_" + phase + "-w4g32.mlir")).read_text()
        ir, args = subgraph(raw, driver, role, phase, rows, dim)
        # Older prefill captures end in a graph residual. A fused MoE
        # can write directly into the caller's disjoint ping-pong buffer.
        fused_prefill = re.search(
            r" = call @k3_moe_m\d+_\w+_res(?:_rhs)?\(", ir
        )
        if role == "lm_head" or (
            role == "decoder_layer" and (rows == 1 or fused_prefill)
        ):
            ir, kernels["k3_kernels-w4g32"] = final_projection(
                ir,
                kernels["k3_kernels-w4g32"],
                1 if role == "lm_head" else rows,
            )
        objs.append(
            compile_ir(
                a,
                symbol,
                ir.replace("@" + captured, "@" + symbol).replace(
                    "@k3_", f"@rb_{role}_k3_"
                ),
                "subgraph_decode" if phase == "decode" else "subgraph",
                cfg,
            )
        )
        pre = ""
        if role == "embed":
            name = "embed_" + phase
            sig = "const float *table, const int64_t *ids, float *out"
            check = "!table || !ids || !out"
            pre = f"  for (int i=0; i<{rows}; ++i) if (ids[i]<0 || ids[i]>={vocab}) return -2;"
        elif role == "decoder_layer":
            name = "layer_" + phase
            sig = "const BJ47WeightsV1 *w, const float *input, float *k, float *v, const BJ47FrameV1 *frame, float *out"
            check = f"!w || !w->f32 || !w->q4 || !input || !k || !v || !frame || !frame->cos_rows || !frame->sin_rows || !out || input==out || frame->cache_position<0 || frame->cache_position>{ctx - rows}"
            pre = "  int64_t pos = frame->cache_position;"
        else:
            name = "lm_head_" + phase
            sig = "const BJ47WeightsV1 *w, const float *input, float *out"
            check = "!w || !w->f32 || !w->q4 || !input || !out"
        cpp += thunk(a, name, symbol, args, sig, check, pre)
        symbols.append(name)
        if role == "decoder_layer":
            name = "prepare_" + phase
            symbol = "rb_" + name
            objs.append(
                compile_ir(
                    a, symbol, prepare_ir(rows, dim, phase), "kernels_a100", cfg
                )
            )
            cpp += thunk(
                a,
                name,
                symbol,
                [
                    ("inv", str(dim // 2)),
                    ("&pos", "1"),
                    ("c", f"{rows}x{dim}"),
                    ("s", f"{rows}x{dim}"),
                ],
                "const float *inv, int64_t pos, float *c, float *s, void *",
                f"!inv || !c || !s || c==s || pos<0 || pos>{ctx - rows}",
            )
            symbols.append(name)
    for stem, kernel in sorted(kernels.items()):
        objs.append(
            compile_ir(
                a,
                stem,
                kernel.replace("@k3_", f"@rb_{role}_k3_"),
                "kernels_ime" if "_ime-" in stem else "kernels_a100",
                cfg,
            )
        )
    if role == "lm_head":
        cpp += 'extern "C" int32_t lm_head(const BJ47WeightsV1 *w, const float *x, float *out) {return lm_head_decode(w, x, out); }\n'
        symbols.append("lm_head")
    shim = a.work / "Shim.cpp"
    shim.write_text(cpp)
    exports = a.work / "exports.map"
    exports.write_text("{ global: " + "; ".join(symbols) + "; local: *; };")
    output = a.work / (role + ".so")
    subprocess.run(
        [
            a.cxx,
            "-O2",
            *a.cxx_flag,
            "-fPIC",
            "-shared",
            "-std=c++17",
            "-I" + str(SRC),
            "-I" + str(ROOT / "runtime/include"),
            shim,
            *objs,
            "-Wl,-z,defs",
            "-Wl,--version-script=" + str(exports),
            "-Wl,-rpath,$ORIGIN",
            "-Wl,-Bsymbolic",
            "-L" + str(a.bundle),
            "-lbj47_kernel_runtime",
            a.crunner,
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            output,
        ],
        check=True,
    )
    tmp = a.bundle / (role + ".so.tmp")
    shutil.copyfile(output, tmp)
    tmp.replace(a.bundle / (role + ".so"))
    record = {
        "role": role,
        "abi_version": abi,
        "path": output.name,
        "nbytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "layout_id": layout,
        "shape_id": m["shape_id"],
        "symbols": symbols,
        "target_flags": ATTRS,
        "cxx_compiler": subprocess.check_output(
            [a.cxx, "--version"], text=True
        ).splitlines()[0],
        "cxx_flags": [
            "-O2",
            *(
                f
                for f in a.cxx_flag
                if not f.startswith(("--sysroot=", "--gcc-toolchain="))
            ),
            "-fPIC",
            "-shared",
            "-std=c++17",
        ],
        "runtime": "libbj47_kernel_runtime.so",
        "kernel_info_struct": "BJ47KernelInfoV1",
    }
    receipt = a.bundle / (role + ".kernel.json")
    pending = receipt.with_suffix(".tmp")
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("bundle", "work"):
        p.add_argument("--" + arg, type=Path, required=True)
    for arg in ("role", "buddy-opt", "llvm", "cxx", "crunner"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--cxx-flag", action="append", default=[])
    build(p.parse_args())
