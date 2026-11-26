#!/usr/bin/env python3
"""Export Qwen3.5 text tensors and build reusable f32 Route B templates."""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import route_b_hybrid_ir as ir
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models/deepseek_r1/pipeline/route_b"
PREFIX = "model.language_model."
ROLE_IDS = {"embed": 1, "lm_head": 3, "gdn_layer": 4, "full_attn_layer": 5}
OWNERS = {
    "embed": "embed",
    "lm_head": "head",
    "gdn_layer": "gdn",
    "full_attn_layer": "full_attn",
}
LAYER_TYPES = {"linear_attention": "gdn", "full_attention": "full_attn"}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_config(path, context, chunk):
    config = json.loads(path.read_text())
    c = config["text_config"]
    dims = {
        "width": c["hidden_size"],
        "intermediate": c["intermediate_size"],
        "keyHeads": c["linear_num_key_heads"],
        "valueHeads": c["linear_num_value_heads"],
        "keyDim": c["linear_key_head_dim"],
        "valueDim": c["linear_value_head_dim"],
        "convKernel": c["linear_conv_kernel_dim"],
        "attentionHeads": c["num_attention_heads"],
        "kvHeads": c["num_key_value_heads"],
        "headDim": c["head_dim"],
        "vocab": c["vocab_size"],
        "chunk": chunk,
        "context": context,
    }
    rope = c["rope_parameters"]
    if (
        config["model_type"] not in ("qwen3_5", "qwen3_8")
        or c["hidden_act"] != "silu"
        or not c["attn_output_gate"]
        or c["attention_bias"]
        or c["tie_word_embeddings"]
        or c.get("mamba_ssm_dtype", "float32") != "float32"
        or rope["rope_type"] != "default"
        or not 0 < rope["partial_rotary_factor"] <= 1
        or rope["rope_theta"] <= 0
        or c["rms_norm_eps"] <= 0
        or c["eos_token_id"] != 248044
        or c["vocab_size"] <= 248069
        or len(c["layer_types"]) != c["num_hidden_layers"]
        or not c["layer_types"]
        or any(kind not in LAYER_TYPES for kind in c["layer_types"])
        or not 0 < context <= c["max_position_embeddings"]
        or chunk != 64
        or context % chunk
        or any(type(x) is not int or x <= 0 for x in dims.values())
    ):
        raise ValueError("unsupported Route B hybrid config")
    dims["rotaryDim"] = int(dims["headDim"] * rope["partial_rotary_factor"])
    dims["keyWidth"] = dims["keyHeads"] * dims["keyDim"]
    dims["valueWidth"] = dims["valueHeads"] * dims["valueDim"]
    dims["convDim"] = 2 * dims["keyWidth"] + dims["valueWidth"]
    if (
        not dims["rotaryDim"]
        or dims["rotaryDim"] % 2
        or dims["valueHeads"] % dims["keyHeads"]
        or dims["attentionHeads"] % dims["kvHeads"]
        or any(
            dims[key] % 8
            for key in ("width", "intermediate", "valueDim", "headDim")
        )
    ):
        raise ValueError("unsupported Route B hybrid head/vector dimensions")
    types = [LAYER_TYPES[kind] for kind in c["layer_types"]]
    return c, dims, types


def schemas(d):
    h, m, k, v, conv = (
        d[key]
        for key in (
            "width",
            "intermediate",
            "keyWidth",
            "valueWidth",
            "convDim",
        )
    )
    q = d["attentionHeads"] * d["headDim"]
    kv = d["kvHeads"] * d["headDim"]
    block = {
        "inputNorm": ("input_layernorm.weight", (h,)),
        "postNorm": ("post_attention_layernorm.weight", (h,)),
        "gateProj": ("mlp.gate_proj.weight", (m, h)),
        "upProj": ("mlp.up_proj.weight", (m, h)),
        "downProj": ("mlp.down_proj.weight", (h, m)),
    }
    return {
        "embed": {"table": (PREFIX + "embed_tokens.weight", (d["vocab"], h))},
        "head": {
            "finalNorm": (PREFIX + "norm.weight", (h,)),
            "lmHead": ("lm_head.weight", (d["vocab"], h)),
        },
        "gdn": block
        | {
            "inProjQKV": ("linear_attn.in_proj_qkv.weight", (conv, h)),
            "inProjZ": ("linear_attn.in_proj_z.weight", (v, h)),
            "inProjB": ("linear_attn.in_proj_b.weight", (d["valueHeads"], h)),
            "inProjA": ("linear_attn.in_proj_a.weight", (d["valueHeads"], h)),
            "convWeight": (
                "linear_attn.conv1d.weight",
                (conv, 1, d["convKernel"]),
            ),
            "aLog": ("linear_attn.A_log", (d["valueHeads"],)),
            "dtBias": ("linear_attn.dt_bias", (d["valueHeads"],)),
            "gatedNorm": ("linear_attn.norm.weight", (d["valueDim"],)),
            "mixerOut": ("linear_attn.out_proj.weight", (h, v)),
        },
        "full_attn": block
        | {
            "qProj": ("self_attn.q_proj.weight", (2 * q, h)),
            "kProj": ("self_attn.k_proj.weight", (kv, h)),
            "vProj": ("self_attn.v_proj.weight", (kv, h)),
            "qNorm": ("self_attn.q_norm.weight", (d["headDim"],)),
            "kNorm": ("self_attn.k_norm.weight", (d["headDim"],)),
            "mixerOut": ("self_attn.o_proj.weight", (h, q)),
        },
    }


def weights_index(checkpoint):
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    shard = "model.safetensors"
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as model:
        return dict.fromkeys(model.keys(), shard)


def pack(path, checkpoint, index, schema, layer, dtype):
    prefix = "" if layer is None else f"{PREFIX}layers.{layer}."
    with path.open("wb") as stream:
        for name, shape in schema.values():
            key = prefix + name
            with safe_open(
                checkpoint / index[key], framework="pt", device="cpu"
            ) as shard:
                raw = shard.get_slice(key)
                if tuple(raw.get_shape()) != shape:
                    raise ValueError(
                        f"{key}: expected {shape}, got {raw.get_shape()}"
                    )
                rows = max(1, (1 << 20) // math.prod(shape[1:]))
                for start in range(0, shape[0], rows):
                    tensor = raw[start : start + rows].to(dtype)
                    stream.write(memoryview(tensor.numpy()).cast("B"))


def receipt(path):
    return {
        "path": path.name,
        "nbytes": path.stat().st_size,
        "sha256": file_digest(path),
    }


def weight_receipt(path, owner, bucket, elements, layout, layer=None):
    result = receipt(path) | {
        "owner": owner,
        "dtype": bucket,
        "bucket": bucket,
        "elements": elements,
        "layout_id": layout,
    }
    if layer is not None:
        result["global_layer_id"] = layer
    return result


def octets(value):
    return "{" + ", ".join(map(str, bytes.fromhex(value))) + "}"


def generate(role, d, c, schema, offsets, count, dtype, layout, shape):
    owner = OWNERS[role]
    role_id = ROLE_IDS[role]
    state_a = state_b = rope = 0
    if owner == "gdn":
        state_a = d["convDim"] * d["convKernel"]
        state_b = d["valueHeads"] * d["keyDim"] * d["valueDim"]
    if owner == "full_attn":
        state_a = state_b = d["kvHeads"] * d["context"] * d["headDim"]
        rope = d["rotaryDim"]
    half = owner == "embed" and dtype == "f16"
    fields = [
        2,
        "sizeof(BJ47KernelInfoV2)",
        role_id,
        d["chunk"],
        d["context"],
        d["width"],
        d["kvHeads"],
        d["headDim"],
        d["vocab"],
        0 if half else count,
        0,
        rope // 2,
        0,
        0,
        64,
        octets(layout),
        octets(shape),
    ]
    extended = [
        state_a,
        state_b,
        count if half else 0,
        rope,
        2 if half else int(owner == "embed"),
    ]
    header = (
        '#pragma once\n#include "Shim.h"\n'
        f"#define HYBRID_ROLE {role_id}\n"
        "static constexpr BJ47KernelInfoV2 kernelInfo = {\n  {"
        + ", ".join(map(str, fields))
        + "},\n  "
        + ", ".join(map(str, extended))
        + "\n};\n"
    )
    for key, value in d.items():
        header += f"static constexpr int {key} = {value};\n"
    header += f"static constexpr float normEpsilon = {c['rms_norm_eps']}f;\n"
    for key, offset in offsets.items():
        header += f"static constexpr size_t {key} = {offset};\n"

    symbols = ["bj47_kernel_info_v1"]
    mlir = []
    if owner == "embed":
        suffix = "_f16" if half else ""
        symbols.extend(["embed_prefill" + suffix, "embed_decode" + suffix])
        header += f"""\
#define EMBED_PREFILL {symbols[1]}
#define EMBED_DECODE {symbols[2]}
using EmbedElement = {"uint16_t" if half else "float"};
template <int R>
int32_t embed(const EmbedElement *, const int64_t *, float *);
"""
        for rows in (d["chunk"], 1):
            mlir.append(ir.embedding(rows, d["width"], d["vocab"], dtype))
            header += f"""\
extern "C" void _mlir_ciface_gather_{rows}(Descriptor<2> *, Descriptor<1> *, Descriptor<2> *);
template <>
inline int32_t embed<{rows}>(const EmbedElement *table, const int64_t *ids, float *out) {{
  if (!table || !ids || !out)
    return -1;
  for (int i = 0; i < {rows}; ++i)
    if (ids[i] < 0 || ids[i] >= vocab)
      return -2;
  Descriptor<2> weights(table, {{vocab, width}}), dst(out, {{{rows}, width}});
  Descriptor<1> tokens(ids, {{{rows}}});
  _mlir_ciface_gather_{rows}(&weights, &tokens, &dst);
  return 0;
}}
"""
    else:
        header += "template <int N, int K> void project(const float *, const float *, float *);\n"
        projections = sorted(
            {shape for _, shape in schema.values() if len(shape) == 2}
        )
        for n, k in projections:
            mlir.append(ir.projection(n, k))
            header += f"""\
extern "C" void _mlir_ciface_project_{n}_{k}(Descriptor<2> *, Descriptor<1> *, Descriptor<1> *);
template <>
inline void project<{n}, {k}>(const float *w, const float *x, float *y) {{
  Descriptor<2> weights(w, {{{n}, {k}}});
  Descriptor<1> input(x, {{{k}}}), out(y, {{{n}}});
  _mlir_ciface_project_{n}_{k}(&weights, &input, &out);
}}
"""
        if owner == "gdn":
            mlir.append(ir.delta_step(d["keyDim"], d["valueDim"]))
            header += """\
extern "C" void _mlir_ciface_delta_step(Descriptor<2> *, Descriptor<1> *,
    Descriptor<1> *, Descriptor<1> *, Descriptor<1> *, float, float);
inline void deltaStep(float *s, const float *q, const float *k, const float *v,
                      float *y, float beta, float alpha) {
  Descriptor<2> state(s, {keyDim, valueDim});
  Descriptor<1> query(q, {keyDim}), key(k, {keyDim}), value(v, {valueDim}), out(y, {valueDim});
  _mlir_ciface_delta_step(&state, &query, &key, &value, &out, beta, alpha);
}
"""
        if owner == "head":
            symbols.extend(["lm_head", "lm_head_prefill", "lm_head_decode"])
        else:
            symbols.extend(["layer_prefill", "layer_decode"])
        if owner == "full_attn":
            symbols.extend(["prepare_prefill", "prepare_decode"])
    return header, "\n".join(mlir), symbols


def compile_template(args, role, header, mlir, symbols):
    dest = args.work / role
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "HybridConfig.h").write_text(header)
    source = dest / "kernel.mlir"
    source.write_text(mlir)
    copy_audit = dest / "kernel.after-copy-elimination.mlir"
    lowered = dest / "kernel.llvm.mlir"
    llvm = dest / "kernel.ll"
    exports = dest / "exports.map"
    exports.write_text("{ global: " + "; ".join(symbols) + "; local: *; };\n")
    start = time.monotonic()
    subprocess.run(
        [
            args.buddy_opt,
            source,
            "--canonicalize",
            "--eliminate-memref-copy",
            "-o",
            copy_audit,
        ],
        check=True,
    )
    if "memref.copy" in copy_audit.read_text():
        raise ValueError(f"{role}: unexpected tensor copy")
    subprocess.run(
        [
            args.buddy_opt,
            copy_audit,
            "--convert-scf-to-cf",
            "--convert-vector-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            lowered,
        ],
        check=True,
    )
    subprocess.run(
        [
            args.llvm_bin / "mlir-translate",
            "--mlir-to-llvmir",
            lowered,
            "-o",
            llvm,
        ],
        check=True,
    )
    if "@llvm.memcpy" in llvm.read_text() or "@memrefCopy" in llvm.read_text():
        raise ValueError(f"{role}: unexpected copy in lowered LLVM IR")
    if any(
        flag.startswith("-fsanitize=") and "address" in flag
        for flag in args.cxx_flag
    ):
        # Clang's ASan pass on input LLVM IR needs the function attribute.
        llvm.write_text(
            re.sub(
                r"^(define .*?) \{$",
                r"\1 sanitize_address {",
                llvm.read_text(),
                flags=re.MULTILINE,
            )
        )
    output = args.output / f"{role}.so"
    tmp = dest / output.name
    subprocess.run(
        [
            args.llvm_bin / "clang++",
            "-O2",
            "-DNDEBUG",
            "-ffp-contract=off",
            "-Wno-override-module",
            *args.cxx_flag,
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-I" + str(ROOT / "runtime/include"),
            "-I" + str(SOURCE),
            "-I" + str(dest),
            SOURCE / "HybridKernel.cpp",
            llvm,
            "-Wl,-z,defs",
            "-Wl,--version-script=" + str(exports),
            "-o",
            tmp,
        ],
        check=True,
    )
    elapsed = time.monotonic() - start
    shutil.copyfile(tmp, output)
    return receipt(output) | {"compile_seconds": elapsed, "symbols": symbols}


def export_bundle(args):
    c, dims, types = read_config(
        args.checkpoint / "config.json", args.context, args.chunk
    )
    schema = schemas(dims)
    rope = c["rope_parameters"]
    shape = digest({"dims": dims, "epsilon": c["rms_norm_eps"], "rope": rope})
    role_names = [
        role for role in ROLE_IDS if OWNERS[role] in {"embed", "head", *types}
    ]
    index = weights_index(args.checkpoint)
    args.work.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output / "vocab.txt"
    shutil.copyfile(args.vocab, vocab_path)
    kernels, weights, layouts = [], [], {}
    compiled = {}
    for role in role_names:
        owner = OWNERS[role]
        keys = schema[owner]
        dtype = args.embed_dtype if owner == "embed" else "f32"
        layout = layouts[owner] = digest(
            {"schema": list(keys.items()), "dtype": dtype}
        )
        offsets, count = {}, 0
        for name, (_, weight_shape) in keys.items():
            offsets[name] = count
            count += math.prod(weight_shape)
        ids = [i for i, kind in enumerate(types) if kind == owner]
        for layer in ids or [None]:
            label = owner if layer is None else f"{owner}-{layer}"
            path = args.output / f"{label}.{dtype}"
            pack(
                path,
                args.checkpoint,
                index,
                keys,
                layer,
                torch.float16 if dtype == "f16" else torch.float32,
            )
            weights.append(
                weight_receipt(path, owner, dtype, count, layout, layer)
            )
        header, mlir, symbols = generate(
            role, dims, c, keys, offsets, count, dtype, layout, shape
        )
        binary = compile_template(args, role, header, mlir, symbols)
        compiled[role] = binary.pop("compile_seconds")
        kernels.append(
            binary
            | {
                "role": role,
                "abi_version": 2,
                "layout_id": layout,
                "shape_id": shape,
            }
        )
    if "full_attn" in types:
        rotary = dims["rotaryDim"]
        freq = 1.0 / (
            rope["rope_theta"]
            ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
        )
        path = args.output / "rope.f32"
        path.write_bytes(freq.numpy().tobytes())
        layouts["rope"] = digest({"inv_freq": rotary // 2})
        weights.append(
            weight_receipt(path, "rope", "f32", rotary // 2, layouts["rope"])
        )
    manifest = {
        "format": "bj47.route-b/1",
        "model_id": args.model_id,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "variant": "f32",
        "kernels": "k3",
        "prefill_ime": False,
        "activation_type": "f32",
        "total_layers": len(types),
        "layer_types": types,
        "activation_width": dims["width"],
        "context": dims["context"],
        "prefill_chunk": dims["chunk"],
        "kv_heads": dims["kvHeads"],
        "head_dim": dims["headDim"],
        "vocab_size": dims["vocab"],
        "eos_id": c["eos_token_id"],
        "tokenizer": args.tokenizer,
        "stop_token_ids": [248044, 248046],
        "vocab_file": vocab_path.name,
        "vocab_sha256": file_digest(vocab_path),
        "pack_version": "qwen3_5-f32/1",
        "shape_id": shape,
        "layouts": layouts,
        "templates": kernels,
        "weights": weights,
    }
    manifest["bundle_id"] = digest(manifest)
    pending = args.output / "route-b.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(args.output / "route-b.json")
    summary = {
        "compile_seconds": compiled,
        "compiler_flags": args.cxx_flag,
        "compiler": subprocess.check_output(
            [args.llvm_bin / "clang++", "--version"], text=True
        ).splitlines()[0],
        "templates": kernels,
    }
    (args.work / "build.json").write_text(json.dumps(summary, indent=2) + "\n")


def build(args):
    # A resident consumer can have every published payload mapped. Prepare a
    # complete bundle on the same filesystem, then publish it in one rename.
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as tmp:
        staged = Path(tmp) / "bundle"
        staged.mkdir()
        export_bundle(argparse.Namespace(**(vars(args) | {"output": staged})))
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in (
        "checkpoint",
        "vocab",
        "output",
        "work",
        "llvm-bin",
        "buddy-opt",
    ):
        parser.add_argument("--" + path, required=True, type=Path)
    parser.add_argument("--embed-dtype", choices=["f16", "f32"], required=True)
    parser.add_argument(
        "--tokenizer", choices=["qwen3_5", "qwen3_8"], default="qwen3_8"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--cxx-flag", action="append", default=[])
    build(parser.parse_args())
