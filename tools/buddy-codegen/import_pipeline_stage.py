#!/usr/bin/env python3
# ===----------------------------------------------------------------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
"""Export one Qwen2 or Qwen3-MoE pipeline stage (FP16 or K3 w4g32) with a stage-local KV ABI."""

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy
import torch
from gen_config import text_decoder_config, w4g32_stage_param_counts
from import_model import (
    apply_fusion,
    apply_k3_w4,
    apply_pre_transforms,
    compile_chunk_graphs,
    compile_graphs,
    create_drivers,
    export_mlir,
    export_plain_weights_direct,
    load_model,
    load_qwen3_moe,
    timed_import_step,
)
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

_LAYER_PARAM_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_MOE_GATE_UP_SUFFIX = "mlp.experts.gate_up_proj"
_MOE_DOWN_SUFFIX = "mlp.experts.down_proj"


def _is_qwen3_moe_config(hf_config):
    """Return True if hf_config describes a Qwen3-MoE text decoder."""
    archs = getattr(hf_config, "architectures", None) or []
    return (
        "Qwen3MoeForCausalLM" in archs
        or getattr(hf_config, "model_type", None) == "qwen3_moe"
    )


def resolve_model_path(config):
    """Resolve the HuggingFace checkpoint directory or repo ID."""
    if config.get("architecture") == "Qwen3MoeForCausalLM" or str(
        config.get("model_family", "")
    ).startswith("qwen3"):
        return (
            os.environ.get("QWEN3_VL_MOE_MODEL_PATH")
            or os.environ.get("BUDDY_MODEL_PATH")
            or config["hf_model_path"]
        )
    return (
        os.environ.get("DEEPSEEKR1_MODEL_PATH")
        or os.environ.get("BUDDY_MODEL_PATH")
        or config["hf_model_path"]
    )


def stage_dtype(config):
    """Return the torch dtype matching config['precision']['activation_type']."""
    activation = config["precision"]["activation_type"]
    if activation == "f16":
        return torch.float16
    if activation == "bf16":
        return torch.bfloat16
    return torch.float32


def validate_stage_interval(hf_config, start, end):
    """Validate [start, end) against model config and return (total, width)."""
    total = hf_config.num_hidden_layers
    if not 0 <= start < end <= total:
        raise ValueError(
            f"invalid layer interval [{start}, {end}) for {total} layers"
        )
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None and any(
        kind != "full_attention" for kind in layer_types
    ):
        raise ValueError("stage export requires full attention layers")
    if _is_qwen3_moe_config(hf_config):
        if getattr(hf_config, "use_sliding_window", False):
            raise ValueError(
                "Qwen3-MoE stage export does not support sliding_window attention"
            )
        if getattr(hf_config, "mlp_only_layers", None) or (
            getattr(hf_config, "decoder_sparse_step", 1) != 1
        ):
            raise ValueError(
                "Qwen3-MoE stage export requires a MoE block in every layer "
                "(empty mlp_only_layers and decoder_sparse_step == 1)"
            )
        if not getattr(hf_config, "norm_topk_prob", False):
            raise ValueError(
                "Qwen3-MoE stage export requires norm_topk_prob=True"
            )
    return total, hf_config.hidden_size


def resolve_checkpoint_dir(model_path):
    """Resolve a local checkpoint directory from a path or cached HF repo ID."""
    local = Path(model_path)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise FileNotFoundError(
            f"lazy loading requires a local checkpoint directory: {model_path}"
        ) from err
    return Path(snapshot_download(model_path, local_files_only=True))


def resolve_safetensors_map(model_dir):
    """Map checkpoint tensor names to their safetensors file paths."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_file}")
        return {key: model_dir / shard for key, shard in weight_map.items()}
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        with safe_open(single_file, framework="pt", device="cpu") as reader:
            return dict.fromkeys(reader.keys(), single_file)
    raise FileNotFoundError(
        f"lazy loading requires model.safetensors or "
        f"model.safetensors.index.json in {model_dir}; "
        f"use --load-mode full as fallback"
    )


def checkpoint_key_for_param(
    local_name,
    start,
    weight_map,
    tie_word_embeddings,
    *,
    model_prefix="model.",
):
    """Map a stage-local parameter name to its global checkpoint tensor key."""
    match = _LAYER_PARAM_RE.match(local_name)
    if match:
        global_idx = start + int(match.group(1))
        ckpt_key = f"{model_prefix}layers.{global_idx}.{match.group(2)}"
    elif local_name.startswith("model."):
        ckpt_key = f"{model_prefix}{local_name[len('model.'):]}"
    else:
        ckpt_key = local_name
    if ckpt_key in weight_map:
        return ckpt_key
    embed_key = f"{model_prefix}embed_tokens.weight"
    if (
        local_name == "lm_head.weight"
        and tie_word_embeddings
        and embed_key in weight_map
    ):
        return embed_key
    raise KeyError(
        f"checkpoint is missing tensor '{ckpt_key}' for stage parameter "
        f"'{local_name}'"
    )


def select_layers(model, start, end, total):
    """Select a half-open layer interval before Dynamo traces decoder.forward."""
    decoder = model.model
    decoder.layers = torch.nn.ModuleList(decoder.layers[start:end])
    if getattr(model.config, "layer_types", None) is not None:
        model.config.layer_types = model.config.layer_types[start:end]
    model.config.num_hidden_layers = end - start
    for local_idx, layer in enumerate(decoder.layers):
        layer.self_attn.layer_idx = local_idx
    if start:
        decoder.embed_tokens = torch.nn.Identity()
    if end != total:
        decoder.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()


def _read_tensor_shapes(weight_map, keys):
    """Read safetensors shapes for `keys` via header metadata only."""
    shapes = {}
    by_shard = defaultdict(list)
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"checkpoint is missing tensor '{key}'")
        by_shard[weight_map[key]].append(key)
    for shard_path, shard_keys in sorted(by_shard.items()):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for key in shard_keys:
                shapes[key] = tuple(reader.get_slice(key).get_shape())
    return shapes


def _resolve_vl_moe_layer_transpose(
    weight_map, model_prefix, global_idx, gate_up_shape, down_shape
):
    """Determine whether a Qwen3-VL-MoE layer stores merged experts transposed.

    HF Hub Qwen3-VL-MoE checkpoints store expert weights transposed on dims
    (1, 2) as [E, H, 2I] / [E, I, H], whereas transformers 5.x save_pretrained
    writes native [E, 2I, H] / [E, H, I]. Because H > 0 and I > 0 cannot
    satisfy both 2*I == H and I == H simultaneously, inspecting both tensors
    together resolves orientation unambiguously even when one tensor is square
    (e.g. 2*I == H). Conflicting orientations raise ValueError.
    """
    gu_key = f"{model_prefix}layers.{global_idx}.{_MOE_GATE_UP_SUFFIX}"
    dn_key = f"{model_prefix}layers.{global_idx}.{_MOE_DOWN_SUFFIX}"
    shapes = _read_tensor_shapes(weight_map, (gu_key, dn_key))
    gu_disk = shapes[gu_key]
    dn_disk = shapes[dn_key]

    gu_exp = tuple(gate_up_shape)
    gu_tr = (gu_exp[0], gu_exp[2], gu_exp[1])
    dn_exp = tuple(down_shape)
    dn_tr = (dn_exp[0], dn_exp[2], dn_exp[1])

    gu_native = gu_disk == gu_exp
    gu_transposed = gu_disk == gu_tr
    dn_native = dn_disk == dn_exp
    dn_transposed = dn_disk == dn_tr

    if not (gu_native or gu_transposed):
        raise ValueError(
            f"unexpected shape {gu_disk} for '{gu_key}'; "
            f"expected {gu_exp} or {gu_tr}"
        )
    if not (dn_native or dn_transposed):
        raise ValueError(
            f"unexpected shape {dn_disk} for '{dn_key}'; "
            f"expected {dn_exp} or {dn_tr}"
        )

    can_be_native = gu_native and dn_native
    can_be_transposed = gu_transposed and dn_transposed
    if can_be_native and not can_be_transposed:
        return False
    if can_be_transposed and not can_be_native:
        return True
    raise ValueError(
        f"inconsistent Qwen3-VL-MoE expert orientation in layer {global_idx}: "
        f"{gu_key} has shape {gu_disk} (native {gu_exp}), "
        f"{dn_key} has shape {dn_disk} (native {dn_exp})"
    )


def _guard_full_vl_moe_square_transpose(model_path, start, end):
    """Reject --load-mode full on Hub-transposed VL-MoE checkpoints with a
    square expert matrix (2*I == H or I == H), where transformers 5.x's
    Transpose(check_dims=True) would silently skip transposing that tensor."""
    try:
        model_dir = resolve_checkpoint_dir(model_path)
        weight_map = resolve_safetensors_map(model_dir)
    except FileNotFoundError:
        return
    raw_hf_dict = AutoConfig.from_pretrained(model_dir).to_dict()
    text_dict = text_decoder_config(raw_hf_dict)
    if text_dict is raw_hf_dict:
        return
    h = text_dict["hidden_size"]
    inter = text_dict["moe_intermediate_size"]
    if 2 * inter != h and inter != h:
        return
    experts = text_dict.get("num_experts")
    if experts is None:
        experts = text_dict.get("num_local_experts")
    if not experts:
        return
    model_prefix = "model.language_model."
    gu_key = f"{model_prefix}layers.{start}.{_MOE_GATE_UP_SUFFIX}"
    dn_key = f"{model_prefix}layers.{start}.{_MOE_DOWN_SUFFIX}"
    if gu_key in weight_map and dn_key in weight_map:
        if _resolve_vl_moe_layer_transpose(
            weight_map,
            model_prefix,
            start,
            (experts, 2 * inter, h),
            (experts, h, inter),
        ):
            raise ValueError(
                f"Qwen3-VL-MoE checkpoint at {model_dir} stores transposed "
                f"square expert weights (hidden_size={h}, "
                f"moe_intermediate_size={inter}); HF transformers "
                f"Transpose(check_dims=True) in --load-mode full would skip "
                f"transposing the square tensor. Use --load-mode lazy."
            )


def load_stage_full(config, start, end):
    """Legacy loader: load the full checkpoint, then crop to [start, end)."""
    model_path = resolve_model_path(config)
    if config.get("architecture") == "Qwen3MoeForCausalLM":
        _guard_full_vl_moe_square_transpose(model_path, start, end)
        model = load_qwen3_moe(
            model_path, config["precision"]["activation_type"]
        )
        model.config.use_cache = False
    else:
        resolved_cfg = dict(config)
        resolved_cfg["hf_model_path"] = model_path
        model = load_model(resolved_cfg)
    total, width = validate_stage_interval(model.config, start, end)
    select_layers(model, start, end, total)
    return model, total, width


def _load_tensor_into_dtype(reader, ckpt_key, expected_shape, target_dtype, transpose_12=False):
    """Load one tensor, validate shape, optionally transpose dims (1, 2), and cast."""
    raw = reader.get_tensor(ckpt_key)
    if transpose_12:
        expected_disk = (
            expected_shape[0],
            expected_shape[2],
            expected_shape[1],
        )
        if tuple(raw.shape) != expected_disk:
            raise ValueError(
                f"tensor '{ckpt_key}' has shape {tuple(raw.shape)}, "
                f"expected transposed {expected_disk}"
            )
        out = torch.empty(expected_shape, dtype=target_dtype)
        out.copy_(raw.transpose(1, 2))
        del raw
        return out
    if tuple(raw.shape) != expected_shape:
        raise ValueError(
            f"tensor '{ckpt_key}' has shape {tuple(raw.shape)}, "
            f"expected {expected_shape}"
        )
    if raw.dtype != target_dtype:
        out = raw.to(dtype=target_dtype)
        del raw
        return out
    return raw


def load_stage_lazy(config, start, end):
    """Build an empty meta model, crop layers, and load only stage tensors.

    Supports Qwen2ForCausalLM and Qwen3-MoE across all three on-disk layouts:
      1. transformers 5.x merged Qwen3MoeForCausalLM
      2. HF Hub per-expert split Qwen3MoeForCausalLM (experts.{e}.{gate,up,down}_proj)
      3. HF Hub / 5.x Qwen3VLMoeForConditionalGeneration (model.language_model.*,
         optional dim-(1,2) expert transpose, ignoring model.visual.*)
    Reads non-layer tables first and then proceeds layer-by-layer, shard-by-shard
    so peak extra memory is bounded by at most one layer buffer.
    """
    model_path = resolve_model_path(config)
    model_dir = resolve_checkpoint_dir(model_path)
    raw_hf_config = AutoConfig.from_pretrained(model_dir)
    raw_hf_dict = raw_hf_config.to_dict()
    text_dict = text_decoder_config(raw_hf_dict)
    is_vl_moe = text_dict is not raw_hf_dict
    if is_vl_moe:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        hf_config = Qwen3MoeConfig(**text_dict)
    else:
        hf_config = raw_hf_config

    total, width = validate_stage_interval(hf_config, start, end)
    target_dtype = stage_dtype(config)
    tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
    model_prefix = "model.language_model." if is_vl_moe else "model."

    with torch.device("meta"):
        if is_vl_moe:
            model = Qwen3MoeForCausalLM(hf_config)
        else:
            model = AutoModelForCausalLM.from_config(hf_config)
    select_layers(model, start, end, total)

    weight_map = resolve_safetensors_map(model_dir)

    non_layer_params = []
    layer_params = defaultdict(dict)
    for local_name, param in model.named_parameters():
        shape = tuple(param.shape)
        match = _LAYER_PARAM_RE.match(local_name)
        if match:
            layer_params[int(match.group(1))][match.group(2)] = (
                local_name,
                shape,
            )
        else:
            non_layer_params.append((local_name, shape))

    state_dict = {}
    shards_used = set()
    loaded_tensors = 0

    # 1. Non-layer tables (embed_tokens, final norm, lm_head), grouped by shard.
    non_layer_by_shard = defaultdict(lambda: defaultdict(list))
    for local_name, shape in non_layer_params:
        ckpt_key = checkpoint_key_for_param(
            local_name,
            start,
            weight_map,
            tie_word_embeddings,
            model_prefix=model_prefix,
        )
        shard_path = weight_map[ckpt_key]
        shards_used.add(shard_path)
        non_layer_by_shard[shard_path][ckpt_key].append((local_name, shape))

    for shard_path, key_to_targets in sorted(non_layer_by_shard.items()):
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            for ckpt_key, targets in key_to_targets.items():
                _, first_shape = targets[0]
                tensor = _load_tensor_into_dtype(
                    reader, ckpt_key, first_shape, target_dtype
                )
                for local_name, expected_shape in targets:
                    if tuple(tensor.shape) != expected_shape:
                        raise ValueError(
                            f"tensor '{ckpt_key}' shape {tuple(tensor.shape)} "
                            f"does not match '{local_name}' {expected_shape}"
                        )
                    state_dict[local_name] = tensor
                loaded_tensors += 1

    # 2. Layer-by-layer, shard-by-shard loading.
    for local_idx in range(end - start):
        global_idx = start + local_idx
        params = layer_params.get(local_idx, {})
        has_moe_experts = (
            _MOE_GATE_UP_SUFFIX in params and _MOE_DOWN_SUFFIX in params
        )
        layer_ops_by_shard = defaultdict(list)

        gate_up_buf = None
        down_buf = None
        gu_local = None
        dn_local = None

        if has_moe_experts:
            gu_local, gu_shape = params[_MOE_GATE_UP_SUFFIX]
            dn_local, dn_shape = params[_MOE_DOWN_SUFFIX]
            gu_merged_key = (
                f"{model_prefix}layers.{global_idx}.{_MOE_GATE_UP_SUFFIX}"
            )
            dn_merged_key = (
                f"{model_prefix}layers.{global_idx}.{_MOE_DOWN_SUFFIX}"
            )
            split_probe = (
                f"{model_prefix}layers.{global_idx}.mlp.experts.0.gate_proj.weight"
            )

            if gu_merged_key in weight_map and dn_merged_key in weight_map:
                transpose_experts = (
                    _resolve_vl_moe_layer_transpose(
                        weight_map, model_prefix, global_idx, gu_shape, dn_shape
                    )
                    if is_vl_moe
                    else False
                )
                for ckpt_key, local_name, expected_shape in (
                    (gu_merged_key, gu_local, gu_shape),
                    (dn_merged_key, dn_local, dn_shape),
                ):
                    shard_path = weight_map[ckpt_key]
                    shards_used.add(shard_path)
                    layer_ops_by_shard[shard_path].append(
                        (
                            "direct",
                            ckpt_key,
                            local_name,
                            expected_shape,
                            transpose_experts,
                        )
                    )
            elif split_probe in weight_map:
                num_experts, two_inter, hidden = gu_shape
                inter = two_inter // 2
                if dn_shape != (num_experts, hidden, inter):
                    raise ValueError(
                        f"unexpected MoE expert shapes in layer {global_idx}: "
                        f"gate_up={gu_shape}, down={dn_shape}"
                    )
                gate_up_buf = torch.empty(gu_shape, dtype=target_dtype)
                down_buf = torch.empty(dn_shape, dtype=target_dtype)
                for exp_idx in range(num_experts):
                    base = f"{model_prefix}layers.{global_idx}.mlp.experts.{exp_idx}"
                    for kind, suffix, expected_shape in (
                        ("split_gate", "gate_proj.weight", (inter, hidden)),
                        ("split_up", "up_proj.weight", (inter, hidden)),
                        ("split_down", "down_proj.weight", (hidden, inter)),
                    ):
                        ckpt_key = f"{base}.{suffix}"
                        if ckpt_key not in weight_map:
                            raise KeyError(
                                f"checkpoint is missing split-expert tensor "
                                f"'{ckpt_key}'"
                            )
                        shard_path = weight_map[ckpt_key]
                        shards_used.add(shard_path)
                        layer_ops_by_shard[shard_path].append(
                            (kind, ckpt_key, exp_idx, expected_shape, inter)
                        )
            else:
                raise KeyError(
                    f"checkpoint is missing MoE expert tensors for layer "
                    f"{global_idx}: expected '{gu_merged_key}' or '{split_probe}'"
                )

        for suffix, (local_name, expected_shape) in params.items():
            if has_moe_experts and suffix in (
                _MOE_GATE_UP_SUFFIX,
                _MOE_DOWN_SUFFIX,
            ):
                continue
            ckpt_key = checkpoint_key_for_param(
                local_name,
                start,
                weight_map,
                tie_word_embeddings,
                model_prefix=model_prefix,
            )
            shard_path = weight_map[ckpt_key]
            shards_used.add(shard_path)
            layer_ops_by_shard[shard_path].append(
                ("direct", ckpt_key, local_name, expected_shape, False)
            )

        for shard_path, ops in sorted(layer_ops_by_shard.items()):
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                for op in ops:
                    kind = op[0]
                    if kind == "direct":
                        _, ckpt_key, local_name, expected_shape, tr12 = op
                        state_dict[local_name] = _load_tensor_into_dtype(
                            reader,
                            ckpt_key,
                            expected_shape,
                            target_dtype,
                            transpose_12=tr12,
                        )
                    else:
                        _, ckpt_key, exp_idx, expected_shape, inter = op
                        raw = reader.get_tensor(ckpt_key)
                        if tuple(raw.shape) != expected_shape:
                            raise ValueError(
                                f"tensor '{ckpt_key}' has shape "
                                f"{tuple(raw.shape)}, expected {expected_shape}"
                            )
                        if kind == "split_gate":
                            gate_up_buf[exp_idx, :inter, :].copy_(raw)
                        elif kind == "split_up":
                            gate_up_buf[exp_idx, inter:, :].copy_(raw)
                        else:
                            down_buf[exp_idx].copy_(raw)
                        del raw
                    loaded_tensors += 1

        if gate_up_buf is not None:
            state_dict[gu_local] = gate_up_buf
            state_dict[dn_local] = down_buf

    # Re-instantiate non-persistent RoPE buffers on CPU and cast to match
    # AutoModelForCausalLM.from_pretrained(...).eval().half().
    rotary_cls = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_cls(config=model.config).to(
        dtype=target_dtype
    )

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict

    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise RuntimeError(
            f"lazy stage load left meta tensors unmaterialized: {meta_tensors}"
        )

    model.config.use_cache = False
    model.eval()
    if _is_qwen3_moe_config(model.config):
        from buddy.compiler.moe import use_buddy_moe_experts

        use_buddy_moe_experts(model)
    print(
        f"[import] Stage [{start}, {end}) lazy-loaded from {model_dir}: "
        f"{loaded_tensors} tensors from {len(shards_used)} shard(s) "
        f"({config['precision']['activation_type']})",
        file=sys.stderr,
    )
    return model, total, width


def check_parameter_layout(prefill, decode):
    """The two entry points share the exact bytes written in prefill order."""
    if len(prefill._params_ref) != len(decode._params_ref):
        raise ValueError("prefill and decode stage parameter counts differ")
    for index, (left, right) in enumerate(
        zip(prefill._params_ref, decode._params_ref)
    ):
        same_view = (
            left.data_ptr() == right.data_ptr()
            and left.stride() == right.stride()
        )
        if left.shape != right.shape or (
            not same_view and not torch.equal(left, right)
        ):
            raise ValueError(
                f"entry point parameter layouts differ at index {index}"
            )


def stage_w4g32_param_counts(hf_config, start, end, total, ime):
    """Expected (f32_params, i8_params) element counts for stage [start, end)."""
    hf_dict = (
        hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    )
    if not hf_dict.get("architectures"):
        model_type = hf_dict.get("model_type")
        if model_type == "qwen3_moe":
            hf_dict["architectures"] = ["Qwen3MoeForCausalLM"]
        elif model_type == "qwen2":
            hf_dict["architectures"] = ["Qwen2ForCausalLM"]
    counts = w4g32_stage_param_counts(hf_dict, start, end, total, ime)
    return {
        "f32_params": counts["f32_elements"],
        "i8_params": counts["i8_elements"],
    }


def export_k3_weights_direct(graph, config, output_dir, expected_counts):
    """Stream w4g32 f32_params and i8_params directly to disk without concatenate."""
    from buddy.compiler.graph.type import TensorDType

    tag_to_weight = {w["tag"]: w for w in config["weights"]}
    for required in ("f32_params", "i8_params"):
        if required not in tag_to_weight:
            raise ValueError(f"w4g32 stage config missing weight bucket {required}")

    f32_path = Path(output_dir) / tag_to_weight["f32_params"]["file"]
    i8_path = Path(output_dir) / tag_to_weight["i8_params"]["file"]
    counts = {"f32_params": 0, "i8_params": 0}

    with open(f32_path, "wb") as f_f32, open(i8_path, "wb") as f_i8:
        for node, ref in zip(graph.params, graph._params_ref):
            data = ref.detach().contiguous().numpy().reshape(-1)
            if node.tensor_meta["dtype"] == TensorDType.Int8:
                arr = numpy.ascontiguousarray(data, dtype=numpy.int8)
                arr.tofile(f_i8)
                counts["i8_params"] += arr.size
            else:
                arr = numpy.ascontiguousarray(data, dtype=numpy.float32)
                arr.tofile(f_f32)
                counts["f32_params"] += arr.size

    for tag in ("f32_params", "i8_params"):
        if counts[tag] != expected_counts[tag]:
            raise RuntimeError(
                f"w4g32 stage {tag} has {counts[tag]} elements, "
                f"expected {expected_counts[tag]}"
            )
        weight = tag_to_weight[tag]
        if int(weight["num_elements"]) != counts[tag]:
            print(
                f"[import] Updated {tag}: {weight['num_elements']} → {counts[tag]}",
                file=sys.stderr,
            )
            weight["num_elements"] = counts[tag]
        path = Path(output_dir) / weight["file"]
        size_mb = path.stat().st_size / (1024 * 1024)
        print(
            f"[import] Written: {weight['file']} ({size_mb:.1f} MB, "
            f"{counts[tag]:,} elements)",
            file=sys.stderr,
        )
    return counts


def export_stage(config, start, end, out, load_mode="lazy"):
    variant = config["variant"]
    arch = config["architecture"]
    is_w4g32 = variant == "w4g32"
    if (
        arch not in ("Qwen2ForCausalLM", "Qwen3MoeForCausalLM")
        or config.get("tiered_kv_cache", {}).get("enabled")
        or config.get("decode_pack", {}).get("enabled")
    ):
        raise ValueError(
            "stage export supports Qwen2ForCausalLM and Qwen3MoeForCausalLM "
            "without tiered KV or decode_pack"
        )
    if arch == "Qwen3MoeForCausalLM" and not is_w4g32:
        raise ValueError(
            f"Qwen3MoeForCausalLM stage export only supports variant='w4g32' "
            f"(got {variant!r}; MoEExpertsOp has no TOSA/Linalg lowering)"
        )
    if variant == "f16":
        if config.get("prefill_chunk") or config.get("arena"):
            raise ValueError(
                "stage export needs the plain FP16 Qwen2 configuration"
            )
    elif is_w4g32:
        if not config.get("prefill_chunk"):
            raise ValueError("w4g32 stage export requires prefill_chunk > 0")
        if (config.get("kernels") or "k3") != "k3":
            raise ValueError(
                f"w4g32 stage export currently supports kernels='k3' only "
                f"(got {config.get('kernels')!r})"
            )
    else:
        raise ValueError(
            f"stage export supports variant 'f16' or 'w4g32', got {variant!r}"
        )
    if load_mode not in ("lazy", "full"):
        raise ValueError(f"unsupported load mode: {load_mode}")

    with timed_import_step("load_stage"):
        if load_mode == "lazy":
            model, total, width = load_stage_lazy(config, start, end)
        else:
            model, total, width = load_stage_full(config, start, end)

    expected_w4_counts = (
        stage_w4g32_param_counts(
            model.config, start, end, total, bool(config.get("prefill_ime"))
        )
        if is_w4g32
        else None
    )

    with timed_import_step("compile_stage_graphs"):
        if is_w4g32:
            (prefill,), (decode,), params = compile_chunk_graphs(
                model,
                config,
                hidden_input=start > 0,
                logits_to_keep=1 if end == total else None,
            )
        else:
            (prefill,), (decode,), params = compile_graphs(
                model, config, hidden_input=start > 0
            )

    del model
    torch._dynamo.reset()
    gc.collect()

    apply_pre_transforms(prefill, decode)
    check_parameter_layout(prefill, decode)
    if is_w4g32:
        # Share the underlying pre-transformed tensors while keeping distinct
        # parameter lists so k3_w4_rewrite can repack prefill and decode
        # independently without doubling float weight RSS first.
        decode._params_ref = list(prefill._params_ref)
        gc.collect()
        out.mkdir(parents=True, exist_ok=True)
        with timed_import_step("k3_w4"):
            apply_k3_w4(prefill, decode, config, str(out))
        check_parameter_layout(prefill, decode)
        decode._params_ref = prefill._params_ref
        gc.collect()
        with timed_import_step("export_stage_weights"):
            export_k3_weights_direct(prefill, config, out, expected_w4_counts)
        apply_fusion(prefill, decode, chunked_prefill=True)
        drivers = create_drivers(prefill, decode)
        export_mlir(*drivers, config, out)
        del drivers, prefill, decode
        gc.collect()
        config["shape"]["num_hidden_layers"] = end - start
        config["shape"]["kv_layers"] = 2 * (end - start)
        config["pipeline_stage"] = {
            "start_layer": start,
            "end_layer": end,
            "total_layers": total,
            "activation_type": config["precision"]["activation_type"],
            "activation_width": width,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return

    decode._params_ref = prefill._params_ref
    gc.collect()
    apply_fusion(prefill, decode)
    drivers = create_drivers(prefill, decode)
    out.mkdir(parents=True, exist_ok=True)
    export_mlir(*drivers, config, out)
    del drivers, prefill, decode
    gc.collect()

    # "hidden_size" in the legacy config is the size per KV head.
    config["shape"]["num_hidden_layers"] = end - start
    config["shape"]["kv_layers"] = 2 * (end - start)
    config["pipeline_stage"] = {
        "start_layer": start,
        "end_layer": end,
        "total_layers": total,
        "activation_type": config["precision"]["activation_type"],
        "activation_width": width,
    }
    with timed_import_step("export_stage_weights"):
        export_plain_weights_direct(params, config, out)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="zero-based interval [START, END)",
    )
    parser.add_argument(
        "--load-mode",
        choices=("lazy", "full"),
        default="lazy",
        help="checkpoint loading strategy: lazy (stage weights only) or full",
    )
    args = parser.parse_args()
    export_stage(
        json.loads(args.config.read_text()),
        *args.layers,
        args.output_dir,
        load_mode=args.load_mode,
    )


if __name__ == "__main__":
    main()
