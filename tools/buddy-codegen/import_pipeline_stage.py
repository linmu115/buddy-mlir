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
"""Export one DeepSeek R1 Distill Qwen FP16 stage with a stage-local KV ABI."""

import argparse
import json
from pathlib import Path

import torch
from import_model import (
    apply_fusion,
    apply_pre_transforms,
    compile_graphs,
    create_drivers,
    export_mlir,
    export_plain_weights_direct,
    load_model,
    timed_import_step,
)


def select_layers(model, start, end, total):
    """Select a half-open layer interval before Dynamo traces Qwen2.forward."""
    decoder = model.model
    decoder.layers = torch.nn.ModuleList(decoder.layers[start:end])
    model.config.layer_types = model.config.layer_types[start:end]
    model.config.num_hidden_layers = end - start
    for local_idx, layer in enumerate(decoder.layers):
        layer.self_attn.layer_idx = local_idx
    if start:
        decoder.embed_tokens = torch.nn.Identity()
    if end != total:
        decoder.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()


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


def export_stage(config, start, end, out):
    if (
        config["architecture"] != "Qwen2ForCausalLM"
        or config["variant"] != "f16"
        or config.get("prefill_chunk")
        or config.get("arena")
        or config.get("tiered_kv_cache", {}).get("enabled")
        or config.get("decode_pack", {}).get("enabled")
    ):
        raise ValueError(
            "stage export needs the plain FP16 Qwen2 configuration"
        )

    with timed_import_step("load_stage"):
        model = load_model(config)
        total = model.config.num_hidden_layers
        if not 0 <= start < end <= total:
            raise ValueError(
                f"invalid layer interval [{start}, {end}) for {total} layers"
            )
        if any(kind != "full_attention" for kind in model.config.layer_types):
            raise ValueError("stage export requires full attention layers")
        width = model.config.hidden_size
        select_layers(model, start, end, total)

    with timed_import_step("compile_stage_graphs"):
        (prefill,), (decode,), params = compile_graphs(
            model, config, hidden_input=start > 0
        )

    apply_pre_transforms(prefill, decode)
    check_parameter_layout(prefill, decode)
    apply_fusion(prefill, decode)
    drivers = create_drivers(prefill, decode)
    out.mkdir(parents=True, exist_ok=True)
    export_mlir(*drivers, config, out)

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
    args = parser.parse_args()
    export_stage(
        json.loads(args.config.read_text()), *args.layers, args.output_dir
    )


if __name__ == "__main__":
    main()
