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
"""Unit tests for Qwen3-MoE and Qwen3-VL-MoE pipeline stage lazy/full export.

Generates tiny random Qwen3-MoE checkpoints in temporary directories (no weight
files checked into the repository) across all three on-disk safetensors layouts:
  1. transformers 5.x merged Qwen3MoeForCausalLM
  2. HF Hub per-expert split Qwen3MoeForCausalLM (sharded)
  3. HF Hub Qwen3VLMoeForConditionalGeneration (model.language_model.* prefix,
     dim-(1,2) transposed experts, separate visual shard ignored by lazy load)

Also tests the 2*I == H square-expert disambiguation rule, negative validation
paths (including Qwen3-MoE + f16), and end-to-end mini stage export parity.
"""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SKIP_EXIT_CODE = 77

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_DIR = REPO_ROOT / "tools" / "buddy-codegen"
if str(CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(CODEGEN_DIR))

try:
    import safetensors  # noqa: F401
    from safetensors.torch import save_file
    import torch
    from transformers import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
    )
    import gen_config
    import import_pipeline_stage as ips
    import verify_pipeline_stage as vps
except ImportError as err:
    print(f"SKIPPED: missing dependency ({err})")
    sys.exit(SKIP_EXIT_CODE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_ref_moe(
    *,
    seed: int = 123,
    num_hidden_layers: int = 3,
    hidden_size: int = 128,
    moe_intermediate_size: int = 256,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 64,
    vocab_size: int = 256,
):
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=True,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=0,
        architectures=["Qwen3MoeForCausalLM"],
    )
    model = Qwen3MoeForCausalLM(cfg).eval().half()
    return cfg, model


def _write_format1_merged(ref_cfg: Qwen3MoeConfig, ref_model, out_dir: Path):
    """Format 1: transformers 5.x merged Qwen3MoeForCausalLM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_cfg.save_pretrained(out_dir)
    tensors = {k: v.detach().contiguous() for k, v in ref_model.state_dict().items()}
    save_file(tensors, out_dir / "model.safetensors")


def _write_format2_hub_split(ref_cfg: Qwen3MoeConfig, ref_model, out_dir: Path):
    """Format 2: HF Hub split experts across multiple safetensors shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_cfg.save_pretrained(out_dir)
    inter = ref_cfg.moe_intermediate_size
    num_experts = ref_cfg.num_experts
    shards: dict[str, dict[str, torch.Tensor]] = {
        "model-00001-of-00003.safetensors": {},
        "model-00002-of-00003.safetensors": {},
        "model-00003-of-00003.safetensors": {},
    }
    for name, tensor in ref_model.state_dict().items():
        t = tensor.detach().contiguous()
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name[: -len(".gate_up_proj")]
            for e in range(num_experts):
                shard_name = (
                    "model-00001-of-00003.safetensors"
                    if e < num_experts // 2
                    else "model-00002-of-00003.safetensors"
                )
                shards[shard_name][f"{prefix}.{e}.gate_proj.weight"] = (
                    t[e, :inter, :].contiguous()
                )
                shards[shard_name][f"{prefix}.{e}.up_proj.weight"] = (
                    t[e, inter:, :].contiguous()
                )
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name[: -len(".down_proj")]
            for e in range(num_experts):
                shard_name = (
                    "model-00002-of-00003.safetensors"
                    if e < num_experts // 2
                    else "model-00003-of-00003.safetensors"
                )
                shards[shard_name][f"{prefix}.{e}.down_proj.weight"] = (
                    t[e].contiguous()
                )
        elif name.startswith("model.layers.0.") or name == "model.embed_tokens.weight":
            shards["model-00001-of-00003.safetensors"][name] = t
        elif name.startswith("model.layers.1."):
            shards["model-00002-of-00003.safetensors"][name] = t
        else:
            shards["model-00003-of-00003.safetensors"][name] = t

    weight_map = {}
    for shard_name, data in shards.items():
        if not data:
            continue
        save_file(data, out_dir / shard_name)
        for key in data:
            weight_map[key] = shard_name
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )


def _write_format3_vl_moe(
    ref_cfg: Qwen3MoeConfig,
    ref_model,
    out_dir: Path,
    *,
    transpose_gate_up: bool = True,
    transpose_down: bool = True,
):
    """Format 3: Qwen3VLMoeForConditionalGeneration with separate visual shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dict = ref_cfg.to_dict()
    text_dict.pop("architectures", None)
    vl_cfg = Qwen3VLMoeConfig(
        text_config=text_dict,
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": ref_cfg.hidden_size,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
            "deepstack_visual_indexes": [],
        },
        tie_word_embeddings=False,
        architectures=["Qwen3VLMoeForConditionalGeneration"],
    )
    vl_cfg.save_pretrained(out_dir)

    # Create dummy visual weights matching Qwen3VLMoeForConditionalGeneration
    # so --load-mode full (Qwen3VLMoeForConditionalGeneration.from_pretrained)
    # can also load this fixture cleanly.
    torch.manual_seed(999)
    vl_dummy = Qwen3VLMoeForConditionalGeneration(vl_cfg).eval().half()
    visual_tensors = {
        k: v.detach().contiguous()
        for k, v in vl_dummy.state_dict().items()
        if k.startswith("model.visual.")
    }
    del vl_dummy

    text_shard_0 = {}
    text_shard_1 = {}
    for name, tensor in ref_model.state_dict().items():
        t = tensor.detach().contiguous()
        if name.endswith(".mlp.experts.gate_up_proj") and transpose_gate_up:
            t = t.transpose(1, 2).contiguous()
        elif name.endswith(".mlp.experts.down_proj") and transpose_down:
            t = t.transpose(1, 2).contiguous()
        if name.startswith("model."):
            disk_key = "model.language_model." + name[len("model.") :]
        else:
            disk_key = name
        if (
            name == "model.embed_tokens.weight"
            or name.startswith("model.layers.0.")
        ):
            text_shard_0[disk_key] = t
        else:
            text_shard_1[disk_key] = t

    shards = {
        "model-visual.safetensors": visual_tensors,
        "model-text-0.safetensors": text_shard_0,
        "model-text-1.safetensors": text_shard_1,
    }
    weight_map = {}
    for shard_name, data in shards.items():
        save_file(data, out_dir / shard_name)
        for key in data:
            weight_map[key] = shard_name
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )


def _make_stage_config(model_dir: Path, *, variant="w4g32", prefill_ime=False):
    chunk = 64 if prefill_ime else 32
    spec = {
        "hf_model_path": str(model_dir),
        "model_family": "qwen3_vl_moe",
        "variant": variant,
        "max_token_len": 64,
        "num_threads": 2,
        "prefill_chunk": chunk if variant == "w4g32" else 0,
        "arena": variant == "w4g32",
        "hugepages": False,
        "thread_pool": variant == "w4g32",
        "prefill_ime": prefill_ime,
    }
    return gen_config.gen_config(spec, str(model_dir / "config.json"))


def _assert_models_equal(tc: unittest.TestCase, left, right):
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())
    tc.assertEqual(list(left_params.keys()), list(right_params.keys()))
    for name, lp in left_params.items():
        rp = right_params[name]
        tc.assertEqual(lp.device.type, "cpu", msg=f"{name} left device")
        tc.assertEqual(rp.device.type, "cpu", msg=f"{name} right device")
        tc.assertEqual(lp.dtype, rp.dtype, msg=f"{name} dtype mismatch")
        tc.assertEqual(lp.shape, rp.shape, msg=f"{name} shape mismatch")
        tc.assertTrue(
            torch.equal(lp, rp), msg=f"parameter {name} values differ"
        )
    for name, buf in left.named_buffers():
        tc.assertNotEqual(buf.device.type, "meta", msg=f"meta buffer {name}")


class TestMoeStageExport(unittest.TestCase):
    def test_three_formats_lazy_matches_full_and_skips_unrelated_shards(self):
        ref_cfg, ref_model = _make_ref_moe(
            seed=17,
            num_hidden_layers=3,
            hidden_size=128,
            moe_intermediate_size=256,
            num_experts=4,
            head_dim=64,
            vocab_size=256,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f1_dir = root / "fmt1_merged"
            f2_dir = root / "fmt2_hub_split"
            f3_dir = root / "fmt3_vl_moe"
            _write_format1_merged(ref_cfg, ref_model, f1_dir)
            _write_format2_hub_split(ref_cfg, ref_model, f2_dir)
            _write_format3_vl_moe(ref_cfg, ref_model, f3_dir)

            # Check lazy == full for every format across first/middle/last stages.
            for fmt_name, model_dir in (
                ("format1_merged", f1_dir),
                ("format2_hub_split", f2_dir),
                ("format3_vl_moe", f3_dir),
            ):
                cfg = _make_stage_config(model_dir)
                for start, end in ((0, 1), (1, 2), (2, 3), (0, 2)):
                    with self.subTest(fmt=fmt_name, interval=(start, end)):
                        lazy_m, lazy_t, lazy_w = ips.load_stage_lazy(
                            cfg, start, end
                        )
                        full_m, full_t, full_w = ips.load_stage_full(
                            cfg, start, end
                        )
                        self.assertEqual((lazy_t, lazy_w), (3, 128))
                        self.assertEqual((full_t, full_w), (3, 128))
                        _assert_models_equal(self, lazy_m, full_m)
                        for layer in lazy_m.model.layers:
                            self.assertEqual(
                                layer.mlp.experts.config._experts_implementation,
                                "buddy",
                            )
                        del lazy_m, full_m

            # Verify Format 3 lazy [0, 1) never opens visual or text-1 shards.
            opened_shards: list[str] = []
            orig_safe_open = ips.safe_open

            def spy_safe_open(path, *args, **kwargs):
                opened_shards.append(Path(path).name)
                return orig_safe_open(path, *args, **kwargs)

            ips.safe_open = spy_safe_open
            try:
                ips.load_stage_lazy(_make_stage_config(f3_dir), 0, 1)
            finally:
                ips.safe_open = orig_safe_open
            self.assertNotIn("model-visual.safetensors", opened_shards)
            self.assertNotIn("model-text-1.safetensors", opened_shards)
            self.assertIn("model-text-0.safetensors", opened_shards)

            # End-to-end export_stage [0, 2) on all 3 formats when buddy_mlir
            # Python bindings are available: verify identical weight SHA-256.
            try:
                import buddy_mlir.execution_engine  # noqa: F401

                has_buddy_mlir = True
            except ImportError:
                has_buddy_mlir = False

            if has_buddy_mlir:
                hashes = {}
                for fmt_name, model_dir in (
                    ("format1_merged", f1_dir),
                    ("format2_hub_split", f2_dir),
                    ("format3_vl_moe", f3_dir),
                ):
                    out_dir = root / f"export_{fmt_name}_0_2"
                    torch._dynamo.reset()
                    ips.export_stage(
                        _make_stage_config(model_dir),
                        0,
                        2,
                        out_dir,
                        load_mode="lazy",
                    )
                    header = out_dir / "stage-config.inc"
                    vps.verify_stage(
                        json.loads((out_dir / "config.json").read_text()),
                        out_dir,
                        header,
                    )
                    self.assertTrue(header.is_file())
                    hashes[fmt_name] = (
                        _sha256(out_dir / "arg0-w4g32-f32.data"),
                        _sha256(out_dir / "arg0-w4g32-q4.data"),
                    )
                self.assertEqual(
                    hashes["format1_merged"], hashes["format2_hub_split"]
                )
                self.assertEqual(
                    hashes["format1_merged"], hashes["format3_vl_moe"]
                )

    def test_square_expert_2i_equals_h_disambiguation_and_errors(self):
        # 2 * moe_intermediate_size == hidden_size (2 * 128 == 256):
        # gate_up_proj is square [E, 256, 256], while down_proj is [E, 256, 128].
        ref_cfg, ref_model = _make_ref_moe(
            seed=29,
            num_hidden_layers=2,
            hidden_size=256,
            moe_intermediate_size=128,
            num_experts=4,
            head_dim=64,
            vocab_size=256,
        )
        # Ensure gate_up_proj is strongly asymmetric so transpose vs identity is
        # unmistakably distinguishable.
        with torch.no_grad():
            ramp = torch.arange(256, dtype=torch.float16)
            asym = (ramp[:, None] * 0.01 - ramp[None, :] * 0.003).unsqueeze(0)
            for layer in ref_model.model.layers:
                layer.mlp.experts.gate_up_proj.add_(asym)
                gu = layer.mlp.experts.gate_up_proj
                self.assertFalse(torch.equal(gu, gu.transpose(1, 2)))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vl_transposed = root / "vl_2i_eq_h_transposed"
            vl_native = root / "vl_2i_eq_h_native"
            _write_format3_vl_moe(
                ref_cfg,
                ref_model,
                vl_transposed,
                transpose_gate_up=True,
                transpose_down=True,
            )
            _write_format3_vl_moe(
                ref_cfg,
                ref_model,
                vl_native,
                transpose_gate_up=False,
                transpose_down=False,
            )

            expected_gu = ref_model.model.layers[0].mlp.experts.gate_up_proj.float()
            expected_dn = ref_model.model.layers[0].mlp.experts.down_proj.float()

            # Both Hub-transposed and 5.x-native VL-MoE resolve unambiguously
            # in lazy mode using the non-square down_proj partner.
            for case_name, model_dir in (
                ("hub_transposed", vl_transposed),
                ("native_5x", vl_native),
            ):
                with self.subTest(case=case_name):
                    m, _, _ = ips.load_stage_lazy(
                        _make_stage_config(model_dir), 0, 1
                    )
                    self.assertTrue(
                        torch.equal(
                            m.model.layers[0].mlp.experts.gate_up_proj,
                            expected_gu,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            m.model.layers[0].mlp.experts.down_proj,
                            expected_dn,
                        )
                    )

            # --load-mode full must refuse Hub-transposed square VL-MoE rather
            # than silently skipping transpose in transformers check_dims=True.
            with self.assertRaisesRegex(ValueError, "lazy"):
                ips.load_stage_full(_make_stage_config(vl_transposed), 0, 1)

        # Inconsistent orientation on non-square VL-MoE must raise ValueError.
        ns_cfg, ns_model = _make_ref_moe(
            seed=31,
            num_hidden_layers=2,
            hidden_size=128,
            moe_intermediate_size=256,
            num_experts=4,
            head_dim=64,
            vocab_size=256,
        )
        with tempfile.TemporaryDirectory() as td:
            bad_vl = Path(td) / "vl_inconsistent"
            _write_format3_vl_moe(
                ns_cfg,
                ns_model,
                bad_vl,
                transpose_gate_up=True,
                transpose_down=False,
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                ips.load_stage_lazy(_make_stage_config(bad_vl), 0, 1)

    def test_negative_validation_moe_f16_and_config_constraints(self):
        ref_cfg, ref_model = _make_ref_moe(seed=41, num_hidden_layers=2)
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "m"
            _write_format1_merged(ref_cfg, ref_model, model_dir)
            cfg_f16 = _make_stage_config(model_dir, variant="f16")
            with self.assertRaisesRegex(ValueError, "w4g32.*MoEExpertsOp"):
                ips.export_stage(cfg_f16, 0, 1, Path(td) / "out")

        for field, bad_val in (
            ("norm_topk_prob", False),
            ("mlp_only_layers", [0]),
            ("decoder_sparse_step", 2),
            ("use_sliding_window", True),
        ):
            bad_cfg = copy.deepcopy(ref_cfg)
            setattr(bad_cfg, field, bad_val)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ips.validate_stage_interval(bad_cfg, 0, 1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("MoE stage export tests passed")
