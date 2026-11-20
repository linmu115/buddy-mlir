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
"""Pure-Python unit tests for Step 1 config, manifest, and stage verification."""

import copy
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_DIR = REPO_ROOT / "tools" / "buddy-codegen"
if str(CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(CODEGEN_DIR))

import gen_config  # noqa: E402
import gen_manifest  # noqa: E402
import verify_pipeline_stage  # noqa: E402


def _qwen3_moe_hf(
    *,
    eos_token_id=151645,
    head_dim=64,
    num_hidden_layers=4,
    hidden_size=256,
    moe_intermediate_size=128,
    num_experts=8,
):
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "vocab_size": 151936,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "max_position_embeddings": 256,
        "tie_word_embeddings": False,
        "rope_theta": 1000000.0,
        "eos_token_id": eos_token_id,
    }


def _qwen3_moe_spec(**overrides):
    spec = {
        "hf_model_path": "/nonexistent/tiny-qwen3-moe",
        "model_family": "qwen3_vl_moe",
        "variant": "w4g32",
        "max_token_len": 128,
        "num_threads": 4,
        "prefill_chunk": 32,
        "arena": True,
        "hugepages": False,
        "thread_pool": True,
        "prefill_ime": False,
    }
    spec.update(overrides)
    return spec


def _gen_with_hf(spec, hf_dict):
    with tempfile.TemporaryDirectory() as td:
        hf_path = Path(td) / "config.json"
        hf_path.write_text(json.dumps(hf_dict))
        return gen_config.gen_config(spec, str(hf_path))


class TestGenConfigNormalization(unittest.TestCase):
    def test_eos_list_normalization_and_dedup(self):
        cfg = _gen_with_hf(
            _qwen3_moe_spec(),
            _qwen3_moe_hf(eos_token_id=[151645, 151643, 151645]),
        )
        self.assertEqual(cfg["tokens"]["eos_id"], 151645)
        self.assertEqual(cfg["tokens"]["stop_token_ids"], [151645, 151643])

    def test_eos_int_keeps_no_stop_token_ids(self):
        cfg = _gen_with_hf(_qwen3_moe_spec(), _qwen3_moe_hf(eos_token_id=151645))
        self.assertEqual(cfg["tokens"]["eos_id"], 151645)
        self.assertNotIn("stop_token_ids", cfg["tokens"])

    def test_eos_invalid_rejected(self):
        for bad in ([], ["151645"], [True], -1, True, "151645"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _gen_with_hf(_qwen3_moe_spec(), _qwen3_moe_hf(eos_token_id=bad))

    def test_tokenizer_normalization(self):
        for family, expected in (
            ("qwen3_vl_moe", "qwen3"),
            ("qwen3_moe", "qwen3"),
            ("qwen3", "qwen3"),
            ("deepseek_r1", "deepseek_r1"),
        ):
            with self.subTest(family=family):
                cfg = _gen_with_hf(
                    _qwen3_moe_spec(model_family=family), _qwen3_moe_hf()
                )
                self.assertEqual(cfg["tokens"]["tokenizer"], expected)

    def test_arena_prefault_mb_passthrough_and_validation(self):
        cfg_without = _gen_with_hf(_qwen3_moe_spec(), _qwen3_moe_hf())
        self.assertNotIn("arena_prefault_mb", cfg_without)

        cfg_with = _gen_with_hf(
            _qwen3_moe_spec(arena_prefault_mb=1536), _qwen3_moe_hf()
        )
        self.assertEqual(cfg_with["arena_prefault_mb"], 1536)

        with self.assertRaises(ValueError):
            _gen_with_hf(
                _qwen3_moe_spec(arena=False, arena_prefault_mb=1536),
                _qwen3_moe_hf(),
            )
        for bad in (-1, True, "1536", 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _gen_with_hf(
                        _qwen3_moe_spec(arena_prefault_mb=bad), _qwen3_moe_hf()
                    )

    def test_w4g32_stage_param_counts_additivity_moe(self):
        for head_dim, ime in ((64, False), (128, False), (128, True)):
            hf = _qwen3_moe_hf(head_dim=head_dim, num_hidden_layers=4)
            chunk = 64 if ime else 32
            full = gen_config.w4g32_param_counts(hf, chunk, ime)

            for splits in (((0, 2), (2, 4)), ((0, 1), (1, 3), (3, 4))):
                i8_sum = 0
                f32_sum = 0
                for start, end in splits:
                    part = gen_config.w4g32_stage_param_counts(
                        hf, start, end, 4, ime
                    )
                    i8_sum += part["i8_elements"]
                    f32_sum += part["f32_elements"]
                self.assertEqual(i8_sum, full["i8_elements"])
                # Each stage carries one copy of RoPE inv_freq (head_dim // 2).
                extra_rope = (len(splits) - 1) * (head_dim // 2)
                self.assertEqual(f32_sum - extra_rope, full["f32_elements"])


class TestGenManifestStageAttrs(unittest.TestCase):
    def _stage_cfg(self, *, eos_token_id=151645, model_family="qwen3_vl_moe"):
        cfg = _gen_with_hf(
            _qwen3_moe_spec(model_family=model_family),
            _qwen3_moe_hf(eos_token_id=eos_token_id),
        )
        cfg["shape"]["num_hidden_layers"] = 2
        cfg["shape"]["kv_layers"] = 4
        cfg["pipeline_stage"] = {
            "start_layer": 0,
            "end_layer": 2,
            "total_layers": 4,
            "activation_type": "f32",
            "activation_width": 256,
        }
        return cfg

    def test_optional_wire_policy(self):
        cfg = self._stage_cfg()
        default = gen_manifest.gen_manifest(cfg)
        self.assertNotIn("pp_wire_dtype", default)
        self.assertNotIn("pp_transport", default)
        for dtype in ("native", "f16"):
            for transport in ("blocking", "async"):
                explicit = gen_manifest.gen_manifest(
                    cfg, wire_dtype=dtype, transport=transport
                )
                self.assertIn(f'pp_wire_dtype = "{dtype}"', explicit)
                self.assertIn(f'pp_transport = "{transport}"', explicit)
        with self.assertRaisesRegex(ValueError, "pp_wire_dtype"):
            gen_manifest.gen_manifest(cfg, wire_dtype="fp8")

    def test_emits_pp_tokenizer_and_stop_token_ids(self):
        cfg = self._stage_cfg(eos_token_id=[151645, 151643])
        text = gen_manifest.gen_manifest(cfg)
        self.assertIn('pp_tokenizer = "qwen3",', text)
        self.assertIn('pp_stop_token_ids = "151645,151643",', text)

    def test_omits_stop_ids_when_absent_and_supports_deepseek_r1(self):
        cfg = self._stage_cfg(eos_token_id=151643, model_family="deepseek_r1")
        text = gen_manifest.gen_manifest(cfg)
        self.assertIn('pp_tokenizer = "deepseek_r1",', text)
        self.assertNotIn("pp_stop_token_ids", text)

    def test_omits_tokenizer_when_missing_or_unknown(self):
        cfg = self._stage_cfg()
        cfg["tokens"].pop("tokenizer")
        text = gen_manifest.gen_manifest(cfg)
        self.assertNotIn("pp_tokenizer", text)

        cfg["tokens"]["tokenizer"] = "unknown"
        text = gen_manifest.gen_manifest(cfg)
        self.assertNotIn("pp_tokenizer", text)

    def test_rejects_invalid_tokenizer_or_stop_ids(self):
        cfg = self._stage_cfg()
        cfg["tokens"]["tokenizer"] = "unsupported_tok"
        with self.assertRaises(ValueError):
            gen_manifest.gen_manifest(cfg)

        for bad_stops in ([], "151645", [True], [-1]):
            bad_cfg = copy.deepcopy(self._stage_cfg())
            bad_cfg["tokens"]["stop_token_ids"] = bad_stops
            with self.subTest(bad_stops=bad_stops):
                with self.assertRaises(ValueError):
                    gen_manifest.gen_manifest(bad_cfg)


class TestVerifyStageDefensiveChecks(unittest.TestCase):
    def test_rejects_non_int_eos_id(self):
        cfg = _gen_with_hf(_qwen3_moe_spec(), _qwen3_moe_hf())
        cfg["shape"]["num_hidden_layers"] = 2
        cfg["shape"]["kv_layers"] = 4
        cfg["pipeline_stage"] = {
            "start_layer": 0,
            "end_layer": 2,
            "total_layers": 4,
            "activation_type": "f32",
            "activation_width": 256,
        }
        with tempfile.TemporaryDirectory() as td:
            for bad_eos in ([151645, 151643], "151645", True, -1, None):
                bad_cfg = copy.deepcopy(cfg)
                bad_cfg["tokens"]["eos_id"] = bad_eos
                with self.subTest(bad_eos=bad_eos):
                    with self.assertRaises(ValueError):
                        verify_pipeline_stage.verify_stage(
                            bad_cfg, Path(td), None
                        )


class TestCheckResultLogitsDtype(unittest.TestCase):
    CHECK_SCRIPT = (
        REPO_ROOT / "examples" / "BuddyDeepSeekR1" / "pipeline" / "check_result.py"
    )

    def _run_check(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.CHECK_SCRIPT), *extra_args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_f32_bit_exact_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_json = root / "ref.json"
            act_json = root / "act.json"
            payload = json.dumps({"prompt_tokens": 9, "token_ids": [10, 20]})
            ref_json.write_text(payload, encoding="utf-8")
            act_json.write_text(payload, encoding="utf-8")

            vals = [0.25, -1.5, 3.0, 0.0, -0.0, 1.25]
            raw = struct.pack("<6f", *vals)
            ref_f32 = root / "ref.f32"
            act_f32 = root / "act.f32"
            ref_f32.write_bytes(raw)
            act_f32.write_bytes(raw)

            ok = self._run_check(
                str(ref_json),
                str(act_json),
                "--logits",
                str(ref_f32),
                str(act_f32),
                "--vocab-size",
                "3",
                "--dtype",
                "f32",
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("6 sampled FP32 logit elements match bit for bit", ok.stdout)

            # Flip 1 bit (-0.0 -> +0.0 at step 2, ID 1) — must fail bit-exact check.
            bad_vals = [0.25, -1.5, 3.0, 0.0, 0.0, 1.25]
            act_f32.write_bytes(struct.pack("<6f", *bad_vals))
            bad = self._run_check(
                str(ref_json),
                str(act_json),
                "--logits",
                str(ref_f32),
                str(act_f32),
                "--vocab-size",
                "3",
                "--dtype",
                "f32",
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("step 2, ID 1", bad.stderr)
            self.assertIn("0x80000000 vs 0x00000000", bad.stderr)

    def test_f16_default_backward_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_json = root / "ref.json"
            act_json = root / "act.json"
            payload = json.dumps({"token_ids": [5]})
            ref_json.write_text(payload, encoding="utf-8")
            act_json.write_text(payload, encoding="utf-8")

            raw = struct.pack("<3H", 0x3C00, 0xBC00, 0x0000)
            ref_f16 = root / "ref.f16"
            act_f16 = root / "act.f16"
            ref_f16.write_bytes(raw)
            act_f16.write_bytes(raw)

            ok = self._run_check(
                str(ref_json),
                str(act_json),
                "--logits",
                str(ref_f16),
                str(act_f16),
                "--vocab-size",
                "3",
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("3 sampled FP16 logit elements match bit for bit", ok.stdout)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("Pipeline codegen config tests passed")
