"""Run directly with the importer venv."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
)
import unittest

import torch
from gen_config import (
    derive_prefill_ime,
    w4g32_component_param_counts,
    w4g32_stage_param_counts,
)
from import_pipeline_stage import checkpoint_key_for_param, select_layers
from transformers import Qwen2Config, Qwen2ForCausalLM


class CountsAndRoles(unittest.TestCase):
    def test_moe_attention_only_has_duplicate_ime_tiles(self):
        hf = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 151936,
            "moe_intermediate_size": 128,
            "num_experts": 8,
            "norm_topk_prob": True,
            "num_hidden_layers": 4,
        }
        for ime in (False, True):
            for start in range(4):
                for end in range(start + 1, 5):
                    layers = end - start
                    counts = w4g32_stage_param_counts(hf, start, end, 4, ime)
                    self.assertEqual(
                        counts["f32_elements"],
                        32
                        + 2688 * layers
                        + (start == 0) * 38895616
                        + (end == 4) * 256,
                    )
                    self.assertEqual(
                        counts["i8_elements"],
                        layers * (442368 + 110592 * (1 + ime))
                        + (end == 4) * 21878784,
                    )
        for extra in (
            {"use_sliding_window": True},
            {"layer_types": ["sliding_attention"]},
            {"decoder_sparse_step": 2},
            {"mlp_only_layers": [0]},
            {"norm_topk_prob": False},
        ):
            with self.assertRaises(ValueError):
                w4g32_component_param_counts({**hf, **extra}, True)

    def test_route_b_counts_and_roles(self):
        hf = {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
        }
        for ime in (False, True):
            for start in range(28):
                for end in range(start + 1, 29):
                    n = end - start
                    # Frozen independent baseline for the 1.5B checkpoint.
                    self.assertEqual(
                        w4g32_stage_param_counts(hf, start, end, 28, ime),
                        {
                            "f32_elements": 64
                            + n * 5120
                            + (start == 0) * 233373696
                            + (end == 28) * 1536,
                            "i8_elements": n * 26320896 * (1 + ime)
                            + (end == 28) * 131272704,
                        },
                    )
        for start, end in ((0, 0), (-1, 1), (1, 29)):
            with self.assertRaises(ValueError):
                w4g32_stage_param_counts(hf, start, end, 28, True)
        for change in [
            {"vocab_size": 151935},
            {"hidden_size": 1535},
            {"tie_word_embeddings": True},
        ]:
            with self.assertRaises(ValueError):
                w4g32_component_param_counts(hf | change, True)
        with self.assertRaises(ValueError):
            derive_prefill_ime({"prefill_ime": True}, "w4g32", 32)
        moe = hf | {
            "architectures": ["Qwen3MoeForCausalLM"],
            "moe_intermediate_size": 256,
            "num_experts": 4,
            "norm_topk_prob": True,
        }
        plain = w4g32_component_param_counts(moe, False)["decoder"]
        ime = w4g32_component_param_counts(moe, True)["decoder"]
        self.assertEqual(plain["f32_elements"], 2 * 1536 + 2 * 128 + 4 * 1536)
        self.assertEqual(
            2 * plain["i8_elements"] - ime["i8_elements"],
            4 * 3 * 1536 * 256 * 9 // 16,
        )
        for start in (0, 27):
            for role in ("stage", "decoder", "embed", "head"):
                with torch.device("meta"):
                    m = Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=28))
                select_layers(m, start, start + 1, 28, role)
                names = dict(m.named_parameters())
                self.assertEqual(
                    "model.embed_tokens.weight" in names,
                    role == "embed" or role == "stage" and start == 0,
                )
                self.assertEqual(
                    "lm_head.weight" in names,
                    role == "head" or role == "stage" and start == 27,
                )
                self.assertEqual(
                    len(m.model.layers), int(role in ("decoder", "stage"))
                )
                if m.model.layers:
                    self.assertEqual(m.model.layers[0].self_attn.layer_idx, 0)
                    key = f"model.layers.{start}.self_attn.q_proj.weight"
                    self.assertEqual(
                        checkpoint_key_for_param(
                            "model.layers.0.self_attn.q_proj.weight",
                            start,
                            {key},
                            False,
                        ),
                        key,
                    )


if __name__ == "__main__":
    unittest.main()
