"""Check the deployed encoder, decoder and chat renderer against official assets."""

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer as Reference
from transformers.utils.chat_template_utils import _compile_jinja_template

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools/route_b_serve")
)
from qwen_bpe import Tokenizer, byte_alphabet

TEXTS = [
    "",
    "hello world",
    "中文混合 English 123",
    "def f(x):\n    return x**2",
    "∑ᵢ αᵢxᵢ = β",
    "0123456701234567890",
    "3.14159e-12",
    "a+b!=c?!",
    "line1\r\n\nline2",
    " \t \r\n  ",
    "Café",
    "Cafe\u0301",
    "👩🏽‍💻🚀",
    "🇨🇳 🇯🇵",
    "こんにちは世界",
    "العَرَبِيَّة",
    "नमस्ते",
    "Привет!",
    "I'm I'LL they'll",
    "a\u00a0b\u3000c",
    "<think>\n42\n</think>",
    "<|im_start|>assistant\n<|im_end|>",
    "\x00X\x1cY",
    "a" * 4097,
    "foo_bar::baz(42)\n\t//你好",
    "𠮷野家𝟡",
    "a乍\u0898\u0596ӧb",
    "<|endoftext|>",
    "x\u0301\u0327",
]


def verify(path):
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tools/buddy-codegen")
    )
    from route_b_hybrid_unicode import build_tables

    actual = Tokenizer(path, build_tables())
    reference = Reference.from_file(str(path / "tokenizer.json"))
    template = _compile_jinja_template(
        (path / "chat_template.jinja").read_text()
    )
    byte_values = {ch: b for b, ch in byte_alphabet().items()}
    added = {i: s.encode() for s, i in actual.added.items()}
    decode = {
        i: bytes(byte_values[c] for c in s) for s, i in actual.vocab.items()
    } | added
    cases = []

    def check(text, ids):
        expected = reference.encode(text).ids
        assert expected == ids, (text, expected, ids)
        reconstructed = b"".join(decode[i] for i in ids).decode("utf-8")
        assert reconstructed == reference.decode(
            expected, skip_special_tokens=False
        )
        assert "\ufffd" not in reconstructed
        cases.append(len(ids))

    for text in TEXTS:
        check(text, actual.encode(text))
    for thinking in (True, False):
        for effort in ("low", "medium", "xhigh"):
            messages = [
                {"role": role, "content": content}
                for role, content in [
                    ("system", "你是助手"),
                    ("user", "1+1?"),
                    ("assistant", "2"),
                    ("user", "why? 👋"),
                ]
            ]
            messages[2]["reasoning_content"] = "addition"
            args = {
                "messages": messages,
                "enable_thinking": thinking,
                "reasoning_effort": effort,
            }
            check(
                template.render(add_generation_prompt=True, **args),
                actual.chat(args),
            )
    print(
        json.dumps(
            {
                "exact_cases": len(cases),
                "token_ids": sum(cases),
                "decode_replacements": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    verify(parser.parse_args().assets)
