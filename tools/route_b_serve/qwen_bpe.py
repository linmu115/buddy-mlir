"""Resident Qwen byte-level BPE and official chat template for Route B."""

import heapq
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from jinja2 import TemplateError
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Generation(Extension):
    """The inference renderer emits the assistant span without training masks."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(
            ("name:endgeneration",), drop_needle=True
        )


def template_error(message):
    raise ValueError(message)


class Normalizer:
    def __init__(self, tables):
        self.decompose = tables["decompose"]
        self.compose = tables["compose"]
        self.order = tables["order"]

    def nfc(self, text):
        if text.isascii():
            return text
        ordered, marks = [], []
        for c in text:
            for char in self.decompose.get(c, c):
                if self.order.get(char):
                    marks.append(char)
                else:
                    ordered.extend(sorted(marks, key=self.order.__getitem__))
                    marks.clear()
                    ordered.append(char)
        ordered.extend(sorted(marks, key=self.order.__getitem__))
        result = []
        starter = None
        last_order = 0
        for char in ordered:
            order = self.order.get(char, 0)
            if starter is not None and (last_order == 0 or last_order < order):
                composite = self.compose.get(result[starter] + char)
                if composite:
                    result[starter] = composite
                    continue
            if order == 0:
                starter = len(result)
            result.append(char)
            last_order = order
        return "".join(result)


def byte_alphabet():
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    missing = sorted(set(range(256)) - set(visible))
    return dict(
        zip(
            visible + missing,
            map(chr, visible + list(range(256, 256 + len(missing)))),
        )
    )


def compile_split(pattern, classes):
    """Expand Unicode property atoms for the stdlib regex engine."""
    properties = {
        name: "".join(
            re.escape(chr(lo))
            if lo == hi
            else re.escape(chr(lo)) + "-" + re.escape(chr(hi))
            for lo, hi in intervals
        )
        for name, intervals in classes.items()
    }
    whitespace = properties["space"]
    in_class = False

    def expand(match):
        nonlocal in_class
        atom = match[0]
        if atom in ("[", "]"):
            in_class = atom == "["
            return atom
        if atom.startswith(r"\p{"):
            ranges = properties[atom[3:-1]]
        elif atom == r"\s":
            ranges = whitespace
        elif atom == r"\S" and not in_class:
            return "[^" + whitespace + "]"
        else:
            return atom
        return ranges if in_class else "[" + ranges + "]"

    return re.compile(re.sub(r"\\p\{[LMN]\}|\\.|\[|\]", expand, pattern))


class Tokenizer:
    def __init__(self, directory, tables=None):
        if tables is None:
            tables = json.loads((directory / "unicode.json").read_bytes())
        self.normalizer = Normalizer(tables)
        data = json.loads((directory / "tokenizer.json").read_bytes())
        model = data["model"]
        split, byte = data["pre_tokenizer"]["pretokenizers"]
        if (
            model["type"] != "BPE"
            or model["ignore_merges"]
            or model["byte_fallback"]
            or model["dropout"]
            or data["normalizer"] != {"type": "NFC"}
            or split["behavior"] != "Isolated"
            or split["invert"]
            or byte["use_regex"]
            or byte["add_prefix_space"]
        ):
            raise ValueError("unsupported tokenizer configuration")
        self.split = compile_split(split["pattern"]["Regex"], tables["classes"])
        self.vocab = model["vocab"]
        self.ranks = {}
        for rank, merge in enumerate(model["merges"]):
            a, b = merge.split() if isinstance(merge, str) else merge
            self.ranks[self.vocab[a], self.vocab[b]] = rank, self.vocab[a + b]
        self.bytes = [
            self.vocab[ch] for _, ch in sorted(byte_alphabet().items())
        ]
        self.added = {
            item["content"]: item["id"] for item in data["added_tokens"]
        }
        self.special = re.compile(
            "("
            + "|".join(
                map(re.escape, sorted(self.added, key=len, reverse=True))
            )
            + ")"
        )
        env = ImmutableSandboxedEnvironment(
            extensions=[Generation], trim_blocks=True, lstrip_blocks=True
        )
        env.filters["tojson"] = lambda x, **kwargs: json.dumps(
            x, ensure_ascii=False, **kwargs
        )
        env.globals["raise_exception"] = template_error
        self.template = env.from_string(
            (directory / "chat_template.jinja").read_text(encoding="utf8")
        )

        self.piece = lru_cache(maxsize=8192)(self.merge_piece)

    def merge_piece(self, text):
        ids = [self.bytes[b] for b in text.encode()]
        n = len(ids)
        left, right = list(range(-1, n - 1)), list(range(1, n + 1))
        queue = []

        def offer(i):
            if i >= 0 and right[i] < n:
                merge = self.ranks.get((ids[i], ids[right[i]]))
                if merge:
                    heapq.heappush(queue, (merge[0], i, right[i], merge[1]))

        for i in range(n - 1):
            offer(i)
        while queue:
            rank, i, j, token = heapq.heappop(queue)
            if (
                right[i] != j
                or ids[j] < 0
                or self.ranks.get((ids[i], ids[j])) != (rank, token)
            ):
                continue
            ids[i], ids[j] = token, -1
            right[i] = right[j]
            if right[i] < n:
                left[right[i]] = i
            offer(left[i])
            offer(i)
        return tuple(i for i in ids if i >= 0)

    def encode(self, text):
        ids = []
        for segment in self.special.split(text):
            if segment in self.added:
                ids.append(self.added[segment])
                continue
            for match in self.split.finditer(self.normalizer.nfc(segment)):
                piece = match[0]
                ids.extend(
                    self.piece(piece)
                    if len(piece) <= 4096
                    else self.merge_piece(piece)
                )
        return ids

    def chat(self, request):
        return self.encode(
            self.template.render(**request, add_generation_prompt=True)
        )


def main(directory):
    tokenizer = Tokenizer(directory)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            ids = tokenizer.chat(request)
            response = {"ids": ids}
        except (ValueError, TypeError, KeyError, TemplateError) as error:
            response = {"error": str(error)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
