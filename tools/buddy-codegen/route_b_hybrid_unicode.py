"""Freeze the Unicode semantics of the HF tokenizer used for the export."""

from functools import cmp_to_key

from tokenizers import Regex, __version__, normalizers, pre_tokenizers


def build_tables():
    # Surrogates are not Unicode scalar values. Match contiguous runs with
    # the same engine as the Split in tokenizer.json.
    intervals = [(0, 0xD800), (0xE000, 0x110000)]
    classes = {}
    for name, atom in (
        ("L", r"\p{L}"),
        ("M", r"\p{M}"),
        ("N", r"\p{N}"),
        ("space", r"\s"),
    ):
        split = pre_tokenizers.Split(Regex(atom + "+"), "removed", invert=True)
        ranges = []
        for lo, hi in intervals:
            for start in range(lo, hi, 16384):
                text = "".join(map(chr, range(start, min(hi, start + 16384))))
                for _, (a, b) in split.pre_tokenize_str(text):
                    ranges.append([start + a, start + b - 1])
        classes[name] = ranges

    nfd = normalizers.NFD().normalize_str
    nfc = normalizers.NFC().normalize_str
    decomposed, compose, marks = {}, {}, []
    high, low = "\u0345", "\u0334"
    for lo, hi in intervals:
        for i in range(lo, hi):
            char = chr(i)
            seq = nfd(char)
            if seq != char:
                decomposed[char] = seq
                if len(seq) > 1 and nfc(seq) == char:
                    prefix = nfc(seq[:-1])
                    if len(prefix) != 1:
                        raise ValueError(
                            f"unexpected canonical decomposition of U+{i:04X}"
                        )
                    compose[prefix + seq[-1]] = char
            # A starter blocks the low-class probe from moving before high.
            elif not nfd(high + char + low).startswith(high):
                marks.append(char)

    # Only class order and equality enter the NFC blocking rule. Derive
    # those comparisons from the exact normalization engine, even when its
    # Unicode version differs from Python's.
    def compare(a, b):
        return int(nfd(a + b) != a + b) - int(nfd(b + a) != b + a)

    marks.sort(key=cmp_to_key(compare))
    order, previous, rank = {}, None, 0
    for char in marks:
        if previous is None or compare(previous, char):
            rank += 1
        order[char] = rank
        previous = char
    return {
        "tokenizers_version": __version__,
        "classes": classes,
        "decompose": decomposed,
        "compose": compose,
        "order": order,
    }
