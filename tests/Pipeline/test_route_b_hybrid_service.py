#!/usr/bin/env python3
"""Gate B1 range, dual-stream and resident slot-reuse tests."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from route_b_test_support import Resident

PROMPTS = {"A": "b" * 65, "B": "a" * 56, "C": "c" * 24}
ROWS = {"A": 73, "B": 64, "C": 32}
TOTAL = 88
RESIDENT_LENGTHS = {"A": 16, "B": 5, "C": 7}
CUTS = ["0:4", "0:2,2:4", "0:3,3:4", "0:1,1:4", "0:1,1:2,2:3,3:4"]


def run(args):
    env = os.environ | {
        "MPIR_CVAR_CH4_SHM_ENABLE": "0",
        "FI_PROVIDER": "tcp",
    }
    launch = shlex.split(args.launcher)
    manifest = json.loads(args.bundle.read_text())

    def hashes():
        return {
            entry["path"]: hashlib.sha256(
                (args.bundle.parent / entry["path"]).read_bytes()
            ).hexdigest()
            for entry in manifest["templates"]
        }

    before = hashes()

    def base(cut):
        return [
            str(args.mpiexec),
            "-n",
            str(cut.count(",") + 1),
            *launch,
            str(args.binary),
            "--bundle",
            str(args.bundle),
            "--ranges",
            cut,
            "--transport",
            args.transport,
        ]

    def invoke(cut, names, total):
        prompts = [arg for key in names for arg in ("--prompt", PROMPTS[key])]
        command = [*base(cut), *prompts, "--total-tokens", str(total)]
        res = subprocess.run(
            command, text=True, capture_output=True, timeout=240, env=env
        )
        label = cut.replace(":", "-").replace(",", "_") + "." + "".join(names)
        (args.work / (label + ".log")).write_text(res.stdout + res.stderr)
        res.check_returncode()
        return json.loads(res.stdout)

    # Token ID 0 is valid data in the fixture. Removing the only match for
    # "a" must raise an encoding error instead of selecting it as a fallback.
    with tempfile.TemporaryDirectory(dir=args.work) as temp:
        folder = Path(temp)
        root = args.bundle.resolve().parent
        for entry in [*manifest["weights"], *manifest["templates"]]:
            (folder / entry["path"]).symlink_to(root / entry["path"])
        vocab = (root / manifest["vocab_file"]).read_text().split("\n")
        assert vocab[ord("a")] == "a"
        vocab[ord("a")] = ""
        bad_vocab = folder / manifest["vocab_file"]
        bad_vocab.write_text("\n".join(vocab))
        modified = manifest.copy()
        modified.pop("bundle_id")
        modified["vocab_sha256"] = hashlib.sha256(
            bad_vocab.read_bytes()
        ).hexdigest()
        modified["bundle_id"] = hashlib.sha256(
            json.dumps(modified, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        bad_bundle = folder / args.bundle.name
        bad_bundle.write_text(json.dumps(modified))
        broken = base("0:4")
        broken[broken.index("--bundle") + 1] = str(bad_bundle)
        rejected = subprocess.run(
            [*broken, "--prompt", "a", "--total-tokens", "9"],
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
        assert (
            rejected.returncode
            and "vocabulary cannot encode the full prompt" in rejected.stderr
        ), rejected

    refs = {}
    for key in PROMPTS:
        total = TOTAL if key != "C" else ROWS["C"] + RESIDENT_LENGTHS["C"] - 1
        ref = invoke("0:4", [key], total)
        assert ref["prompt_tokens"] == ROWS[key], ref
        assert len(ref["token_ids"]) == total - ROWS[key] + 1, ref
        refs[key] = ref["token_ids"]

    if args.oracle:
        oracle = json.loads(args.oracle.read_text())
        expected = {
            case["prompt_rows"]: case["tokens"] for case in oracle["cases"]
        }
        for key in "AB":
            assert refs[key][:16] == expected[ROWS[key]], (key, refs[key])

    def verify(cut):
        single = invoke(cut, ["A"], TOTAL)
        assert (
            single["token_ids"] == refs["A"]
            and single["prompt_tokens"] == ROWS["A"]
        )
        batch = invoke(cut, ["A", "B"], TOTAL)
        for key, stream in zip("AB", batch["streams"], strict=True):
            assert (
                stream["token_ids"] == refs[key]
                and stream["prompt_tokens"] == ROWS[key]
            )
        wanted = {
            key: refs[key][:length] for key, length in RESIDENT_LENGTHS.items()
        }
        with Resident([*base(cut), "--serve", "--slots", "2"], env) as server:
            assert server.until(lambda e: e["event"] == "ready")["slots"] == 2
            server.submit(
                "A", PROMPTS["A"], max_new_tokens=RESIDENT_LENGTHS["A"]
            )
            server.until(lambda e: e["event"] == "token" and e["step"] == 2)
            server.submit(
                "B", PROMPTS["B"], max_new_tokens=RESIDENT_LENGTHS["B"]
            )
            first_b = server.until(
                lambda e: e["event"] == "token" and e["req_id"] == "B"
            )
            assert not any(
                e.get("req_id") == "A" and e["event"] == "done"
                for e in server.log
            )
            server.done("B")
            server.submit(
                "C", PROMPTS["C"], max_new_tokens=RESIDENT_LENGTHS["C"]
            )
            first_c = server.until(
                lambda e: e["event"] == "token" and e["req_id"] == "C"
            )
            assert first_c["slot"] == first_b["slot"], (first_b, first_c)
            server.until(
                lambda _: sum(e["event"] == "done" for e in server.log) == 3
            )
            server.sequences(wanted, "ABC")
            label = cut.replace(":", "-").replace(",", "_")
            (args.work / (label + ".resident.json")).write_text(
                json.dumps(server.log, indent=2) + "\n"
            )
        return {
            "ranges": cut,
            "ranks": cut.count(",") + 1,
            "single_tokens": len(refs["A"]),
            "dual_tokens": sum(len(refs[k]) for k in "AB"),
            "resident_tokens": sum(map(len, wanted.values())),
            "token_diff": 0,
            "reused_slot": first_c["slot"],
            "staggered": True,
        }

    result = {
        "manifest": str(args.bundle),
        "templates_sha256": before,
        "transport": args.transport,
        "oracle_checked": bool(args.oracle),
        "unencodable_prompt_rejected": True,
        "single_rank_references": refs,
        "ranges": [],
    }
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(verify, args.cuts):
            result["ranges"].append(row)
            print(json.dumps(row), flush=True)
    assert before == hashes()
    result["unchanged_templates"] = True
    (args.work / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "bundle", "mpiexec", "work"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cuts", nargs="+", default=CUTS)
    parser.add_argument(
        "--transport", choices=["blocking", "async"], default="blocking"
    )
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    run(args)
