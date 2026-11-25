"""Exercise the real bundle loader, stage and MPI CLI with tiny native DSOs."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TEST, BUILD, MPI = sys.argv[1:]
BUILD = Path(BUILD)
ZERO = "00" * 32


def save(path, index):
    index = index.copy()
    index["bundle_id"] = hashlib.sha256(
        json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(index))


def fixture(root):
    index = {
        "format": "bj47.route-b/1",
        "model_id": "fixture",
        "shape_id": ZERO,
        "variant": "w4g32",
        "kernels": "k3",
        "prefill_ime": True,
        "architecture": "Qwen2ForCausalLM",
        "activation_type": "f32",
        "pack_version": "k3-w4g32/1",
        "total_layers": 28,
        "context": 128,
        "activation_width": 16,
        "vocab_size": 151936,
        "eos_id": 151643,
        "prefill_chunk": 64,
        "kv_heads": 1,
        "head_dim": 16,
        "tokenizer": "deepseek_r1",
        "stop_token_ids": [],
        "vocab_file": "vocab.txt",
        "layouts": dict.fromkeys(("embed", "decoder", "head", "rope"), ZERO),
        "templates": [],
        "weights": [],
    }
    vocab = (
        Path(__file__).resolve().parents[2]
        / "examples/BuddyDeepSeekR1/vocab.txt"
    )
    (root / "vocab.txt").symlink_to(vocab)
    for i, role in enumerate(("embed", "decoder_layer", "lm_head"), 1):
        name = f"{role}.so"
        shutil.copy(BUILD / f"libroute_b_mock_{i}.so", root / name)
        index["templates"].append(
            {
                "role": role,
                "path": name,
                "abi_version": 2 if i == 3 else 1,
                "layout_id": ZERO,
                "shape_id": ZERO,
                "nbytes": (root / name).stat().st_size,
                "sha256": hashlib.sha256(
                    (root / name).read_bytes()
                ).hexdigest(),
            }
        )
    for owner, ids in [
        ("embed", [-1]),
        ("decoder", range(28)),
        ("head", [-1]),
        ("rope", [-1]),
    ]:
        for i in ids:
            for bucket in (
                ["f32", "q4"] if owner in ("decoder", "head") else ["f32"]
            ):
                path = f"{owner}-{i}.{bucket}"
                data = (
                    struct.pack("f", float(i))
                    if bucket == "f32"
                    else struct.pack("b", i)
                )
                if owner == "rope":
                    data *= 8
                (root / path).write_bytes(data)
                w = {
                    "owner": owner,
                    "bucket": bucket,
                    "path": path,
                    "elements": 8 if owner == "rope" else 1,
                    "dtype": "f32" if bucket == "f32" else "i8",
                    "nbytes": len(data),
                    "layout_id": ZERO,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                if i >= 0:
                    w["global_layer_id"] = i
                index["weights"].append(w)
    save(root / "index.json", index)
    return index


def run(cmd, error=None, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
    if (p.returncode != 0) != (error is not None) or (
        error and error not in p.stderr + p.stdout
    ):
        raise AssertionError((cmd, p.returncode, p.stderr, p.stdout))
    return p.stdout


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    index = fixture(root)
    manifest = root / "index.json"
    local = [BUILD / "route_b_local", manifest]
    if TEST == "local_stage":
        for part in ("0:28", "0:1", "0:7", "2:5", "2:6", "27:28", "22:28"):
            start, end = map(int, part.split(":"))
            run(
                [*local, part],
                env=dict(
                    os.environ,
                    CHECK_ROUTE_B_KV="1",
                    CHECK_ROUTE_B_LAYERS=str(end - start),
                ),
            )
    elif TEST == "storage_and_dso":
        part = [*local, "14:16"]
        for name in ("embed.so", "lm_head.so", "decoder-0.q4", "head--1.f32"):
            (root / name).unlink()
        run(part)
        w = root / "decoder-15.q4"
        for data in (b"", b"12"):
            w.write_bytes(data)
            run(part, "weight length")
        w.unlink()
        run(part, "open")
        w.write_bytes(bytes([15]))
        for key, value, error in [
            ("abi_version", 2, "template"),
            ("shape_id", "11" * 32, "template"),
            ("layout_id", "11" * 32, "template"),
        ]:
            item = index["templates"][1]
            old = item[key]
            item[key] = value
            save(manifest, index)
            run(part, error)
            item[key] = old
        for key, value, error in [
            ("dtype", "f16", "bucket"),
            ("nbytes", 0, "nbytes"),
            ("elements", 1.5, "elements"),
        ]:
            item = index["weights"][0]
            old = item[key]
            item[key] = value
            save(manifest, index)
            run(part, error)
            item[key] = old
        save(manifest, index)
        j = json.loads(manifest.read_text())
        j["total_layers"] = 7
        manifest.write_text(json.dumps(j))
        run(part, "bundle_id")
        save(manifest, index)
        dso = root / "decoder_layer.so"
        dso.write_bytes(dso.read_bytes()[:-1])
        run(part, "kernel size")
        shutil.copy(BUILD / "libroute_b_mock_1.so", dso)

        def receipt():
            index["templates"][1].update(
                nbytes=dso.stat().st_size,
                sha256=hashlib.sha256(dso.read_bytes()).hexdigest(),
            )
            save(manifest, index)

        receipt()
        run(part, "descriptor")
        shutil.copy(BUILD / "libroute_b_mock_incomplete.so", dso)
        receipt()
        run(part, "layer_decode")
        shutil.copy(BUILD / "libroute_b_mock_2.so", dso)
        receipt()
        w.write_bytes(b"X")
        run([*part, "verify"], "weight digest")
    else:

        def cli(path=manifest):
            return [
                BUILD / "buddy-pp-route-b",
                "--bundle",
                path,
                "--prompt",
                "hello",
                "--total-tokens",
                "18",
            ]

        for cuts in (
            [28],
            [14, 28],
            [16, 28],
            [7, 14, 21, 28],
            [4, 8, 12, 16, 19, 22, 25, 28],
        ):
            ranges = ",".join(
                f"{a}:{b}" for a, b in zip([0] + list(cuts), cuts)
            )
            trace = root / ("trace-" + ranges)
            out = run(
                [MPI, "-n", str(len(cuts)), *cli(), "--ranges", ranges],
                env=dict(
                    os.environ,
                    BUDDY_PP_TEST_EOS_STEP="3",
                    ROUTE_B_MPI_TRACE=str(trace),
                    LD_PRELOAD=str(BUILD / "libroute_b_mpi_trace.so"),
                ),
            )
            assert json.loads(out)["token_ids"] == [43, 43, 43], out
            for rank in range(len(cuts)):
                logfile = Path(f"{trace}.{rank}")
                events = (
                    logfile.read_text().splitlines() if logfile.exists() else []
                )
                expected = []
                if len(cuts) > 1:
                    if rank < len(cuts) - 1:
                        for count in (64 * 16, 16, 16, 16):
                            expected.extend(
                                [
                                    f"header {rank + 1} 4",
                                    f"hidden {rank + 1} {count}",
                                ]
                            )
                    else:
                        expected.extend(["token 0 1"] * 3)
                    expected.extend(
                        [f"stop {len(cuts) - 1} {val}" for val in (0, 0, 1)]
                    )
                assert sorted(events) == sorted(expected), (events, expected)
        for bad in (
            "0:15,14:28",
            "0:13,14:28",
            "14:28,0:14",
            "0:0,0:28",
            "0:14z,14:28",
            "0:14,14:27",
        ):
            run([MPI, "-n", "2", *cli(), "--ranges", bad], "range")
        other = root / "other.json"
        index["model_id"] = "other"
        save(other, index)

        def mpmd(right, extra=("0:14", "14:28")):
            return [
                MPI,
                "-n",
                "1",
                *cli(),
                "--stage-range",
                extra[0],
                ":",
                "-n",
                "1",
                *cli(right),
                "--stage-range",
                extra[1],
            ]

        run(mpmd(other), "model_name")
        run(mpmd(manifest, ("0:13", "14:28")), "gap")
        run(mpmd(manifest))
