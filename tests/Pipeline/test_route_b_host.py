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

from route_b_test_support import Resident

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


def run(cmd, error=None, env=None, diagnostic=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
    if (p.returncode != 0) != (error is not None) or (
        error and error not in p.stderr + p.stdout
    ):
        raise AssertionError((cmd, p.returncode, p.stderr, p.stdout))
    if diagnostic and diagnostic not in p.stderr:
        raise AssertionError((cmd, diagnostic, p.stderr))
    return p.stdout


def test_resident(root, manifest, index):
    env = dict(
        os.environ,
        CHECK_ROUTE_B_BATCH="1",
        CHECK_ROUTE_B_FRAME="1",
        CHECK_ROUTE_B_SERVE="1",
    )
    # A has the shortest decode. B and C cover multi-frame/overlapped prefill.
    prompts = {
        "A": "hello" + " x" * 17,
        "B": "hello" + " x" * 72,
        "C": "hello" + " x" * 89,
    }
    limits = {"A": 18, "B": 30, "C": 7}
    index["vocab_sha256"] = hashlib.sha256(
        (root / "vocab.txt").read_bytes()
    ).hexdigest()
    runtime = root / "libbj47_kernel_runtime.so"
    shutil.copy(BUILD / "libroute_b_mock_1.so", runtime)
    index["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    save(manifest, index)

    for ranks in (1, 2, 4):
        ranges = ",".join(
            f"{i * 28 // ranks}:{(i + 1) * 28 // ranks}" for i in range(ranks)
        )
        for wire in ("f32", "f16"):
            base = [
                MPI,
                "-n",
                str(ranks),
                BUILD / "buddy-pp-route-b-serve",
                "--bundle",
                manifest,
                "--ranges",
                ranges,
                "--wire",
                wire,
            ]
            refs = {}
            totals = {}
            for key, prompt in prompts.items():
                length = json.loads(
                    run(
                        [*base, "--prompt", prompt, "--total-tokens", "1"],
                        env=env,
                    )
                )["prompt_tokens"]
                totals[key] = str(length + limits[key] - 1)
                refs[key] = json.loads(
                    run(
                        [
                            *base,
                            "--prompt",
                            prompt,
                            "--total-tokens",
                            totals[key],
                        ],
                        env=env,
                    )
                )["token_ids"]
                assert len(refs[key]) == limits[key]

            with Resident([*base, "--serve", "--slots", "1"], env) as server:
                assert (
                    server.until(lambda e: e["event"] == "ready")["slots"] == 1
                )
                server.submit("A", prompts["A"], total_tokens=int(totals["A"]))
                assert server.done("A")["slot"] == 0
                server.submit("B", prompts["B"], max_new_tokens=limits["B"])
                assert server.done("B")["slot"] == 0
                server.sequences(refs, "AB")

            with Resident([*base, "--serve"], env) as server:
                server.until(lambda e: e["event"] == "ready")
                server.submit("A", prompts["A"], max_new_tokens=limits["A"])
                server.until(lambda e: e["event"] == "token" and e["step"] == 4)
                server.submit("bad", "x " * 200, max_new_tokens=1)
                server.proc.stdin.write("{ malformed json\n")
                server.proc.stdin.flush()
                server.submit("empty", "")
                for key in ("bad", "", "empty"):
                    error = server.until(lambda e: e["event"] == "error")
                    assert error["req_id"] == key
                    if key == "bad":
                        assert "exceeds model context" in error["message"]
                server.submit("B", prompts["B"], max_new_tokens=limits["B"])
                server.until(
                    lambda e: e["event"] == "accepted" and e["req_id"] == "B"
                )
                server.done("A")
                server.done("B")
                server.sequences(refs, "AB")

            with Resident([*base, "--serve", "--slots", "2"], env) as server:
                server.until(lambda e: e["event"] == "ready")
                for key, prompt in prompts.items():
                    server.submit(key, prompt, max_new_tokens=limits[key])
                accepted_a = server.until(lambda e: e["event"] == "accepted")
                accepted_b = server.until(lambda e: e["event"] == "accepted")
                assert (accepted_a["req_id"], accepted_b["req_id"]) == (
                    "A",
                    "B",
                )
                assert accepted_a["slot"] != accepted_b["slot"]
                server.done("A")
                accepted_c = server.until(lambda e: e["event"] == "accepted")
                assert (
                    accepted_c["req_id"] == "C"
                    and accepted_c["slot"] == accepted_a["slot"]
                )
                server.until(
                    lambda _: (
                        len([e for e in server.log if e["event"] == "done"])
                        == 3
                    )
                )
                server.sequences(refs, "ABC")

        original = index["vocab_sha256"]
        index["vocab_sha256"] = ZERO
        save(manifest, index)
        run([*base, "--serve"], "rank 0: vocab digest", env=env)
        index["vocab_sha256"] = original
        save(manifest, index)
        data = runtime.read_bytes()
        runtime.write_bytes(data + b"tampered")
        try:
            run([*base, "--serve"], "runtime digest", env=env)
        finally:
            runtime.write_bytes(data)
        assert (
            run([*base, "--serve", "--slots", "8"], env=env).count(
                '"event":"ready"'
            )
            == 1
        )

    # Inject a non-std exception after Prefill and one Decode step. Both
    # async paths must abort the original communicator with their diagnostic.
    inject_env = dict(
        env,
        CHECK_ROUTE_B_THROW_DECODE_POSITION=str(
            int(totals["A"]) - limits["A"] + 2
        ),
    )
    distributed = [
        MPI,
        "-n",
        "2",
        BUILD / "buddy-pp-route-b-serve",
        "--bundle",
        manifest,
        "--ranges",
        "0:14,14:28",
        "--wire",
        "f16",
    ]
    run(
        [
            *distributed,
            "--prompt",
            prompts["A"],
            "--prompt",
            prompts["B"],
            "--total-tokens",
            totals["A"],
        ],
        "batch decode: unknown exception",
        env=inject_env,
    )
    p = subprocess.run(
        [*distributed, "--serve"],
        input=json.dumps(
            {
                "cmd": "submit",
                "req_id": "throw",
                "prompt": prompts["A"],
                "max_new_tokens": limits["A"],
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        env=inject_env,
        timeout=20,
    )
    assert (
        p.returncode != 0 and "resident serve: unknown exception" in p.stderr
    ), p.stderr

    base = [BUILD / "buddy-pp-route-b-serve", "--bundle", manifest]
    for first, second in (
        (["--serve"], ["--prompt", "hello"]),
        (["--serve", "--slots", "1"], ["--serve", "--slots", "2"]),
    ):
        run(
            [
                MPI,
                "-n",
                "1",
                *base,
                "--stage-range",
                "0:14",
                *first,
                ":",
                "-n",
                "1",
                *base,
                "--stage-range",
                "14:28",
                *second,
            ],
            "field model_name mismatch",
            env=env,
        )


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    index = fixture(root)
    manifest = root / "index.json"
    local = [BUILD / "route_b_local", manifest]
    if TEST == "mpi_serve":
        test_resident(root, manifest, index)
    elif TEST == "local_stage":
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
        original = dso.read_bytes()
        dso.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        run(part, "kernel digest")
        dso.write_bytes(original[:-1])
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

        def cli(path=manifest, prompt="hello", limit="18"):
            return [
                BUILD / "buddy-pp-route-b",
                "--bundle",
                path,
                "--prompt",
                prompt,
                "--total-tokens",
                limit,
            ]

        if TEST == "mpi_batch":
            prompts = [
                "hello",
                "hello" + " x" * 20,
                "hello" + " x" * 67,
                "hello" + " x" * 92,
            ]
            for ranks in (1, 2, 4, 8):
                ranges = ",".join(
                    f"{i * 28 // ranks}:{(i + 1) * 28 // ranks}"
                    for i in range(ranks)
                )
                for wire in ("f32", "f16"):
                    for transport in ("blocking", "async"):
                        env = dict(
                            os.environ,
                            CHECK_ROUTE_B_BATCH="1",
                            CHECK_ROUTE_B_FRAME="1",
                        )
                        base = [
                            MPI,
                            "-n",
                            str(ranks),
                            BUILD / "buddy-pp-route-b",
                            "--bundle",
                            manifest,
                            "--ranges",
                            ranges,
                            "--wire",
                            wire,
                            "--transport",
                            transport,
                            "--total-tokens",
                            "109",
                        ]
                        refs = [
                            json.loads(
                                run([*base, "--prompt", prompt], env=env)
                            )
                            for prompt in prompts
                        ]
                        assert refs[0]["token_ids"][-1] == 151643
                        assert len(refs[1]["token_ids"]) > len(
                            refs[0]["token_ids"]
                        )
                        for count in (2, 4):
                            trace = root / "batch-trace"
                            args = [
                                arg
                                for prompt in prompts[:count]
                                for arg in ("--prompt", prompt)
                            ]
                            result = json.loads(
                                run(
                                    [*base, *args],
                                    env=dict(
                                        env, BUDDY_PP_TRACE_STAGE=str(trace)
                                    ),
                                )
                            )
                            assert len(result["streams"]) == count
                            total = 0
                            for slot, (stream, ref) in enumerate(
                                zip(result["streams"], refs)
                            ):
                                assert stream["slot"] == slot
                                assert stream["token_ids"] == ref["token_ids"]
                                rows = ref["prompt_tokens"]
                                assert stream["prompt_tokens"] == rows
                                steps = len(ref["token_ids"]) - 1
                                assert stream["decode_steps"] == steps
                                assert stream["ttft_us"] >= 0
                                assert (
                                    stream["stream_decode_us"]
                                    <= result["decode_wall_us"]
                                )
                                total += steps
                                for rank in range(ranks):
                                    log = Path(
                                        f"{trace}.slot{slot}.rank{rank}.log"
                                    )
                                    events = [
                                        dict(
                                            field.split("=")
                                            for field in line.split()
                                        )
                                        for line in log.read_text().splitlines()
                                    ]
                                    assert [
                                        int(e["pos"])
                                        for e in events
                                        if int(e["pos"]) >= rows
                                    ] == list(range(rows, rows + steps))
                                    log.unlink()
                            assert result["aggregate_decode_tokens"] == total
                        # Both streams finish in prefill. Shutdown still traverses all ranks.
                        result = json.loads(
                            run(
                                [
                                    *base,
                                    "--total-tokens",
                                    "1",
                                    "--prompt",
                                    prompts[0],
                                    "--prompt",
                                    prompts[2],
                                ],
                                env=env,
                            )
                        )
                        assert result["aggregate_decode_tokens"] == 0
                        assert all(
                            s["decode_steps"] == 0 for s in result["streams"]
                        )
            run(
                [
                    MPI,
                    "-n",
                    "1",
                    *cli(),
                    "--stage-range",
                    "0:14",
                    ":",
                    "-n",
                    "1",
                    *cli(),
                    "--stage-range",
                    "14:28",
                    "--prompt",
                    "a second stream",
                ],
                "streamCount",
            )
            sys.exit(0)

        if TEST == "mpi_wire":
            for ranks in (1, 2, 4, 8):
                ranges = ",".join(
                    f"{i * 28 // ranks}:{(i + 1) * 28 // ranks}"
                    for i in range(ranks)
                )
                for wire in ("f32", "f16"):
                    for transport in ("blocking", "async"):
                        out = run(
                            [
                                MPI,
                                "-n",
                                str(ranks),
                                *cli(prompt="x" + " x" * 85, limit="128"),
                                "--ranges",
                                ranges,
                                "--wire",
                                wire,
                                "--transport",
                                transport,
                            ],
                            env=dict(
                                os.environ,
                                BUDDY_PP_TEST_EOS_STEP="3",
                                CHECK_ROUTE_B_FRAME="1",
                            ),
                            diagnostic=(
                                f"transport={transport} requested_wire="
                                f"{'native' if wire == 'f32' else 'f16'} "
                                f"effective_wire={wire if ranks > 1 else 'none'}"
                            ),
                        )
                        result = json.loads(out)
                        assert 64 < result["prompt_tokens"] < 126, result
                        assert result["token_ids"] == [42, 43, 43], result
            for opt, value, error in (
                ("--wire", "f16", "pp_wire_dtype mismatch"),
                ("--transport", "async", "pp_transport mismatch"),
            ):
                run(
                    [
                        MPI,
                        "-n",
                        "1",
                        *cli(),
                        "--stage-range",
                        "0:14",
                        ":",
                        "-n",
                        "1",
                        *cli(),
                        "--stage-range",
                        "14:28",
                        opt,
                        value,
                    ],
                    error,
                )
            for opt in ("--wire", "--transport"):
                run([MPI, "-n", "1", *cli(), opt, "invalid"], opt + " expects")
            sys.exit(0)

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

        # Fail one rank before its first collective, then after plan consensus
        # while its peer is entering the next collective.
        index["weights"][0]["dtype"] = "f16"
        save(other, index)
        run(mpmd(other), "bucket")
        (root / "decoder-27.q4").unlink()
        run(mpmd(manifest), "open")

        # Both ranks' own ranges agree with their neighbours, but the third
        # rank presents a different full plan for the other two processes.
        list_job = []
        for plan in ("0:7,7:14,14:28", "0:7,7:14,14:28", "0:8,8:14,14:28"):
            if list_job:
                list_job.append(":")
            list_job.extend(["-n", "1", *cli(), "--ranges", plan])
        run([MPI, *list_job], "model_name")
