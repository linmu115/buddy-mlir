"""Compare the EXP-19 rank-0 records collected by bench-dual."""

import json
import sys
from pathlib import Path


def read_run(directory, label, mode, workload):
    path = directory / f"28-{mode}-f32-{workload}-{label}.json"
    return json.loads(path.read_bytes())["requests"]


def outcome(request):
    events = request["events"]
    tokens = [e for e in events if e["event"] == "token"]
    done = events[-1]
    if not tokens or done["event"] != "done":
        raise ValueError("incomplete token stream")
    signature = (
        request.get("input_ids"),
        events[0]["prompt_tokens"],
        [e["token_id"] for e in tokens],
        done["reason"],
        done.get("mtp_accepted"),
        done.get("mtp_proposed"),
    )
    return signature, tokens[0]["client_ns"] / 1e9, done["decode_tok_s"]


def compare(directory, tag):
    for mode in ("off", "on"):
        for workload in ("quality", "grid"):
            runs = [
                read_run(directory, f"{tag}-{phase}", mode, workload)
                for phase in ("A1", "B1", "A2", "B2")
            ]
            cases = {"short": 30, "long": 164}
            if workload == "quality":
                cases.update(other=40, boundary=384)
            for name, length in cases.items():
                rows = [outcome(run[name]) for run in runs]
                signature = rows[0][0]
                if workload == "quality" and len(signature[0] or []) != length:
                    raise ValueError(f"incomplete input IDs: {mode} {name}")
                if signature[1] != length or any(
                    r[0] != signature for r in rows
                ):
                    raise ValueError(
                        f"parity failure: {mode} {workload} {name}"
                    )
                times = " ".join(f"{r[1]:.3f}" for r in rows)
                rates = " ".join(f"{r[2]:.2f}" for r in rows)
                print(
                    f"{mode} {workload} {name}: {len(signature[2])} IDs match;"
                    f" TTFT(s) {times}, decode(tok/s) {rates}"
                )


if __name__ == "__main__":
    compare(Path(sys.argv[1]), sys.argv[2])
