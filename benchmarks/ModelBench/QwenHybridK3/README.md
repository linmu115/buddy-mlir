# Dual-rank Qwen hybrid benchmark

Place the launchers and Python scripts alongside the existing Route B bundle,
engine, gateway, tokenizer assets, Python dependencies in `pydeps/`, and VLEN
probe in `vlen`. Both ranks use the same release path. On the first rank,
`k3-hosts` holds the two comma-separated end0 addresses in rank order and
`mpiexec.path` holds the absolute path to the MPICH executable. The launcher
also accepts `K3_HOSTS` and `MPIEXEC` environment overrides.

Run `./start_27b_server.sh` for greedy decoding or add `--mtp`. The gateway
listens on loopback port 8081 by default. Run `./chat_27b.py --think` or
`./chat_27b.py --no-think` in another terminal. Both scripts accept `--port`.
The client accepts `/clear` and `/quit`. Stop the foreground server with Ctrl+C.

For the benchmark, expose the two entrypoints as symlinks in the first rank's
home directory and run `PYTHONPATH=pydeps python3 -u board-suite.py` there
from the release directory. It uses port 18082 for HTTP and writes case JSON,
stderr and per-rank memory samples under `exp16-resume/`. A restart reuses
completed case files for the same bundle and host configuration. For a fresh
measurement, archive that directory on both ranks before running the suite.
