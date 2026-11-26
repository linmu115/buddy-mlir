#!/bin/bash
set -euo pipefail
root="$(dirname "$(readlink -f "$0")")"
mpiexec="${MPIEXEC:-$(cat "$root/mpiexec.path")}"
hosts="${K3_HOSTS:-$(cat "$root/k3-hosts")}"
export PATH="$(dirname "$mpiexec"):$PATH"
cd "$root"
binary="$root/rank-27b"
if [[ -n ${EXP16_BENCHMARK:-} ]]; then binary="$root/rank-test"; fi
args=()
for arg; do
  case "$arg" in --mtp) binary="$binary-mtp";; *) args+=("$arg");; esac
done
exec python3 "$root/server.py" --bundle "$root/route-b.json" --binary "$binary" \
  --mpiexec "$mpiexec" --hosts "$hosts" \
  --vlen-probe "$root/vlen" --ranges 0:20,20:64 --slots 2 --wire f32 \
  --bind "${BIND_ADDRESS:-127.0.0.1}" --port "${PORT:-8081}" --startup-timeout 900 "${args[@]}"
