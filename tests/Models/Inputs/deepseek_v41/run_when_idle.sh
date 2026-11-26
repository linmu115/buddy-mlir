#!/usr/bin/env bash
# Run from the board benchmark directory. Exit 75 when Route B needs the board.
set -eu
busy() { pgrep -f '[b]uddy-pp-route-b' >/dev/null; }
if busy; then exit 75; fi
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close
echo 0 > /proc/set_ai_thread
./bench "$@" & job=$!
trap 'kill "$job" 2>/dev/null || true' EXIT HUP INT TERM
while kill -0 "$job" 2>/dev/null; do
  if busy; then
    kill "$job" 2>/dev/null || true
    wait "$job" || true
    exit 75
  fi
  sleep .1
done
wait "$job"
trap - EXIT
