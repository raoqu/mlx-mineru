#!/usr/bin/env bash
# Dev mode: run the C++ API/backend (mlx-mineru) and the Vite dev server together.
# The Vite dev server (http://localhost:5173) proxies /file_parse, /info, /health to the
# C++ backend on :8000, so frontend edits hot-reload while the real backend does the work.
#
# Usage:
#   ./dev.sh                       # backend: pipeline (fast); frontend: vite dev
#   ./dev.sh --backend vlm         # backend: VLM (loads the 2.3GB model)
#   BACKEND=vlm PORT=8000 ./dev.sh
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${BACKEND:-pipeline}"
PORT="${PORT:-8000}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:-pipeline}"; shift 2;;
    --backend=*) BACKEND="${1#*=}"; shift;;
    --port) PORT="${2:-8000}"; shift 2;;
    *) shift;;
  esac
done

if [[ ! -x build/mlx-mineru ]]; then
  echo "[dev] build/mlx-mineru not found — building once ..."
  ./build.sh
fi

# Frontend deps (pnpm; allow esbuild's build script in pnpm v11).
if [[ ! -d web/node_modules ]]; then
  echo "[dev] installing web deps (pnpm) ..."
  ( cd web && pnpm install )
fi

cleanup() { kill "${BACK_PID:-}" "${FRONT_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[dev] backend: mlx-mineru --server --backend $BACKEND --port $PORT"
./build/mlx-mineru --server --backend "$BACKEND" --port "$PORT" &
BACK_PID=$!

echo "[dev] frontend: vite dev (http://localhost:5173, proxy -> :$PORT)"
( cd web && pnpm dev ) &
FRONT_PID=$!

echo "[dev] open http://localhost:5173  (Ctrl-C to stop both)"
wait
