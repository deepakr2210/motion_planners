#!/usr/bin/env bash
# launch.sh — Start sim, controller, and planner in separate processes.
#
# Usage:
#   bash launch.sh           # Python planner (default)
#   bash launch.sh cpp       # C++ planner
#   bash launch.sh --no-render  # headless sim

set -euo pipefail

PLANNER="${1:-python}"
RENDER_FLAG=""
[[ "${*}" == *"--no-render"* ]] && RENDER_FLAG="--no-render"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLU}[launch]${NC} $*"; }
err() { echo -e "${RED}[launch]${NC} $*" >&2; }

cleanup() {
    log "shutting down all processes..."
    kill "${SIM_PID:-}" "${CTRL_PID:-}" "${PLAN_PID:-}" 2>/dev/null || true
    wait "${SIM_PID:-}" "${CTRL_PID:-}" "${PLAN_PID:-}" 2>/dev/null || true
    log "done."
}
trap cleanup EXIT INT TERM

# ── Validate environment ──────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    err "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

CPP_BIN="$ROOT/build/planners/cpp/cpp_planner"
if [[ "$PLANNER" == "cpp" ]] && [[ ! -f "$CPP_BIN" ]]; then
    err "C++ planner not built. Run:"
    err "  bash scripts/setup_cpp_deps.sh"
    err "  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc) && cd .."
    exit 1
fi

# ── Start sim ─────────────────────────────────────────────────────────────────
log "starting ${GRN}MuJoCo sim${NC} ${RENDER_FLAG}..."
uv run python -m sim.mujoco_sim $RENDER_FLAG &
SIM_PID=$!

sleep 1.5   # wait for ZMQ sockets to bind

# ── Start controller ──────────────────────────────────────────────────────────
log "starting ${YLW}controller${NC}..."
uv run python -m control.controller &
CTRL_PID=$!

sleep 0.5

# ── Start planner ─────────────────────────────────────────────────────────────
if [[ "$PLANNER" == "cpp" ]]; then
    log "starting ${RED}C++ planner${NC}..."
    "$CPP_BIN" &
else
    log "starting ${GRN}Python planner${NC}..."
    uv run python -m planners.python.planner &
fi
PLAN_PID=$!

log "all processes started  (sim=${SIM_PID} ctrl=${CTRL_PID} plan=${PLAN_PID})"
log "press Ctrl-C to stop"

# Wait for any process to exit unexpectedly
wait -n "${SIM_PID}" "${CTRL_PID}" "${PLAN_PID}" 2>/dev/null || true
