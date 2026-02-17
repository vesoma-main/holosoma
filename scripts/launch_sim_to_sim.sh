#!/bin/bash
# Sim-to-Sim: launch MuJoCo simulator + inference policy in a single terminal.
#
# Usage (from repo root):
#   bash thirdparty/holosoma/scripts/launch_sim_to_sim.sh
#
# The simulator runs in the `hsmujoco` conda environment (setup via setup_mujoco.sh).
# The policy runs in the `hsinference` conda environment (setup via setup_inference.sh).
# On macOS the simulator must run under mjpython (MuJoCo viewer requirement).

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")

source "${SCRIPT_DIR}/source_common.sh"

MODEL_PATH="${REPO_ROOT}/src/holosoma_inference_ext/holosoma_inference_ext/models/model_go2.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH"
    exit 1
fi

cleanup() {
    echo ""
    echo "Stopping simulation..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT

export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== Sim-to-Sim ==="
echo "  Model: $MODEL_PATH"
echo ""

# 1. Simulator (background, hsmujoco) — mjpython required on macOS for MuJoCo viewer
echo "[1/2] Starting Simulator (hsmujoco, background)..."
source "${CONDA_ROOT}/bin/activate" hsmujoco
mjpython -m holosoma_ext.run_sim robot:go2-12dof \
    --simulator.config.virtual-gantry.enabled False &
SIM_PID=$!

echo "  Simulator PID: $SIM_PID"
echo "  Waiting 8 seconds for simulator to initialize..."
sleep 8

# 2. Policy (foreground, hsinference) — reads DDS state from simulator, publishes commands back
echo "[2/2] Starting Policy (hsinference, foreground)..."
echo "  Press ']' to start the policy, 'o' to stop, '-' to quit."
echo ""
source "${CONDA_ROOT}/bin/activate" hsinference
python3 -m holosoma_inference_ext.run_policy inference:go2-12dof-loco \
    --task.model-path "$MODEL_PATH" \
    --task.interface lo
