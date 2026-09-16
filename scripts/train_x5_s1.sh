#!/usr/bin/env bash
# Run from an activated Isaac Lab Python environment. Extra CLI flags override defaults.
set -euo pipefail
MUJICA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUJICA_PROJECT_DIR"
exec python -m mujica.train \
    --backend isaaclab --robot x5 --stage s1 \
    --num-envs 512 --headless --seed 42 \
    --config configs/mujica_default.json \
    --iterations 30000 "$@"
