#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
unset VIRTUAL_ENV
../.boot/bin/uv run --frozen python -m rl.train --model outputs/smoke/supercombo.pt --steps 1 --rollout-frames 4 --sampling-steps 15 -o outputs/smoke/on_policy.pt --cache-dir outputs/rl/cache
