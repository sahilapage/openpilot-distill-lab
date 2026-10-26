#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
PYTHONPATH=. ../.boot/bin/uv run --frozen python ../check_env.py
echo "SMOKE_EXIT=$?"
