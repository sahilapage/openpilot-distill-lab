#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
unset VIRTUAL_ENV
CUDA_VISIBLE_DEVICES="" ../.boot/bin/uv run --frozen python ../inspect_wm.py
