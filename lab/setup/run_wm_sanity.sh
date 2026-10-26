#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
unset VIRTUAL_ENV
PYTHONPATH=. ../.boot/bin/uv run --frozen python ../wm_sanity.py 8 40
