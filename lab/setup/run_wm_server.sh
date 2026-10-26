#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
unset VIRTUAL_ENV
../.boot/bin/uv run --frozen python -m rl.server
