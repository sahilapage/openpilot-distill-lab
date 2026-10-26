#!/usr/bin/env bash
source /home/sra/sahil_distill/env.sh
unset VIRTUAL_ENV
mkdir -p outputs/smoke
../.boot/bin/uv run --frozen torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 20 --batch-size 8 --workers 1 -o outputs/smoke/supercombo.pt
