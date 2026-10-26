#!/usr/bin/env bash
# After the pipeline finishes: pull results + saved frames from the 4090 and render every cut.
#   ./render_all.sh            (uses the venv python in $PY)
set -euo pipefail
PY=${PY:-python3}
BOX=${BOX:-gpu}  # ssh host of the GPU box
SRC=/home/sra/sahil_distill
OUT=${OUT:-$HOME/openpilot-distill-work/showcase/out}
mkdir -p "$OUT/frames"
rsync -a "$BOX:$SRC/results.jsonl" "$BOX:$SRC/analysis.txt" "$OUT/"
rsync -a "$BOX:$SRC/openpilot.distill/outputs/eval/frames/" "$OUT/frames/"
cd "$(dirname "$0")"
$PY pick_clips.py "$OUT/results.jsonl" "$OUT/frames" | tee "$OUT/picks.txt"
echo "now run:  $PY make_video.py hero --results $OUT/results.jsonl --frames $OUT/frames --segment SEG --start N --out $OUT/hero.mp4"
echo "          $PY make_video.py strip --results ... (same flags)   |   wall   |   stats"
