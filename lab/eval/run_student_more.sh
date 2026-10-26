#!/usr/bin/env bash
# Student on ALL comma-four comma1M segments (user's go 2026-10-26 ~21:30). Queued after run_anchor.sh (never overlaps
# a GPU job) and the staging download. Moves complete staged segments into openpilot.distill/data/, writes the list of
# every never-trained-on segment (repo val split: hash % 10 == 0) to unseen_all.txt, then trains a NEW run initialised
# from the 40k student (fresh AdamW, cosine 3e-4 -> 3e-5 over STEPS), snapshots every 5k. Resumes after crashes.
#   tmux new-session -d -s more "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/more.log /home/sra/sahil_distill/eval/run_student_more.sh"
STEPS=${STEPS:-40000}
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=. WANDB_MODE=offline LR_DECAY_START=0 LR_MIN_RATIO=0.1 SNAPSHOT_EVERY=5000 SAVE_EVERY=500
export INIT_WEIGHTS=outputs/supervised_long/supercombo_step40000.pt
OUT=outputs/supervised_more/supercombo.pt
log() { echo "[$(date '+%F %T')] $*"; }
foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}

log "more: waiting for ANCHOR_DONE (anchor.log) and DOWNLOAD_DONE (download_more.log)"
until grep -q ANCHOR_DONE $W/anchor.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_anchor\.sh" > /dev/null || { log "anchor test is not running and never finished: not starting"; exit 2; }
  sleep 120
done
until grep -q DOWNLOAD_DONE $W/download_more.log 2>/dev/null; do
  pgrep -f "download_more[.]sh" > /dev/null || { log "download is not running and never finished: not starting"; exit 2; }
  sleep 120
done

# move only complete segments (all four files, no partial downloads) into the training data
moved=0; skipped=0
for d in $W/data_staging/*/; do
  seg=$(basename $d)
  if [ -s $d/fcamera.hevc ] && [ -s $d/ecamera.hevc ] && [ -s $d/frame_info.safetensors ] && [ -s $d/localizer.safetensors ] \
     && [ -z "$(ls $d | grep '\.part$')" ]; then
    [ -e data/$seg ] || { mv $d data/$seg && moved=$((moved + 1)); }
  else
    skipped=$((skipped + 1))
  fi
done
log "moved $moved segments into data/ ($skipped incomplete left in data_staging); data/ now $(ls data | wc -l)"
python3 - <<'PY' > $W/unseen_all.txt
import os
segs = sorted(s for s in os.listdir("data") if os.path.exists(f"data/{s}/fcamera.hevc"))
print("\n".join(s for s in segs if hash(int(s, 16)) % 10 == 0))
PY
log "unseen (never trained on) segments: $(wc -l < $W/unseen_all.txt)"

[ -f $INIT_WEIGHTS ] || { log "missing $INIT_WEIGHTS"; exit 3; }
log "student-more: $STEPS steps from $INIT_WEIGHTS -> $OUT"
for attempt in $(seq 1 10); do
  [ -f $OUT ] && { log "MORE_DONE: $OUT"; exit 0; }
  while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done
  log "attempt $attempt"
  $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --steps $STEPS --batch-size 8 --workers 8 -o $OUT \
    >> $W/stage13_more.log 2>&1 || { log "attempt $attempt failed at $(grep -o 'step [0-9]*' $W/stage13_more.log | tail -1)"; sleep 60; }
done
[ -f $OUT ] && log "MORE_DONE: $OUT" || { log "student-more FAILED"; exit 13; }
