#!/usr/bin/env bash
# Better supervised student: continue the 10k-step student (it stopped mid-descent: lane_lines / road_edges never learned)
# to STEPS with cosine LR decay from step 10000 (3e-4 -> 3e-5) and weight snapshots every 5k for model selection.
# Writes outputs/supervised_long/ (the original outputs/supervised/ student is kept for comparison). Resumes from
# supercombo.resume.pt (saved every 500 steps) after any crash. Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s student "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/student.log /home/sra/sahil_distill/eval/run_student.sh"
STEPS=${STEPS:-60000}
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=. WANDB_MODE=offline LR_DECAY_START=10000 LR_MIN_RATIO=0.1 SNAPSHOT_EVERY=5000 SAVE_EVERY=500
OUT=outputs/supervised_long/supercombo.pt
log() { echo "[$(date '+%F %T')] $*"; }
foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
log "student: training to $STEPS steps -> $OUT"
for attempt in $(seq 1 10); do
  [ -f $OUT ] && { log "STUDENT_DONE: $OUT"; exit 0; }
  while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done
  log "attempt $attempt"
  $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --steps $STEPS --batch-size 8 --workers 8 -o $OUT \
    >> $W/stage10_student.log 2>&1 || { log "attempt $attempt failed at $(grep -o 'step [0-9]*' $W/stage10_student.log | tail -1)"; sleep 60; }
done
[ -f $OUT ] && log "STUDENT_DONE: $OUT" || { log "student FAILED"; exit 10; }
