#!/usr/bin/env bash
# Queued after run_student_more.sh (user's go 2026-10-26 ~21:30, "check it after each 5k"): open-loop check of every
# snapshot of the bigger-data student on every never-trained-on drive (unseen_all.txt) plus the 12 eval drives the
# students trained on (dagger_holdout.txt, the memorisation reference), against the teacher and the 40k student.
# Never overlaps: starts only after MORE_DONE. Resumable. Logs MORE_EVAL_DONE.
#   tmux new-session -d -s moreeval "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/moreeval.log /home/sra/sahil_distill/eval/run_more_eval.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }
foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
log "moreeval: waiting for MORE_DONE in more.log"
until grep -q MORE_DONE $W/more.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_(anchor|student_more)\.sh" > /dev/null || { log "student-more is not running and never finished: not starting"; exit 2; }
  sleep 120
done
sleep 30
while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done

cat $W/unseen_all.txt $W/dagger_holdout.txt | sort -u > $W/more_eval_segments.txt
POLICIES="--policy teacher=teacher --policy student40k=student:outputs/supervised_long/supercombo_step40000.pt"
for f in $(ls outputs/supervised_more/supercombo_step*.pt 2>/dev/null | sort -V); do
  n=$(basename $f .pt | sed 's/.*step\([0-9]*\)/\1/')
  POLICIES="$POLICIES --policy more$((n / 1000))k=student:$f"
done
POLICIES="$POLICIES --policy morefinal=student:outputs/supervised_more/supercombo.pt"
log "$(wc -l < $W/more_eval_segments.txt) drives; policies:$POLICIES"

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
setsid $UV python -m rl.server > $W/moreeval_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
  curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
done
curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }

for attempt in 1 2 3; do
  $UV python $W/eval/eval_open_loop.py --segments $W/more_eval_segments.txt --out $W/results_open_loop_more.jsonl $POLICIES \
    >> $W/stage14_moreeval.log 2>&1 && break
  log "attempt $attempt failed, retrying"; sleep 30
  [ $attempt = 3 ] && { log "more open-loop FAILED"; exit 14; }
done
stop_server
$UV python -I $W/eval/analyze_open_loop.py $W/results_open_loop_more.jsonl $W/unseen_all.txt > $W/analysis_open_loop_more.txt 2>&1 \
  || log "analyze FAILED"
log "MORE_EVAL_DONE: see $W/analysis_open_loop_more.txt"
