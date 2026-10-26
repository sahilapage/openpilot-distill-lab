#!/usr/bin/env bash
# Queued after run_after.sh: waits for AFTER_DONE in after.log, then the open-loop check (every policy on the REAL drive,
# raw and VAE frames, no world model in the loop) and rescoring of the closed-loop files in the repo frame.
# Resumable by rerunning (finished rows are skipped). Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s openloop "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/openloop.log /home/sra/sahil_distill/eval/run_openloop.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
R=$W/openpilot.distill
cd $R
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }
SUP=outputs/supervised/supercombo.pt

foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}

log "openloop: waiting for AFTER_DONE in after.log"
until grep -q AFTER_DONE $W/after.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_(after|coach)\.sh" > /dev/null || { log "run_after.sh is not running and never finished: not starting"; exit 2; }
  sleep 120
done
sleep 20
while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
setsid $UV python -m rl.server > $W/openloop_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
  curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
done
curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
log "server ready: open-loop check on the real drives"

for attempt in 1 2 3; do
  $UV python $W/eval/eval_open_loop.py --segments $W/eval_segments.txt --out $W/results_open_loop.jsonl \
    --policy teacher=teacher --policy student=student:$SUP \
    --policy dagger50=dagger:$SUP:outputs/dagger/on_policy_step50.pt \
    --policy dagger800=dagger:$SUP:outputs/dagger/on_policy_step800.pt \
    --policy coach800=dagger:$SUP:outputs/dagger_coach/on_policy_step800.pt \
    >> $W/stage9_openloop.log 2>&1 && break
  log "open-loop attempt $attempt failed, retrying"; sleep 30
  [ $attempt = 3 ] && { log "open-loop FAILED"; exit 9; }
done
stop_server
$UV python -I $W/eval/analyze_open_loop.py $W/results_open_loop.jsonl $W/val_segments.txt > $W/analysis_open_loop.txt 2>&1 || log "analyze_open_loop FAILED"
for f in results results_pull; do
  $UV python -I $W/eval/rescore.py $W/$f.jsonl $W/${f}_fixed.jsonl >> $W/stage9_rescore.log 2>&1 || log "rescore $f FAILED"
done
log "OPENLOOP_DONE: see $W/analysis_open_loop.txt"
