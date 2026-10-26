#!/usr/bin/env bash
# Queued after run_openloop.sh: adds every new-student snapshot present at start (outputs/supervised_long/
# supercombo_step*.pt) to the open-loop check, appending to results_open_loop.jsonl (finished rows are skipped), and
# re-runs the analysis. Rerun any time later to add newer snapshots. Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s snaps "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/snaps.log /home/sra/sahil_distill/eval/run_openloop_snapshots.sh"
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

log "snaps: waiting for OPENLOOP_DONE in openloop.log"
until grep -q OPENLOOP_DONE $W/openloop.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_(after|coach|openloop)\.sh" > /dev/null || { log "open-loop queue is not running and never finished: not starting"; exit 2; }
  sleep 120
done
sleep 20
while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done

POLICIES=""
for f in $(ls outputs/supervised_long/supercombo_step*.pt 2>/dev/null | sort -V); do
  n=$(basename $f .pt | sed 's/.*step\([0-9]*\)/\1/')
  POLICIES="$POLICIES --policy student$((n / 1000))k=student:$f"
done
[ -z "$POLICIES" ] && { log "no snapshots yet"; exit 0; }
log "snapshots:$POLICIES"

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
setsid $UV python -m rl.server > $W/snaps_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
  curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
done
curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }

for attempt in 1 2 3; do
  $UV python $W/eval/eval_open_loop.py --segments $W/eval_segments.txt --out $W/results_open_loop.jsonl $POLICIES \
    >> $W/stage11_snaps.log 2>&1 && break
  log "attempt $attempt failed, retrying"; sleep 30
  [ $attempt = 3 ] && { log "snapshot open-loop FAILED"; exit 11; }
done
stop_server
$UV python -I $W/eval/analyze_open_loop.py $W/results_open_loop.jsonl $W/val_segments.txt > $W/analysis_open_loop.txt 2>&1 || log "analyze FAILED"
log "SNAPS_DONE: see $W/analysis_open_loop.txt"
