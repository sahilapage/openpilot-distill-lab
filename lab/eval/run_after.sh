#!/usr/bin/env bash
# Queued after run_coach.sh: waits for ALL_DONE in pipeline.log, then measures how the world model rendered every saved
# steering-pull rollout (teacher lane-line head on generated vs real frames) and writes the heading/bend analysis.
# Resumable by rerunning (outputs are rewritten). Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s after "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/after.log /home/sra/sahil_distill/eval/run_after.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
R=$W/openpilot.distill
cd $R
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }
GPU_ORT="--providers CUDAExecutionProvider,CPUExecutionProvider"

foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}

log "after: waiting for ALL_DONE in pipeline.log"
until grep -q ALL_DONE $W/pipeline.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_coach\.sh" > /dev/null || { log "run_coach.sh is not running and never finished: not starting"; exit 2; }
  sleep 120
done
sleep 20
while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
# run_coach.sh's server may still be shutting down and answering /health: wait until the port is free
for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
setsid $UV python -m rl.server > $W/after_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
  curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
done
curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
log "server ready: measuring rendered geometry of the steering-pull rollouts"

$UV python $W/eval/measure_forgiveness.py $GPU_ORT --results $W/results_pull.jsonl --frames outputs/pull/frames \
  --out $W/forgiveness_pull.jsonl > $W/stage8_measure.log 2>&1 || { log "measure FAILED"; exit 8; }
$UV python $W/eval/calibrate_forgiveness.py $GPU_ORT --results $W/results_pull.jsonl \
  --out $W/calib_pull.jsonl > $W/stage8_calib.log 2>&1 || { log "calibrate FAILED"; exit 8; }
stop_server
$UV python -I $W/eval/analyze_bend.py $W/forgiveness_pull.jsonl $W/calib_pull.jsonl $W/results_pull.jsonl \
  > $W/analysis_bend.txt 2>&1 || log "analyze_bend FAILED (inputs are kept)"
log "AFTER_DONE: see $W/analysis_bend.txt"
