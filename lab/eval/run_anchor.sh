#!/usr/bin/env bash
# Anchor test (user's go 2026-10-26 ~21:30): does the real-future anchor pull the rendered scene back onto the lane?
# Steering pull 3 deg/s left and right, 2 s rollouts, anchor 10 / 20 / max frames after the start, straight + teacher,
# 3 starts on each of the 24 eval drives (864 rollouts, frames saved for all), then the lane readout on the generated
# frames and on the real frames, and analyze_anchor.py. Resumable (finished rows are skipped). Logs ANCHOR_DONE.
#   tmux new-session -d -s anchor "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/anchor.log /home/sra/sahil_distill/eval/run_anchor.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }
GPU_ORT="--providers CUDAExecutionProvider,CPUExecutionProvider"
foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
start_server() {
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
  curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
  setsid $UV python -m rl.server > $W/anchor_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
    curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
  done
  curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
}
trap stop_server EXIT

for attempt in 1 2 3; do
  start_server
  log "anchor rollouts, attempt $attempt"
  $UV python $W/eval/eval_shove.py --segments $W/eval_segments.txt --out $W/results_anchor.jsonl \
    --pull-dps 3 --rollout-frames 10 --anchor-frames 10,20,max --starts-per-segment 3 \
    --save-frames 3 --frames-dir outputs/anchor/frames \
    --policy straight=straight --policy teacher=teacher >> $W/stage12_anchor.log 2>&1 && break
  stop_server
  log "attempt $attempt failed"; sleep 30
  [ $attempt = 3 ] && { log "anchor rollouts FAILED"; exit 12; }
done
log "rollouts done: $(wc -l < $W/results_anchor.jsonl) rows; measuring"
$UV python $W/eval/measure_forgiveness.py $GPU_ORT --results $W/results_anchor.jsonl --frames outputs/anchor/frames \
  --out $W/forgiveness_anchor.jsonl > $W/stage12_measure.log 2>&1 || { log "measure FAILED"; exit 12; }
$UV python $W/eval/calibrate_forgiveness.py $GPU_ORT --results $W/results_anchor.jsonl \
  --out $W/calib_anchor.jsonl > $W/stage12_calib.log 2>&1 || { log "calibrate FAILED"; exit 12; }
stop_server
$UV python -I $W/eval/analyze_anchor.py $W/forgiveness_anchor.jsonl $W/calib_anchor.jsonl $W/results_anchor.jsonl \
  > $W/analysis_anchor.txt 2>&1 || log "analyze_anchor FAILED (inputs kept)"
log "ANCHOR_DONE: see $W/analysis_anchor.txt"
