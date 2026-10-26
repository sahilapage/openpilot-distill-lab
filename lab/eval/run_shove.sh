#!/usr/bin/env bash
# The shove test, chained after the main pipeline: waits for PIPELINE_DONE in pipeline.log (so it never overlaps our own
# eval), then starts the world-model server and runs eval_shove.py. Resumable. Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s shove "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/shove.log /home/sra/sahil_distill/eval/run_shove.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
R=$W/openpilot.distill
cd $R
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }

foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
wait_for_free_gpu() {
  while [ -n "$(foreign_gpu_procs)" ]; do log "GPU has someone else's compute process(es) $(foreign_gpu_procs | tr '\n' ' '), waiting 60 s"; sleep 60; done
}

log "shove: waiting for the main pipeline to finish"
until grep -q PIPELINE_DONE $W/pipeline.log 2>/dev/null; do
  pgrep -f "[r]un_extend.sh|[r]un_pipeline.sh" > /dev/null || { log "main pipeline is not running and never finished: not starting"; exit 2; }
  sleep 120
done
sleep 20  # let the pipeline's server release the GPU

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
wait_for_free_gpu
setsid $UV python -m rl.server > $W/shove_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5; done
curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
log "world-model server ready, shove test start"

SUP=outputs/supervised/supercombo.pt
$UV python $W/eval/eval_shove.py --segments $W/eval_segments.txt --out $W/results_shove.jsonl --save-frames 1 \
  --policy straight=straight --policy student=student:$SUP \
  --policy dagger200=dagger:$SUP:outputs/dagger/on_policy_step200.pt \
  --policy dagger1400=dagger:$SUP:outputs/dagger/on_policy_step1400.pt \
  --policy teacher=teacher \
  > $W/stage7_shove.log 2>&1 || { log "shove test FAILED"; exit 7; }
log "SHOVE_DONE"
