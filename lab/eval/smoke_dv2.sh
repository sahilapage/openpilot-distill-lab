#!/usr/bin/env bash
# DAgger v2 smoke test (~15 min of GPU): input check, 4 short rollouts, 40 mixed training steps. Throwaway outputs.
#   tmux new-session -d -s dv2smoke "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/dv2smoke.log /home/sra/sahil_distill/eval/smoke_dv2.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=. WANDB_MODE=offline
MORE=outputs/supervised_more/supercombo.pt
log() { echo "[$(date '+%F %T')] $*"; }
SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; sleep 5; }
trap stop_server EXIT
rm -rf outputs/wm_data_smoke outputs/dv2_smoke
curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 busy"; exit 4; }
setsid $UV python -m rl.server > $W/dv2smoke_server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5; done
log "1. sample == actor input check"
$UV python $W/eval/check_wm_inputs.py $MORE $(head -1 $W/wm_collect_pool.txt) 2>&1 | grep -v onnxruntime | tail -5
log "2. collect 4 rollouts (10 frames)"
t0=$(date +%s)
$UV python $W/eval/collect_wm.py --segments $W/dagger_train.txt --out-dir outputs/wm_data_smoke --rollouts 4 --rollout-frames 10 \
  --driver student=student:$MORE --driver teacher=teacher --driver-weights 0.5,0.5 2>&1 | grep -v onnxruntime | tail -4
log "collect took $(( $(date +%s) - t0 )) s; $(ls outputs/wm_data_smoke/*.npz | wc -l) files, $(du -sh outputs/wm_data_smoke | cut -f1)"
stop_server
log "3. 40 mixed training steps"
WM_DATA=outputs/wm_data_smoke INIT_WEIGHTS=$MORE LR_DECAY_START=0 SAVE_EVERY=100000 \
  $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 40 --batch-size 8 --workers 8 --learning-rate 1e-4 \
  -o outputs/dv2_smoke/supercombo.pt > $W/dv2smoke_train.log 2>&1 &
TP=$!
peak=0
while kill -0 $TP 2>/dev/null; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$peak" ] && peak=$m; sleep 3
done
wait $TP; rc=$?
grep -E "^step|Error|error" $W/dv2smoke_train.log | grep -v onnxruntime | tail -6
log "train exit $rc, peak GPU memory ${peak} MiB, checkpoint: $(ls outputs/dv2_smoke/ 2>/dev/null | tr '\n' ' ')"
rm -rf outputs/wm_data_smoke outputs/dv2_smoke
log "SMOKE_DONE"
