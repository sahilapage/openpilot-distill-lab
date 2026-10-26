#!/usr/bin/env bash
# The full experiment, sequential and resumable (finished stages are skipped). Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s pipeline "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/pipeline.log /home/sra/sahil_distill/eval/run_pipeline.sh"
# Stages: 1 supervised student -> 2 world-model server -> 3 latent precompute -> 4 DAgger training -> 5 eval -> 6 analysis
SUP_STEPS=${SUP_STEPS:-10000}
WORKERS=${WORKERS:-8}
DAGGER_STEPS=${DAGGER_STEPS:-200}
STARTS=${STARTS:-5}
EXTRA_POLICIES=${EXTRA_POLICIES:-}  # e.g. "--policy dagger200=dagger:...": more points on the curve, set by run_extend.sh

W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
R=$W/openpilot.distill
cd $R
export PYTHONPATH=.
log() { echo "[$(date '+%F %T')] $*"; }

foreign_gpu_procs() {  # compute processes on the GPU that are not ours
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
wait_for_free_gpu() {  # never share the GPU with someone else's job: wait until they are gone
  while [ -n "$(foreign_gpu_procs)" ]; do log "GPU has someone else's compute process(es) $(foreign_gpu_procs | tr '\n' ' '), waiting 60 s"; sleep 60; done
}

SERVER_PID=""
start_server() {
  [ -n "$SERVER_PID" ] && return
  wait_for_free_gpu
  setsid $UV python -m rl.server > $W/pipeline_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do curl -s http://127.0.0.1:8006/health > /dev/null && { log "world-model server ready"; return; }; sleep 5; done
  log "server did not come up"; exit 4
}
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT

log "pipeline start: SUP_STEPS=$SUP_STEPS WORKERS=$WORKERS DAGGER_STEPS=$DAGGER_STEPS STARTS=$STARTS"

# ---- 1. supervised student (train split only: Comma1MDataset(val=False) excludes hash%10==0 segments)
if [ ! -f outputs/supervised/supercombo.pt ]; then
  wait_for_free_gpu
  log "stage 1: supervised ($SUP_STEPS steps)"
  $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --steps $SUP_STEPS --batch-size 8 --workers $WORKERS \
    -o outputs/supervised/supercombo.pt > $W/stage1_supervised.log 2>&1 || { log "stage 1 FAILED"; exit 1; }
fi
log "stage 1 done"

# ---- 2/3. server + latent cache for every segment we will roll out
start_server
cat $W/eval_segments.txt $W/dagger_train.txt | sort -u > $W/precompute_segments.txt
log "stage 3: precompute latents for $(wc -l < $W/precompute_segments.txt) segments"
$UV python $W/eval/precompute.py $W/precompute_segments.txt > $W/stage3_precompute.log 2>&1 || { log "stage 3 FAILED"; exit 3; }
grep -c FAILED $W/stage3_precompute.log | xargs -I{} log "stage 3 done, {} segment(s) failed to encode (they are skipped)"

# ---- 4. DAgger training (checkpoint every 10 steps)
if [ ! -f outputs/dagger/on_policy_step$DAGGER_STEPS.pt ]; then
  log "stage 4: DAgger ($DAGGER_STEPS steps)"
  $UV python $W/eval/train_dagger.py --segments $W/dagger_train.txt --model outputs/supervised/supercombo.pt \
    --steps $DAGGER_STEPS --save-every 10 > $W/stage4_dagger.log 2>&1 || { log "stage 4 FAILED"; exit 4; }
fi
log "stage 4 done"

# ---- 5. closed-loop evaluation, paired across policies, on segments the student/DAgger never trained on
SUP=outputs/supervised/supercombo.pt
log "stage 5: closed-loop eval"
$UV python $W/eval/eval_closed_loop.py --segments $W/eval_segments.txt --out $W/results.jsonl \
  --starts-per-segment $STARTS --save-frames 1 \
  --policy straight=straight --policy teacher=teacher --policy student=student:$SUP \
  --policy dagger10=dagger:$SUP:outputs/dagger/on_policy_step10.pt \
  --policy dagger50=dagger:$SUP:outputs/dagger/on_policy_step50.pt \
  --policy dagger100=dagger:$SUP:outputs/dagger/on_policy_step100.pt \
  $EXTRA_POLICIES \
  --policy dagger$DAGGER_STEPS=dagger:$SUP:outputs/dagger/on_policy_step$DAGGER_STEPS.pt \
  > $W/stage5_eval.log 2>&1 || { log "stage 5 FAILED"; exit 5; }
log "stage 5 done"

# ---- 6. analysis (CPU)
$UV python -I $W/eval/analyze.py $W/results.jsonl --baseline student --val $W/val_segments.txt > $W/analysis.txt 2>&1
log "stage 6 done: see $W/analysis.txt"
stop_server
log "PIPELINE_DONE"
