#!/usr/bin/env bash
# DAgger v2 (user's go 2026-10-27 ~18:00): full-network on-policy distillation inside comma's world model.
# Per iteration k = 1..ITERS:
#   collect  ROLLOUTS world-model rollouts (collect_wm.py) on the 348 TRAIN drives that are not eval drives; the driver
#            is the current student (or comma's teacher, beta mixing: TEACHER_FRAC[k]); steering pulls in 75% of them
#   train    the WHOLE student (vision included) from the previous iterate: real comma1M batches as before + one
#            world-model batch per step (all iterations so far, aggregated), teacher labels computed online
#   eval     steering-pull test (same drives/starts as results_pull_scale.jsonl) + open-loop on the 42 eval drives
# Base: the 360-drive student. Resumable at every level; markers in $W/dv2_markers. Queued after run_scale.sh.
#   tmux new-session -d -s dv2 "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/dv2.log /home/sra/sahil_distill/eval/run_dagger_v2.sh"
ITERS=${ITERS:-3}
ROLLOUTS=${ROLLOUTS:-600}
STEPS=${STEPS:-6000}
LR=${LR:-1e-4}
TEACHER_FRAC=(0 0.5 0.2 0.2 0.2 0.2)
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=. WANDB_MODE=offline
M=$W/dv2_markers
mkdir -p $M
BASE=outputs/supervised_more/supercombo.pt
log() { echo "[$(date '+%F %T')] $*"; }
foreign_gpu_procs() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W/*) ;; *) echo $p;; esac
  done
}
wait_gpu() { while [ -n "$(foreign_gpu_procs)" ]; do log "GPU busy with someone else's job, waiting 60 s"; sleep 60; done; }
SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; sleep 5; }
start_server() {
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
  curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
  setsid $UV python -m rl.server > $W/dv2_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
    curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
  done
  curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
}
trap stop_server EXIT
retry() {  # retry <name> <cmd...>: 3 attempts, restarting the server in between
  local name=$1; shift
  for attempt in 1 2 3; do
    "$@" && return 0
    log "$name attempt $attempt failed"; stop_server; sleep 30; start_server
  done
  log "$name FAILED"; exit 20
}

if [ -z "$NO_WAIT" ]; then
  log "dv2: waiting for the scaling queue (run_scale.sh) to finish its control evaluation"
  until [ -f $W/scale_markers/ctleval ] && ! pgrep -f "^bash /home/sra/sahil_distill/eval/run_scale\.sh" > /dev/null; do
    if ! pgrep -f "^bash /home/sra/sahil_distill/eval/run_scale\.sh" > /dev/null && [ ! -f $W/scale_markers/ctleval ]; then
      log "scaling queue stopped before its control evaluation: not starting"; exit 2
    fi
    sleep 120
  done
fi

prev=$BASE
for k in $(seq 1 $ITERS); do
  out=outputs/dagger_v2/iter$k/supercombo.pt
  tf=${TEACHER_FRAC[$k]}
  sf=$(python3 -c "print(1 - $tf)")
  if [ ! -f $M/collect$k ]; then
    wait_gpu; start_server
    log "iter $k: collecting $ROLLOUTS rollouts, driver student $sf / teacher $tf, student = $prev"
    retry collect$k $UV python $W/eval/collect_wm.py --segments $W/wm_collect_pool.txt --out-dir outputs/wm_data/iter$k \
      --rollouts $ROLLOUTS --seed $k --driver student=student:$prev --driver teacher=teacher --driver-weights $sf,$tf \
      >> $W/stage20_collect.log 2>&1
    stop_server
    touch $M/collect$k; log "iter $k: $(ls outputs/wm_data/iter$k/r*.npz | wc -l) rollouts, $(du -sh outputs/wm_data | cut -f1) world-model data"
  fi
  if [ ! -f $M/train$k ]; then
    log "iter $k: training $STEPS steps from $prev on real + all world-model data"
    for attempt in $(seq 1 10); do
      [ -f $out ] && break
      wait_gpu
      WM_DATA=outputs/wm_data WM_WEIGHT=1.0 INIT_WEIGHTS=$prev LR_DECAY_START=0 LR_MIN_RATIO=0.1 SNAPSHOT_EVERY=0 SAVE_EVERY=500 \
        $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --steps $STEPS --batch-size 8 --workers 8 \
        --learning-rate $LR -o $out >> $W/stage20_train.log 2>&1 || { log "train attempt $attempt failed at $(grep -o 'step [0-9]*' $W/stage20_train.log | tail -1)"; sleep 60; }
    done
    [ -f $out ] || { log "iter $k training FAILED"; exit 21; }
    touch $M/train$k; log "iter $k: trained $out"
  fi
  if [ ! -f $M/eval$k ]; then
    wait_gpu; start_server
    log "iter $k: pull test + open-loop"
    retry pull$k $UV python $W/eval/eval_shove.py --segments $W/eval_segments.txt --out $W/results_pull_scale.jsonl \
      --pull-dps 3 --save-frames 3 --frames-dir outputs/pull_scale/frames --policy dv2_$k=student:$out >> $W/stage20_eval.log 2>&1
    retry openloop$k $UV python $W/eval/eval_open_loop.py --segments $W/more_eval_segments.txt \
      --out $W/results_open_loop_dv2.jsonl --policy dv2_$k=student:$out >> $W/stage20_eval.log 2>&1
    stop_server
    PULL_REFS=straight,more $UV python -I $W/eval/analyze_pull.py $W/results_pull_scale.jsonl $W/val_segments.txt \
      > $W/analysis_pull_dv2.txt 2>&1 || log "analyze_pull FAILED"
    cat $W/results_open_loop_more.jsonl $W/results_open_loop_dv2.jsonl > $W/results_open_loop_dv2_all.jsonl
    $UV python -I $W/eval/analyze_open_loop.py $W/results_open_loop_dv2_all.jsonl $W/unseen_all.txt > $W/analysis_open_loop_dv2.txt 2>&1 \
      || log "analyze_open_loop FAILED"
    touch $M/eval$k; log "iter $k: eval done ($W/analysis_pull_dv2.txt, $W/analysis_open_loop_dv2.txt)"
  fi
  prev=$out
done
log "DV2_DONE"
