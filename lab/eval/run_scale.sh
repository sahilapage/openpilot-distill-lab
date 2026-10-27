#!/usr/bin/env bash
# Data-scaling queue (user's go 2026-10-27 ~17:30, "start ahead"). One GPU job at a time, in order; every stage is
# resumable and marked done in $W/scale_markers/, so re-running the script skips finished work.
#   1 pull     steering-pull test (3 deg/s, same drives/starts/seed as results_pull_fixed) for straight, teacher, the
#              40k student (128 training drives) and the bigger-data student (360)  -> results_pull_scale.jsonl
#   2 big      wait for download_big.sh; load every staged comma 3/3X segment once through the training loader; train
#              the same recipe (init 40k student, 40k steps, cosine 3e-4 -> 3e-5, batch 8) on all of data/
#   3 bigeval  open-loop on the same 42 drives as results_open_loop_more.jsonl (big snapshots), + pull test of bigfinal
#   4 control  the same recipe on ONLY the original 128 training drives (does data or the extra 40k steps help?)
#   5 ctleval  open-loop + pull test of the control
# Logs SCALE_DONE.
#   tmux new-session -d -s scale "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/scale.log /home/sra/sahil_distill/eval/run_scale.sh"
W=/home/sra/sahil_distill
source $W/env.sh
unset VIRTUAL_ENV
UV="../.boot/bin/uv run --frozen"
cd $W/openpilot.distill
export PYTHONPATH=. WANDB_MODE=offline
M=$W/scale_markers
mkdir -p $M
STUDENT40K=outputs/supervised_long/supercombo_step40000.pt
MORE=outputs/supervised_more/supercombo.pt
BIG=outputs/supervised_big/supercombo.pt
CONTROL=outputs/supervised_control/supercombo.pt
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
  setsid $UV python -m rl.server > $W/scale_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
    curl -s http://127.0.0.1:8006/health > /dev/null && break; sleep 5
  done
  curl -s http://127.0.0.1:8006/health > /dev/null || { log "server did not come up"; exit 4; }
}
trap stop_server EXIT

# pull test: same settings as results_pull_fixed.jsonl (25 frames, 3 starts per drive, seed 0, 3 deg/s, left + right)
pull() {  # pull <policy args...>
  wait_gpu; start_server
  for attempt in 1 2 3; do
    $UV python $W/eval/eval_shove.py --segments $W/eval_segments.txt --out $W/results_pull_scale.jsonl \
      --pull-dps 3 --save-frames 3 --frames-dir outputs/pull_scale/frames "$@" >> $W/stage16_pull.log 2>&1 && break
    log "pull attempt $attempt failed"; stop_server; sleep 30; start_server
    [ $attempt = 3 ] && { log "pull test FAILED"; exit 16; }
  done
  stop_server
  PULL_REFS=straight,student40k $UV python -I $W/eval/analyze_pull.py $W/results_pull_scale.jsonl $W/val_segments.txt \
    > $W/analysis_pull_scale.txt 2>&1 || log "analyze_pull FAILED"
}

# open-loop on the 42 drives of the bigger-data eval; analysis over every open-loop row so far
openloop() {  # openloop <out.jsonl> <policy args...>
  local out=$1; shift
  wait_gpu; start_server
  for attempt in 1 2 3; do
    $UV python $W/eval/eval_open_loop.py --segments $W/more_eval_segments.txt --out $out "$@" >> $W/stage16_openloop.log 2>&1 && break
    log "open-loop attempt $attempt failed"; sleep 30
    [ $attempt = 3 ] && { log "open-loop FAILED"; exit 17; }
  done
  stop_server
  cat $W/results_open_loop_more.jsonl $W/results_open_loop_big.jsonl $W/results_open_loop_control.jsonl 2>/dev/null > $W/results_open_loop_scale.jsonl
  $UV python -I $W/eval/analyze_open_loop.py $W/results_open_loop_scale.jsonl $W/unseen_all.txt > $W/analysis_open_loop_scale.txt 2>&1 \
    || log "analyze_open_loop FAILED"
}

snapshot_policies() {  # snapshot_policies <name> <dir>
  local p=""
  for f in $(ls $2/supercombo_step*.pt 2>/dev/null | sort -V); do
    n=$(basename $f .pt | sed 's/.*step\([0-9]*\)/\1/'); p="$p --policy $1$((n / 1000))k=student:$f"
  done
  echo "$p --policy $1final=student:$2/supercombo.pt"
}

train() {  # train <out.pt> <dataset root> <log>
  export LR_DECAY_START=0 LR_MIN_RATIO=0.1 SNAPSHOT_EVERY=5000 SAVE_EVERY=500 INIT_WEIGHTS=$STUDENT40K
  [ -f $INIT_WEIGHTS ] || { log "missing $INIT_WEIGHTS"; exit 3; }
  for attempt in $(seq 1 10); do
    [ -f $1 ] && return 0
    wait_gpu
    log "train $1 attempt $attempt"
    $UV torchrun --standalone --nproc-per-node=1 -m supervised.train --dataset $2 --steps 40000 --batch-size 8 --workers 8 \
      -o $1 >> $3 2>&1 || { log "attempt $attempt failed at $(grep -o 'step [0-9]*' $3 | tail -1)"; sleep 60; }
  done
  [ -f $1 ] || { log "training $1 FAILED"; exit 18; }
}

# ---- 1. pull test of the 128-drive and 360-drive students
if [ ! -f $M/pull ]; then
  log "stage 1: pull test straight / teacher / student40k / more"
  pull --policy straight=straight --policy teacher=teacher --policy student40k=student:$STUDENT40K --policy more=student:$MORE
  touch $M/pull; log "stage 1 done: $W/analysis_pull_scale.txt"
fi

# ---- 2. bigger data: comma four + comma 3/3X
if [ ! -f $M/big ]; then
  log "stage 2: waiting for DOWNLOAD_BIG_DONE"
  until grep -q DOWNLOAD_BIG_DONE $W/download_big.log 2>/dev/null; do
    pgrep -f "download_big[.]sh" > /dev/null || { log "download is not running and never finished: stopping"; exit 2; }
    sleep 120
  done
  if ! grep -q CHECK_DONE $W/big_checked.txt 2>/dev/null; then
    wait_gpu
    log "loading every staged segment once through the training loader"
    $UV python $W/eval/check_segments.py $W/data_staging_big --procs 4 > $W/big_checked.tmp 2> $W/big_check.err \
      || log "check_segments exited non-zero (per-segment results kept)"
    echo CHECK_DONE >> $W/big_checked.tmp && mv $W/big_checked.tmp $W/big_checked.txt
  fi
  log "check: $(grep -c '^OK' $W/big_checked.txt) OK (by device type: $(awk '$1=="OK"{print $3}' $W/big_checked.txt | sort | uniq -c | tr -s ' \n' ' ')), $(grep -c '^FAIL' $W/big_checked.txt) FAIL"
  grep '^FAIL' $W/big_checked.txt | head -5
  [ "$(grep -c '^OK' $W/big_checked.txt)" -ge 100 ] || { log "fewer than 100 segments passed: not training"; exit 3; }
  moved=0
  for seg in $(awk '$1=="OK"{print $2}' $W/big_checked.txt); do
    [ -d $W/data_staging_big/$seg ] && [ ! -e data/$seg ] && mv $W/data_staging_big/$seg data/$seg && moved=$((moved + 1))
  done
  log "moved $moved segments into data/; data/ now $(ls data | wc -l) segments"
  train $BIG $W/openpilot.distill $W/stage16_big.log
  touch $M/big; log "stage 2 done: $BIG"
fi

# ---- 3. evaluate the bigger-data student
if [ ! -f $M/bigeval ]; then
  log "stage 3: open-loop + pull test of the big student"
  openloop $W/results_open_loop_big.jsonl $(snapshot_policies big outputs/supervised_big)
  pull --policy big=student:$BIG
  touch $M/bigeval; log "stage 3 done: $W/analysis_open_loop_scale.txt, $W/analysis_pull_scale.txt"
fi

# ---- 4. control: same recipe, only the original 128 training drives (+ their 12 val, which training skips)
if [ ! -f $M/control ]; then
  log "stage 4: control run on the original drives"
  mkdir -p $W/data_control/data
  for seg in $(cat $W/chosen_segments.txt); do
    [ -e $W/data_control/data/$seg ] || ln -s $W/openpilot.distill/data/$seg $W/data_control/data/$seg
  done
  log "control dataset: $(ls $W/data_control/data | wc -l) segments"
  train $CONTROL $W/data_control $W/stage16_control.log
  touch $M/control; log "stage 4 done: $CONTROL"
fi

# ---- 5. evaluate the control
if [ ! -f $M/ctleval ]; then
  log "stage 5: open-loop + pull test of the control"
  openloop $W/results_open_loop_control.jsonl $(snapshot_policies control outputs/supervised_control)
  pull --policy control=student:$CONTROL
  touch $M/ctleval; log "stage 5 done"
fi
log "SCALE_DONE"
