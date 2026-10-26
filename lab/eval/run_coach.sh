#!/usr/bin/env bash
# Two coaches, equal budget, then every eval. Each stage resumes from its latest checkpoint, so a crash or a kill only
# costs the steps since the last save. Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s pipeline "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/pipeline.log /home/sra/sahil_distill/eval/run_coach.sh"
# 1. vanilla DAgger (world-model plan labels) resumed to BUDGET steps     -> outputs/dagger
# 2. coached DAgger (teacher labels + injected steering pulls) 0..BUDGET  -> outputs/dagger_coach
# 3. main closed-loop eval + analysis (run_pipeline.sh, both coaches added)
# 4. steering-pull test (eval_shove.py --pull-dps)                        -> results_pull.jsonl
BUDGET=${BUDGET:-800}
PULL_DPS=${PULL_DPS:-3}
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

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill -TERM -- -$SERVER_PID 2>/dev/null; SERVER_PID=""; }
trap stop_server EXIT
start_server() {
  if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null && curl -s http://127.0.0.1:8006/health > /dev/null; then return; fi
  stop_server
  # an old server (e.g. run_pipeline.sh's, shutting down) can still answer /health: wait until the port is free
  for i in $(seq 1 90); do curl -s http://127.0.0.1:8006/health > /dev/null || break; sleep 2; done
  curl -s http://127.0.0.1:8006/health > /dev/null && { log "port 8006 still served by an old server"; exit 4; }
  wait_for_free_gpu
  setsid $UV python -m rl.server > $W/pipeline_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 $SERVER_PID 2>/dev/null || { log "server exited during startup"; exit 4; }
    curl -s http://127.0.0.1:8006/health > /dev/null && { log "world-model server ready"; return; }
    sleep 5
  done
  log "server did not come up"; exit 4
}
latest_step() { ls $1/on_policy_step*.pt 2>/dev/null | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }

# train <name> <out_dir> <extra args...>: resume <out_dir> to BUDGET steps, fresh seed per resume
train() {
  local name=$1 dir=$2; shift 2
  for attempt in 1 2 3 4 5 6 7 8; do
    local last=$(latest_step $dir)
    [ -n "$last" ] && [ "$last" -ge "$BUDGET" ] && { log "$name done at step $BUDGET"; return 0; }
    start_server
    local resume=""
    [ -n "$last" ] && resume="--on-policy $dir/on_policy_step$last.pt --start-step $last"
    log "$name attempt $attempt: from step ${last:-0}, $((BUDGET - ${last:-0})) steps to go"
    $UV python $W/eval/train_dagger.py --segments $W/dagger_train.txt --model outputs/supervised/supercombo.pt \
      $resume --steps $((BUDGET - ${last:-0})) --save-every 50 --seed $((${last:-0} + 1 + ${SEED_OFFSET:-0})) --out-dir $dir "$@" \
      >> $W/stage4_${name}.log 2>&1 || { log "$name attempt $attempt failed at step $(latest_step $dir)"; stop_server; sleep 30; }
  done
  log "$name FAILED, latest step $(latest_step $dir)"; exit 4
}

log "coach pipeline start: BUDGET=$BUDGET PULL_DPS=$PULL_DPS"
train vanilla outputs/dagger
SEED_OFFSET=100000 train coach outputs/dagger_coach --expert teacher --disturb-prob 0.8 --pull-min-dps 1 --pull-max-dps 4
stop_server
sleep 10

# ---- main eval: run_pipeline.sh skips its finished stages 1-4 and adds the extra checkpoints
SUP=outputs/supervised/supercombo.pt
EXTRA="--policy dagger200=dagger:$SUP:outputs/dagger/on_policy_step200.pt"
for n in 200 400 $BUDGET; do EXTRA="$EXTRA --policy coach$n=dagger:$SUP:outputs/dagger_coach/on_policy_step$n.pt"; done
DAGGER_STEPS=$BUDGET EXTRA_POLICIES="$EXTRA" bash $W/eval/run_pipeline.sh || { log "main eval FAILED"; exit 5; }

# ---- steering-pull test
start_server
log "pull test: $PULL_DPS deg/s, left and right"
$UV python $W/eval/eval_shove.py --segments $W/eval_segments.txt --out $W/results_pull.jsonl --save-frames 1 \
  --frames-dir outputs/pull/frames --pull-dps $PULL_DPS \
  --policy straight=straight --policy teacher=teacher --policy student=student:$SUP \
  --policy dagger$BUDGET=dagger:$SUP:outputs/dagger/on_policy_step$BUDGET.pt \
  --policy coach200=dagger:$SUP:outputs/dagger_coach/on_policy_step200.pt \
  --policy coach400=dagger:$SUP:outputs/dagger_coach/on_policy_step400.pt \
  --policy coach$BUDGET=dagger:$SUP:outputs/dagger_coach/on_policy_step$BUDGET.pt \
  > $W/stage7_pull.log 2>&1 || { log "pull test FAILED"; exit 7; }
stop_server
log "ALL_DONE"
