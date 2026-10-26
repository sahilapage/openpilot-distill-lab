#!/usr/bin/env bash
# Extend DAgger from its latest checkpoint to TARGET rollouts (auto-resumes after a crash, fresh rollouts each resume),
# then run the closed-loop eval + analysis through run_pipeline.sh. Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s pipeline "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/pipeline.log /home/sra/sahil_distill/eval/run_extend.sh"
TARGET=${TARGET:-1400}
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
  if [ -n "$SERVER_PID" ] && curl -s http://127.0.0.1:8006/health > /dev/null; then return; fi
  stop_server
  wait_for_free_gpu
  setsid $UV python -m rl.server > $W/pipeline_server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do curl -s http://127.0.0.1:8006/health > /dev/null && { log "world-model server ready"; return; }; sleep 5; done
  log "server did not come up"; exit 4
}
latest_step() { ls outputs/dagger/on_policy_step*.pt | sed 's/.*step\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }

log "extend: DAgger to $TARGET rollouts, from step $(latest_step)"
for attempt in 1 2 3 4 5 6 7 8; do
  last=$(latest_step)
  [ "$last" -ge "$TARGET" ] && break
  start_server
  log "extend attempt $attempt: resume from step $last, $((TARGET - last)) steps to go"
  # new seed per resume so the continuation does not replay the first 200 rollouts; AdamW state restarts (it is not saved)
  $UV python $W/eval/train_dagger.py --segments $W/dagger_train.txt --model outputs/supervised/supercombo.pt \
    --on-policy outputs/dagger/on_policy_step$last.pt --start-step $last --steps $((TARGET - last)) \
    --save-every 50 --seed $((last + 1)) >> $W/stage4b_dagger.log 2>&1 || { log "extend attempt $attempt failed at step $(latest_step)"; stop_server; sleep 30; }
done
[ "$(latest_step)" -ge "$TARGET" ] || { log "extend FAILED, latest step $(latest_step)"; exit 4; }
log "extend done at step $TARGET"
stop_server
sleep 10

SUP=outputs/supervised/supercombo.pt
EXTRA=""
for n in 200 400 800; do EXTRA="$EXTRA --policy dagger$n=dagger:$SUP:outputs/dagger/on_policy_step$n.pt"; done
DAGGER_STEPS=$TARGET EXTRA_POLICIES="$EXTRA" bash $W/eval/run_pipeline.sh
