#!/usr/bin/env bash
# Slot the DAgger v2 smoke test in between scaling stage 1 (pull test) and stage 2 (control run), never overlapping:
# wait for stage 1's marker, stop the scaling queue (the control has just started; it restarts from scratch), run
# smoke_dv2.sh, restart the scaling queue, and if the smoke test passed, queue run_dagger_v2.sh (it waits for the
# control evaluation by itself).
#   tmux new-session -d -s pausesmoke "bash /home/sra/sahil_distill/eval/pause_for_smoke.sh >> /home/sra/sahil_distill/pausesmoke.log 2>&1"
W=/home/sra/sahil_distill
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for scaling stage 1 (pull test)"
until [ -f $W/scale_markers/pull ]; do sleep 20; done
sleep 20
for g in $(ps -eo pgid,args | grep "[b]ash /home/sra/sahil_distill/eval/run_scale\.sh" | awk '{print $1}' | sort -u); do
  log "stopping scaling queue process group $g"; kill -TERM -- -$g
done
tmux kill-session -t scale 2>/dev/null
sleep 20
while pgrep -f "[s]upervised[.]train|[r]l[.]server" > /dev/null; do log "waiting for GPU jobs to exit"; sleep 10; done
rm -rf $W/openpilot.distill/outputs/supervised_control  # a few hundred steps at most; the control restarts cleanly
log "running smoke test"
$W/guard.sh $W/dv2smoke.log $W/eval/smoke_dv2.sh
log "smoke finished; restarting the scaling queue"
tmux new-session -d -s scale "$W/guard.sh $W/scale.log $W/eval/run_scale.sh"
if grep -q WM_INPUTS_OK $W/dv2smoke.log && grep -q "train exit 0" $W/dv2smoke.log; then
  tmux new-session -d -s dv2 "$W/guard.sh $W/dv2.log $W/eval/run_dagger_v2.sh"
  log "smoke passed: DAgger v2 queued (tmux dv2, starts after the control evaluation)"
else
  log "smoke FAILED: DAgger v2 not queued"
fi
