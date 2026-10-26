#!/usr/bin/env bash
# One-shot (user's go, 2026-10-26 ~19:00): pause the student at its next resume save, then run the eval queue
# (run_after -> run_openloop -> run_openloop_snapshots) and resume the student after SNAPS_DONE.
# Runs detached (setsid nohup) so a dropped ssh cannot leave it half-done. Log: pause_and_queue.log
W=/home/sra/sahil_distill
G=$W/guard.sh
R=$W/openpilot.distill/outputs/supervised_long/supercombo.resume.pt
log() { echo "[$(date '+%F %T')] $*"; }
m0=$(stat -c %Y $R)
log "waiting for the next resume save (now $(grep -o 'step [0-9]*' $W/stage10_student.log | tail -1))"
for i in $(seq 1 120); do [ "$(stat -c %Y $R)" != "$m0" ] && break; sleep 5; done
sleep 8  # let the atomic rename settle
log "saved at $(grep -o 'step [0-9]*' $W/stage10_student.log | tail -1); stopping the student"
bash $W/eval/stop_pipeline.sh
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -n "$left" ] && { log "GPU still has compute processes ($left): not starting the queue"; exit 3; }
log "GPU free"
tmux kill-session -t student 2>/dev/null
tmux new-session -d -s after "$G $W/after.log $W/eval/run_after.sh"
sleep 5
tmux new-session -d -s openloop "$G $W/openloop.log $W/eval/run_openloop.sh"
sleep 5
tmux new-session -d -s snaps "$G $W/snaps.log $W/eval/run_openloop_snapshots.sh"
sleep 5
tmux new-session -d -s student "$G $W/student.log $W/eval/run_resume_student.sh"
sleep 5
tmux ls
log "queued: after -> openloop -> snaps -> student resume"
