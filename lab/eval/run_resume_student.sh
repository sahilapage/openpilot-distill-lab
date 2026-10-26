#!/usr/bin/env bash
# After a pause of the student run (to free the GPU for the measure / open-loop / snapshot queue): waits for SNAPS_DONE
# in snaps.log, then resumes training from supercombo.resume.pt (run_student.sh continues the same step-indexed LR
# schedule, so the pause does not change it). Launch ONLY with the user's go-ahead:
#   tmux new-session -d -s student "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/student.log /home/sra/sahil_distill/eval/run_resume_student.sh"
W=/home/sra/sahil_distill
log() { echo "[$(date '+%F %T')] $*"; }
log "resume-student: waiting for SNAPS_DONE in snaps.log"
until grep -q SNAPS_DONE $W/snaps.log 2>/dev/null; do
  pgrep -f "^bash /home/sra/sahil_distill/eval/run_(after|openloop|openloop_snapshots)\.sh" > /dev/null || { log "eval queue is not running and never finished: not resuming"; exit 2; }
  sleep 120
done
sleep 30
exec bash $W/eval/run_student.sh
