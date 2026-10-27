#!/usr/bin/env bash
# Stop every process of this experiment (pipeline, training, evals, world-model server) and nothing else.
# DRY=1 only lists what it would stop. Safe to run over ssh: it never matches its own shell. Training resumes exactly on relaunch (resume.pt), evals skip
# finished rows. Relaunch:
#   tmux new-session -d -s pipeline "/home/sra/sahil_distill/guard.sh /home/sra/sahil_distill/pipeline.log /home/sra/sahil_distill/eval/run_coach.sh"
W=/home/sra/sahil_distill
me=$$
[ -z "$DRY" ] && { tmux kill-session -t pipeline 2>/dev/null; tmux kill-session -t after 2>/dev/null; tmux kill-session -t openloop 2>/dev/null; tmux kill-session -t student 2>/dev/null; tmux kill-session -t snaps 2>/dev/null; tmux kill-session -t anchor 2>/dev/null; tmux kill-session -t more 2>/dev/null; tmux kill-session -t moreeval 2>/dev/null; tmux kill-session -t scale 2>/dev/null; tmux kill-session -t dv2 2>/dev/null; }
groups=""
for p in $(pgrep -u "$(id -u)" -f "guard\.sh|run_(coach|pipeline|extend|shove|after|openloop|openloop_snapshots|student|resume_student|anchor|student_more|more_eval|scale|dagger_v2)\.sh|collect_wm\.py|download_big\.sh|check_segments\.py|train_dagger\.py|eval_(closed_loop|shove|open_loop)\.py|measure_forgiveness\.py|calibrate_forgiveness\.py|rl\.server|supervised\.train"); do
  [ "$p" = "$me" ] && continue
  case "$(readlink /proc/$p/cwd 2>/dev/null)" in $W|$W/*) ;; *) continue;; esac
  case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in *stop_pipeline*|*ssh*) continue;; esac
  groups="$groups $(ps -o pgid= -p $p | tr -d ' ')"
done
for g in $(echo $groups | tr ' ' '\n' | sort -u); do
  [ "$g" = "$(ps -o pgid= -p $me | tr -d ' ')" ] && continue
  if [ -n "$DRY" ]; then echo "would stop group $g: $(ps -o args= -g $g | cut -c1-80 | tr "\n" "|")"; else kill -TERM -- -$g 2>/dev/null && echo "stopped process group $g"; fi
done
[ -z "$DRY" ] && sleep 10
echo "GPU compute processes left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
