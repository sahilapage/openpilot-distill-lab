#!/usr/bin/env bash
# Usage: guard.sh <log> <command...>  -- runs the command in its own process group; kills ONLY it if MemAvailable < 6 GB.
LOG="$1"; shift
setsid "$@" > "$LOG" 2>&1 &
PID=$!
PGID=$(ps -o pgid= $PID | tr -d " ")
while kill -0 $PID 2>/dev/null; do
  AVAIL=$(awk "/MemAvailable/ {print int(\$2/1024)}" /proc/meminfo)
  if [ "$AVAIL" -lt 6000 ]; then
    echo "GUARD: MemAvailable ${AVAIL} MB < 6000 -> killing our process group $PGID" >> "$LOG"
    kill -TERM -- -$PGID; sleep 5; kill -KILL -- -$PGID 2>/dev/null
    break
  fi
  sleep 3
done
wait $PID 2>/dev/null; echo "EXIT=$?" >> "$LOG"
