#!/usr/bin/env bash
# Download the chosen comma1M segments into <repo>/data/<seg>/ with 8 parallel resumable curls.
# Usage: download_segments.sh <workdir> <repo_dir>
set -u
WORK="$1"; REPO="$2"
BASE="https://huggingface.co/datasets/commaai/comma1M/resolve/main/data"
FILES="fcamera.hevc ecamera.hevc frame_info.safetensors localizer.safetensors"

fetch() {
  local seg="$1" name="$2" dest="$REPO/data/$1/$2"
  mkdir -p "$REPO/data/$seg"
  [ -f "$dest" ] && return 0
  for attempt in 1 2 3 4 5 6; do
    if curl -sSL --fail -C - --retry 3 --connect-timeout 30 -o "$dest.part" "$BASE/$seg/$name"; then
      mv "$dest.part" "$dest" && return 0
    fi
    sleep 5
  done
  echo "FAILED $seg/$name" >> "$WORK/download_failures.txt"
}
export -f fetch; export BASE REPO WORK

: > "$WORK/download_failures.txt"
for seg in $(cat "$WORK/chosen_segments.txt"); do
  for name in $FILES; do echo "$seg $name"; done
done | xargs -P 8 -L 1 bash -c 'fetch "$0" "$1"'
echo "download finished: $(date)"
echo "failures: $(wc -l < "$WORK/download_failures.txt")"
du -sh "$REPO/data"
