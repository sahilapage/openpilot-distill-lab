#!/usr/bin/env bash
# Download every comma-four (device_type 7) comma1M segment we do not have yet into a STAGING dir
# ($W/data_staging/<seg>/), so running jobs never see half-downloaded segments. Move complete ones into
# openpilot.distill/data/ only when starting a new training run. Network/CPU only (no GPU). Resumable.
#   setsid nohup bash $W/eval/download_more.sh > $W/download_more.log 2>&1 &
set -u
W=/home/sra/sahil_distill
cd $W
BASE="https://huggingface.co/datasets/commaai/comma1M/resolve/main/data"
python3 - <<'PY' > $W/more_segments.txt
import os, json, struct
have = set(os.listdir("openpilot.distill/data"))
out = []
for f in sorted(os.listdir("scan")):
    b = open("scan/" + f, "rb").read(); n = struct.unpack("<Q", b[:8])[0]; h = json.loads(b[8:8 + n])
    lo, hi = h["device_type"]["data_offsets"]
    if int.from_bytes(b[8 + n + lo:8 + n + hi], "little", signed=True) == 7:
        s = f.split(".")[0]
        if s not in have: out.append(s)
print("\n".join(out))
PY
echo "[$(date '+%F %T')] $(wc -l < $W/more_segments.txt) new comma-four segments -> data_staging/"
fetch() {
  local seg="$1" name="$2" dest="$W/data_staging/$1/$2"
  mkdir -p "$W/data_staging/$seg"
  [ -f "$dest" ] && return 0
  for attempt in 1 2 3 4 5 6; do
    if curl -sSL --fail -C - --retry 3 --connect-timeout 30 -o "$dest.part" "$BASE/$seg/$name"; then
      mv "$dest.part" "$dest" && return 0
    fi
    sleep 5
  done
  echo "FAILED $seg/$name" >> "$W/download_more_failures.txt"
}
export -f fetch; export BASE W
: > $W/download_more_failures.txt
for seg in $(cat $W/more_segments.txt); do
  for name in fcamera.hevc ecamera.hevc frame_info.safetensors localizer.safetensors; do echo "$seg $name"; done
done | nice -n 15 ionice -c3 xargs -P 6 -L 1 bash -c 'fetch "$0" "$1"'
echo "[$(date '+%F %T')] DOWNLOAD_DONE: failures $(wc -l < $W/download_more_failures.txt); $(du -sh $W/data_staging | cut -f1)"
