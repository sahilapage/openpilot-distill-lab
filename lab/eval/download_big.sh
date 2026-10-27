#!/usr/bin/env bash
# Data-scaling step 3 (user's go 2026-10-27 ~01:30): download comma 3 / 3X (tici device_type 4, tizi 6) comma1M segments
# from the TRAIN split only (hash % 10 != 0; never-seen eval drives stay comma four), N_PER_TYPE of each, seed 0, into
# $W/data_staging_big/<seg>/. The supervised loader already warps every device's cameras into the model's view. Network
# and disk only (no GPU); stops taking new segments when /home would drop below MIN_FREE_GB. Resumable. Logs
# DOWNLOAD_BIG_DONE.
#   setsid nohup bash $W/eval/download_big.sh > $W/download_big.log 2>&1 &
set -u
N_PER_TYPE=${N_PER_TYPE:-250}
MIN_FREE_GB=${MIN_FREE_GB:-35}
W=/home/sra/sahil_distill
cd $W
BASE="https://huggingface.co/datasets/commaai/comma1M/resolve/main/data"
N_PER_TYPE=$N_PER_TYPE python3 - <<'PY' > $W/big_segments.txt
import json, os, random, struct
have = set(os.listdir("openpilot.distill/data"))
by = {4: [], 6: []}
for f in sorted(os.listdir("scan")):
    b = open("scan/" + f, "rb").read(); n = struct.unpack("<Q", b[:8])[0]; h = json.loads(b[8:8 + n])
    lo, hi = h["device_type"]["data_offsets"]
    d = int.from_bytes(b[8 + n + lo:8 + n + hi], "little", signed=True)
    s = f.split(".")[0]
    if d in by and hash(int(s, 16)) % 10 != 0 and s not in have:
        by[d].append(s)
rng = random.Random(0)
k = int(os.environ["N_PER_TYPE"])
print("\n".join(rng.sample(by[4], k) + rng.sample(by[6], k)))
PY
echo "[$(date '+%F %T')] $(wc -l < $W/big_segments.txt) comma 3/3X train-split segments -> data_staging_big/ (stop below ${MIN_FREE_GB} GB free)"
fetch_segment() {
  local seg="$1" dir="$W/data_staging_big/$1"
  local free_gb=$(( $(df --output=avail -k /home | tail -1) / 1048576 ))
  if [ ! -f "$dir/localizer.safetensors" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "SKIPPED_DISK $seg" >> "$W/download_big_failures.txt"; return 0
  fi
  mkdir -p "$dir"
  for name in fcamera.hevc ecamera.hevc frame_info.safetensors localizer.safetensors; do
    [ -f "$dir/$name" ] && continue
    local ok=0
    for attempt in 1 2 3 4 5 6; do
      if curl -sSL --fail -C - --retry 3 --connect-timeout 30 -o "$dir/$name.part" "$BASE/$seg/$name"; then
        mv "$dir/$name.part" "$dir/$name"; ok=1; break
      fi
      sleep 5
    done
    [ $ok = 1 ] || { echo "FAILED $seg/$name" >> "$W/download_big_failures.txt"; return 0; }
  done
}
export -f fetch_segment; export BASE W MIN_FREE_GB
: > $W/download_big_failures.txt
nice -n 15 ionice -c3 xargs -P 6 -n 1 bash -c 'fetch_segment "$0"' < $W/big_segments.txt
complete=$(for d in $W/data_staging_big/*/; do [ -f $d/localizer.safetensors ] && [ -f $d/fcamera.hevc ] && [ -f $d/ecamera.hevc ] && [ -f $d/frame_info.safetensors ] && echo; done | wc -l)
echo "[$(date '+%F %T')] DOWNLOAD_BIG_DONE: $complete complete; failures/skips $(wc -l < $W/download_big_failures.txt); $(du -sh $W/data_staging_big | cut -f1); /home free $(df -h --output=avail /home | tail -1)"
