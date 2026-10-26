"""Pick comma four (mici, device_type 7) segments from comma1M and cache their frame_info files.

Usage: python3 select_segments.py <workdir> <count>
Writes <workdir>/chosen_segments.txt (one id per line) and <workdir>/scan/<seg>.frame_info.safetensors.
"""
import json
import os
import random
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co"
MICI = 7
KNOWN = "2333e255f1de1fe83ea5975b24a3cc67"  # segment used by the repo's README / env defaults

workdir, count = sys.argv[1], int(sys.argv[2])
scan_dir = os.path.join(workdir, "scan")
os.makedirs(scan_dir, exist_ok=True)

info = json.load(urllib.request.urlopen(f"{BASE}/api/datasets/commaai/comma1M"))
segs = sorted(
    {s["rfilename"].split("/")[1] for s in info["siblings"] if s["rfilename"].endswith("frame_info.safetensors")}
)
print("segments:", len(segs), flush=True)


def device_type(seg: str) -> tuple[str, int]:
    path = os.path.join(scan_dir, f"{seg}.frame_info.safetensors")
    if not os.path.exists(path):
        url = f"{BASE}/datasets/commaai/comma1M/resolve/main/data/{seg}/frame_info.safetensors"
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                break
            except Exception:
                if attempt == 4:
                    raise
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    header = json.loads(b[8 : 8 + n])
    lo, hi = header["device_type"]["data_offsets"]
    return seg, int.from_bytes(b[8 + n + lo : 8 + n + hi], "little", signed=True)


with ThreadPoolExecutor(16) as pool:
    results = list(pool.map(device_type, segs))

mici = [s for s, d in results if d == MICI]
counts = {}
for _, d in results:
    counts[d] = counts.get(d, 0) + 1
print("device_type counts (all segments):", counts, flush=True)
print("comma four segments:", len(mici), flush=True)

random.seed(0)
chosen = random.sample(mici, min(count, len(mici)))
if KNOWN in mici and KNOWN not in chosen:
    chosen[0] = KNOWN
chosen = sorted(chosen)
with open(os.path.join(workdir, "chosen_segments.txt"), "w") as f:
    f.write("\n".join(chosen) + "\n")
print("chosen:", len(chosen), "| includes README segment:", KNOWN in chosen, flush=True)
