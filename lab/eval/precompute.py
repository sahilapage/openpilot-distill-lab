"""Encode segments to world-model latents once (cached on disk). Needs the world-model server running.

Usage (repo root): PYTHONPATH=. python ../eval/precompute.py <segments_file>
"""
import sys
import time
from pathlib import Path

from rl.env import Episode
from rl.server import SERVER_URL, RuntimeClient

segments = [s.strip() for s in Path(sys.argv[1]).read_text().split() if s.strip()]
rt = RuntimeClient(SERVER_URL)
cache = Path("outputs/rl/cache")
start = time.time()
for i, seg in enumerate(segments, 1):
    t = time.time()
    try:
        ep = Episode(rt, seg, cache_dir=cache)
        print(f"[{i}/{len(segments)}] {seg} ok ({len(ep.latents)} latents, {time.time() - t:.0f}s)", flush=True)
    except Exception as e:  # keep going; a bad segment must not stop the batch
        print(f"[{i}/{len(segments)}] {seg} FAILED: {type(e).__name__}: {e}", flush=True)
print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)
