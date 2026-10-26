"""Does the world model (with dequantized NVFP4 weights) still generate plausible driving frames?

Rolls out N frames with a constant actor from a real segment, then writes a grid:
top row = generated narrow-camera frames, bottom row = the real frames (through the same VAE).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from rl.actor import ConstantActor
from rl.env import FRAME_SKIP, SEGMENT, Env, Episode, rollout
from rl.server import SERVER_URL, RuntimeClient

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 40  # 5 Hz frames into the segment

rt = RuntimeClient(SERVER_URL, 15, 2.0)
ep = Episode(rt, SEGMENT, cache_dir=Path("outputs/rl/cache"))
start = (rt.history_frames + OFFSET) * FRAME_SKIP
end = start + N * FRAME_SKIP
env = Env(rt, ep, start, end)

t = time.time()
res = rollout(env, ConstantActor(0.0, 0.0))
dt = time.time() - t
print(f"rollout {N} frames in {dt:.1f}s -> {dt / N:.2f}s/frame", flush=True)


def narrow(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape[0] == 6:  # CHW -> HWC
        frame = frame.transpose(1, 2, 0)
    return frame[..., :3]


gen = [narrow(f) for f in res.worldmodel_outputs["frames"]]
first = start // FRAME_SKIP
real = [narrow(rt.decode(ep.latents[first + i : first + i + 1])[0]) for i in range(N)]
mse = [float(np.mean((g.astype(np.float32) - r.astype(np.float32)) ** 2)) for g, r in zip(gen, real)]
print("per-frame MSE generated vs real:", [round(m, 1) for m in mse])
print("generated frame mean/std:", float(np.mean(gen)), float(np.std(gen)))
plans = np.asarray(res.worldmodel_outputs["plan"])
print("plan array", plans.shape, "finite:", bool(np.isfinite(plans).all()))

out = Path("outputs/smoke")
out.mkdir(parents=True, exist_ok=True)
grid = np.concatenate([np.concatenate(gen, axis=1), np.concatenate(real, axis=1)], axis=0)
cv2.imwrite(str(out / "wm_sanity.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("wrote", out / "wm_sanity.png", grid.shape)
print("SANITY_OK")
