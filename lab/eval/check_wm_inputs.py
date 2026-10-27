"""DAgger v2 check: a world-model training sample (supervised/wm_dataset.py) must be byte-identical to what the student
actor itself fed its network at the same moment of a closed-loop rollout. Runs one short rollout with the student,
then rebuilds the last step's image pair from the saved frames and compares. Needs the world-model server.
  PYTHONPATH=. python check_wm_inputs.py STUDENT.pt SEGMENT
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_closed_loop import build_actor  # noqa: E402

from rl.env import ENCODE_BATCH_SIZE, FRAME_SKIP, Env, Episode, rollout  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402
from supervised.wm_dataset import N_STEPS, make_sample  # noqa: E402


def main() -> None:
    student, seg = sys.argv[1], sys.argv[2]
    rt = RuntimeClient(SERVER_URL, 15, 2.0)
    ep = Episode(rt, seg, encode_batch_size=ENCODE_BATCH_SIZE, cache_dir=Path("outputs/rl/cache"))
    start, frames_n = 30, 6
    actor = build_actor("student", [student])
    actor.reset()
    env = Env(rt, ep, start * FRAME_SKIP, (start + frames_n) * FRAME_SKIP)
    history = np.asarray(rt.decode(ep.latents[start - rt.history_frames : start]))
    res = rollout(env, actor)
    frames = np.concatenate((history, np.stack([np.asarray(f) for f in res.worldmodel_outputs["frames"]])))
    meta = {"n_history": len(history), "lht": False}
    inputs, targets = make_sample(frames, meta, len(frames) - 1, np.array([0.2, 0.2]))
    seen = actor.model_inputs  # after the last act(): the pair (frames[-2], frames[-1])
    ok = True
    for key in ("img", "big_img"):
        mine = inputs[key][-1]
        theirs = np.asarray(seen[key][0])
        same = mine.shape == theirs.shape and np.array_equal(mine, theirs)
        print(f"{key}: sample {mine.shape} actor {theirs.shape} identical={same} max|diff|="
              f"{np.abs(mine.astype(int) - theirs.astype(int)).max() if mine.shape == theirs.shape else 'shape'}")
        ok &= same
    print(f"frames {frames.shape} {frames.dtype}; sample img {inputs['img'].shape} (N_STEPS={N_STEPS}); imgs target {targets['imgs'].shape}")
    print("WM_INPUTS_OK" if ok else "WM_INPUTS_MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
