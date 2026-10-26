"""Multi-segment DAgger-style fine-tuning: rl/train.py generalised to many segments, with periodic checkpoints.

Same update as rl/train.py (only the on-policy temporal head trains; target = action derived from the world model's own
plan for the states the student actually reached), but each step draws a random segment from --segments, and a
checkpoint is written every --save-every steps so one run gives the whole "number of DAgger steps" curve.
Needs the world-model server. Run from the repo root with PYTHONPATH=. .

--disturb-prob > 0 turns on disturbance injection (DART-style): in that fraction of rollouts the steering is pulled to
one side by a random yaw rate from some random frame on, on top of what the student commands. The label is unchanged
(the world model's plan for the state actually reached), so the student practises recovering from drift it would not
cause on its own.

--expert teacher replaces the world model's plan as the label with comma's teacher (the ONNX Supercombo the student was
distilled from), run on the same frames the student sees: textbook DAgger for a distilled student. Reason: under a slow
steering pull the world model's plan bends along with the drift (it reads it as the road curving), while the teacher
counter-steers, so only the teacher's labels teach recovery.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_accel_from_plan, get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

from rl.actor import SupercomboActor
from rl.env import ENCODE_BATCH_SIZE, FRAME_SKIP, ROLLOUT_FRAMES, Env, Episode, rollout
from rl.server import SERVER_URL, RuntimeClient
from supervised.model import ModelInputs

T_IDXS = np.asarray(ModelConstants.T_IDXS)
FPS = 5


class PullActor:
    """Adds a constant yaw rate (rad/s) to the actor's steering from frame `onset` on. Model inputs/outputs pass through."""

    def __init__(self, actor, pull: float, onset: int):
        self.actor, self.pull, self.onset, self.i = actor, pull, onset, 0

    def act(self, frame=None, speed=0.0):
        curvature, accel = self.actor.act(frame, speed)
        self.i += 1
        if self.i > self.onset:
            curvature += self.pull / max(speed, 1.0)
        return curvature, accel

    @property
    def model_inputs(self):
        return self.actor.model_inputs

    @property
    def model_outputs(self):
        return self.actor.model_outputs


class CoachActor:
    """Drives with the student; the teacher watches the same frames and its (lat_accel, accel) rides along in model_outputs."""

    def __init__(self, actor, teacher):
        self.actor, self.teacher = actor, teacher
        self.teacher.reset()

    def act(self, frame=None, speed=0.0):
        self.teacher.act(frame, speed)
        return self.actor.act(frame, speed)

    @property
    def model_inputs(self):
        return self.actor.model_inputs

    @property
    def model_outputs(self):
        out = self.actor.model_outputs
        if out is None or self.teacher.model_outputs is None:
            return out
        return {**out, "teacher_action": self.teacher.model_outputs["action"][0, :2].float()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--server", default=SERVER_URL)
    p.add_argument("--segments", type=Path, required=True, help="file with one TRAIN segment id per line")
    p.add_argument("--model", type=Path, required=True, help="supervised Torch Supercombo state dict")
    p.add_argument("--on-policy", type=Path, help="resume from this on-policy state dict")
    p.add_argument("--start-step", type=int, default=0, help="step number of the resumed checkpoint")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--rollout-frames", type=int)
    p.add_argument("--sampling-steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/dagger"))
    p.add_argument("--expert", choices=("worldmodel", "teacher"), default="worldmodel", help="who labels the student's states")
    p.add_argument("--disturb-prob", type=float, default=0.0, help="fraction of rollouts with an injected steering pull")
    p.add_argument("--pull-min-dps", type=float, default=1.0)
    p.add_argument("--pull-max-dps", type=float, default=4.0)
    p.add_argument("--resume-every", type=int, default=5, help="write <out-dir>/resume.pt (head + optimizer + RNG) every N steps")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", args.gpu_id)
    rt = RuntimeClient(args.server, args.sampling_steps, args.cfg)
    max_frames = ROLLOUT_FRAMES - rt.history_frames - rt.future_frames
    rollout_frames = max_frames if args.rollout_frames is None else args.rollout_frames
    if not 2 <= rollout_frames <= max_frames:
        raise SystemExit(f"rollout frames must be in [2, {max_frames}]")

    segments = [s.strip() for s in args.segments.read_text().split() if s.strip()]
    episodes: dict[str, Episode] = {}

    actor = SupercomboActor.from_torch_checkpoint(args.model, on_policy=args.on_policy, device=device)
    supercombo = actor.model.model
    supercombo.requires_grad_(False).eval()
    model = supercombo.on_policy_temporal
    teacher = SupercomboActor.from_onnx() if args.expert == "teacher" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out_dir / "train_log.jsonl").open("a")

    def atomic_save(obj, path: Path) -> None:  # a kill mid-write never leaves a truncated checkpoint behind
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save(step: int) -> None:
        atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.out_dir / f"on_policy_step{step}.pt")

    # Exact resume: resume.pt holds the head, the AdamW state and both RNG states as of its step. If it is ahead of
    # --start-step (the run was killed between periodic checkpoints), continue from it; the end step stays the same.
    run_config = {"expert": args.expert, "disturb_prob": args.disturb_prob, "pull": (args.pull_min_dps, args.pull_max_dps),
                  "learning_rate": args.learning_rate, "segments": str(args.segments)}
    end_step = args.start_step + args.steps
    first_step = args.start_step + 1
    resume_path = args.out_dir / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["config"] != run_config:
            raise SystemExit(f"{resume_path} is from a different run config: {state['config']} vs {run_config}")
        if args.start_step < state["step"] < end_step:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.bit_generator.state = state["rng"]
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            first_step = state["step"] + 1
            print(f"resumed exactly from {resume_path} at step {state['step']}", flush=True)

    def save_resume(step: int) -> None:
        atomic_save({"step": step, "config": run_config, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all()},
                    resume_path)

    if first_step == 1:
        save(0)  # step 0 == the supervised student's own head, for the curve's origin

    for step in range(first_step, end_step + 1):
        t0 = time.perf_counter()
        seg = segments[int(rng.integers(len(segments)))]
        if seg not in episodes:
            episodes[seg] = Episode(rt, seg, gpu_id=args.gpu_id, encode_batch_size=ENCODE_BATCH_SIZE,
                                    cache_dir=Path("outputs/rl/cache"))
        episode = episodes[seg]
        actor.reset()
        last_start = len(episode.latents) - rt.future_frames - rollout_frames
        start_index = int(rng.integers(rt.history_frames, last_start + 1))
        end_index = start_index + rollout_frames
        pull_dps, onset = 0.0, 0
        if args.disturb_prob > 0 and rng.random() < args.disturb_prob:  # no extra draws in vanilla runs
            pull_dps = float(rng.uniform(args.pull_min_dps, args.pull_max_dps) * rng.choice([-1.0, 1.0]))
            onset = int(rng.integers(0, rollout_frames // 3 + 1))
        driver = CoachActor(actor, teacher) if teacher is not None else actor
        driver = PullActor(driver, np.deg2rad(pull_dps), onset) if pull_dps else driver
        result = rollout(Env(rt, episode, start_index * FRAME_SKIP, end_index * FRAME_SKIP), driver)

        keys = (ModelInputs.FEATURES, ModelInputs.DESIRE, ModelInputs.TRAFFIC, ModelInputs.ACTION_T)
        rollout_inputs: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
        target_actions: list[torch.Tensor] = []
        for model_input, model_output, plan in zip(
            result.model_inputs, result.model_outputs, result.worldmodel_outputs["plan"], strict=True
        ):
            if model_input is None or model_output is None or (teacher is not None and "teacher_action" not in model_output):
                continue
            rollout_inputs[ModelInputs.FEATURES].append(
                torch.cat((model_input[ModelInputs.FEATURES], model_output["hidden_state"][:, None]), dim=1)
            )
            for k in keys[1:]:
                rollout_inputs[k].append(model_input[k])
            if teacher is not None:
                target_actions.append(model_output["teacher_action"].cpu())
                continue
            plan = np.asarray(plan)
            action_t = model_input[ModelInputs.ACTION_T][0, -1]
            speed = max(float(plan[0, Plan.VELOCITY.start]), MIN_SPEED)
            curvature = get_curvature_from_plan(
                plan[:, Plan.T_FROM_CURRENT_EULER.start + 2], plan[:, Plan.ORIENTATION_RATE.start + 2],
                T_IDXS, speed, float(action_t[0]),
            )
            accel = get_accel_from_plan(
                plan[:, Plan.VELOCITY.start], plan[:, Plan.ACCELERATION.start], T_IDXS, action_t=float(action_t[1])
            )
            target = torch.tensor((curvature * speed**2, accel), dtype=torch.float32)
            if not torch.isfinite(target).all():
                raise ValueError(f"world-model plan produced a non-finite target: {target.tolist()}")
            target_actions.append(target)

        inputs = {k: torch.cat(v) for k, v in rollout_inputs.items()}
        targets = torch.stack(target_actions)
        model.requires_grad_(True).train()
        total = 0.0
        for b in range(0, len(targets), args.batch_size):
            batch = {k: v[b : b + args.batch_size].to(device) for k, v in inputs.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch[ModelInputs.FEATURES], batch[ModelInputs.DESIRE], batch[ModelInputs.TRAFFIC],
                         batch[ModelInputs.ACTION_T])["action"][:, -1, :2]
            loss = F.mse_loss(pred.float(), targets[b : b + args.batch_size].to(device))
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(batch[ModelInputs.FEATURES])
        model.requires_grad_(False).eval()

        mean_loss = total / len(targets)
        if step % args.save_every == 0:
            save(step)
        if step % args.resume_every == 0 or step % args.save_every == 0:
            save_resume(step)
        log.write(json.dumps({"step": step, "segment": seg, "start": start_index, "loss": mean_loss, "pull_dps": pull_dps,
                              "onset": onset, "seconds": time.perf_counter() - t0}) + "\n")
        log.flush()
        print(f"step {step}: loss={mean_loss:.6g} seg={seg[:8]} start={start_index} {time.perf_counter() - t0:.1f}s", flush=True)
    save(end_step)
    save_resume(end_step)


if __name__ == "__main__":
    main()
