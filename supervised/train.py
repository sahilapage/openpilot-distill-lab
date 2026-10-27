"""Train Supercombo on logged comma1M segments."""

import argparse
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb

from .dataset import Comma1MDataset, DatasetConfig, prefill_supercombo_batch
from .wm_dataset import WorldModelDataset
from .model import Supercombo
from .model_for_inference import InputQueues, ORTSupercomboForInference


ROOT = Path(__file__).resolve().parents[1]
TARGET_OUTPUTS = ("wide_from_device_euler", "pose", "plan", "action")
LOSS_WEIGHTS = {"gt/imgs": 100.0, "distill/lead": 0.05, "distill/hidden_state": 10.0}


def loss_fn(predictions, targets, teacher_targets, outputs=TARGET_OUTPUTS):
    predicted_images = predictions["imgs"][:, -1].float() / 127.5 - 1
    target_images = targets["imgs"].permute(0, 3, 1, 2).float() / 127.5 - 1
    target_losses = {"imgs": F.mse_loss(predicted_images, target_images)}
    target_losses |= {
        name: F.mse_loss(
            predictions[name][..., : targets[name].shape[-1]].float(),
            targets[name].float(),
        )
        for name in outputs
    }
    distill_losses = {
        name: F.mse_loss(
            (predictions["hidden_state"].mean(-2) if name == "hidden_state" else predictions[name]).float(),
            target.float(),
        )
        for name, target in teacher_targets.items()
    }
    target_loss = sum(LOSS_WEIGHTS.get(f"gt/{name}", 1.0) * value for name, value in target_losses.items())
    distill_loss = sum(LOSS_WEIGHTS.get(f"distill/{name}", 1.0) * value for name, value in distill_losses.items())
    return target_loss + distill_loss, {
        "gt": target_loss,
        "distill": distill_loss,
        **{f"gt/{name}": value for name, value in target_losses.items()},
        **{f"distill/{name}": value for name, value in distill_losses.items()},
    }


def _identity(value):
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT), help="comma1M dataset root")
    parser.add_argument(
        "--teacher",
        default=str(ROOT / "models/big_driving_supercombo.onnx"),
        help="teacher Supercombo ONNX model",
    )
    parser.add_argument("--steps", type=int, default=1, help="training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="samples per training batch")
    parser.add_argument("--workers", type=int, default=0, help="data-loader workers")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs/supervised/supercombo.pt",
        help="checkpoint output path",
    )
    args = parser.parse_args()

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    train_dataset = Comma1MDataset(
        DatasetConfig(args.dataset, batch_size=args.batch_size),
        local_rank=local_rank,
        global_rank=rank,
        global_world_size=world_size,
    )
    batch_iterator = iter(
        DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=args.workers,
            collate_fn=_identity,
            multiprocessing_context="spawn" if args.workers else None,
            persistent_workers=bool(args.workers),
        )
    )

    # WM_PATCH (DAgger v2): with WM_DATA set, every step also trains on a batch of world-model rollouts
    # (wm_dataset.py), labelled by the teacher online; only the image reconstruction has a ground truth there.
    wm_root = os.environ.get("WM_DATA")
    wm_weight = float(os.environ.get("WM_WEIGHT", 1.0))
    wm_iterator = (
        iter(
            DataLoader(
                WorldModelDataset(wm_root, batch_size=args.batch_size),
                batch_size=None,
                num_workers=int(os.environ.get("WM_WORKERS", 4)),
                collate_fn=_identity,
                multiprocessing_context="spawn",
                persistent_workers=True,
            )
        )
        if wm_root
        else None
    )

    raw_model = Supercombo().to(device)
    model = DDP(raw_model, device_ids=[local_rank]) if distributed else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
        fused=True,
    )

    if is_main:
        print(f"student: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M parameters")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    teacher = ORTSupercomboForInference.from_supercombo(
        args.teacher,
        providers=[("CUDAExecutionProvider", {"device_id": local_rank}), "CPUExecutionProvider"],
    )
    run = (
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "openpilot.distill.supervised"),
            dir=args.output.parent,
            config={**vars(args), "world_size": world_size, "loss_weights": LOSS_WEIGHTS},
        )
        if is_main
        else None
    )

    # RESUME_PATCH: periodic resumable checkpoint (model + optimizer + step), written atomically
    resume_path = args.output.with_suffix(".resume.pt")
    save_every = int(os.environ.get("SAVE_EVERY", 500))
    start_step = 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)  # own checkpoint; holds numpy scalars
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        if is_main:
            print(f"resumed from {resume_path} at step {start_step}")

    # INIT_PATCH: start a NEW run (no resume file yet) from another run's weights, fresh optimizer and LR schedule
    elif os.environ.get("INIT_WEIGHTS"):
        raw_model.load_state_dict(torch.load(os.environ["INIT_WEIGHTS"], map_location=device))
        if is_main:
            print(f"initialised weights from {os.environ['INIT_WEIGHTS']}")

    # LR_PATCH: optional cosine decay from LR_DECAY_START to the last step (down to LR_MIN_RATIO x base), fixed by env
    # vars so it is identical across resumes; and SNAPSHOT_EVERY weight snapshots (supercombo_step{N}.pt) for model selection.
    decay_start = int(os.environ.get("LR_DECAY_START", -1))
    min_ratio = float(os.environ.get("LR_MIN_RATIO", 0.1))
    snapshot_every = int(os.environ.get("SNAPSHOT_EVERY", 0))
    for step in range(start_step + 1, args.steps + 1):
        step_start = time.perf_counter()
        if decay_start >= 0:
            progress = min(max((step - decay_start) / max(args.steps - decay_start, 1), 0.0), 1.0)
            lr = float(args.learning_rate) * float(min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
        inputs, targets = next(batch_iterator)
        while not len(inputs["img"]):
            inputs, targets = next(batch_iterator)

        queues = InputQueues.from_model(teacher, batch_size=len(inputs["img"]))
        teacher_predictions = prefill_supercombo_batch(teacher, queues, inputs)
        teacher_targets = {name: torch.from_numpy(value).to(device) for name, value in teacher_predictions.items()}
        inputs = {name: torch.from_numpy(value).to(device) for name, value in inputs.items()}
        targets = {name: torch.from_numpy(value).to(device) for name, value in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(inputs)
            loss, loss_metrics = loss_fn(predictions, targets, teacher_targets)
        loss.backward()
        if wm_iterator is not None:
            wm_inputs, wm_targets = next(wm_iterator)
            wm_queues = InputQueues.from_model(teacher, batch_size=len(wm_inputs["img"]))
            wm_teacher = {k: torch.from_numpy(v).to(device) for k, v in prefill_supercombo_batch(teacher, wm_queues, wm_inputs).items()}
            wm_inputs = {k: torch.from_numpy(v).to(device) for k, v in wm_inputs.items()}
            wm_targets = {k: torch.from_numpy(v).to(device) for k, v in wm_targets.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                wm_loss, wm_metrics = loss_fn(model(wm_inputs), wm_targets, wm_teacher, outputs=())
            (wm_weight * wm_loss).backward()
            loss_metrics = {**loss_metrics, "wm": wm_loss.detach(), "wm/distill": wm_metrics["distill"].detach(),
                            **{f"wm/{k}": v.detach() for k, v in wm_metrics.items() if k.startswith("distill/")}}
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if is_main:
            torch.cuda.synchronize()
            metrics = {
                "loss": loss.item(),
                **{name: value.item() for name, value in loss_metrics.items()},
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "step_time": time.perf_counter() - step_start,
            }
            if step == 1 or step % 100 == 0:
                target_images = targets["imgs"][0].cpu().numpy()
                predicted_images = (
                    predictions["imgs"][0, -1]
                    .detach()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                metrics |= {
                    "imgs/gt_narrow": wandb.Image(np.ascontiguousarray(target_images[..., :3])),
                    "imgs/pred_narrow": wandb.Image(np.ascontiguousarray(predicted_images[..., :3])),
                    "imgs/gt_wide": wandb.Image(np.ascontiguousarray(target_images[..., 3:])),
                    "imgs/pred_wide": wandb.Image(np.ascontiguousarray(predicted_images[..., 3:])),
                }
            print(
                f"step {step}/{args.steps}: loss={metrics['loss']:.6g} "
                f"gt={metrics['gt']:.6g} distill={metrics['distill']:.6g} "
                + (f"wm={metrics['wm']:.6g} wm_distill={metrics['wm/distill']:.6g} " if "wm" in metrics else "")
                + f"time={metrics['step_time']:.2f}s"
            )
            run.log(metrics, step=step)
            if snapshot_every and step % snapshot_every == 0:
                snap = args.output.with_name(f"{args.output.stem}_step{step}.pt")
                torch.save({name: value.detach().cpu() for name, value in raw_model.state_dict().items()}, str(snap) + ".tmp")
                os.replace(str(snap) + ".tmp", snap)
            if step % save_every == 0:
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                           str(resume_path) + ".tmp")
                os.replace(str(resume_path) + ".tmp", resume_path)

    if is_main:
        torch.save(
            {name: value.detach().cpu() for name, value in raw_model.state_dict().items()},
            args.output,
        )
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
