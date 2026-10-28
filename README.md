# 🌰 openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, distill the openpilot driving model into a new model and make it drive your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

You don't have a comma four, no problem! You can still download videos and labels from https://huggingface.co/datasets/commaai/comma1M and train your own driving models.

## Step 1: Run a simple localization pipeline

Run the localizer.

<img width="512" height="256" alt="2333e255f1de1fe83ea5975b24a3cc67_frame_700" src="https://github.com/user-attachments/assets/395683ff-7d96-4e07-bf6e-12a3969b487f" />

## Step 2: Distill the model

Train a smaller driving model using the localized targets, and distillation targets from the openpilot model.

<img width="1366" height="777" alt="Screenshot from 2026-08-31 17-28-31" src="https://github.com/user-attachments/assets/d53a1e0e-3b79-4923-a777-9368c34234ed" />

https://api.wandb.ai/links/yassiney/qme6ln4z

## Step 3: Fine tune the driving model in the simulator

DAgger style post-training in a learned world model as a simulator.

https://github.com/user-attachments/assets/45ba3950-9a90-45c2-af15-f7e55f87785e

Have fun!

## This fork

Built on [openpilot.distill](https://github.com/YassineYousfi/openpilot.distill). Same three steps, plus:

### Runs on a 4090

The world model ships NVFP4 weights, which need a Blackwell GPU. On older GPUs `rl/server.py` dequantizes them to bf16 once at load. Nothing else changes.

### Check your student on roads it never trained on

Every model watches the same real, logged drive and steers. The human's steering comes from the localizer. Steering gain 1.0 = the model turns the wheel as much as the human did. Drives in the training split and unseen drives (the dataset's validation split) are reported separately, so memorisation shows up.

```
python -m rl.server &
python -m eval.open_loop run --segments eval.txt --out open_loop.jsonl \
    --policy teacher=teacher --policy student=student:outputs/supervised/supercombo.pt
python -m eval.open_loop summary open_loop.jsonl
```

And watch it: every model's steering drawn on the real camera, the human as white dots.

```
python -m eval.realroad_video video.mp4 open_loop.jsonl SEGMENT:START:STEPS \
    --policy teacher="comma's model" --policy student="my student"
```

### DAgger with the whole network

A student trained on logged drives only ever sees the car where a human put it. So let it drive inside the world model, push it off course with a random steering pull, and save what it sees. The teacher labels those frames online, and every training step mixes a logged batch with a world-model batch.

```
python -m rl.server &
python -m rl.collect --segments train.txt --out-dir outputs/wm_data/iter1 --rollouts 600 \
    --driver student=student:outputs/supervised/supercombo.pt --driver teacher=teacher --driver-weights 0.5,0.5
WM_DATA=outputs/wm_data INIT_WEIGHTS=outputs/supervised/supercombo.pt \
    torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 6000 --learning-rate 1e-4 \
    -o outputs/dagger/iter1/supercombo.pt
```

Repeat with the new student as the driver. Rollouts also save the world model's own plan, which is anchored to where the human really drove; `WM_PLAN_LABELS=1` steers by that plan instead of the teacher on generated frames. `supervised/train.py` also takes `INIT_WEIGHTS` (start from a checkpoint), `SAVE_EVERY` (resumable checkpoints), `LR_DECAY_START` / `LR_MIN_RATIO` (cosine decay) and `SNAPSHOT_EVERY` (weight snapshots).

On real drives, `AUGMENT=1` adds photometric augmentation (exposure, gamma, contrast, white balance, saturation, blur, sensor noise; one look per clip). The teacher labels the same augmented frames online, so its labels stay consistent with what the student sees. `FRAME_JITTER=40` lets a training clip start on any frame instead of about 25 fixed points per segment. `YAW_AUG=4` gives half the real samples a random heading error of up to 4° (an exact view rotation); the human-path targets rotate with it and the teacher labels the rotated frames. All three are off by default.
