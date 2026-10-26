# lab: experiments on top of openpilot.distill

Code for testing the distilled student inside comma's world model, on a single RTX 4090. Only code lives here: no
comma1M data, results, renders or checkpoints.

## What is here

| folder | what |
|---|---|
| `setup/` | box setup: model download, segment selection, NVFP4 → bf16 patch for pre-Blackwell GPUs, smoke tests, `guard.sh` (memory watchdog) |
| `eval/` | experiments and their analysis scripts |
| `showcase/` | video rendering from saved world-model frames |
| `patches/` | `ada_nvfp4_dequant.patch` (also applied to `rl/server.py` in this fork) |

Changes to the repo itself: `supervised/train.py` gains resumable checkpoints (`SAVE_EVERY`), `INIT_WEIGHTS` (start a
new run from another run's weights), cosine LR decay (`LR_DECAY_START`, `LR_MIN_RATIO`) and weight snapshots
(`SNAPSHOT_EVERY`). `rl/server.py` dequantises the NVFP4 weights to bf16 on GPUs older than Blackwell.

## Experiments (`eval/`)

- **Steering-pull test** (`eval_shove.py`, `analyze_pull.py`): a constant yaw rate is added to whatever the driver
  commands, as a misalignment or crosswind would. Drivers compared: nobody steering, comma's model (the teacher), the
  distilled student, and DAgger-tuned students. Drift is measured against the human's logged path.
  `--anchor-frames` moves the real-future frames the world model is conditioned on (`analyze_anchor.py`).
- **Open-loop steering** (`eval_open_loop.py`, `analyze_open_loop.py`): on real frames, does the model's lateral
  acceleration track the human's? Correlation and gain with lag search, cluster bootstrap over drives, split by drives
  the student trained on vs never saw.
- **DAgger in the world model** (`train_dagger.py`): the student drives, and a coach labels the states it reaches.
  The coach is either the world model's own plan or comma's model. Optional DART-style pull injection, and exact
  resume.
- **Closed-loop drift** (`eval_closed_loop.py`, `measure_forgiveness.py`, `calibrate_forgiveness.py`): lane readout
  on generated frames. It is only valid for small deviations: it saturates near half a lane.

Runs go through `run_*.sh` queues: one GPU job at a time, a resumable rollout server, and `guard.sh` around everything.
They assume the box layout `W=/home/sra/sahil_distill` (repo at `$W/openpilot.distill`, these scripts at `$W/eval`).

Frame convention used throughout: x forward, y right, positive curvature = right turn.
