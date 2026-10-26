#!/usr/bin/env bash
# Copy the lab code (never data, results, renders or checkpoints: comma1M-derived) into the fork and push.
#   bash ~/openpilot-distill-work/sync_to_github.sh "commit message"
# Box-side edits to repo files (supervised/, rl/) are pulled from the box's openpilot.distill checkout first.
set -euo pipefail
SRC=~/openpilot-distill-work
REPO=~/openpilot-distill-lab
MSG=${1:?commit message}

rs() { rsync -a --delete --exclude '__pycache__' --exclude '.DS_Store' --exclude '*.jsonl' --exclude '*.npz' \
         --exclude '*.mp4' --exclude '*.pt' --exclude '*.png' --exclude '*.txt' "$@"; }
mkdir -p $REPO/lab/patches
rs $SRC/eval/ $REPO/lab/eval/
rs $SRC/scripts/ $REPO/lab/setup/
rs $SRC/showcase/ $REPO/lab/showcase/
cp $SRC/ada_nvfp4_dequant.patch $REPO/lab/patches/
cp $SRC/sync_to_github.sh $REPO/lab/

# repo files edited on the box (train.py patches, server dequant): take the box's working tree, retrying ZeroTier drops
for f in supervised/train.py rl/server.py; do
  for i in 1 2 3 4 5; do scp -q -o ConnectTimeout=8 gpu:/home/sra/sahil_distill/openpilot.distill/$f $REPO/$f && break; sleep 5; done
done

cd $REPO
git add -A
# refuse to push anything that looks like data, a lab address or a token
if git diff --cached --name-only | grep -E '\.(jsonl|npz|mp4|pt|hevc|safetensors|onnx|txt)$'; then
  echo "data file staged: abort"; git reset -q; exit 1
fi
if git diff --cached -U0 -- . ':!lab/sync_to_github.sh' | grep -nE '^\+.*(192\.168\.|10\.27\.|gho_|ghp_|hf_[A-Za-z0-9]{20})'; then
  echo "private address or token in diff: abort"; git reset -q; exit 1
fi
git diff --cached --quiet && { echo "nothing to commit"; exit 0; }
git -c user.name=sahilapage -c user.email=sahilvapage@gmail.com commit -q -m "$MSG" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin HEAD
git log --oneline -1
