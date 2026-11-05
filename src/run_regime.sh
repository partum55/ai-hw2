#!/bin/bash
# Training-regime screening on the plain (unmasked) data, 6 epochs each, everything else = baseline (yolo11s @1280,
# batch 8, seed 0). Motivation: baseline val AP peaked at epoch 3 of 15 and then degraded while train loss kept falling,
# and Ultralytics' default close_mosaic=10 switched mosaic off after epoch 5. Optimizer is set explicitly because
# `auto` picks AdamW for short runs and SGD for long ones, which would confound the comparison.
# Run from repo root:  bash src/run_regime.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
D=$PWD/data/yolo/data_s3.yaml
run() { # name, extra args
  n=$1; shift
  yolo detect train model=yolo11s.pt data=$D imgsz=1280 epochs=6 batch=8 workers=8 seed=0 deterministic=True \
       project=$PWD/runs name=$n exist_ok=True close_mosaic=0 "$@"
  python src/eval_coco.py --weights runs/$n/weights/best.pt --imgsz 1280 --tag $n
}
run reg_sgd   optimizer=SGD   lr0=0.01
run reg_adamw optimizer=AdamW lr0=0.001 cos_lr=True
run reg_aug   optimizer=SGD   lr0=0.01 mixup=0.15 scale=0.9
