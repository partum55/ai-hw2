#!/bin/bash
# Baseline pair: yolo11s @1280, 15 epochs, plain train vs ignore-masked train (same everything else).
# Per-epoch val uses every 4th val frame (cheap curves); final numbers come from eval_coco.py on all val frames.
# Run from repo root on the training box:  bash src/run_baselines.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
for cfg in "base_plain data/yolo/data_s3.yaml" "base_masked data/yolo_masked/data_s3.yaml"; do
  set -- $cfg
  yolo detect train model=yolo11s.pt data=$PWD/$2 imgsz=1280 epochs=15 batch=8 workers=8 seed=0 deterministic=True \
       project=$PWD/runs name=$1 exist_ok=True
  python src/eval_coco.py --weights runs/$1/weights/best.pt --imgsz 1280 --tag $1
done
