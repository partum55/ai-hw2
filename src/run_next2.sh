#!/bin/bash
# Round 4: yolo11m + aug recipe on the ignore-MASKED train set (masking gave +0.014 AP at yolo11s, see notes #20).
# Run from repo root: bash src/run_next2.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
yolo detect train model=yolo11m.pt data=$PWD/data/yolo_masked/data_s3.yaml batch=4 optimizer=SGD lr0=0.01 mixup=0.15 \
     scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True imgsz=1280 workers=8 \
     project=$PWD/runs name=hyp_m_aug_masked exist_ok=True
for s in 1280 1600; do
  python src/eval_coco.py --weights runs/hyp_m_aug_masked/weights/best.pt --imgsz $s --tag hyp_m_aug_masked_i$s
done
