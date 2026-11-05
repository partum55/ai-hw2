#!/bin/bash
# Round 5: what limits us now - train resolution or model size? Same recipe as hyp_m_aug_masked (masked data, SGD,
# mixup .15, scale .9, 6 ep, seed 0) with ONE change each. Batch 3 because 1600px x batch 4 would not fit 16 GB
# (a CUDA OOM crashes the WSL VM).
#   A: yolo11m trained at imgsz 1600     (vs trained at 1280)
#   B: yolo11l trained at imgsz 1280     (vs yolo11m)
# Run from repo root: bash src/run_next3.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
R="optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True workers=8 batch=3"
D=$PWD/data/yolo_masked/data_s3.yaml
yolo detect train model=yolo11m.pt data=$D imgsz=1600 $R project=$PWD/runs name=m1600 exist_ok=True
for s in 1600 1920; do python src/eval_coco.py --weights runs/m1600/weights/best.pt --imgsz $s --tag m1600_i$s; done
yolo detect train model=yolo11l.pt data=$D imgsz=1280 $R project=$PWD/runs name=l1280 exist_ok=True
for s in 1280 1600; do python src/eval_coco.py --weights runs/l1280/weights/best.pt --imgsz $s --tag l1280_i$s; done
