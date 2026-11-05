#!/bin/bash
# Hypothesis queue (runs after run_regime.sh). Run from repo root: bash src/run_hyp.sh
# H1: does a different inference size help? (weights fixed = reg_sgd; inference only, ~3-6 min each)
# H2: is the model the bottleneck? yolo11m instead of yolo11s, same recipe as reg_sgd (batch 4: 16 GB card, a GPU OOM crashes WSL)
set -e
. .venv/bin/activate; export PYTHONPATH=src
for s in 960 1600 1920; do
  python src/eval_coco.py --weights runs/reg_sgd/weights/best.pt --imgsz $s --tag reg_sgd_i$s
done
yolo detect train model=yolo11m.pt data=$PWD/data/yolo/data_s3.yaml imgsz=1280 epochs=6 batch=4 workers=8 seed=0 \
     deterministic=True project=$PWD/runs name=hyp_m exist_ok=True close_mosaic=0 optimizer=SGD lr0=0.01
python src/eval_coco.py --weights runs/hyp_m/weights/best.pt --imgsz 1280 --tag hyp_m
