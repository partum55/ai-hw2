#!/bin/bash
# Pseudo-label (self-training) round. Idempotent like run_final2.sh. Run from repo root.
#  1. validation of the idea: train + our ensemble's pseudo-labels on the (unlabelled) val frames, scored on val with the real
#     labels; same recipe as the teammate's yolo_dfl3 run (.2611 @1600, no PL) -> the difference is the effect of PL.
#  2. final models: train+val + pseudo-labelled test frames (thr .4 of the best public ensemble sub_merge_sl.csv).
set -e
. .venv/bin/activate; export PYTHONPATH=src
R="optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True workers=8 dfl=3.0"
pred() { [ -f runs/testdets_$1.npy ] || python src/predict_test.py --tag "$@"; }
train() { # name, model, batch, imgsz, yaml
  [ -f runs/$1/done ] && return
  if [ -f runs/$1/weights/last.pt ]; then yolo detect train resume model=$PWD/runs/$1/weights/last.pt
  else yolo detect train model=$2.pt data=$5 batch=$3 imgsz=$4 $R project=$PWD/runs name=$1 exist_ok=True; fi
  touch runs/$1/done; }
# 1. val experiment
train plv_m yolo11m 4 1280 $PWD/data/yolo_pl_val_t0.4/data_pl.yaml
[ -f runs/eval_plv_m_i1600.json ] || python src/eval_coco.py --weights runs/plv_m/weights/best.pt --imgsz 1600 --bs 8 --tag plv_m_i1600
# 2. final models
[ -d data/yolo_pl_test ] || python src/pseudo_label.py --split test --src sub_merge_sl.csv --thr 0.4 --base data/yolo_masked_tv/trainval_s3.txt --out data/yolo_pl_test
Y=$PWD/data/yolo_pl_test/data_pl.yaml
train tv_pl_m yolo11m 4 1280 $Y
pred tv_pl_m_tta --weights runs/tv_pl_m/weights/last.pt --imgsz 1600 --tta
pred tv_pl_m_ml_tta --weights runs/tv_pl_m/weights/last.pt --imgsz 1600 --tta --multi-label
train tv_pl_l yolo11l 3 1280 $Y
pred tv_pl_l_i1600 --weights runs/tv_pl_l/weights/last.pt --imgsz 1600
pred tv_pl_l_ml --weights runs/tv_pl_l/weights/last.pt --imgsz 1600 --multi-label
train tv_pl_m16 yolo11m 4 1600 $Y
pred tv_pl_m16_i1920 --weights runs/tv_pl_m16/weights/last.pt --imgsz 1920
pred tv_pl_m16_ml --weights runs/tv_pl_m16/weights/last.pt --imgsz 1920 --multi-label
echo ALLDONE_PL
