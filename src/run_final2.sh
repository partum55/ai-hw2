#!/bin/bash
# Round 2 of the final train+val members: same frozen recipe as run_final.sh but with the DFL loss gain 3.0 (the only
# tuning change that helped on val in the A100 pack: yolo11m @1600 .2523 -> .2611), plus multi-label test detections for
# the members from run_final.sh. Test detections only (no held-out data left). Run from repo root.
# Idempotent: the WSL VM sometimes restarts and kills the queue; re-running skips finished steps and resumes a half-trained model.
set -e
. .venv/bin/activate; export PYTHONPATH=src
D=$PWD/data/yolo_masked_tv/data_trainval_s3.yaml
R="optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True workers=8 dfl=3.0"
pred() { # tag, then predict_test.py args
  [ -f runs/testdets_$1.npy ] || python src/predict_test.py --tag "$@"; }
train() { # name, model, batch, imgsz
  [ -f runs/$1/done ] && return
  if [ -f runs/$1/weights/last.pt ]; then
    yolo detect train resume model=$PWD/runs/$1/weights/last.pt
  else
    yolo detect train model=$2.pt data=$D batch=$3 imgsz=$4 $R project=$PWD/runs name=$1 exist_ok=True
  fi
  touch runs/$1/done; }
# cheap first: multi-label dets of the already trained members
pred tv_y_ml_tta --weights runs/tv_y/weights/last.pt --imgsz 1600 --tta --multi-label
pred tv_l_ml --weights runs/tv_l/weights/last.pt --imgsz 1600 --multi-label
pred tv_s_ml --weights runs/tv_s/weights/last.pt --imgsz 1280 --multi-label
# M16': yolo11m trained at 1600 (train-only twin m1600_dfl3 was one of the best val members) -> infer 1920
train tv_m16d yolo11m 4 1600
pred tv_m16d_i1920 --weights runs/tv_m16d/weights/last.pt --imgsz 1920
pred tv_m16d_ml --weights runs/tv_m16d/weights/last.pt --imgsz 1920 --multi-label
# S': yolo11s dfl3 @1280
train tv_sd yolo11s 8 1280
pred tv_sd_i1280 --weights runs/tv_sd/weights/last.pt --imgsz 1280
pred tv_sd_ml --weights runs/tv_sd/weights/last.pt --imgsz 1280 --multi-label
# L'': yolo11l dfl3 @1280 -> 1600 (own copy of the teammate's tv_l_dfl3, so the box alone can reproduce the ensemble)
train tv_ld yolo11l 3 1280
pred tv_ld_i1600 --weights runs/tv_ld/weights/last.pt --imgsz 1600
pred tv_ld_ml --weights runs/tv_ld/weights/last.pt --imgsz 1600 --multi-label
echo ALLDONE2
