#!/bin/bash
# Round 3 (from the screening results): combine the winners and measure the ignore-mask effect on the same recipe.
#  1. inference-size check at 1600 for reg_aug / hyp_m (no retraining)
#  2. submission files from hyp_m at 1280 and 1600 (pick by AP_no4k, submit by hand)
#  3. reg_aug_masked: yolo11s + aug recipe on the ignore-masked train set  -> pair with reg_aug (plain data)
#  4. hyp_m_aug: yolo11m + aug recipe (mixup .15, scale .9)
# Run from repo root: bash src/run_next.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
python src/eval_coco.py --weights runs/reg_aug/weights/best.pt --imgsz 1600 --tag reg_aug_i1600
python src/eval_coco.py --weights runs/hyp_m/weights/best.pt   --imgsz 1600 --tag hyp_m_i1600
for s in 1280 1600; do
  python src/make_submission.py --weights runs/hyp_m/weights/best.pt --imgsz $s --conf 0.001 --out runs/sub_hyp_m_i$s.csv
done
AUG="optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True imgsz=1280 workers=8"
yolo detect train model=yolo11s.pt data=$PWD/data/yolo_masked/data_s3.yaml batch=8 $AUG project=$PWD/runs name=reg_aug_masked exist_ok=True
python src/eval_coco.py --weights runs/reg_aug_masked/weights/best.pt --imgsz 1280 --tag reg_aug_masked
yolo detect train model=yolo11m.pt data=$PWD/data/yolo/data_s3.yaml batch=4 $AUG project=$PWD/runs name=hyp_m_aug exist_ok=True
python src/eval_coco.py --weights runs/hyp_m_aug/weights/best.pt --imgsz 1280 --tag hyp_m_aug
python src/eval_coco.py --weights runs/hyp_m_aug/weights/best.pt --imgsz 1600 --tag hyp_m_aug_i1600
