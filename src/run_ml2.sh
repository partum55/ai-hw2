#!/bin/bash
# Multi-label NMS (+.015 AP on the best single YOLO) applied to every YOLO ensemble member: val detections (for choosing
# the ensemble) and test detections (for the submission), same tag in both namespaces. Run from repo root.
set -e
. .venv/bin/activate; export PYTHONPATH=src
both() { # tag weights imgsz [extra flags]
  t=$1; w=$2; s=$3; shift 3
  python src/eval_coco.py --weights $w --imgsz $s --bs 8 --multi-label --tag $t "$@"
  python src/predict_test.py --weights $w --imgsz $s --bs 8 --multi-label --tag $t "$@"
}
both y_ml_tta runs/hyp_m_aug_masked/weights/best.pt 1600 --tta
both l_ml     runs/l1280/weights/best.pt              1600
both m19_ml   runs/m1600/weights/best.pt              1920
both s_ml     runs/reg_aug_masked/weights/best.pt     1280
both mp_ml    runs/hyp_m/weights/best.pt              1600
