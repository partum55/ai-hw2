#!/bin/bash
# Final members retrained on train+val (recipes frozen from the train-only experiments; last.pt is used because there is
# no held-out data left to pick a best checkpoint). Test detections only. Run from repo root: bash src/run_final.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
D=$PWD/data/yolo_masked_tv/data_trainval_s3.yaml
R="optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True workers=8 imgsz=1280"
# Y': yolo11m (the best single recipe)  -> TTA @1600
yolo detect train model=yolo11m.pt data=$D batch=4 $R project=$PWD/runs name=tv_y exist_ok=True
python src/predict_test.py --weights runs/tv_y/weights/last.pt --imgsz 1600 --tta --tag tv_y_tta
# L': yolo11l -> @1600
yolo detect train model=yolo11l.pt data=$D batch=3 $R project=$PWD/runs name=tv_l exist_ok=True
python src/predict_test.py --weights runs/tv_l/weights/last.pt --imgsz 1600 --tag tv_l_i1600
# torchvision members (Faster R-CNN is trained on Kaggle: notebooks/torchvision_trainval.ipynb)
for m in retina fcos; do
  python src/train_tv.py --model $m --train-list data/yolo_masked_tv/trainval_s3.txt --tag tvtv_$m --out runs --predict-test --no-val
done
# S': yolo11s -> @1280
yolo detect train model=yolo11s.pt data=$D batch=8 $R project=$PWD/runs name=tv_s exist_ok=True
python src/predict_test.py --weights runs/tv_s/weights/last.pt --imgsz 1280 --tag tv_s_i1280
