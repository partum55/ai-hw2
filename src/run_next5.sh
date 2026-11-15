#!/bin/bash
# Round 7: diagnostics + rare-class balancing for the "higher score" plan. Run from repo root: bash src/run_next5.sh
#  0. TEST detections of every ensemble member (so a real ensemble submission can be made early)
#  1. train+val masked set for the final model (CPU, used later)
#  2. best YOLO on the synthetic low-res val sets (test has 680x382 / 960x540 clips) at two inference sizes
#  3. NMS IoU 0.5 / 0.7 for the best YOLO (the default 0.6 was never tuned)
#  4. repeat-factor sampling of rare classes: yolo11m, same recipe as hyp_m_aug_masked, but 4 epochs over the 1.54x larger
#     list = the same number of iterations (6 epochs x 8 090 frames ~ 4 epochs x 12 454 frames)
set -e
. .venv/bin/activate; export PYTHONPATH=src
# 0. test detections (the torchvision models resume from last.pt at epoch 6, skip training and only run inference)
python src/predict_test.py --weights runs/hyp_m_aug_masked/weights/best.pt --imgsz 1600 --tta --tag y_tta
python src/predict_test.py --weights runs/l1280/weights/best.pt --imgsz 1600 --tag l_i1600
for m in retina frcnn fcos; do
  python src/train_tv.py --model $m --train-list data/yolo_masked/train_s3.txt --tag tv_$m --out runs --predict-test --no-val
done
python src/mask_ignore.py --stride 3 --splits train val --out data/yolo_masked_tv
Y=runs/hyp_m_aug_masked/weights/best.pt
for lw in 960 680; do for s in 1280 1600; do
  python src/eval_coco.py --weights $Y --imgsz $s --split val_lr$lw --bs 8 --tag m_lr${lw}_i$s
done; done
for iou in 0.5 0.7; do python src/eval_coco.py --weights $Y --imgsz 1600 --iou $iou --bs 8 --tag m_i1600_nms$iou; done
yolo detect train model=yolo11m.pt data=$PWD/data/yolo_masked/data_s3_rfs.yaml batch=4 optimizer=SGD lr0=0.01 mixup=0.15 \
     scale=0.9 close_mosaic=0 epochs=4 seed=0 deterministic=True imgsz=1280 workers=8 project=$PWD/runs name=m_rfs exist_ok=True
for s in 1280 1600; do python src/eval_coco.py --weights runs/m_rfs/weights/best.pt --imgsz $s --bs 8 --tag m_rfs_i$s; done
