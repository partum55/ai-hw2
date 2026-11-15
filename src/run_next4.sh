#!/bin/bash
# Round 6 (GPU box was idle after round 5): (1) does test-time augmentation help the best YOLO? (2) the two remaining
# detector families from torchvision under the same protocol: RetinaNet (one-stage anchor-based), FCOS (anchor-free).
# Run from repo root: bash src/run_next4.sh
set -e
. .venv/bin/activate; export PYTHONPATH=src
python src/eval_coco.py --weights runs/hyp_m_aug_masked/weights/best.pt --imgsz 1600 --tta --bs 8 --tag hyp_m_aug_masked_i1600_tta
for m in retina fcos; do
  python src/train_tv.py --model $m --train-list data/yolo_masked/train_s3.txt --tag tv_$m --out runs
done
