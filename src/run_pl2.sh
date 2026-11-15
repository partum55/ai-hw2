#!/bin/bash
# Pseudo-label round, torchvision members (box). Same data as the YOLO PL models: train+val masked + test pseudo-labels (list
# data/yolo_pl_test/train_pl.txt). Faster R-CNN first (strongest non-YOLO member), then RetinaNet and FCOS. Idempotent (train_tv.py resumes).
set -e
. .venv/bin/activate; export PYTHONPATH=src
for m in frcnn retina fcos; do
  [ -f runs/testdets_tvpl_$m.npy ] || python src/train_tv.py --model $m --train-list data/yolo_pl_test/train_pl.txt --tag tvpl_$m --out runs --workers 8 --predict-test --no-val
done
echo ALLDONE_PL2
