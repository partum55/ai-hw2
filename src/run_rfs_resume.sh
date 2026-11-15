#!/bin/bash
# Resume the repeat-factor-sampling run `m_rfs` after the WSL restart (last.pt = end of epoch 3 of 4), then score it.
set -e
. .venv/bin/activate; export PYTHONPATH=src
yolo detect train resume model=$PWD/runs/m_rfs/weights/last.pt
for s in 1280 1600; do python src/eval_coco.py --weights runs/m_rfs/weights/best.pt --imgsz $s --bs 8 --tag m_rfs_i$s; done
