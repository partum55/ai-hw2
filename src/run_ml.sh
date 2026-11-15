#!/bin/bash
# Does multi-label NMS (several class hypotheses per box, as in Ultralytics val) raise AP on val? Best YOLO, no TTA, @1600.
set -e
. .venv/bin/activate; export PYTHONPATH=src
python src/eval_coco.py --weights runs/hyp_m_aug_masked/weights/best.pt --imgsz 1600 --bs 8 --multi-label --tag m_i1600_ml
python src/confusion.py --tag m_i1600_ml
