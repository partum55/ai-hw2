#!/bin/bash
# WBF screening, round 2 (CPU only): higher merge IoU, model weights, more members. Run from repo root.
. .venv/bin/activate; export PYTHONPATH=src
Y=hyp_m_aug_masked_i1600_tta; L=l1280_i1600; F=tv_frcnn; R=tv_retina
run() { python src/ensemble.py --tag "$@" 2>&1 | grep -E '^ "(tag|AP|AP_no4k|AP50|AP_small)"' | tr -d '\n'; echo; }
run e2_a   --tags $Y,$L,$F,$R --weights 2,2,1,1 --iou 0.7
run e2_b   --tags $Y,$L,$F,$R --weights 2,2,1,1 --iou 0.8
run e2_c   --tags $Y,$L,$F,$R --weights 3,3,2,1 --iou 0.7
run e2_d   --tags $Y,$L,m1600_i1920,$F,$R --weights 2,2,2,1,1 --iou 0.7
run e2_e   --tags $Y,$L,$F,$R --weights 2,2,1,1 --iou 0.7 --conf-type absent_model_aware_avg
run e2_f   --tags $Y,$L,$F,$R --weights 2,2,1,1 --iou 0.7 --skip 0.01
