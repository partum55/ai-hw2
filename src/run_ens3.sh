#!/bin/bash
# WBF screening, round 3 (CPU only): which extra members still help? Run from repo root.
. .venv/bin/activate; export PYTHONPATH=src
Y=hyp_m_aug_masked_i1600_tta; L=l1280_i1600; M=m1600_i1920; F=tv_frcnn; R=tv_retina; C=tv_fcos
run() { python src/ensemble.py --tag "$@" 2>&1 | grep -E '^ "(tag|AP|AP_no4k|AP50|AP_small)"' | tr -d '\n'; echo; }
run e3_a --tags $Y,$L,$M,$F,$R,$C --weights 2,2,2,1,1,1 --iou 0.7                                   # round-2 best + FCOS
run e3_b --tags $Y,$L,$M,$F,$R,$C,hyp_m_i1600 --weights 2,2,2,1,1,1,1 --iou 0.7                    # + yolo11m (no aug/mask)
run e3_c --tags $Y,$L,$M,$F,$R,$C,hyp_m_i1600,hyp_m_aug_masked_i1280 --weights 2,2,2,1,1,1,1,1 --iou 0.7
run e3_d --tags $Y,$L,$M,$F,$R,$C,hyp_m_i1600,reg_aug_masked --weights 2,2,2,1,1,1,1,1 --iou 0.7   # + yolo11s
run e3_e --tags $Y,$L,$M,$F,$R,$C --weights 2,2,2,2,1,1 --iou 0.7                                   # FRCNN weight up
