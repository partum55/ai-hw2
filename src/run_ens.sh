#!/bin/bash
# WBF ensemble screening on val from saved detections (CPU only). Run from repo root: bash src/run_ens.sh
. .venv/bin/activate; export PYTHONPATH=src
Y=hyp_m_aug_masked_i1600_tta   # best single YOLO (m, aug, mask, TTA)
run() { python src/ensemble.py --tag "$@" 2>&1 | grep -E '^ "(tag|AP|AP_no4k|AP50|AP_small)"' | tr -d '\n'; echo; }
run ens_self    --tags $Y --iou 0.6                                    # sanity: fusing one model with itself
run ens_y_f_60  --tags $Y,tv_frcnn --weights 2,1 --iou 0.6
run ens_y_f_70  --tags $Y,tv_frcnn --weights 2,1 --iou 0.7
run ens_y_f_55  --tags $Y,tv_frcnn --weights 2,1 --iou 0.55
run ens_y_f_11  --tags $Y,tv_frcnn --weights 1,1 --iou 0.6
run ens_yl_f    --tags $Y,l1280_i1600,tv_frcnn --weights 2,2,1 --iou 0.6
run ens_yl_f_r  --tags $Y,l1280_i1600,tv_frcnn,tv_retina --weights 2,2,1,1 --iou 0.6
run ens_yy      --tags $Y,l1280_i1600 --weights 1,1 --iou 0.6
