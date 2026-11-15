"""Save an Ultralytics model's detections for the TEST set as runs/testdets_<tag>.npy
(rows: test_images.csv row index, category 1-based, score, x, y, w, h in source pixels) so ensemble.py --split test can fuse
several models. Same inference settings as eval_coco.py (chunks of --bs frames, conf .001, up to 500 detections).

Usage: python src/predict_test.py --weights runs/x/weights/best.pt --imgsz 1600 --tta --tag y_tta
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import RTDETR, YOLO

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--iou", type=float, default=0.6)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--multi-label", action="store_true", help="NMS keeps several classes per box (as Ultralytics val does; predict() default keeps only the argmax class)")
ap.add_argument("--tta", action="store_true")
ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
ap.add_argument("--tag", required=True)
a = ap.parse_args()
if a.multi_label:  # predict() calls nms.non_max_suppression without multi_label; val() passes multi_label=True
    import functools
    from ultralytics.utils import nms
    nms.non_max_suppression = functools.partial(nms.non_max_suppression, multi_label=True)

t = pd.read_csv(DATA / "test_images.csv")
files = [str(DATA / f) for f in t.file_name]
model, rows = (RTDETR if a.arch == "rtdetr" else YOLO)(a.weights), []
for s in range(0, len(files), a.bs):  # chunks: passing every path at once -> CUDA OOM
    for r, i in zip(model.predict(files[s:s + a.bs], imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=500, batch=a.bs,
                                  augment=a.tta, verbose=False), range(s, s + a.bs)):
        xyxy, sc, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        rows += [[i, int(k) + 1, p, x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), p, k in zip(xyxy, sc, c)]
Path("runs").mkdir(exist_ok=True)
np.save(f"runs/testdets_{a.tag}.npy", np.array(rows, np.float32))
print("saved", len(rows), "detections for", len(files), "test frames ->", f"runs/testdets_{a.tag}.npy")
