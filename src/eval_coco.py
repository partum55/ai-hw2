"""COCO-style evaluation of an Ultralytics model on `val` that honours ignore regions (iscrowd=1 in the GT json).

Ultralytics' own val does not know about ignore regions, so false positives inside them are counted against the model;
the Kaggle scorer ignores them. This script is the project's single evaluation protocol for every experiment:
AP (0.5:0.95), AP50, AP75, AP_small/medium/large (COCO bins on source-pixel area), AR@500, per-class AP, speed.

Usage: python src/eval_coco.py --weights runs/x/weights/best.pt --imgsz 1280 --tag smoke
"""
import argparse
import time

from pycocotools.coco import COCO
from ultralytics import RTDETR, YOLO

from coco_utils import score
from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--iou", type=float, default=0.6)     # NMS IoU
ap.add_argument("--split", default="val")
ap.add_argument("--tag", default="eval")
ap.add_argument("--bs", type=int, default=16, help="inference chunk size; lower it for heavy models / small GPUs")
ap.add_argument("--arch", default="yolo", choices=["yolo", "rtdetr"])
ap.add_argument("--limit", type=int, default=0, help="debug: evaluate only the first N frames")
a = ap.parse_args()

gt = COCO(str(DATA / a.split / "annotations_coco.json"))
img_ids = sorted(gt.getImgIds())[: a.limit or None]
files = [str(DATA / a.split / gt.imgs[i]["file_name"]) for i in img_ids]
model, B, dets = (RTDETR if a.arch == "rtdetr" else YOLO)(a.weights), a.bs, []  # chunks of 16: passing every path at once -> CUDA OOM

t0 = time.time()
for s in range(0, len(files), B):
    for r, iid in zip(model.predict(files[s:s + B], imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=500, batch=B,
                                    verbose=False), img_ids[s:s + B]):
        xyxy, sc, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        dets += [dict(image_id=iid, category_id=int(k) + 1, score=float(p),
                      bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                 for (x1, y1, x2, y2), p, k in zip(xyxy, sc, c)]
ms = (time.time() - t0) / len(files) * 1000  # end-to-end per frame (decode + preprocess + model + NMS)
score(gt, dets, img_ids, a.tag, ms, weights=a.weights, imgsz=a.imgsz, conf=a.conf, nms_iou=a.iou)
