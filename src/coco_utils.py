"""COCO-style scoring shared by every detector family (Ultralytics models in eval_coco.py, torchvision in train_tv.py).

score() = the project's single evaluation protocol: AP (0.5:0.95), AP50, AP75, AP_small/medium/large (COCO bins on
source-pixel area), AR@500, per-class AP and AP without the 4K clip; ignore regions are the iscrowd=1 GT boxes,
maxDets=500 because the Kaggle scorer keeps up to 500 detections per image.
"""
import json
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval

from common import CLASSES


def _mean(x):
    return float(np.mean(x[x > -1]))


def score(gt, dets, img_ids, tag, ms_per_frame, out_dir="runs", **meta):
    """dets: list of dict(image_id, category_id (1-based), score, bbox [x, y, w, h] in source pixels)."""
    out = Path(out_dir); out.mkdir(exist_ok=True)
    # raw detections for CPU-only analysis later (analyze.py): image_id, cat, score, x, y, w, h
    np.save(out / f"dets_{tag}.npy", np.array([[d["image_id"], d["category_id"], d["score"], *d["bbox"]] for d in dets], np.float32))
    dt = gt.loadRes(dets)
    E = COCOeval(gt, dt, "bbox")
    E.params.imgIds, E.params.maxDets = img_ids, [1, 10, 500]
    E.evaluate(); E.accumulate(); E.summarize()
    st, P = E.stats, E.eval["precision"][:, :, :, 0, 2]  # P: [iou, recall, class, area=all, maxDets=500]
    st[0] = _mean(P)  # stats[0] is -1 because COCOeval's default summary looks for maxDets=100
    # test has no 4K frames (max 1920x1080) and one 4K val clip is 34% of val frames -> also report AP without it
    E2 = COCOeval(gt, dt, "bbox")
    E2.params.imgIds, E2.params.maxDets = [i for i in img_ids if gt.imgs[i]["width"] < 3000], [1, 10, 500]
    E2.evaluate(); E2.accumulate()
    res = dict(tag=tag, **meta, frames=len(img_ids), AP=st[0], AP_no4k=_mean(E2.eval["precision"][:, :, :, 0, 2]),
               AP50=st[1], AP75=st[2], AP_small=st[3], AP_medium=st[4], AP_large=st[5], AR500=st[8],
               ms_per_frame=ms_per_frame, per_class_AP={n: _mean(P[:, :, k]) for k, n in enumerate(CLASSES)})
    print(json.dumps(res, indent=1))
    json.dump(res, open(out / f"eval_{tag}.json", "w"), indent=1)
    return res
