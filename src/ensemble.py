"""Fuse the saved detections of several models with Weighted Boxes Fusion and score the result on val.

Works from runs/dets_<tag>.npy (image_id, cat, score, x, y, w, h in source pixels) written by eval_coco.py / train_tv.py,
so trying an ensemble costs no GPU time. Also used on test: the same fusion is applied to per-model test detections.

Usage: python src/ensemble.py --tags hyp_m_aug_masked_i1600_tta,tv_frcnn --weights 2,1 --iou 0.6 --tag ens_a
       python src/ensemble.py --split test --tags y_tta,l1600,frcnn,retina --weights 2,2,1,1 --iou 0.7 --tag final --out submission.csv
  (test mode reads runs/testdets_<tag>.npy from predict_test.py / train_tv.py --predict-test and writes the Kaggle CSV)
"""
import argparse
import contextlib
import io
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO

from coco_utils import score
from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True, help="comma-separated run tags (runs/dets_<tag>.npy)")
ap.add_argument("--weights", default="", help="comma-separated model weights (default all 1)")
ap.add_argument("--iou", type=float, default=0.6, help="WBF IoU threshold for merging boxes")
ap.add_argument("--skip", type=float, default=0.001, help="drop boxes below this score before fusion")
ap.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--out", default="submission.csv", help="test mode: Kaggle CSV to write")
ap.add_argument("--max-det", type=int, default=500, help="keep the top-N fused boxes per image (smaller file / scorer load)")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

tags = a.tags.split(",")
w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(tags)
if a.split == "val":
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(DATA / "val" / "annotations_coco.json"))
    img_ids = sorted(gt.getImgIds())
    size = {i: (gt.imgs[i]["width"], gt.imgs[i]["height"]) for i in img_ids}
    PREFIX = "dets"
else:  # test: image_id = row index of test_images.csv
    test = pd.read_csv(DATA / "test_images.csv")
    img_ids = list(range(len(test)))
    size = {i: (int(r.width), int(r.height)) for i, r in enumerate(test.itertuples())}
    PREFIX = "testdets"


def per_image(tag):
    D = np.load(f"runs/{PREFIX}_{tag}.npy")
    D = D[np.argsort(D[:, 0], kind="stable")]
    ids, start = np.unique(D[:, 0], return_index=True)
    return {int(i): c for i, c in zip(ids, np.split(D, start[1:]))}


MODELS = [per_image(t) for t in tags]


def fuse(iid):
    W, H = size[iid]
    bl, sl, ll = [], [], []
    for m in MODELS:
        d = m.get(iid)
        if d is None or len(d) == 0:
            bl.append(np.zeros((0, 4))); sl.append(np.zeros(0)); ll.append(np.zeros(0)); continue
        b = np.stack([d[:, 3], d[:, 4], d[:, 3] + d[:, 5], d[:, 4] + d[:, 6]], 1) / [W, H, W, H]
        bl.append(np.clip(b, 0, 1)); sl.append(d[:, 2]); ll.append(d[:, 1] - 1)
    if sum(len(s) for s in sl) == 0:
        return []
    b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=w, iou_thr=a.iou, skip_box_thr=a.skip, conf_type=a.conf_type)
    keep = np.argsort(-s)[: a.max_det]  # the Kaggle scorer keeps at most 500 detections per image
    b = b[keep] * [W, H, W, H]
    return [dict(image_id=iid, category_id=int(c) + 1, score=float(p), bbox=[x1, y1, x2 - x1, y2 - y1])
            for (x1, y1, x2, y2), p, c in zip(b, s[keep], l[keep])]


if __name__ == "__main__":
    with Pool(a.workers) as p:
        out = p.map(fuse, img_ids, chunksize=16)
    if a.split == "val":
        score(gt, [d for o in out for d in o], img_ids, a.tag, 0.0, tags=a.tags, weights=w, wbf_iou=a.iou,
              skip=a.skip, conf_type=a.conf_type)
    else:  # Kaggle format: "cls score x y w h ..." with 0-based class, top-left x/y in source pixels
        rows = [(test.image_id[i], " ".join(f"{d['category_id'] - 1} {d['score']:.4f} {d['bbox'][0]:.1f} {d['bbox'][1]:.1f} "
                                            f"{d['bbox'][2]:.1f} {d['bbox'][3]:.1f}" for d in o)) for i, o in zip(img_ids, out)]
        pd.DataFrame(rows, columns=["image_id", "PredictionString"]).to_csv(a.out, index=False)
        print(f"wrote {a.out}: {len(rows)} rows, {sum(len(o) for o in out) / len(out):.0f} dets/image")
