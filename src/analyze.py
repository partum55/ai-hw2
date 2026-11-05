"""CPU-only breakdown of a saved evaluation (runs/dets_<tag>.npy written by eval_coco.py): AP per val clip and per
frame resolution. Answers "is the val score dominated by a few clips / the 4K clips that the test set does not contain?".

Usage: python src/analyze.py --tag base_plain
"""
import argparse
import contextlib
import io
from collections import Counter

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="val")
a = ap.parse_args()

with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(str(DATA / a.split / "annotations_coco.json"))
D = np.load(f"runs/dets_{a.tag}.npy")
dets = D[:, [0, 3, 4, 5, 6, 2, 1]]  # -> [image_id, x, y, w, h, score, cat] as expected by loadRes
dt = gt.loadRes(dets) if len(dets) else None
imgs = pd.DataFrame(gt.imgs.values())
imgs["res"] = imgs.width.astype(str) + "x" + imgs.height.astype(str)


def AP(ids):
    """COCO AP (0.5:0.95), AP50, AP_small on a subset of frames, maxDets=500, iscrowd regions ignored."""
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds, E.params.maxDets = list(ids), [1, 10, 500]
        E.evaluate(); E.accumulate(); E.summarize()
    P = E.eval["precision"][:, :, :, 0, 2]
    return float(np.mean(P[P > -1])) if (P > -1).any() else float("nan"), E.stats[1], E.stats[3]


nobj = Counter(a_["image_id"] for a_ in gt.dataset["annotations"] if not a_["iscrowd"])
rows = []
for key in ("video", "res"):
    for g, sub in imgs.groupby(key):
        ap_, ap50, aps = AP(sub.id)
        n_obj = sum(nobj[i] for i in sub.id)
        rows.append(dict(by=key, group=g, frames=len(sub), AP=ap_, AP50=ap50, AP_small=aps,
                         res=sub.res.iloc[0] if key == "video" else "", objs_per_frame=n_obj / len(sub)))
sub4 = imgs[imgs.width < 3000]
rows.append(dict(by="no4k", group="width<3000", frames=len(sub4), **dict(zip(["AP", "AP50", "AP_small"], AP(sub4.id)))))
rows.append(dict(by="all", group="all", frames=len(imgs), **dict(zip(["AP", "AP50", "AP_small"], AP(imgs.id)))))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv(f"runs/analysis_{a.tag}.csv", index=False)
