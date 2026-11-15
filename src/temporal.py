"""Temporal score smoothing on fused detections (bonus: video context). A box that has a same-class partner (IoU >= thr) in
the neighbouring frames keeps its score, one that flickers for a single frame is down-weighted; missed frames are not invented.
    s' = (1-a)*s + a*mean_over_neighbours(best matched score, 0 if no match)

val:  python src/temporal.py --tag ens_x --alpha .3 --out-tag ens_x_t   (reads runs/dets_<tag>.npy, scores on val)
test: python src/temporal.py --split test --csv in.csv --alpha .3 --out sub_t.csv  (Kaggle CSV in -> Kaggle CSV out)
"""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--tag", help="val: runs/dets_<tag>.npy")
ap.add_argument("--csv", help="test: Kaggle CSV to smooth")
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--thr", type=float, default=0.5, help="IoU for a neighbour match")
ap.add_argument("--k", type=int, default=1, help="neighbours on each side")
ap.add_argument("--out-tag", default="temporal")
ap.add_argument("--out", default="submission_t.csv")
ap.add_argument("--keep-unmatched", action="store_true", help="an unmatched neighbour counts as the box's own score (no flicker penalty, only matched boxes change)")
ap.add_argument("--max-det", type=int, default=300)
a = ap.parse_args()


def iou(A, B):  # A: n x 4 (x, y, w, h), B: m x 4
    ax2, ay2, bx2, by2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3], B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(A[:, :1], B[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(A[:, 1:2], B[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (A[:, 2:3] * A[:, 3:4] + (B[:, 2] * B[:, 3])[None] - inter + 1e-9)


def smooth(D, order):
    """D: rows (image_id, cat, score, x, y, w, h); order: {image_id: (clip, frame)}. Returns D with new scores."""
    per = {int(i): D[D[:, 0] == i] for i in np.unique(D[:, 0])}
    pos = {order[i]: i for i in per}  # (clip, frame) -> image_id
    out = []
    for i, d in per.items():
        clip, fr = order[i]
        acc, n = np.zeros(len(d)), 0
        for off in [o for k in range(1, a.k + 1) for o in (-k, k)]:
            j = pos.get((clip, fr + off))
            if j is None:
                continue
            n += 1
            e = per[j]
            M = iou(d[:, 3:7], e[:, 3:7]) * (d[:, 1][:, None] == e[:, 1][None]) * (1)
            M = np.where(M >= a.thr, e[:, 2][None], 0)
            m = M.max(1) if M.shape[1] else np.zeros(len(d))
            acc += np.where(m > 0, m, d[:, 2]) if a.keep_unmatched else m
        d = d.copy()
        if n:
            d[:, 2] = (1 - a.alpha) * d[:, 2] + a.alpha * acc / n
        out.append(d)
    return np.concatenate(out)


if a.split == "val":
    from pycocotools.coco import COCO
    from coco_utils import score
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(DATA / "val" / "annotations_coco.json"))
    order = {i: (m["video"], m["frame_id"]) for i, m in gt.imgs.items()}
    D = smooth(np.load(f"runs/dets_{a.tag}.npy"), order)
    dets = []
    for i in sorted(gt.getImgIds()):
        d = D[D[:, 0] == i]
        d = d[np.argsort(-d[:, 2])[: a.max_det]]
        dets += [dict(image_id=i, category_id=int(r[1]), score=float(r[2]), bbox=[float(v) for v in r[3:7]]) for r in d]
    score(gt, dets, sorted(gt.getImgIds()), a.out_tag, 0.0, src=a.tag, alpha=a.alpha, thr=a.thr, k=a.k)
else:
    test = pd.read_csv(DATA / "test_images.csv")
    order = {i: (r.clip, int(r.frame)) for i, r in enumerate(test.itertuples())}
    sub = pd.read_csv(a.csv).fillna("")
    rows = []
    for i, s in enumerate(sub.PredictionString):
        v = np.array(s.split(), float).reshape(-1, 6)
        rows.append(np.column_stack([np.full(len(v), i), v[:, 0] + 1, v[:, 1:2], v[:, 2:6]]))
    D = smooth(np.concatenate(rows), order)
    res = []
    for i in range(len(test)):
        d = D[D[:, 0] == i]
        d = d[np.argsort(-d[:, 2])[: a.max_det]]
        res.append((test.image_id[i], " ".join(f"{int(r[1]) - 1} {r[2]:.4f} {r[3]:.1f} {r[4]:.1f} {r[5]:.1f} {r[6]:.1f}" for r in d)))
    pd.DataFrame(res, columns=["image_id", "PredictionString"]).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
