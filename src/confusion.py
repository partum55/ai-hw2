"""Class confusion on val from saved detections: for every GT object take the best-IoU detection (IoU >= 0.5, score >= thr,
ANY class) and record its class. Rows = GT class, columns = predicted class (+ 'miss' when nothing overlaps).
Shows whether a weak class (bus) is lost to another class (truck/van) or simply not found. Failure-case analysis for the report.

Usage: python src/confusion.py --tag hyp_m_aug_masked_i1600_tta [--thr 0.3]
"""
import argparse

import numpy as np

from common import CLASSES, load_coco

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--thr", type=float, default=0.3)
a = ap.parse_args()

_, objs, _ = load_coco("val")
D = np.load(f"runs/dets_{a.tag}.npy")
D = D[D[:, 2] >= a.thr]
D = D[np.argsort(D[:, 0], kind="stable")]
ids, st = np.unique(D[:, 0], return_index=True)
dets = {int(i): c for i, c in zip(ids, np.split(D, st[1:]))}
M = np.zeros((8, 9), int)  # last column = miss
for iid, g in objs.groupby("image_id"):
    G = g[["x", "y", "w", "h"]].to_numpy(float); gc = g.cls.to_numpy(int)
    d = dets.get(int(iid))
    if d is None:
        for c in gc: M[c, 8] += 1
        continue
    P = d[:, 3:7]
    ix = np.clip(np.minimum(G[:, None, 0] + G[:, None, 2], P[None, :, 0] + P[None, :, 2]) - np.maximum(G[:, None, 0], P[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(G[:, None, 1] + G[:, None, 3], P[None, :, 1] + P[None, :, 3]) - np.maximum(G[:, None, 1], P[None, :, 1]), 0, None)
    inter = ix * iy
    iou = inter / (G[:, None, 2] * G[:, None, 3] + P[None, :, 2] * P[None, :, 3] - inter)
    j = iou.argmax(1); ok = iou[np.arange(len(G)), j] >= 0.5
    for c, k, o in zip(gc, j, ok):
        M[c, int(d[k, 1]) - 1 if o else 8] += 1
print(f"{a.tag}  (thr {a.thr}); % of GT objects of each class -> predicted class / miss")
print(f"{'GT \\ pred':10s}" + "".join(f"{n[:6]:>8s}" for n in CLASSES) + f"{'miss':>8s}{'n':>9s}")
for c in range(8):
    n = M[c].sum()
    print(f"{CLASSES[c]:10s}" + "".join(f"{100 * M[c, k] / n:8.1f}" for k in range(9)) + f"{n:9d}")
