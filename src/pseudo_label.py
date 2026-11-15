"""Pseudo-labels from our own ensemble predictions on UNLABELED frames, written as a YOLO dataset that is concatenated with
the real training list (self-training / transductive domain adaptation). No test labels are involved: only the test images and
the detections we predicted for them.

val  (validation of the idea, uses no val labels for training):
    python src/pseudo_label.py --split val --src ens_x --thr .45 --base data/yolo_masked/train_s3.txt --out data/yolo_pl_val
    -> train on train + pseudo-labelled val frames, evaluate on val with the real labels (eval_coco.py)
test (final):
    python src/pseudo_label.py --split test --src sub_merge_sl.csv --thr .45 --base data/yolo_masked_tv/trainval_s3.txt --out data/yolo_pl_test
For split=val it also prints precision/recall of the kept boxes against the real val labels (diagnostic for choosing --thr).
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from common import CLASSES, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["val", "test"])
ap.add_argument("--src", required=True, help="val: runs/dets_<tag>.npy tag; test: Kaggle CSV")
ap.add_argument("--thr", type=float, default=0.45, help="keep boxes with score >= thr")
ap.add_argument("--stride", type=int, default=1, help="use every N-th frame of each clip")
ap.add_argument("--base", required=True, help="real training list to concatenate")
ap.add_argument("--out", required=True)
a = ap.parse_args()
OUT = Path(a.out).resolve()

if a.split == "val":
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(DATA / "val" / "annotations_coco.json"))
    D = np.load(f"runs/dets_{a.src}.npy")
    frames = {i: (DATA / "val" / m["file_name"], m["width"], m["height"], m["video"], m["frame_id"]) for i, m in gt.imgs.items()}
    dets = {i: D[D[:, 0] == i] for i in frames}
else:
    test = pd.read_csv(DATA / "test_images.csv")
    frames = {i: (DATA / r.file_name, r.width, r.height, r.clip, int(r.frame)) for i, r in enumerate(test.itertuples())}
    sub = pd.read_csv(a.src).fillna("").set_index("image_id").PredictionString
    dets = {}
    for i, r in enumerate(test.itertuples()):
        v = np.array(sub[r.image_id].split(), float).reshape(-1, 6)
        dets[i] = np.column_stack([np.full(len(v), i), v[:, 0] + 1, v[:, 1], v[:, 2:6]])  # image_id, cat(1-based), score, x y w h

lst, kept_n, tp, n_gt = [], 0, 0, 0
for i, (img, W, H, clip, fr) in frames.items():
    if (fr - 1) % a.stride:
        continue
    d = dets[i][dets[i][:, 2] >= a.thr]
    (OUT / "images" / clip).mkdir(parents=True, exist_ok=True); (OUT / "labels" / clip).mkdir(parents=True, exist_ok=True)
    dst = OUT / "images" / clip / img.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(img.resolve())
    (OUT / "labels" / clip / f"{img.stem}.txt").write_text("\n".join(
        f"{int(r[1]) - 1} {(r[3] + r[5] / 2) / W:.6f} {(r[4] + r[6] / 2) / H:.6f} {r[5] / W:.6f} {r[6] / H:.6f}" for r in d))
    lst.append(str(dst)); kept_n += len(d)
    if a.split == "val":  # diagnostic: greedy IoU>=.5 same-class match against the real labels
        g = [x for x in gt.imgToAnns[i] if not x["iscrowd"]]
        n_gt += len(g); used = set()
        for r in d[np.argsort(-d[:, 2])]:
            best, bj = 0.5, None
            for j, x in enumerate(g):
                if j in used or x["category_id"] != int(r[1]):
                    continue
                gx, gy, gw, gh = x["bbox"]
                iw = min(r[3] + r[5], gx + gw) - max(r[3], gx); ih = min(r[4] + r[6], gy + gh) - max(r[4], gy)
                if iw > 0 and ih > 0:
                    iou = iw * ih / (r[5] * r[6] + gw * gh - iw * ih)
                    if iou >= best:
                        best, bj = iou, j
            if bj is not None:
                used.add(bj); tp += 1
base = Path(a.base).read_text().split()
(OUT / "train_pl.txt").write_text("\n".join(base + lst))
from_yaml = Path(a.base).parent / next(p for p in ["data_trainval_s3.yaml", "data_s3.yaml"] if (Path(a.base).parent / p).exists())
val_line = [l for l in from_yaml.read_text().splitlines() if l.startswith("val:")][0]
(OUT / "data_pl.yaml").write_text(f"path: {OUT}\ntrain: train_pl.txt\n{val_line}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
msg = f"{len(lst)} pseudo-labelled frames, {kept_n} boxes ({kept_n / max(len(lst), 1):.1f}/frame), base {len(base)} frames -> {OUT}"
if a.split == "val":
    msg += f" | precision {tp / max(kept_n, 1):.3f} recall {tp / max(n_gt, 1):.3f} (vs real val labels, diagnostic only)"
print(msg)
