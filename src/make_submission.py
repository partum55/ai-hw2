"""Predict the test set with an Ultralytics model and write a Kaggle submission.csv.

Format (see Kaggle Data tab): one row per test image, `image_id,PredictionString`, where PredictionString is
"cls score x y w h cls score x y w h ..." with 0-based class ids and (x, y) the top-left corner in ORIGINAL pixels.
At most 500 detections per image are scored.

Usage:  python src/make_submission.py --weights runs/x/weights/best.pt --imgsz 1280 --out submission.csv
Then:   kaggle competitions submit -c multiple-object-video-detection-hw-2-ucu-ai-2026 -f submission.csv -m "msg"
"""
import argparse

import pandas as pd
from ultralytics import YOLO

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.001)  # low threshold: mAP is computed over the whole PR curve
ap.add_argument("--iou", type=float, default=0.6)     # NMS IoU
ap.add_argument("--out", default="submission.csv")
a = ap.parse_args()

test = pd.read_csv(DATA / "test_images.csv")
model = YOLO(a.weights)
rows = []
B = 16  # chunks: passing the whole list makes Ultralytics batch everything at once -> CUDA OOM (crashes WSL)
files = [str(DATA / f) for f in test.file_name]
for i in range(0, len(files), B):
    res = model.predict(files[i:i + B], imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=500, batch=B, verbose=False)
    for r, image_id in zip(res, test.image_id[i:i + B]):
        xyxy, s, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        rows.append((image_id, " ".join(
            f"{int(k)} {p:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}" for (x1, y1, x2, y2), p, k in zip(xyxy, s, c))))
    if (i // B) % 20 == 0:
        print(f"{len(rows)}/{len(test)}", flush=True)

sub = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
assert len(sub) == len(test) and (sub.image_id == test.image_id).all()
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} rows, {sub.PredictionString.str.split().str.len().div(6).mean():.1f} dets/image")
