"""Train + evaluate torchvision detectors under the same protocol as the YOLO runs.

  frcnn  : Faster R-CNN R50-FPN v2   (two-stage, anchor-based)
  retina : RetinaNet  R50-FPN v2     (one-stage, anchor-based)
  fcos   : FCOS       R50-FPN        (one-stage, anchor-free)
All start from COCO-pretrained weights (new 8-class heads), train on the YOLO-format lists made by prepare.py /
mask_ignore.py (labels found by replacing /images/ -> /labels/), and are scored on all val frames with coco_utils.score.
Frames are pre-resized so the long side is --max-size (default 1280, like imgsz=1280 for YOLO); boxes are mapped back
to source pixels before scoring.

Usage: python src/train_tv.py --model frcnn --train-list data/yolo_masked/train_s3.txt --tag tv_frcnn
"""
import argparse
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import nn
from torchvision.models import detection as D
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import to_tensor

from coco_utils import score
from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["frcnn", "retina", "fcos"])
ap.add_argument("--train-list", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=0.005)
ap.add_argument("--min-size", type=int, default=720)
ap.add_argument("--max-size", type=int, default=1280)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default="runs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rpn-post-nms", type=int, default=1000, help="frcnn: proposals kept per image after RPN NMS")
ap.add_argument("--roi-batch", type=int, default=512, help="frcnn: RoIs sampled per image for the box head")
ap.add_argument("--viz", type=int, default=0, help="debug: save N augmented training samples with their boxes and exit")
ap.add_argument("--limit-train", type=int, default=0, help="debug: stop after N iterations")
ap.add_argument("--limit-val", type=int, default=0, help="debug: evaluate only the first N frames")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
NC = 8 + 1  # 8 classes + background (torchvision convention: label 0 = background)


class DS(torch.utils.data.Dataset):
    """Image -> (tensor, target, scale, id). Frames are pre-resized to the working size; hflip only when training."""

    def __init__(s, files, train, ids=None):
        s.files, s.train, s.ids = files, train, ids

    def __len__(s):
        return len(s.files)

    def __getitem__(s, i):
        im = Image.open(s.files[i]).convert("RGB"); W, H = im.size
        sc = min(a.min_size / min(W, H), a.max_size / max(W, H))
        im = im.resize((round(W * sc), round(H * sc)), Image.BILINEAR)
        tgt = {}
        if s.train:
            lab = Path(s.files[i].replace("/images/", "/labels/")).with_suffix(".txt")
            r = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
            w, h = im.size
            cx, cy, bw, bh = (r[:, 1:5] * np.array([w, h, w, h])).T
            b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).reshape(-1, 4)
            keep = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
            b, c = b[keep], r[keep, 0].astype(np.int64) + 1
            x = to_tensor(im)
            if torch.rand(1).item() < 0.5:  # horizontal flip (torch RNG is re-seeded per dataloader worker, numpy is not)
                x = x.flip(-1); b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
            tgt = dict(boxes=torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4), labels=torch.as_tensor(c))
            return x, tgt, sc, -1
        return to_tensor(im), tgt, sc, s.ids[i]


def build():
    """COCO-pretrained detector with a fresh 8-class head; inference thresholds set for mAP (low score, up to 500 dets)."""
    size = dict(min_size=(a.min_size - 80, a.min_size - 40, a.min_size + 40, a.min_size + 80, a.min_size),  # multi-scale train; eval uses the last
                max_size=int(a.max_size * 1.12))
    if a.model == "frcnn":
        m = D.fasterrcnn_resnet50_fpn_v2(weights=D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=0.001,
                                         box_detections_per_img=500, rpn_post_nms_top_n_train=a.rpn_post_nms,
                                         rpn_post_nms_top_n_test=a.rpn_post_nms, box_batch_size_per_image=a.roi_batch, **size)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NC)
    elif a.model == "retina":
        m = D.retinanet_resnet50_fpn_v2(weights=D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, score_thresh=0.001,
                                        detections_per_img=500, **size)
        m.head.classification_head = RetinaNetClassificationHead(
            256, m.head.classification_head.num_anchors, NC, norm_layer=functools.partial(nn.GroupNorm, 32))
    else:
        m = D.fcos_resnet50_fpn(weights=D.FCOS_ResNet50_FPN_Weights.COCO_V1, score_thresh=0.001,
                                detections_per_img=500, **size)
        m.head.classification_head = FCOSClassificationHead(256, m.head.classification_head.num_anchors, NC)
    return m.to(dev)


files = Path(a.train_list).read_text().split()
loader = torch.utils.data.DataLoader(DS(files, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     collate_fn=lambda b: tuple(zip(*b)), persistent_workers=a.workers > 0)
if a.viz:  # sanity check of labels after resize + flip
    from PIL import ImageDraw
    ds = DS(files, True)
    for k in range(a.viz):
        x, t, _, _ = ds[k * 997 % len(ds)]
        im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype("uint8")); d = ImageDraw.Draw(im)
        for b in t["boxes"].tolist():
            d.rectangle(b, outline=(0, 255, 0), width=2)
        im.save(f"{a.out}/viz_{k}.jpg"); print("saved", k, len(t["boxes"]), "boxes", im.size)
    raise SystemExit
model = build()
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=a.lr, momentum=0.9, weight_decay=1e-4)
total = a.epochs * len(loader)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1, (i + 1) / 500) * (0.5 * (1 + math.cos(math.pi * min(i, total) / total))))  # warmup + cosine
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
out = Path(a.out) / a.tag; out.mkdir(parents=True, exist_ok=True)
start, log = 0, []
if (out / "last.pt").exists():  # resume after an interrupted session
    ck = torch.load(out / "last.pt", map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, log = ck["epoch"], ck["log"]
    print("resumed from epoch", start)

it = 0
for ep in range(start, a.epochs):
    model.train(); t0, run = time.time(), []
    for x, tgt, _, _ in loader:
        x = [i.to(dev) for i in x]; tgt = [{k: v.to(dev) for k, v in t.items()} for t in tgt]
        with torch.autocast(dev, enabled=dev == "cuda"):
            losses = model(x, tgt)
            loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # RetinaNet/FCOS can spike early with AMP
        scaler.step(opt); scaler.update(); sched.step()
        run.append({k: v.item() for k, v in losses.items()}); it += 1
        if it % 100 == 0:
            print(f"ep {ep + 1}/{a.epochs} it {it} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.5f} "
                  f"{(time.time() - t0) / len(run):.2f}s/it", flush=True)
        if a.limit_train and it >= a.limit_train:
            break
    log.append(dict(epoch=ep + 1, minutes=(time.time() - t0) / 60, **{k: float(np.mean([r[k] for r in run])) for k in run[0]}))
    print(log[-1], flush=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, log=log), out / "last.pt")
    if a.limit_train and it >= a.limit_train:
        break
json.dump(log, open(out / "train_log.json", "w"), indent=1)

# ---- evaluation on val with the project protocol -------------------------------------------------------------
gt = COCO(str(DATA / "val" / "annotations_coco.json"))
ids = sorted(gt.getImgIds())[: a.limit_val or None]
vl = torch.utils.data.DataLoader(DS([str(DATA / "val" / gt.imgs[i]["file_name"]) for i in ids], False, ids), batch_size=4,
                                 num_workers=a.workers, collate_fn=lambda b: tuple(zip(*b)))
model.eval(); dets, t0 = [], time.time()
with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
    for x, _, scs, iids in vl:
        for o, sc, iid in zip(model([i.to(dev) for i in x]), scs, iids):
            for (x1, y1, x2, y2), p, c in zip((o["boxes"] / sc).tolist(), o["scores"].tolist(), o["labels"].tolist()):
                dets.append(dict(image_id=iid, category_id=c, score=p, bbox=[x1, y1, x2 - x1, y2 - y1]))
ms = (time.time() - t0) / len(ids) * 1000
score(gt, dets, ids, a.tag, ms, out_dir=a.out, **{k: v for k, v in vars(a).items() if k not in ("train_list", "out", "tag")})
