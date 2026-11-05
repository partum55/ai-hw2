"""Make a train set where ignore regions are "don't care": paint them gray and drop labels that sit inside them.

Why: unlabeled crowds/parking lots inside ignore regions would otherwise be learned as background. The scorer does not
penalise detections there (COCO iscrowd=1), so we also should not teach the model to suppress them.
Rule: fill region with gray (114) on the image; drop a label if >50% of its box area is inside a region (~1% of boxes).
Frames without ignore regions are symlinked (no re-encode). Val is left untouched: it is scored with iscrowd (eval_coco.py).

Usage: python src/mask_ignore.py --stride 3      (writes DATA/../yolo_masked/{images,labels,train_sN.txt,data_sN.yaml})
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

from common import CLASSES, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--out", default=str(DATA.parent / "yolo_masked"))
a = ap.parse_args()
OUT = Path(a.out)


def regions(p):  # normalized cx cy w h lines -> list of (x0, y0, x1, y1) normalized
    if not p.exists():
        return []
    return [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            for cx, cy, w, h in (map(float, l.split()) for l in p.read_text().splitlines() if l.strip())]


def inside_frac(b, r):  # share of box b's area inside region r
    ix = max(0.0, min(b[2], r[2]) - max(b[0], r[0])); iy = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
    return ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)


def work(img):
    vid, name = img.parent.name, img.stem
    dst_img, dst_lab = OUT / "images" / vid / img.name, OUT / "labels" / vid / f"{name}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True); dst_lab.parent.mkdir(parents=True, exist_ok=True)
    regs = regions(DATA / "train" / "ignore" / vid / f"{name}.txt")
    src_lab = DATA / "train" / "labels" / vid / f"{name}.txt"
    lines, kept = (src_lab.read_text().splitlines() if src_lab.exists() else []), []
    for l in lines:
        c, cx, cy, w, h = l.split(); cx, cy, w, h = map(float, (cx, cy, w, h))
        b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if not any(inside_frac(b, r) > 0.5 for r in regs):
            kept.append(l)
    dst_lab.write_text("\n".join(kept))
    if dst_img.exists() or dst_img.is_symlink():
        dst_img.unlink()
    if regs:
        im = Image.open(img).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
        for x0, y0, x1, y1 in regs:
            d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=(114, 114, 114))
        im.save(dst_img, quality=95)
    else:
        dst_img.symlink_to(img)
    return len(lines) - len(kept), bool(regs)


if __name__ == "__main__":
    imgs = []
    for vid in sorted((DATA / "train" / "images").iterdir()):
        imgs += sorted(vid.glob("*.jpg"))[::a.stride]  # same subsampling as prepare.py
    with Pool() as p:
        res = p.map(work, imgs, chunksize=32)
    (OUT / f"train_s{a.stride}.txt").write_text("\n".join(str(OUT / "images" / i.parent.name / i.name) for i in imgs))
    (OUT / f"data_s{a.stride}.yaml").write_text(
        f"path: {OUT}\ntrain: train_s{a.stride}.txt\nval: {Path(a.out).parent / 'yolo' / 'val_s4.txt'}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(f"{len(imgs)} frames, {sum(r[1] for r in res)} masked, {sum(r[0] for r in res)} labels dropped -> {OUT}")
