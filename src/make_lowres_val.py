"""Synthetic low-resolution val set: the test set has 680x382 and 960x540 clips (~16 % of frames) that look nothing like
train/val (>=1344 px wide). Downscale val frames to a given width and rescale the COCO GT, so `eval_coco.py --split val_lr<W>`
measures how a model copes. Writes DATA/val_lr<W>/{images/<clip>/*.jpg, annotations_coco.json}.

Usage: python src/make_lowres_val.py --width 960
"""
import argparse
import json
from multiprocessing import Pool

from PIL import Image

from common import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, required=True)
a = ap.parse_args()
j = json.load(open(DATA / "val" / "annotations_coco.json"))
out = DATA / f"val_lr{a.width}"
scale = {i["id"]: a.width / i["width"] for i in j["images"]}


def work(im):
    dst = out / im["file_name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Image.open(DATA / "val" / im["file_name"]).convert("RGB")
    h = round(src.height * a.width / src.width)
    src.resize((a.width, h), Image.BILINEAR).save(dst, quality=92)
    return im["id"], h


if __name__ == "__main__":
    with Pool() as p:
        heights = dict(p.map(work, j["images"], chunksize=16))
    for im in j["images"]:
        im["width"], im["height"] = a.width, heights[im["id"]]
    for an in j["annotations"]:
        s = scale[an["image_id"]]
        an["bbox"] = [v * s for v in an["bbox"]]
        an["area"] = an["area"] * s * s
    json.dump(j, open(out / "annotations_coco.json", "w"))
    print("wrote", out, len(j["images"]), "frames")
