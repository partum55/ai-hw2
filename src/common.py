"""Shared helpers: dataset paths and COCO-json -> DataFrame loading."""
import json
import os
from pathlib import Path

import pandas as pd

# override with HW2_DATA=/path/to/raw when running on another machine
DATA = Path(os.environ.get("HW2_DATA", Path(__file__).resolve().parents[1] / "data" / "raw"))
CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]


def load_coco(split):
    """Return (images_df, objects_df, ignore_df) for 'train' or 'val'.

    Ignore regions are stored in the COCO json as iscrowd=1 boxes copied once per
    category (4 regions x 8 classes), so we dedupe them to one row per region.
    objects_df: real objects only (iscrowd=0), category 0..7 (0-based).
    """
    j = json.load(open(DATA / split / "annotations_coco.json"))
    imgs = pd.DataFrame(j["images"]).rename(columns={"id": "image_id"})
    ann = pd.DataFrame(j["annotations"])
    ann[["x", "y", "w", "h"]] = pd.DataFrame(ann.bbox.tolist(), index=ann.index)
    ann = ann.drop(columns=["bbox", "area"]).merge(
        imgs[["image_id", "video", "width", "height"]], on="image_id")
    objs = ann[ann.iscrowd == 0].copy()
    objs["cls"] = objs.category_id - 1
    ign = (ann[ann.iscrowd == 1]
           .drop_duplicates(["image_id", "x", "y", "w", "h"])
           [["image_id", "video", "x", "y", "w", "h", "width", "height"]])
    return imgs, objs, ign
