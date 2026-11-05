"""Build Ultralytics-ready lists + data.yaml (the provided layout images/<video>/N.jpg <-> labels/<video>/N.txt
already matches the YOLO convention, so no label conversion is needed).

- Split: official `train` (56 videos) / `val` (7 videos); no video appears in both, so it is a sequence-level split.
- Frame subsampling: keep every STRIDE-th frame of train (neighbouring frames are near-duplicates).
- Ignore regions are NOT in the YOLO labels; they are only used at evaluation (COCO iscrowd=1, see common.load_coco).

Usage: python src/prepare.py [--stride 3] [--out data/yolo]
"""
import argparse
from pathlib import Path

from common import CLASSES, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--out", default=str(DATA.parent / "yolo"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def frames(split, stride):
    keep = []
    for vid in sorted((DATA / split / "images").iterdir()):
        fr = sorted(vid.glob("*.jpg"))
        keep += fr[::stride]  # same phase for every video -> deterministic
    return keep


tr, va, va4 = frames("train", a.stride), frames("val", 1), frames("val", 4)  # val4: cheap per-epoch curves during training
(out / f"train_s{a.stride}.txt").write_text("\n".join(map(str, tr)))
(out / "val.txt").write_text("\n".join(map(str, va)))
(out / "val_s4.txt").write_text("\n".join(map(str, va4)))
(out / f"data_s{a.stride}.yaml").write_text(
    f"path: {out}\ntrain: train_s{a.stride}.txt\nval: val_s4.txt\nnames:\n"
    + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
print(f"train {len(tr)} frames (stride {a.stride}), val {len(va)} frames -> {out}")
