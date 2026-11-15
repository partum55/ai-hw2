"""Repeat-factor sampling of the masked train list (LVIS, Gupta et al. 2019) to up-weight rare classes.

AP is a mean over classes and bus/motor/van/tricycle are our weakest, rarest classes. A frame is repeated
r_i = max over classes c in the frame of max(1, sqrt(t / f_c)), f_c = share of all *objects* that are class c (frame-level
frequency is useless here: bus is in 27 % of frames but is only 0.9 % of the boxes); fractional parts are rounded
stochastically with a fixed seed. Only the *list* changes (images are listed several times), nothing is copied.

Usage: python src/rfs.py --src data/yolo_masked --stride 3 --t 0.3
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir made by mask_ignore.py (has labels/ and train_sN.txt)")
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--t", type=float, default=0.05, help="object-share threshold: classes with fewer than t of all boxes are repeated")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
src = Path(a.src)
files = (src / f"train_s{a.stride}.txt").read_text().split()


def classes(f):
    lab = Path(f.replace("/images/", "/labels/")).with_suffix(".txt")
    return [int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []


cls_per = [classes(f) for f in files]
freq = Counter(c for s in cls_per for c in s)
f_c = {c: freq[c] / sum(freq.values()) for c in range(8)}
r_c = {c: max(1.0, np.sqrt(a.t / f_c[c])) for c in range(8)}
rng = np.random.default_rng(a.seed)
out = []
for f, s in zip(files, cls_per):
    r = max([1.0] + [r_c[c] for c in s])
    out += [f] * (int(r) + (rng.random() < r - int(r)))
(src / f"train_s{a.stride}_rfs.txt").write_text("\n".join(out))
txt = (src / f"data_s{a.stride}.yaml").read_text().replace(f"train_s{a.stride}.txt", f"train_s{a.stride}_rfs.txt")
(src / f"data_s{a.stride}_rfs.yaml").write_text(txt)
print(f"frames {len(files)} -> {len(out)} (x{len(out) / len(files):.2f})")
print("share of boxes per class:", {CLASSES[c]: round(f_c[c], 3) for c in range(8)})
print("repeat factor per class:", {CLASSES[c]: round(float(r_c[c]), 2) for c in range(8)})
after = Counter(c for f in out for c in classes(f))
print("boxes per class before -> after:", {CLASSES[c]: (freq[c], after[c]) for c in range(8)})
