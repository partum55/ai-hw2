# %% [markdown]
# # 01 EDA: class balance, object sizes, density, per-sequence variation
# Run: `python src/eda.py` (works as a plain script; `# %%` cells also open as a notebook in VS Code / Jupytext).
# Figures go to `images/`, summary tables are printed.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

from common import CLASSES, DATA, load_coco

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)
D = {s: load_coco(s) for s in ("train", "val")}
for s, (imgs, objs, ign) in D.items():
    objs["size"] = np.sqrt(objs.w * objs.h)  # sqrt(area) in source pixels
    print(f"{s}: {len(imgs)} frames, {imgs.video.nunique()} videos, {len(objs)} objects, {len(ign)} ignore regions")

# %% [markdown]
# ## Class balance

# %%
cls = pd.DataFrame({s: D[s][1].cls.value_counts().sort_index() for s in D}).set_axis(CLASSES)
print((cls / cls.sum()).round(3))
cls.plot.bar(logy=True, figsize=(7, 3.5), title="objects per class (log)")
plt.tight_layout(); plt.savefig(OUT / "class_dist.png", dpi=120); plt.close()

# %% [markdown]
# ## Object size (sqrt(area), px) and the effect of the network input size
# COCO buckets: small < 32px, medium 32-96px, large > 96px (measured at the *input* resolution).

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for s in D:
    ax[0].hist(np.log2(D[s][1]["size"]), bins=60, alpha=.6, density=True, label=s)
ax[0].set_xlabel("log2 sqrt(area), source px"); ax[0].legend()
rows = []
for S in (640, 960, 1280, 1920):  # longest side after resize
    for s, (imgs, objs, _) in D.items():
        sz = objs["size"] * S / objs.width
        rows.append(dict(imgsz=S, split=s, lt8=(sz < 8).mean(), lt16=(sz < 16).mean(),
                         small=(sz < 32).mean(), medium=((sz >= 32) & (sz < 96)).mean(), large=(sz >= 96).mean()))
share = pd.DataFrame(rows).round(3)
print(share.to_string(index=False))
pv = share[share.split == "train"].set_index("imgsz")[["lt8", "lt16", "small"]]
pv.plot.bar(ax=ax[1], title="train: share of objects below N px vs input size"); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(OUT / "object_size.png", dpi=120); plt.close()

# per-class median size
print(D["train"][1].groupby("cls")["size"].median().set_axis(CLASSES).round(1))

# %% [markdown]
# ## Density and per-sequence variation

# %%
rows = []
for s, (imgs, objs, ign) in D.items():
    per_frame = objs.groupby("image_id").size().reindex(imgs.image_id, fill_value=0)
    print(f"{s}: objects/frame mean {per_frame.mean():.1f}, median {per_frame.median():.0f}, "
          f"p95 {per_frame.quantile(.95):.0f}, max {per_frame.max()}")
    pf = per_frame.rename("n").reset_index().merge(imgs[["image_id", "video", "width"]], on="image_id")
    g = pf.groupby("video").agg(frames=("n", "size"), objs_per_frame=("n", "mean"), width=("width", "first"))
    g["median_size"] = objs.groupby("video")["size"].median()
    g["person_share"] = objs.groupby("video").cls.apply(lambda c: (c == 0).mean())
    g["car_share"] = objs.groupby("video").cls.apply(lambda c: (c == 2).mean())
    g["split"] = s
    rows.append(g)
seq = pd.concat(rows).round(2)
seq.to_csv(OUT.parent / "data" / "sequence_stats.csv")
print(seq.describe().round(1).to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(seq.objs_per_frame, bins=25); ax[0].set_title("objects/frame, per sequence")
ax[1].scatter(seq.objs_per_frame, seq.median_size, c=(seq.split == "val"), cmap="coolwarm")
ax[1].set_xlabel("objects/frame"); ax[1].set_ylabel("median size px"); ax[1].set_title("dense scenes = small objects (red=val)")
seq.groupby("width").size().plot.bar(ax=ax[2], title="sequences per frame width")
plt.tight_layout(); plt.savefig(OUT / "sequences.png", dpi=120); plt.close()

# %% [markdown]
# ## Ignore regions: how much of the frame area and of the frames they affect

# %%
ign = D["train"][2]
ign = ign.assign(frac=ign.w * ign.h / (ign.width * ign.height))
imgs_tr = D["train"][0]
print(f"frames with ignore regions: {ign.image_id.nunique() / len(imgs_tr):.2%}; "
      f"median region = {ign.frac.median():.2%} of frame, p95 = {ign.frac.quantile(.95):.2%}")

# %% [markdown]
# ## Verify conversion: draw the YOLO labels (green) and ignore regions (red) on frames from different sequences

# %%
def draw(split, video, frame, ax):
    p = DATA / split / "images" / video / f"{frame:07d}.jpg"
    im = Image.open(p).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    lw = max(2, W // 640)
    lab = DATA / split / "labels" / video / f"{frame:07d}.txt"
    for l in (lab.read_text().split("\n") if lab.exists() else []):
        if l:
            c, cx, cy, w, h = l.split(); cx, cy, w, h = (float(cx) * W, float(cy) * H, float(w) * W, float(h) * H)
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(0, 255, 0), width=lw)
            d.text((cx - w / 2, cy - h / 2 - 10), CLASSES[int(c)], fill=(255, 255, 0))
    ig = DATA / split / "ignore" / video / f"{frame:07d}.txt"
    for l in (ig.read_text().split("\n") if ig.exists() else []):
        if l:
            cx, cy, w, h = (float(v) for v in l.split()); cx, cy, w, h = cx * W, cy * H, w * W, h * H
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(255, 0, 0), width=lw)
    ax.imshow(im); ax.set_title(f"{video} #{frame} ({W}x{H})", fontsize=8); ax.axis("off")

dense = seq[seq.split == "train"].sort_values("objs_per_frame")
picks = [dense.index[0], dense.index[len(dense) // 2], dense.index[-1], seq[seq.split == "val"].index[0]]
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
for ax, v in zip(axs.flat, picks):
    draw("val" if v in D["val"][0].video.values else "train", v, 50, ax)
plt.tight_layout(); plt.savefig(OUT / "gt_samples.jpg", dpi=110); plt.close()
print("saved figures to", OUT)
