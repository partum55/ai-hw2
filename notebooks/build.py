"""Generate self-contained notebooks that run on Colab OR Kaggle (the platform is detected at run time).
The needed src/ files are embedded via %%writefile cells, so a notebook always matches src/.

Usage: python notebooks/build.py     (re-run after changing anything in src/)
Writes notebooks/rtdetr.ipynb (query-based) and notebooks/torchvision.ipynb (Faster R-CNN / RetinaNet / FCOS).
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
SRC = ["common.py", "coco_utils.py", "prepare.py", "mask_ignore.py", "eval_coco.py", "train_tv.py"]

SETUP_MD = """
**Setup (once)** — Colab: Runtime → T4 GPU. Secrets (🔑, left bar, "Notebook access" on): `KAGGLE_USERNAME` + `KAGGLE_KEY`
from kaggle.com → Settings → API → *Create New Token* (or a single `KGAT_...` token as `KAGGLE_API_TOKEN` / `kaggle`);
you must have joined the competition with that account. Checkpoints go to Google Drive and training resumes after a disconnect.
Kaggle — Session options: GPU T4 x2, Internet On, Add data → Competitions → `multiple-object-video-detection-hw-2-ucu-ai-2026`.
Protocol = the YOLO runs: ignore-masked stride-3 train, 6 epochs, seed 0; scored by `coco_utils.score` on all 2 846 val frames.
"""

ENV = '''
import glob, os, pathlib, subprocess
ON_KAGGLE = os.path.exists("/kaggle/input")
if ON_KAGGLE:
    RAW = pathlib.Path(glob.glob("/kaggle/input/**/train/annotations_coco.json", recursive=True)[0]).parents[1]
    WORK, OUT = "/kaggle/working", "/kaggle/working/outputs"
else:  # Colab: data from the Kaggle API, outputs on Drive
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        try: os.environ[k] = userdata.get(k)
        except Exception: pass
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        try: os.environ["KAGGLE_API_TOKEN"] = userdata.get("kaggle")  # one secret named `kaggle` holding the KGAT_... token
        except Exception: pass
    RAW, WORK, OUT = pathlib.Path("/content/raw"), "/content", "/content/drive/MyDrive/hw2_runs"
    if not (RAW / "train/annotations_coco.json").exists():
        subprocess.run("mkdir -p /content/dl /content/raw && cd /content/dl && pip install -q kaggle && "
                       "kaggle competitions download -c multiple-object-video-detection-hw-2-ucu-ai-2026 && "
                       "unzip -q -o *.zip -d /content/raw && rm -f *.zip", shell=True, check=True)
os.makedirs(OUT, exist_ok=True)
os.environ["HW2_DATA"] = str(RAW); os.environ["PYTHONPATH"] = "src"
print("data:", RAW, "| work:", WORK, "| outputs:", OUT)
'''


def build(name, title, install, extra_cells):
    cells = []

    def md(t):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)})

    def code(t):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": t.strip().splitlines(True)})

    md(f"# {title}\n" + SETUP_MD)
    code(install)
    md("## Environment: locate / download the data, set paths")
    code(ENV)
    md("## Source files (embedded copies of `src/`)")
    code("!mkdir -p src")
    for f in SRC:
        code(f"%%writefile src/{f}\n" + (root / "src" / f).read_text())
    md("## Stride-3 lists and the ignore-masked train set (local disk, rebuilt in every new session)")
    code("""
!python src/prepare.py --stride 3 --out {WORK}/yolo
!python src/mask_ignore.py --stride 3 --out {WORK}/yolo_masked
""")
    for kind, text in extra_cells:
        (md if kind == "md" else code)(text)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (root / "notebooks" / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote notebooks/%s.ipynb with %d cells" % (name, len(cells)))


build("rtdetr", "HW2 — RT-DETR-L (query-based detector)", "!pip install -q -U ultralytics pycocotools", [
    ("md", "## Train RT-DETR-L (AdamW 1e-4 as is usual for DETR; batch 2 so a 16 GB T4 does not run out of memory)"),
    ("code", """
from pathlib import Path
from ultralytics import RTDETR
last = Path(OUT) / "rtdetr_l/weights/last.pt"
if last.exists():
    RTDETR(str(last)).train(resume=True)             # continue an interrupted run
else:
    RTDETR("rtdetr-l.pt").train(data=f"{WORK}/yolo_masked/data_s3.yaml", imgsz=1280, epochs=6, batch=2, workers=2,
        optimizer="AdamW", lr0=1e-4, mixup=0.15, scale=0.9, close_mosaic=0, seed=0, deterministic=True,
        project=OUT, name="rtdetr_l", exist_ok=True, plots=False)
"""),
    ("md", "## Score on all val frames (COCO AP with ignore regions)"),
    ("code", """
!python src/eval_coco.py --arch rtdetr --bs 4 --weights {OUT}/rtdetr_l/weights/best.pt --imgsz 1280 --tag rtdetr_l
!cp runs/eval_rtdetr_l.json runs/dets_rtdetr_l.npy {OUT}/
import json, pandas as pd
r = json.load(open("runs/eval_rtdetr_l.json")); print({k: v for k, v in r.items() if k != "per_class_AP"})
d = pd.read_csv(f"{OUT}/rtdetr_l/results.csv"); d.columns = d.columns.str.strip()
print("epochs completed:", len(d)); d.filter(regex="epoch|loss|mAP")
"""),
])

build("torchvision", "HW2 — Faster R-CNN / RetinaNet / FCOS (torchvision, COCO-pretrained)", "!pip install -q pycocotools", [
    ("md", "## Train + score one detector. Set `MODEL` to `frcnn` (two-stage), `retina` (one-stage anchor-based) or `fcos` (anchor-free). "
           "Resumes from `last.pt` after an interrupted session. Extra flags: `--rpn-post-nms`, `--roi-batch` (frcnn), `--lr`, `--epochs`."),
    ("code", """
MODEL = "frcnn"
!python src/train_tv.py --model {MODEL} --train-list {WORK}/yolo_masked/train_s3.txt --tag tv_{MODEL} --out {OUT} --workers 2
"""),
    ("code", """
import json
r = json.load(open(f"{OUT}/eval_tv_{MODEL}.json")); print({k: v for k, v in r.items() if k != "per_class_AP"})
print(json.load(open(f"{OUT}/tv_{MODEL}/train_log.json"))[-1])
"""),
])
