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
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ   # Colab also has an (empty) /kaggle/input, so test the env variable
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


def build(name, title, install, extra_cells, trainval=False):
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
!python src/mask_ignore.py --stride 3 %s--out {WORK}/yolo_masked
""" % ("--splits train val " if trainval else ""))
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
MODEL, EXTRA, TAG = "frcnn", "", "tv_frcnn"     # EXTRA = extra train_tv.py flags of a tuning variant, TAG must differ per variant
!python src/train_tv.py --model {MODEL} --train-list {WORK}/yolo_masked/train_s3.txt --tag {TAG} --out {OUT} --workers 2 {EXTRA}
"""),
    ("code", """
import json
r = json.load(open(f"{OUT}/eval_{TAG}.json")); print({k: v for k, v in r.items() if k != "per_class_AP"})
print(json.load(open(f"{OUT}/{TAG}/train_log.json"))[-1])
"""),
])

# Final model on train + val (no held-out data any more: val numbers of these runs are meaningless, only test detections are used)
build("torchvision_trainval", "HW2 — final torchvision detector trained on train+val (test detections only)", "!pip install -q pycocotools", [
    ("md", "## Train on train+val, then save TEST detections. Set `MODEL` to `frcnn`, `retina` or `fcos`. Resumes from `last.pt` after an interrupted session."),
    ("code", """
MODEL = "frcnn"
!python src/train_tv.py --model {MODEL} --train-list {WORK}/yolo_masked/trainval_s3.txt --tag tvtv_{MODEL} --out {OUT} --workers 2 --predict-test --no-val
"""),
    ("code", """
import numpy as np
d = np.load(f"{OUT}/testdets_tvtv_{MODEL}.npy"); print("test detections:", d.shape, "frames:", len(np.unique(d[:, 0])))
"""),
], trainval=True)

# RT-DETR tuning on Colab: number of queries and denoising, compared epoch-by-epoch with the 6-epoch baseline (rtdetr.ipynb)
build("rtdetr_tune", "HW2 — RT-DETR-L tuning: number of queries and denoising", "!pip install -q -U ultralytics pycocotools", [
    ("md", """
## Variants (everything else = the baseline `rtdetr.ipynb`: masked stride-3 train, 1280 px, batch 2, AdamW 1e-4, mixup .15, scale .9, seed 0)
* `nq100`: 100 object queries instead of 300 (scenes have 46 objects/frame on average, up to 180 -> dense scenes may saturate 100 queries).
* `nodn`: no denoising queries (`nd=0`): removes the contrastive-denoising auxiliary loss, which is what stabilises early Hungarian matching.

Each variant keeps the 6-epoch LR schedule but is **stopped after `STOP` epochs** (callback), so the per-epoch validation curve is
directly comparable with the baseline (baseline val mAP50-95 per epoch: .015 .054 .090 .111 .110 .126) at 40 % of the cost."""),
    ("code", """
import re, pathlib, ultralytics
from ultralytics import RTDETR
STOP, VARIANTS = 4, {"nq100": "[nc, 256, 100]", "nodn": "[nc, 256, 300, 4, 8, 6, 1024, 0.0, None, -1, 0]"}
base = pathlib.Path(ultralytics.__file__).parent / "cfg/models/rt-detr/rtdetr-l.yaml"
for name, args in VARIANTS.items():
    y = base.read_text().replace("RTDETRDecoder, [nc]]", f"RTDETRDecoder, {args}]")
    assert y != base.read_text(); pathlib.Path(f"rtdetr-l-{name}.yaml").write_text(y)

def n_epochs_done(name):
    f = pathlib.Path(OUT) / name / "results.csv"
    return len(f.read_text().strip().splitlines()) - 1 if f.exists() else 0

def stop_after(trainer):                     # stop after STOP epochs; last.pt is saved every epoch
    if trainer.epoch + 1 >= STOP: trainer.stop = True

for name in VARIANTS:
    last = pathlib.Path(OUT) / name / "weights/last.pt"
    if n_epochs_done(name) >= STOP: print(name, "already done"); continue
    if last.exists():
        m = RTDETR(str(last)); m.add_callback("on_fit_epoch_end", stop_after); m.train(resume=True)
    else:
        m = RTDETR(f"rtdetr-l-{name}.yaml").load("rtdetr-l.pt"); m.add_callback("on_fit_epoch_end", stop_after)
        m.train(data=f"{WORK}/yolo_masked/data_s3.yaml", imgsz=1280, epochs=6, batch=2, workers=2, optimizer="AdamW", lr0=1e-4,
                mixup=0.15, scale=0.9, close_mosaic=0, seed=0, deterministic=True, project=OUT, name=name, exist_ok=True, plots=False)
"""),
    ("md", "## Per-epoch validation curves (compare with the baseline numbers above) and full-val COCO scoring of the last checkpoint"),
    ("code", """
import pandas as pd
for name in VARIANTS:
    d = pd.read_csv(f"{OUT}/{name}/results.csv"); d.columns = d.columns.str.strip()
    print(name, "mAP50-95 per epoch:", [round(v, 3) for v in d["metrics/mAP50-95(B)"]], "| giou", [round(v, 2) for v in d["train/giou_loss"]])
"""),
    ("code", """
for name in VARIANTS:
    !python src/eval_coco.py --arch rtdetr --bs 4 --weights {OUT}/{name}/weights/last.pt --imgsz 1280 --tag rtdetr_{name}
    !cp runs/eval_rtdetr_{name}.json runs/dets_rtdetr_{name}.npy {OUT}/
"""),
])

# YOLO loss-weight tuning (anchor-free rubric: "loss modification"): Ultralytics loss gains box / cls / dfl
build("yolo_tune", "HW2 — YOLO11m loss-weight tuning (box / cls / dfl gains)", "!pip install -q -U ultralytics pycocotools", [
    ("md", "## Jobs: every entry = the best YOLO recipe (yolo11m, masked stride-3, 1280, SGD lr 0.01, mixup .15, scale .9, 6 epochs, seed 0) with ONE changed loss gain. "
           "Defaults: `box=7.5`, `cls=0.5`, `dfl=1.5`. Edit `JOBS` to add runs; finished jobs are skipped. Batch 4 as in the reference run (`hyp_m_aug_masked`: AP .2413 @1280)."),
    ("code", """
import pathlib
from ultralytics import YOLO
JOBS = {"box12": dict(box=12.0), "box4": dict(box=4.0), "cls1": dict(cls=1.0), "cls025": dict(cls=0.25), "dfl3": dict(dfl=3.0)}
for name, extra in JOBS.items():
    if (pathlib.Path(OUT) / f"yolo_{name}" / "weights/last.pt").exists() and (pathlib.Path(OUT) / f"eval_yolo_{name}.json").exists():
        print(name, "done"); continue
    YOLO("yolo11m.pt").train(data=f"{WORK}/yolo_masked/data_s3.yaml", imgsz=1280, epochs=6, batch=4, workers=4, optimizer="SGD", lr0=0.01,
                             mixup=0.15, scale=0.9, close_mosaic=0, seed=0, deterministic=True, project=OUT, name=f"yolo_{name}",
                             exist_ok=True, plots=False, **extra)
"""),
    ("md", "## Score every job on all val frames (COCO AP with ignore regions) at inference size 1600, like the reference"),
    ("code", """
for name in JOBS:
    !python src/eval_coco.py --weights {OUT}/yolo_{name}/weights/best.pt --imgsz 1600 --bs 8 --tag yolo_{name}
    !cp runs/eval_yolo_{name}.json runs/dets_yolo_{name}.npy {OUT}/
"""),
])

# A100 job launcher: runs the tuning jobs of notebooks/JOBS.md (torchvision + YOLO) several at a time on one big GPU
build("a100_jobs", "HW2 — tuning jobs launcher (several jobs in parallel on one A100)", "!pip install -q -U ultralytics pycocotools", [
    ("md", """
## How it works
`JOBS` maps a tag to a shell command (same protocol as the reference runs, one changed setting each). The scheduler runs `P` jobs at a time
(12 vCPU -> 3 jobs x 3 dataloader workers), skips jobs whose `eval_<tag>.json` already exists in `OUT` and resumes torchvision jobs from `last.pt`
after a disconnect (YOLO jobs restart). Logs: `OUT/logs/<tag>.log`. Run the last cell for the results table. RT-DETR jobs live in `rtdetr_tune.ipynb`."""),
    ("code", """
import os, pathlib, subprocess, time, json
from concurrent.futures import ThreadPoolExecutor
P, W = 3, 3          # parallel jobs, dataloader workers per job
TV = f"python src/train_tv.py --train-list {WORK}/yolo_masked/train_s3.txt --out {OUT} --workers {W}"
YOLO = (f"yolo detect train model=yolo11m.pt data={WORK}/yolo_masked/data_s3.yaml imgsz=1280 epochs=6 batch=4 workers={W} cache=ram "
        f"optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 seed=0 deterministic=True project={OUT} exist_ok=True plots=False")
def yolo(name, args):
    return (f"{YOLO} name=yolo_{name} {args} && python src/eval_coco.py --weights {OUT}/yolo_{name}/weights/best.pt --imgsz 1600 --bs 8 "
            f"--tag yolo_{name} && cp runs/eval_yolo_{name}.json {OUT}/")
JOBS = {
    "tv_retina_a16":  f"{TV} --model retina --tag tv_retina_a16 --anchor-base 16",
    "tv_retina_giou": f"{TV} --model retina --tag tv_retina_giou --retina-loss giou",
    "tv_retina_g1":   f"{TV} --model retina --tag tv_retina_g1 --focal-gamma 1.0",
    "tv_fcos_r25":    f"{TV} --model fcos --tag tv_fcos_r25 --fcos-radius 2.5",
    "tv_fcos_r10":    f"{TV} --model fcos --tag tv_fcos_r10 --fcos-radius 1.0",
    "tv_frcnn_beta1": f"{TV} --model frcnn --tag tv_frcnn_beta1 --frcnn-beta 1.0",
    "yolo_box12": yolo("box12", "box=12.0"), "yolo_box4": yolo("box4", "box=4.0"),
    "yolo_cls1": yolo("cls1", "cls=1.0"), "yolo_cls025": yolo("cls025", "cls=0.25"), "yolo_dfl3": yolo("dfl3", "dfl=3.0"),
}
def run(item):
    tag, cmd = item
    if (pathlib.Path(OUT) / f"eval_{tag}.json").exists(): return tag, "skipped (done)", 0
    t0 = time.time(); logs = pathlib.Path(OUT) / "logs"; logs.mkdir(exist_ok=True)
    with open(logs / f"{tag}.log", "w") as f:
        rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
    return tag, "ok" if rc == 0 else f"FAILED rc={rc} (see log)", round((time.time() - t0) / 60, 1)
with ThreadPoolExecutor(P) as ex:
    for r in ex.map(run, JOBS.items()):
        print(r, flush=True)
"""),
    ("md", "## Progress (run in a second cell while the scheduler is busy) and results"),
    ("code", """
!tail -qn 1 {OUT}/logs/*.log 2>/dev/null | cut -c1-140
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
"""),
    ("code", """
import pandas as pd
rows = [json.load(open(f)) for f in sorted(pathlib.Path(OUT).glob("eval_*.json"))]
pd.DataFrame([{k: r[k] for k in ("tag", "AP", "AP_no4k", "AP50", "AP_small", "AP_medium", "AP_large", "ms_per_frame")} for r in rows]).round(4).sort_values("AP", ascending=False)
"""),
])

# Pseudo-label final models (A100): train+val + our ensemble's pseudo-labels on the unlabelled TEST frames (src/pseudo_label.py, thr .4)
import base64
PL_B64 = base64.b64encode((Path(__file__).resolve().parent / "pl_labels.tgz").read_bytes()).decode()
build("pl_jobs", "HW2 — final YOLO models with pseudo-labelled test frames (A100, 3 jobs in parallel)", "!pip install -q -U ultralytics pycocotools", [
    ("md", """
## What this is
Self-training: our best ensemble (`sub_merge_sl.csv`, public .3277) predicted the unlabelled TEST frames; boxes with score >= 0.4 are used as
labels (97 k boxes on 3 811 frames, precision ~.78 measured on val with the same recipe). Only test *images* and our own predictions are used,
never test labels. Validation of the idea on val (pseudo-labelled val frames, scored with the real labels): yolo11m AP .2611 -> .2729.
Each job = train on train+val (9 041 masked frames) + the 3 811 pseudo-labelled test frames, `dfl=3.0`, 6 epochs, seed 0, then TEST detections
(single-label and multi-label) are written to `OUT` as `testdets_<tag>.npy`. Do not change any setting."""),
    ("code", "%%writefile src/predict_test.py\n" + (root / "src" / "predict_test.py").read_text()),
    ("code", f'''
import base64, io, tarfile, pathlib
PL_B64 = "{PL_B64}"
PL = pathlib.Path(WORK) / "yolo_pl"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PL_B64)), mode="r:gz").extractall(PL)     # -> yolo_pl/labels/<clip>/<frame>.txt
lst = []
for lab in sorted((PL / "labels").glob("*/*.txt")):
    dst = PL / "images" / lab.parent.name / (lab.stem + ".jpg"); dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_symlink(): dst.symlink_to(RAW / "test" / "images" / lab.parent.name / (lab.stem + ".jpg"))
    lst.append(str(dst))
M = pathlib.Path(WORK) / "yolo_masked"
(PL / "train_pl.txt").write_text((M / "trainval_s3.txt").read_text() + "\\n" + "\\n".join(lst))
val = [l for l in (M / "data_trainval_s3.yaml").read_text().splitlines() if l.startswith("val:")][0]
(PL / "data_pl.yaml").write_text(f"path: {{PL}}\\ntrain: train_pl.txt\\n{{val}}\\nnames:\\n" + "".join(f"  {{i}}: {{n}}\\n" for i, n in enumerate(
    ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"])))
print(len(lst), "pseudo-labelled test frames;", len((PL / "train_pl.txt").read_text().split()), "training frames in total")
'''),
    ("md", "## Jobs (3 in parallel x 4 dataloader workers = 12 vCPU). Finished steps are skipped, an interrupted training resumes. Logs: `OUT/logs/<tag>.log`"),
    ("code", """
import os, pathlib, subprocess, time
from concurrent.futures import ThreadPoolExecutor
P, W = 3, 4
D = f"{WORK}/yolo_pl/data_pl.yaml"
R = f"optimizer=SGD lr0=0.01 mixup=0.15 scale=0.9 close_mosaic=0 epochs=6 seed=0 deterministic=True workers={W} dfl=3.0 cache=False project={OUT} exist_ok=True plots=False"
def job(tag, model, batch, imgsz, preds):          # preds: list of (testdets tag, imgsz, extra flags)
    w = f"{OUT}/{tag}/weights/last.pt"
    train = (f"yolo detect train resume model={w}" if os.path.exists(w) else
             f"yolo detect train model={model}.pt data={D} batch={batch} imgsz={imgsz} {R} name={tag}")
    steps = [] if os.path.exists(f"{OUT}/{tag}/done") else [train, f"touch {OUT}/{tag}/done"]
    for t, sz, extra in preds:
        if not os.path.exists(f"{OUT}/testdets_{t}.npy"):
            steps.append(f"python src/predict_test.py --weights {w} --imgsz {sz} --bs 8 {extra} --tag {t} && cp runs/testdets_{t}.npy {OUT}/")
    return " && ".join(steps)
JOBS = {
    "tv_pl_m16": job("tv_pl_m16", "yolo11m", 4, 1600, [("tv_pl_m16_i1920", 1920, ""), ("tv_pl_m16_ml", 1920, "--multi-label")]),
    "tv_pl_l":   job("tv_pl_l",   "yolo11l", 4, 1280, [("tv_pl_l_i1600", 1600, ""), ("tv_pl_l_ml", 1600, "--multi-label")]),
    "tv_pl_s":   job("tv_pl_s",   "yolo11s", 8, 1280, [("tv_pl_s_i1280", 1280, ""), ("tv_pl_s_ml", 1280, "--multi-label")]),
}
def run(item):
    tag, cmd = item
    if not cmd: return tag, "skipped (done)", 0
    t0 = time.time(); logs = pathlib.Path(OUT) / "logs"; logs.mkdir(exist_ok=True)
    with open(logs / f"{tag}.log", "a") as f:
        rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
    return tag, "ok" if rc == 0 else f"FAILED rc={rc} (see log)", round((time.time() - t0) / 60, 1)
with ThreadPoolExecutor(P) as ex:
    for r in ex.map(run, JOBS.items()):
        print(r, flush=True)
"""),
    ("md", "## Check: the files to send back (they are on Drive in `hw2_runs/`)"),
    ("code", "!ls -la {OUT}/testdets_tv_pl_*.npy"),
], trainval=True)
