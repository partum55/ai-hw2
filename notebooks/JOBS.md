# Experiment jobs (who runs what)

All jobs use the same protocol as the reference runs: official `train`/`val` split by clip, ignore-masked stride-3 train frames,
6 epochs, seed 0, scored by `src/coco_utils.py` on all 2 846 val frames (COCO AP with ignore regions). A job = one notebook + one
changed setting. Record **AP, AP50, AP_small, AP_no4k, ms/frame** from the printed JSON and keep the JSON file.

Notebooks are generated from `src/` by `python notebooks/build.py`. Run on Colab (T4/L4/A100) or Kaggle (T4); see the first cell of each.

| # | Family | Notebook | Setting(s) | Tag | Reference to beat |
|---|---|---|---|---|---|
| 1 | two-stage | `torchvision.ipynb` (`MODEL="frcnn"`) | `--frcnn-beta 1.0` (box loss: smooth-L1 beta 1/9 → 1) | `tv_frcnn_beta1` | `tv_frcnn` AP .2367 |
| 2 | two-stage | same | `--rpn-post-nms 2000` / `--roi-batch 128` | running on Kaggle | `tv_frcnn` |
| 3 | one-stage anchor-based | `MODEL="retina"` | `--anchor-base 16` (anchors from 16 px, objects are small) | `tv_retina_a16` | `tv_retina` AP .1934 |
| 4 | one-stage anchor-based | same | `--retina-loss giou` | `tv_retina_giou` | `tv_retina` |
| 5 | one-stage anchor-based | same | `--focal-gamma 1.0` | `tv_retina_g1` | `tv_retina` |
| 6 | anchor-free | `MODEL="fcos"` | `--fcos-radius 2.5` (more positives per object) | `tv_fcos_r25` | `tv_fcos` AP .1812 |
| 7 | anchor-free | same | `--fcos-radius 1.0` | `tv_fcos_r10` | `tv_fcos` |
| 8 | anchor-free (YOLO) | `yolo_tune.ipynb` | loss gains `box` 12 / 4, `cls` 1 / .25, `dfl` 3 | `yolo_box12`, … | `hyp_m_aug_masked` AP .2413 @1280 |
| 9 | query-based | `rtdetr_tune.ipynb` | `nq=100` (fewer queries), `nd=0` (no denoising); 4 of 6 epochs | `rtdetr_nq100`, `rtdetr_nodn` | baseline val mAP50-95 per epoch .015 .054 .090 .111 |
| 10 | query-based | `rtdetr.ipynb` with more epochs | does RT-DETR converge with a larger budget? (e.g. 24 epochs) | `rtdetr_long` | `rtdetr_l` AP .1295 (6 ep, not converged) |

**Fastest way on one big GPU: `a100_jobs.ipynb`** runs jobs 1, 3–8 (6 torchvision + 5 YOLO jobs, 11 runs) with 3 jobs in parallel
(12 vCPU: 3 jobs x 3 workers); finished jobs are skipped, torchvision jobs resume after a disconnect; the last cell prints the results table.
RT-DETR jobs (9, 10) stay in their own notebooks.

For the torchvision notebook set the cell `MODEL, EXTRA, TAG = "retina", "--anchor-base 16", "tv_retina_a16"` (one run per notebook session).
Keep batch/lr of the reference runs for tuning jobs, otherwise the comparison is not clean.
