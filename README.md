# Cattle Detection

YOLO26 cattle detectors for drone imagery. This repo holds the best-performing
checkpoints from the project, plus the training, pruning, distillation,
pseudo-labelling and evaluation code that produced them.

**Author:** Leo ([@leo-whe](https://github.com/leo-whe)), who designed and built the
pipeline.
**Access:** private repository. Code, model weights and results are not licensed for
redistribution without the author's permission.

The companion repository `cattle-tracking` uses `weights/cattle_nadir_m.pt` from here
for tracking and counting in video.

## Which model to use

| You have | Use | `imgsz` | Why |
|---|---|---|---|
| Drone **video** frames: low-altitude nadir, or an oblique herd | `cattle_nadir_m.pt` | 1920 | the only model that works on nadir footage; recall 0.978 / 0.941 on the two verified frames |
| High-altitude UAV **stills** like cow_uav, best accuracy | `yolo26m_p5drop_cowuav_2560.pt` | 2560 | best test mAP50-95 (0.6558), 25% smaller and 13% faster than the unpruned twin |
| The same, as an exact stock-architecture graph | `yolo26m_cowuav_2560.pt` | 2560 | identical accuracy; stock YOLO26m, so it exports and fine-tunes like any ultralytics model |
| The same, throughput-bound (edge GPU, batch) | `yolo26s_kd_cowuav_1280.pt` | 1280 | 0.6027 at 215 FPS: yolo26m-level accuracy at a third of the size |
| The same, size-bound (3.6 MiB) | `yolo26n_p5drop_cowuav_3200.pt` | 3200 | 0.6349 in 3.6 MiB, 5.4x smaller than yolo26s and 0.058 more accurate |

The two families do not substitute for each other. The cow_uav models were trained on
high-altitude stills where an animal is a ~34 px speck. On the two drone frames
verified by hand they reach recall of at most 0.74 (nadir) and 0.22 (oblique herd),
against 0.98 and 0.94 for `cattle_nadir_m`. `cattle_nadir_m` was in turn never scored
on the cow_uav benchmark.

## Model zoo

All checkpoints are standard ultralytics `.pt` files and load with `YOLO(path)`.

| File | Arch | Classes | Train data | `imgsz` | Size | Params | Key metric |
|---|---|---|---|---|---|---|---|
| `cattle_nadir_m.pt` | YOLO26m | `cattle` | 840 teacher-labelled drone frames | 1920 | 42.2 MiB | 20.4 M | recall 0.978 nadir / 0.941 herd |
| `yolo26m_p5drop_cowuav_2560.pt` | YOLO26m, P5 head removed | 6 (cow_uav) | cow_uav_images-5 | 2560 | 31.5 MiB | 15.5 M | test mAP50-95 **0.6558** |
| `yolo26m_cowuav_2560.pt` | YOLO26m | 6 (cow_uav) | cow_uav_images-5 | 2560 | 42.5 MiB | 20.4 M | test mAP50-95 **0.6558** |
| `yolo26s_kd_cowuav_1280.pt` | YOLO26s, P5 removed, distilled | 6 (cow_uav) | cow_uav_images-5 | 1280 | 13.8 MiB | 6.8 M | test mAP50-95 0.6027 |
| `yolo26n_p5drop_cowuav_3200.pt` | YOLO26n, P5 head removed | 6 (cow_uav) | cow_uav_images-5 | 3200 | 3.6 MiB | 1.7 M | test mAP50-95 0.6349 |

The cow_uav classes are `0`, `cattle`, `cow`, `cowrotation`, `red_squirrelrotation` and
`unknownrotation`. The first four are all cattle. `detect.py` keeps those four and
reports them as one class.

## Results

### Drone video: frames verified by hand

`eval_detectors.py`, conf 0.25, IoU match 0.4. Ground truth is in `gt/`. Each frame was
labelled box by box from detector and teacher proposals, and recall was checked on
magnified tiles.

| Detector | `imgsz` | Nadir frame 10501 (46 cattle) | Oblique herd frame 1787 (51 cattle) |
|---|---|---|---|
| COCO `yolo26m`, `cow` class | best | recall **0.022** | recall 0.922, precision 0.940 |
| cow_uav `yolo26m_p5drop_cowuav_2560` | best | recall 0.217 | recall 0.118 |
| cow_uav `yolo26n_p5drop_cowuav_3200` | best | recall 0.739 | recall 0.216 |
| **`cattle_nadir_m`** | **1920** | **recall 0.978, precision 0.938** | **recall 0.941, precision 0.923** |
| `cattle_nadir_m` | 2560 | recall 0.978, precision 0.957 | recall 0.902, precision 1.000 |

`cattle_nadir_m` @1920 runs at 28.2 ms/frame (35 FPS) on an A100. The LocateAnything-3B
teacher it was distilled from takes 2.2-7.9 s/frame. For the cow_uav models, "best" is
the best of `imgsz` 1280-3200. Full rows are in `results/cattle_nadir_m_eval_*.json`
and `results/cowuav_models_eval_frame*.json`.

### cow_uav_images-5 test split (201 images, 2469 instances)

| Model | `imgsz` | Size | mAP50-95 | mAP50 | mAP75 | P | R | A100 batch-8 FPS |
|---|---|---|---|---|---|---|---|---|
| **`yolo26m_p5drop_cowuav_2560`** | 2560 | 31.5 MiB | **0.6558** | 0.9462 | 0.7150 | 0.938 | 0.934 | 24.0 |
| `yolo26m_cowuav_2560` | 2560 | 42.5 MiB | **0.6558** | 0.9462 | 0.7150 | 0.938 | 0.934 | 21.2 |
| `yolo26n_p5drop_cowuav_3200` | 3200 | 3.6 MiB | 0.6350 | 0.9350 | 0.7033 | 0.928 | 0.918 | 52.6 |
| `yolo26s_kd_cowuav_1280` | 1280 | 13.8 MiB | 0.6027 | 0.9282 | 0.6648 | 0.935 | 0.902 | 214.7 |
| *baseline* yolo26m | 1280 | 42.1 MiB | 0.6031 | 0.9263 | – | – | – | 94.4 |
| *baseline* yolo26s | 1280 | 19.5 MiB | 0.5764 | 0.9211 | – | – | – | 189.0 |
| *best DINOv3 backbone* (ConvNeXt, 2-stage) | 1280 | – | 0.5587 | – | – | – | – | – |

The shipped rows were re-scored from the files in `weights/` with `eval_model.py`
(`results/summary_*.json`). Latency comes from `results/latency_idle_gpu_a100.json`,
measured on an otherwise idle A100. All 69 runs of the study are listed in
`results/cow_uav_all_runs.json`. `results/size_vs_accuracy.png` plots the frontier.

### What moved the numbers

1. **Input resolution is the largest lever.** The median cow_uav box is 34 px at
   `imgsz` 1280. Raising `imgsz` adds no parameters and had not saturated at
   2560-3200: yolo26n goes from 0.512 at 1280 to 0.637 at 3200.
2. **The P5/32 head is dead weight on small targets** (`prune_p5.py`). Of the boxes,
   99.7% are under 128 px at 1280. Deleting the stride-32 branch keeps every other
   trained tensor, needs no retraining, and cuts the model by 20-28%. Accuracy is
   unchanged at `imgsz` ≤ 1920 and for n/m at 2560. It is not free for s at 3200
   (−0.018), where the largest objects grow into the P5 range.
3. **Distillation helps a compact student** (`train_distill.py`). The student is the
   s-sized P5-dropped model. Prediction-level KD from a two-teacher ensemble (pruned m
   + l) raises it from 0.5764 to 0.6027 at an unchanged size, with most of the gain in
   localisation (mAP75 +0.027). The teacher should be only a few times larger than the
   student: for the nano student a 9x teacher did worse than a 4x one, and at the lower
   KD weight it fell below the undistilled baseline.
4. **Pseudo-labels close a domain gap** (`pseudo_label/`). LocateAnything-3B is far
   too slow to deploy, but it does fire on nadir cattle. It labelled 840 frames,
   sampled so as to over-weight the frames the deployed detector got wrong, and
   finetuning yolo26m on them took nadir recall from 0.022 to 0.978 without hurting the
   herd frame.
5. **What did not help.** DINOv3 backbones (ViT and ConvNeXt, frozen or finetuned)
   scored 0.39-0.56 at 1280, all below stock yolo26s. Narrower-width and
   retrained-from-partial-init graphs were dominated by the weight-preserving P5
   transplant. Fine-tuning a transplant, or re-distilling an already-distilled model,
   lowered accuracy. Generic structured pruning (torch-pruning) cannot trace YOLO26.

`docs/compact_study.md` and `docs/nadir_detector.md` are the full research logs.

## Quick start

Python 3.11 and one CUDA GPU.

```bash
conda create -n cattle-det python=3.11 -y
conda activate cattle-det
pip install -r requirements.txt

# images, a folder, or a video; labels/ + counts.csv, and annotated output with --save-vis
python detect.py --source path/to/frames/ --out runs/detect/frames --device 0 --save-vis
python detect.py --source flight.MP4 --vid-stride 30 --save-vis --device 0
python detect.py --source stills/ --weights weights/yolo26m_p5drop_cowuav_2560.pt --device 0
```

Each checkpoint defaults to the `imgsz` it was trained at; override with `--imgsz`. For
footage over gravel or hillsides, raise `--conf` (see Limitations).

In Python:

```python
from ultralytics import YOLO
model = YOLO("weights/cattle_nadir_m.pt")
result = model.predict("frame.jpg", imgsz=1920, conf=0.25)[0]
```

## Reproduce

### Data layout

The configs keep dataset paths relative to the repo root (`common.resolve_data`
resolves them), so place or symlink the data here. `datasets/` is gitignored.

```
datasets/cow_uav_images-5/{train,valid,test}/{images,labels}   # Roboflow cow_uav_images v5
datasets/texas_pseudo/images/{train,val}, labels/{train,val}   # built by pseudo_label/
```

### cow_uav models

```bash
# yolo26m_cowuav_2560.pt  (300 epochs, early stop patience 50)
python train.py --model yolo26m.pt --name yolo26m_2560 --imgsz 2560 --batch 4 --device 0

# yolo26m_p5drop_cowuav_2560.pt  (no training: transplant into configs/yolo26-p34keep.yaml)
python prune_p5.py --src runs/yolo26m_2560/weights/best.pt --scale m \
    --out runs/p5drop_yolo26m_2560/weights/best.pt
python eval_model.py --weights runs/p5drop_yolo26m_2560/weights/best.pt \
    --name p5drop_yolo26m_2560 --imgsz 2560 --batch 4 --device 0

# yolo26n_p5drop_cowuav_3200.pt
python train.py --model yolo26n.pt --name yolo26n_3200 --imgsz 3200 --batch 4 --device 0
python prune_p5.py --src runs/yolo26n_3200/weights/best.pt --scale n \
    --out runs/p5drop_yolo26n_3200/weights/best.pt

# yolo26s_kd_cowuav_1280.pt: train s/m/l at 1280, prune each, then distil s from m+l
for s in s m l; do
  python train.py --model yolo26$s.pt --name yolo26${s}_1280 --device 0
  python prune_p5.py --src runs/yolo26${s}_1280/weights/best.pt --scale $s \
      --out runs/p5drop_yolo26${s}/weights/best.pt
done
python train_distill.py --model configs/yolo26s-p34keep.yaml \
    --teacher runs/p5drop_yolo26m/weights/best.pt,runs/p5drop_yolo26l/weights/best.pt \
    --init runs/p5drop_yolo26s/weights/best.pt \
    --name kd_s_ens_w8 --w-cls 8 --w-box 4 --epochs 200 --lr0 0.0005 --device 0
```

Every training script writes `runs/summary_<name>.json` with the test metrics, the
parameter count and the checkpoint size. `bench_latency.py` times the shipped
checkpoints, or any `--models name=path.pt:imgsz`. Run it on an idle GPU, because
batch-1 latency at this model size is launch-bound and sensitive to CPU load.

### `cattle_nadir_m.pt`

The teacher, LocateAnything-3B, runs in its own environment. It needs the
`NVlabs/Eagle` repo (its `Embodied/` directory provides `locateanything_worker`) and
the LocateAnything-3B checkpoint, by default under `third_party/` and `models/` here.
Both are third-party and not included.

```bash
# 1. teacher pseudo-labels, stratified by how many boxes the deployed detector finds
#    (the counts CSV is track_count_cattle.py output from cattle-tracking)
conda activate locateanything
python pseudo_label/build_pseudo_labels.py --video DJI_0056.MP4 \
    --frame-counts DJI_0056_frame_counts.csv --out-dir datasets/texas_pseudo \
    --tag v0056 --n-fail 220 --n-mid 90 --n-ok 110 --device cuda:0
#    (same for DJI_0057 with --n-fail 120 --n-mid 110 --n-ok 190)

# 2. split by contiguous 900-frame blocks, never at random: at 60 fps neighbouring
#    frames are near-duplicates
conda activate cattle-det
python pseudo_label/split_pseudo_dataset.py --dataset datasets/texas_pseudo

# 3. finetune COCO yolo26m: 60 epochs, imgsz 1920, AdamW lr0 0.002, degrees=180, flipud=0.5
python pseudo_label/train_cattle_nadir.py --device 0      # ~1 h on one A100

# 4. score against the frames verified by hand, not against the teacher
python eval_detectors.py --gt-json gt/frame10501_gt.json --gt-boxes gt/frame10501_gt_boxes.npy \
    --image gt/frame10501.jpg --models nadir=runs/cattle_nadir_m/weights/best.pt:cattle \
    coco=yolo26m.pt:cow --imgsz 1280 1920 2560 --device 0
```

The validation mAP printed during finetuning (mAP50 0.983) measures agreement with the
teacher, not accuracy. Only the frames in `gt/` score the model.

## Layout

```
detect.py              inference on images / folders / video -> YOLO labels + counts
train.py               train + test-split summary for any YOLO26 model or config
prune_p5.py            drop the P5/32 head from a trained checkpoint, keeping all other weights
train_distill.py       prediction-level KD from one or several P3/P4 teachers
eval_model.py          cow_uav test-split metrics for a checkpoint
eval_detectors.py      recall / precision / count error against frames verified by hand
bench_latency.py       batch-1 latency and batch-8 throughput
common.py              repo paths; resolves relative dataset paths
configs/               dataset yamls; yolo26-p34keep (P5 removed), -p34, -p234 graphs
pseudo_label/          LocateAnything teacher labelling, block split, nadir finetune
weights/               shipped checkpoints (table above)
gt/                    two frames verified by hand: images, box arrays, provenance JSON
results/               metric summaries, latency, all-runs table, size-vs-accuracy plot
docs/                  research logs for the compaction study and the nadir detector
```

## Limitations

- **`cattle_nadir_m` is domain-bound.** It was trained on two flights from one Texas
  ranch, on one day, at one sun angle. On terrain it never saw it produces false
  positives. On bare gravel the false-positive count grows steeply with input size:
  2, 21, 135 and 366 over five frames verified to be empty, at `imgsz` 1280, 1920, 2560
  and 3840. Use `imgsz` ≤ 1920 and conf ≥ 0.45 there. Distant hillsides give large
  low-confidence boxes (0.24-0.41). Extending it means running the teacher over those
  terrains and retraining.
- **The verified ground truth is two frames,** each labelled once by one annotator and
  derived from detector and teacher proposals. An animal that every source missed is
  absent from it. Broader detection ground truth and hard-negative packages have been
  prepared for annotation but are not yet labelled.
- **The cow_uav classes are noisy**: four labels for cattle, plus two unrelated
  classes. Per-class mAP for `cow` (0.31) is far below the others, so the overall mAP
  understates single-class cattle accuracy.
- fp16 ONNX export (`half=True`) fails in ultralytics 8.4.113 for every YOLO26
  checkpoint; fp32 export works.
