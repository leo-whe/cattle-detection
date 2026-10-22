> Research log kept during development, reproduced unchanged. Script and run names refer to
> the original experiment tree: `train_compact.py` is now `train.py`, `cfg/` is `configs/`,
> and `runs/<name>/weights/best.pt` checkpoints that ship here are in `weights/` (see README).

# Closing the Nadir Detection Gap

The deployed detector found 1 of 46 cattle on low-altitude straight-down frames. A
detector finetuned on teacher pseudo-labels finds 45 of 46, at 35 FPS.

## The gap

On strict nadir frames shot from low altitude, an animal is a bare back: no legs, no
side profile, often blending with dry grass. Four detectors were measured on the same
frames, and three of them fail identically:

| | herd frame 1787 | nadir frame 10501 |
|---|---|---|
| COCO `yolo26m` (`cow`) | 51 detections | **1** |
| YOLOE open-vocabulary | 48 | **0** |
| `runs/yolo26m_cow_uav` | ~0 | **0** |
| LocateAnything-3B | 46 | **46** |

`cow_uav` fails here despite being trained on nadir imagery, because its training data
is *high-altitude* nadir where an animal is a 15 px speck. Same viewpoint, different
scale regime.

Fusing COCO with `cow_uav` was implemented and measured (`sweep_detectors.py`): at
matched per-model input scales it added 2.6 boxes per frame, and every one of those
was a badly-localised duplicate on an animal already detected. The two models' blind
spots overlap, so fusion has nothing to contribute.

## Ground truth

Two frames were annotated, one per domain, so a change could be checked for gain in
the failing domain *and* damage in the working one.

| frame | domain | cattle | how |
|---|---|---|---|
| 10501 | low-altitude nadir | **46** | teacher proposals, every box inspected as a crop, recall checked on magnified tiles |
| 1787 | oblique herd | **51** | union of multi-scale COCO and the teacher, 54 proposals, 3 rejected (two bushes, one trough rail at conf 0.66) |

Both are single-frame, single-annotator, and inherit whatever the proposal sources
missed. They are in `runs/texas_gt/`.

A note on reading nadir footage: at this sun angle every animal throws a cow-shaped
shadow, and at thumbnail zoom the shadows are indistinguishable from animals. An early
pass over these frames wrongly counted shadows as teacher misses. Verification has to
be done zoomed in far enough to see grass texture through the shadow.

## Pseudo-labels

LocateAnything-3B is 2.2-7.9 s/frame, far too slow to deploy, which is exactly the
teacher role the methods plan gives it. 840 frames were sampled in strata by how many
boxes the deployed detector already found, so the set is weighted towards the failure
regime without dropping the regime that works.

```
840 frames, 17,567 boxes, 1.3 % with no boxes
v0056  frames where deployed detector found <5:  mean 1.0 -> teacher 9.9
v0057  same stratum:                             mean 2.1 -> teacher 3.1
```

v0057's low-count frames are mostly genuinely sparse rather than a detection failure,
with one real gap pocket around frame 6800-6900. The recovered gap comes mainly from
v0056.

Split into train and val by contiguous 900-frame blocks, never randomly: at 59.94 fps
frames a fraction of a second apart are near-identical, and a random split would put
near-duplicates of training frames into validation.

## Result

Finetuned `yolo26m` from COCO weights, 60 epochs, imgsz 1920, AdamW lr0 0.002, nadir
augmentation (`degrees=180`, `flipud=0.5`). 1.0 h on one A100.

Scored against the hand-verified frames, not against the teacher:

| | nadir 10501 (46) | herd 1787 (51) |
|---|---|---|
| COCO best | recall **0.022** | recall 0.922, prec 0.940 |
| finetuned @1920 | recall **0.978**, prec 0.938 | recall **0.941**, prec 0.923 |
| finetuned @2560 | recall 0.978, prec 0.957 | recall 0.902, prec 1.000 |

Recall in the failing domain went from 0.022 to 0.978, and the working domain was not
damaged: at imgsz 1920 it is slightly better than COCO's best there.

It generalises past the annotated frame. On the five frames that failed at every input
size, counts now match or slightly exceed the teacher:

| frame | COCO@2560 | teacher | finetuned@1920 |
|---|---|---|---|
| 5361 | 1 | 3 | 4 |
| 7000 | 0 | 8 | 8 |
| 8935 | 2 | 3 | 5 |
| 9500 | 2 | 26 | 27 |
| 10500 | 1 | 46 | 47 |

And it is deployable, which the teacher is not:

| | ms/frame | FPS |
|---|---|---|
| LocateAnything-3B | 2200-7900 | 0.13-0.45 |
| finetuned @1920 | **28.2** | **35.4** |
| finetuned @2560 | 52.3 | 19.1 |

**Recommended operating point: `runs/cattle_nadir_m/weights/best.pt`, imgsz 1920,
conf 0.25.** It is the only setting that is near-best in both domains at once.

## Limits

The training data is two flights, one ranch, one day, one sun angle. This model should
be expected to work on this footage and to generalise poorly elsewhere, including to
the New Zealand nadir segments, which have different ground cover and all-black cattle.
Extending it means running the teacher over those flights too.

The teacher-agreement metric printed during training (mAP50 0.983) measures imitation,
not accuracy, and is not evidence the model is right. Only the two verified frames are.

The herd-domain labels are slightly incomplete: on frame 1787 the teacher found 46 of
51 with perfect precision, while COCO found 50 of 51 with 3 false positives. In domains
where the deployed detector already works, pseudo-labelling from the union of both
would supervise more completely than the teacher alone. Not done here, to keep one
change per iteration.

## Reproduce

```bash
conda activate locateanything
bash run_pseudo_label.sh                      # teacher labels, ~50 min on 2 GPUs
conda activate yolo26
python split_pseudo_dataset.py --dataset datasets/texas_pseudo
python train_cattle_nadir.py --imgsz 1920 --epochs 60 --batch 12 --device 0
python eval_detectors.py \
  --gt-json runs/texas_gt/frame10501_gt.json \
  --gt-boxes runs/texas_gt/frame10501_gt_boxes.npy \
  --image runs/texas_gt/frame10501.jpg \
  --models coco=yolo26m.pt:cow finetuned=runs/cattle_nadir_m/weights/best.pt:cattle
```
