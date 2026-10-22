> Research log kept during development, reproduced unchanged. Script and run names refer to
> the original experiment tree: `train_compact.py` is now `train.py`, `cfg/` is `configs/`,
> and `runs/<name>/weights/best.pt` checkpoints that ship here are in `weights/` (see README).

# Making the cow detector smaller

Goal: either (a) get under yolo26s's size while holding its accuracy, or
(b) beat yolo26s's accuracy while staying under yolo26m's size.

Baselines on the held-out test split (201 images / 2469 instances, imgsz 1280):

| model | fused params | best.pt | test mAP50-95 |
|---|---|---|---|
| yolo26s | 9,467,502 | 19.49 MiB | 0.5764 |
| yolo26m | 20,354,078 | 42.11 MiB | 0.6031 |

## Why there is slack: the P5/32 head has no targets

`sqrt(w*h)` of every annotation, measured at imgsz=1280:

| split | median | p95 | max | share ≥ 128 px |
|---|---|---|---|---|
| train | 33.7 px | 74.7 px | 288.7 px | 0.3% |
| valid | 33.9 px | 76.5 px | 573.9 px | 0.3% |
| test | 34.1 px | 73.0 px | **142.6 px** | 0.2% |

YOLO26's third detection branch predicts on a stride-32 grid sized for
256–1280 px objects. On this dataset it is fed 0.2–0.3% of the boxes and, in
the test split, nothing at all above 143 px. Meanwhile the P5 neck (stock
layers 20–22) plus the Detect P5 branch is 27–29% of the parameters.

## Step 1 — delete the P5 branch, keep the trained weights (`prune_p5.py`)

`cfg/yolo26-p34keep.yaml` reproduces stock layers 0–19 verbatim and ends at
`Detect(P3, P4)`. Every trained tensor therefore transfers by name and shape
(568/568 for s, 628/628 for m, 952/952 for l) — no retraining, no fine-tuning.

| model | fused params | best.pt | test mAP50-95 | test mAP50 | inference |
|---|---|---|---|---|---|
| yolo26s | 9,467,502 | 19.49 MiB | 0.5764 | 0.9211 | 3.3 ms |
| **p5drop_yolo26s** | **6,790,692** (−28%) | **13.66 MiB** (−30%) | **0.5764** | 0.9211 | **2.9 ms** |
| yolo26m | 20,354,078 | 42.11 MiB | 0.6031 | 0.9263 | 6.5 ms |
| **p5drop_yolo26m** | **15,484,180** (−24%) | **31.53 MiB** (−25%) | **0.6032** | 0.9263 | **5.9 ms** |
| yolo26l | 24,750,366 | 50.64 MiB | 0.5974 | 0.9272 | 11.9 ms |
| p5drop_yolo26l | 19,880,468 (−20%) | 40.07 MiB (−21%) | 0.5974 | 0.9272 | 7.9 ms |

The metrics are not merely close, they are the same run: on the val split
(426 images) base and pruned agree to five decimals on mAP50-95, mAP50,
precision and recall alike. Per-class test mAP differs by ~1e-5, which is the
fp16 round-trip of the re-saved checkpoint.

Both goals are already met at this point, with no training:

- (a) **p5drop_yolo26s** — 13.66 MiB vs 19.49 MiB, mAP50-95 unchanged at 0.5764.
- (b) **p5drop_yolo26m** — 31.53 MiB vs yolo26m's 42.11 MiB, mAP50-95 0.6032,
  well above yolo26s's 0.5764.

## Step 2 — resolution is the cheapest accuracy, and it costs nothing in size

Retraining the same graph at a higher `imgsz` changes no parameters at all. On a
dataset whose median box is 34 px at 1280 and whose source stills are 4000x3000
or larger, it is the single largest lever, and it had not saturated at 2560.

Test mAP50-95, after the P5 transplant (so every column is one fixed size):

| model | 3.61 MiB (nano) | 13.66 MiB (small) | 31.53 MiB (medium) | 40.07 MiB (large) |
|---|---|---|---|---|
| @1280 | 0.5114 | 0.5764 | 0.6032 | 0.5974 |
| @1536 | 0.5588 | 0.5987 | 0.6178 | 0.6292 |
| @1920 | 0.5872 | 0.6229 | 0.6348 | — |
| @2560 | **0.6198** | **0.6344** | — | — |

The nano column is the striking one: 3.61 MiB goes from 0.5114 to 0.6198 without
adding a single parameter, passing both yolo26s (0.5764 at 19.49 MiB) and
yolo26m (0.6031 at 42.11 MiB) on the way. The cost is compute, not size —
2560 is 4x the pixels of 1280, which `bench_latency.py` quantifies.

The transplant's cost grows with resolution, for a reason the size statistics
predict. Delta = base minus transplanted, test mAP50-95, over all 16 checkpoints:

| imgsz | n | s | m | l |
|---|---|---|---|---|
| 1280 | +0.0007 | 0.0000 | -0.0000 | 0.0000 |
| 1536 | +0.0007 | -0.0001 | -0.0000 | -0.0001 |
| 1920 | +0.0023 | +0.0015 | +0.0001 | — |
| 2560 | +0.0003 | +0.0065 | +0.0002 | — |
| 3200 | +0.0025 | **+0.0178** | — | — |

At 1280 the stride-32 branch is genuinely dead and the transplant is exact. But
the largest test object is 143 px at 1280, which becomes 214 px at 1920, 286 px at
2560 and **357 px at 3200** — by then squarely inside the P5 range, and the branch
is doing real work. For yolo26s at 3200 dropping it costs 0.0178, enough that
`p5drop_yolo26s_3200` (0.6309) falls *below* `p5drop_yolo26s_2560` (0.6344).

So the transplant is free only while the branch is starved. Practically:

- **imgsz ≤ 1920:** free for every scale — take the 24–29% size cut unconditionally.
- **imgsz 2560:** free for n and m, costs 0.0065 for s — still a good trade for −30% size.
- **imgsz 3200:** free for n (0.0025), but not for s (0.0178) — keep P5 at that
  resolution, or accept that 2560 is the better operating point for the s scale.

The nano scale stays nearly free throughout, which is what makes
`p5drop_yolo26n_3200` (3.61 MiB, 0.6349) the headline result.

## Step 3 — retraining a smaller graph loses to transferring weights

Removing the *whole* P5 stage (`cfg/yolo26-p34.yaml`: SPPF/C2PSA move onto P4,
whose C3k2 is deepened) shrinks the graph far more, but the neck no longer
matches stock, so only 38% of the parameters warm-start and the rest must be
learned from 1462 images. `-p34w` sweeps intermediate widths (0.375-0.75) and
`-p234` adds a stride-4 branch for the ~10% of boxes that are only 8-16 px.

| model | fused params | best.pt | test mAP50-95 |
|---|---|---|---|
| p234_m | 11,224,286 | 23.31 MiB | 0.5872 |
| p34_m | 10,958,612 | 23.03 MiB | 0.5857 |
| p234_s | 2,804,366 | 6.53 MiB | 0.5637 |
| p34w_m (width 0.75) | 6,170,708 | 13.17 MiB | 0.5610 |
| p34w_l (width 0.75, deeper) | 7,685,396 | 16.15 MiB | 0.5549 |
| p34_s | 2,737,828 | 6.05 MiB | 0.5504 |
| p34w_s (width 0.625) | 4,272,520 | 9.22 MiB | 0.5105 |
| p34_n | 688,732 | 1.80 MiB | 0.4664 |

Removing the P5 *stage* beats narrowing the whole network at the same budget —
p34_s is 0.038 above yolo26n at a comparable size, and the stride-4 branch adds
another 0.013 (p234_s over p34_s). But every intermediate-width variant is
dominated, and nothing retrained from a partial init reaches the transplanted
models: p34w_s needs 9.22 MiB for 0.5105 while p5drop_yolo26n gets 0.5114 in
3.61 MiB untrained. On 1462 images, keeping trained weights beats a better graph.

Fine-tuning a transplant does not help either: `ft_p5drop_m` (120 epochs,
lr0 1e-3 from p5drop_yolo26m) lands at 0.5971, *below* the 0.6032 it started
from. The transplant is already at the optimum; more gradient steps perturb it.

## Step 4 — distillation buys accuracy at zero size (`train_distill.py`)

Teacher and student both end at `Detect(P3, P4)` with nc=6 and reg_max=1, so their
raw head outputs are shape-identical — boxes (B, 4, A) and scores (B, 6, A) over
the same anchor grid. Predictions distil directly, with no projection layers, so
the student's parameter count is untouched. The loss is soft-BCE on scores plus
teacher-confidence-weighted L1 on boxes, over both the one2many and one2one
branches. A student with extra levels (p234) is trimmed to the teacher's trailing
anchors, since Detect concatenates levels finest-first.

### Teacher count and KD weight, on the 6,790,692-parameter student at 1280

| student | teachers | w_cls | test mAP50-95 | mAP50 | mAP75 |
|---|---|---|---|---|---|
| p5drop_yolo26s | — | — | 0.5764 | 0.9211 | 0.6377 |
| kd_s | 1 | 4 | 0.5866 | 0.9248 | 0.6538 |
| kd_s_w8 | 1 | 8 | 0.5892 | 0.9245 | 0.6553 |
| kd_s_ens | 2 | 4 | 0.5940 | 0.9270 | 0.6543 |
| **kd_s_ens_w8** | **2** | **8** | **0.6027** | 0.9283 | 0.6648 |
| kd_s_ens3_w8 | 3 | 8 | 0.5970 | 0.9281 | 0.6650 |

Every row is the same size. The gain is mostly localisation: mAP75 moves
0.6377 -> 0.6648 while mAP50 moves only 0.9211 -> 0.9283. The two-teacher
ensemble (p5drop_yolo26m 0.6032 + p5drop_yolo26l 0.5974) beats three, because the
third was ft_p5drop_m at 0.5971 and averaging a weaker model into the target
pulls it down — teacher quality matters more than teacher count.

`kd_s_ens_w8` reaches 0.6027 at 13.83 MiB, matching yolo26m's 0.6031 in a third
of the size at yolo26s's resolution.

### The teacher must not be too much larger than the student

Same 1,703,452-parameter nano student, same 1536 resolution, baseline 0.5588:

| teacher | w_cls 4 | w_cls 8 |
|---|---|---|
| p5drop_yolo26s_1536, 0.5987 (4x the student) | **0.5698** | 0.5658 |
| p5drop_yolo26m_1536, 0.6178 (9x the student) | 0.5524 | 0.5608 |

The better teacher is worse at both weights, and at w_cls 4 it drops *below* the
undistilled baseline. The s student (6.8M) gains from an m teacher (15.5M, 2.3x),
but the nano student (1.7M) cannot follow a 9x teacher — a capacity gap, not a
weight artefact, since it holds at both weights. Pick a teacher a few times the
student's size, not the strongest one available.

### Distillation composes with the P5 transplant and with resolution

| model | best.pt | test mAP50-95 |
|---|---|---|
| yolo26n @1280 | 5.25 MiB | 0.5121 |
| + P5 transplant | 3.61 MiB | 0.5114 |
| + 1536 | 3.61 MiB | 0.5588 |
| + distillation | 3.84 MiB | 0.5698 |
| P5 transplant + 1920 | 3.61 MiB | 0.5872 |
| **P5 transplant + 1920 + distillation** | **3.95 MiB** | **0.5942** |

and on the 6.79M student, the same four combinations at that size:

| model | imgsz | best.pt | test mAP50-95 |
|---|---|---|---|
| p5drop_yolo26s | 1280 | 13.66 MiB | 0.5764 |
| kd_s_ens_w8 | 1280 | 13.83 MiB | 0.6027 |
| p5drop_yolo26s_1536 | 1536 | 13.66 MiB | 0.5987 |
| kd_s_1536_w8 | 1536 | 13.89 MiB | 0.6139 |
| **p5drop_yolo26s_1920** | **1920** | **13.66 MiB** | **0.6229** |

Resolution outruns distillation on this dataset: 1920 with no teacher beats 1536
with one. They are additive, so 1920 + distillation is the remaining combination.

Other students: kd_p34s_ens_w8 0.5755 at 6.04 MiB (from p34_s's 0.5504),
kd_p234s_1536 0.5959 at 6.77 MiB with the highest mAP50 of any model (0.9314).

One caveat: distilling a checkpoint that has *already* been distilled does not
help. `kd_p34s_ens` starts from kd_p34s and lands at 0.5624, below its own 0.5672
starting point — the same perturbation seen with `ft_p5drop_m`. Always distil from
the undistilled weights.

## Negative results

**Generic structured pruning.** torch-pruning 1.6 cannot build a dependency graph
for YOLO26. Switching C3k2/C2f to `forward_split` fixes the first `IndexError`
from the hidden `chunk()`, but the split/concat index mapping then fails for every
configuration tried (global and local importance, with and without SPPF pinned).
The end-to-end Detect head consumes each neck tensor twice, through the one2many
and one2one branches, and the yolo26 SPPF adds a residual around its concat.
Architecture surgery driven by the measured target-size distribution is what
worked instead.

**Intermediate widths.** Every `-p34w` variant (width 0.375-0.75) is dominated:
p34w_m needs 13.17 MiB for 0.5610 where p5drop_yolo26s_1536 gets 0.5987 in
13.66 MiB, and p34w_s needs 9.22 MiB for 0.5105 where p5drop_yolo26n reaches
0.5114 in 3.61 MiB untrained.

**fp16 ONNX export.** `half=True` raises "tensor does not have a device" in
ultralytics 8.4.113, for stock yolo26 checkpoints as well as transplanted ones.
fp32 export works; see `runs/export_sizes.json`, where every model lands at
~4 bytes/parameter in ONNX against ~2 in the checkpoint. The size ranking is a
property of the weights, not of checkpoint bookkeeping.

## Latency, measured on an idle GPU (`bench_latency.py`)

The milliseconds in the per-model summary JSONs were recorded while up to ten
training jobs shared the machine and must not be compared. These were taken with
nothing else running, batch-1 for latency and batch-8 for throughput, on one A100.
Batch-1 on an A100 is largely kernel-launch bound for a 2–7 M parameter detector,
so batch-8 is the number that reflects compute.

| model | imgsz | best.pt | mAP50-95 | batch-1 ms | batch-8 ms/img | fps |
|---|---|---|---|---|---|---|
| yolo26s | 1280 | 19.49 MiB | 0.5764 | 16.51 | 5.29 | 189.0 |
| yolo26m | 1280 | 42.11 MiB | 0.6031 | 18.04 | 10.59 | 94.4 |
| yolo26l | 1280 | 50.64 MiB | 0.5974 | 25.24 | 13.51 | 74.0 |
| **kd_s_ens_w8** | 1280 | **13.83 MiB** | **0.6027** | 13.33 | **4.66** | **214.7** |
| p5drop_yolo26s_1920 | 1920 | 13.66 MiB | 0.6229 | 15.27 | 11.09 | 90.2 |
| p5drop_yolo26n_2560 | 2560 | 3.61 MiB | 0.6198 | 16.64 | 11.07 | 90.3 |
| p5drop_yolo26s_2560 | 2560 | 13.66 MiB | 0.6344 | 26.13 | 21.15 | 47.3 |
| **p5drop_yolo26n_3200** | 3200 | **3.61 MiB** | **0.6349** | 28.46 | 19.00 | 52.6 |
| yolo26n_3200 | 3200 | 6.03 MiB | 0.6373 | 35.25 | 24.90 | 40.2 |
| yolo26s_2560 | 2560 | 19.94 MiB | 0.6409 | 31.51 | 26.41 | 37.9 |
| yolo26s_3200 | 3200 | 20.26 MiB | 0.6486 | 56.21 | 48.49 | 20.6 |
| p5drop_yolo26m_2560 | 2560 | 31.53 MiB | 0.6556 | 47.13 | 41.64 | 24.0 |
| yolo26m_2560 | 2560 | 42.55 MiB | 0.6558 | 52.91 | 47.20 | 21.2 |
| p34_n | 1280 | 1.80 MiB | 0.4664 | 11.72 | 3.25 | 307.6 |

This splits the answer into two regimes.

**If compute is fixed at yolo26s's budget**, `kd_s_ens_w8` wins outright: 13.83 MiB
against yolo26s's 19.49, 0.6027 against 0.5764, and 4.66 ms/img against 5.29 —
smaller, more accurate *and* faster. It also matches yolo26m's accuracy (0.6031)
at 2.3x yolo26m's throughput and a third of its size.

**If size is the binding constraint and compute is available**,
`p5drop_yolo26n_3200` gives 0.6349 in 3.61 MiB — 5.4x smaller than yolo26s and
0.0585 more accurate — but at 19.00 ms/img it is 3.6x slower than yolo26s. The
resolution levers buy accuracy with compute, not with parameters, and that
trade is only worth making when the deployment is size-bound rather than
throughput-bound.

## Measurement caveat

Inference milliseconds inside `runs/summary_*.json` were recorded under load and
are not comparable across runs; quote `runs/latency_idle_gpu.json` instead.
