"""Score detectors against a hand-verified frame, not against the teacher.

Finetuning on teacher pseudo-labels makes the student converge towards the teacher, so
scoring it on teacher output measures imitation and hides whatever the teacher got
wrong. The reference here is a frame whose every box was inspected individually and
whose recall was checked on magnified tiles, so it is independent of both.

Reports recall, precision and count error per model and per input size, plus recall
restricted to the smallest ground-truth boxes, which is where input resolution bites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--gt-boxes", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Each as name=path/to.pt or name=path/to.pt:classname")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1280, 1920, 2560])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def score(det: np.ndarray, gt: np.ndarray, thr: float, small: np.ndarray) -> dict:
    if len(det) == 0:
        return {"n": 0, "tp": 0, "recall": 0.0, "precision": 0.0, "small_recall": 0.0}
    m = iou_matrix(det[:, :4], gt)
    taken: set[int] = set()
    hit = np.zeros(len(gt), bool)
    for di in np.argsort(-det[:, 4]):
        row = m[di].copy()
        for j in taken:
            row[j] = -1
        j = int(np.argmax(row))
        if row[j] >= thr:
            taken.add(j)
            hit[j] = True
    tp = int(hit.sum())
    return {"n": len(det), "tp": tp, "recall": tp / len(gt), "precision": tp / len(det),
            "small_recall": float(hit[small].sum() / max(1, small.sum()))}


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    meta = json.loads(args.gt_json.read_text())
    gt = np.load(args.gt_boxes).astype(np.float32)
    side = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
    small = side <= np.percentile(side, 33)
    image = cv2.imread(str(args.image))
    print(f"ground truth: {len(gt)} cattle in {args.image.name} "
          f"(median box side {np.median(side):.0f} px)\n")

    rows = []
    print(f"{'model':<22} {'imgsz':>6} {'nDet':>5} {'Recall':>7} {'SmallR':>7} {'Prec':>7} {'CountErr':>9}")
    for spec in args.models:
        name, _, rest = spec.partition("=")
        path, _, cls = rest.partition(":")
        model = YOLO(path)
        wanted = {cls.lower()} if cls else {"cow", "cattle", "cowrotation"}
        ids = [i for i, n in model.names.items() if n.lower() in wanted] or None
        for size in args.imgsz:
            r = model.predict(image, imgsz=size, conf=args.conf, iou=0.6,
                              classes=ids, device=args.device, verbose=False)[0]
            b = r.boxes
            det = (np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32)
                   if b is not None and len(b) else np.zeros((0, 5), np.float32))
            s = score(det, gt, args.iou_match, small)
            s.update({"model": name, "imgsz": size, "count_err": s["n"] - len(gt)})
            rows.append(s)
            print(f"{name:<22} {size:>6} {s['n']:>5} {s['recall']:>7.3f} {s['small_recall']:>7.3f} "
                  f"{s['precision']:>7.3f} {s['count_err']:>+9d}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"gt": meta, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
