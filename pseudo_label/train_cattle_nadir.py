"""Finetune a fast detector on teacher-labelled frames to cover the nadir blind spot.

Starts from the COCO-pretrained checkpoint rather than from scratch: the pseudo-label
set is small, and the COCO features already handle the oblique and moderate-altitude
views that the deployed detector gets right. The training set deliberately mixes those
frames with the low-altitude nadir frames it gets wrong, so the result should cover
both rather than trade one for the other.

Validation uses the held-out time blocks, which measures agreement with the teacher.
That is a sanity check, not the real score: the honest score comes from
`eval_detectors.py` against the hand-verified frame, since a student trained on teacher
output will agree with the teacher whether or not the teacher was right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import resolve_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "texas_pseudo.yaml")
    parser.add_argument("--weights", default="yolo26m.pt",
                        help="Start checkpoint; the bare name downloads COCO yolo26m")
    parser.add_argument("--name", default="cattle_nadir_m")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.002, help="Low: finetuning, not training from scratch")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"dataset yaml missing: {args.data}")
    data = resolve_data(args.data)
    model = YOLO(str(args.weights))
    model.train(
        data=data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, project=str(args.project), name=args.name,
        patience=args.patience, lr0=args.lr0, optimizer="AdamW", cos_lr=True,
        # the animals are seen from directly above, so orientation carries no information
        degrees=180.0, fliplr=0.5, flipud=0.5, scale=0.5, mosaic=1.0, close_mosaic=10,
        # colour varies a lot in this herd (black, white, red, brindle); keep it mild
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        val=True, plots=True, seed=0,
    )
    metrics = model.val(data=data, imgsz=args.imgsz, device=args.device)
    print(f"\nagreement with teacher on held-out blocks: "
          f"mAP50 {metrics.box.map50:.4f}  mAP50-95 {metrics.box.map:.4f}  "
          f"P {metrics.box.mp:.4f}  R {metrics.box.mr:.4f}")
    print("This measures imitation of the teacher. Score the model with eval_detectors.py.")


if __name__ == "__main__":
    main()
