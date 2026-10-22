"""Train a YOLO26 detector on cow_uav_images-5 and score it on the test split.

Takes a stock checkpoint (yolo26m.pt), any model yaml (the scale letter is read
from the filename, e.g. configs/yolo26s-p34.yaml) or a .pt to resume from,
optionally warm-starts from a donor checkpoint, and writes a summary JSON carrying
the test metrics alongside the parameter count and on-disk size, so accuracy and
model size can be compared in one place.

    # the best benchmark model, weights/yolo26m_cowuav_2560.pt
    python train.py --model yolo26m.pt --name yolo26m_2560 --imgsz 2560 --batch 4 --device 0
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

from common import DEFAULT_DATA, RUNS, resolve_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model yaml (scale from filename) or .pt")
    p.add_argument("--name", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset yaml")
    p.add_argument("--init", default=None, help="donor .pt for partial weight transfer")
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--test-imgsz", type=int, default=None)
    args = p.parse_args()
    data = resolve_data(args.data)

    model = YOLO(args.model)
    if args.init:
        before = sum(1 for _ in model.model.state_dict())
        model.load(args.init)
        print(f"warm-started {args.model} from {args.init} ({before} target tensors)")
    print(f"{args.name}: {get_num_params(model.model):,} parameters")

    kwargs = dict(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        cache="ram",
        patience=args.patience,
        seed=0,
        deterministic=False,
        plots=True,
        val=True,
    )
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0
    model.train(**kwargs)

    weights = RUNS / args.name / "weights" / "best.pt"
    best = YOLO(str(weights))
    test_imgsz = args.test_imgsz or args.imgsz
    metrics = best.val(
        data=data,
        split="test",
        imgsz=test_imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS),
        name=f"{args.name}_test",
        exist_ok=True,
        plots=True,
    )

    names = best.model.names
    size = weights.stat().st_size
    summary = {
        "name": args.name,
        "model": args.model,
        "init": args.init,
        "imgsz": args.imgsz,
        "test_imgsz": test_imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "params": get_num_params(best.model),
        "checkpoint_bytes": size,
        "checkpoint_MiB": round(size / 1024**2, 2),
        "test": {
            "mAP50-95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "test_per_class": {
            names[c]: {"mAP50-95": float(metrics.box.maps[c])}
            for c in metrics.box.ap_class_index.tolist()
        },
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
    out = RUNS / f"summary_{args.name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.name} test summary written to {out} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
