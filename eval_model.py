"""Evaluate a checkpoint on the cow_uav_images-5 test split and write a summary JSON.

    python eval_model.py --weights weights/yolo26m_p5drop_cowuav_2560.pt \
        --name yolo26m_p5drop_2560 --imgsz 2560 --device 0
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

from common import DEFAULT_DATA, RUNS, resolve_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=resolve_data(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS),
        name=f"{args.name}_{args.split}",
        exist_ok=True,
        plots=False,
    )

    names = model.model.names
    size = Path(args.weights).stat().st_size
    summary = {
        "name": args.name,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "params": get_num_params(model.model),
        "checkpoint_bytes": size,
        "checkpoint_MiB": round(size / 1024**2, 2),
        args.split: {
            "mAP50-95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        f"{args.split}_per_class": {
            names[c]: {"mAP50-95": float(metrics.box.maps[c])}
            for c in metrics.box.ap_class_index.tolist()
        },
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
    out = RUNS / f"summary_{args.name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.name} -> {out} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
