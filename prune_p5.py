"""Drop the unused P5/32 detection branch from a trained YOLO26 cow checkpoint.

On cow_uav_images-5 the stride-32 head covers 256-1280 px objects while 99.7% of
boxes are under 128 px at imgsz=1280 (the test split tops out at 143 px), so the
branch costs parameters and predicts nothing. configs/yolo26-p34keep.yaml keeps stock
layers 0-19 verbatim, which lets every trained weight transfer by name; only the
stock P5 neck (layers 20-22) and the Detect P5 branch are discarded.

    python prune_p5.py --src runs/yolo26m_2560/weights/best.pt --scale m \
        --out runs/p5drop_yolo26m_2560/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.torch_utils import get_num_params

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs" / "yolo26-p34keep.yaml"
STOCK_DETECT, NEW_DETECT = 23, 20
DROPPED_BRANCH = "2"  # third Detect branch == P5


def remap(src_sd: dict, dst_sd: dict) -> tuple[dict, list[str]]:
    """Rename trained keys onto the pruned graph, skipping the P5 tensors."""
    out, missing = {}, []
    for key in dst_sd:
        if key.startswith(f"model.{NEW_DETECT}."):
            src_key = key.replace(f"model.{NEW_DETECT}.", f"model.{STOCK_DETECT}.", 1)
        else:
            src_key = key
        if src_key in src_sd and src_sd[src_key].shape == dst_sd[key].shape:
            out[key] = src_sd[src_key]
        else:
            missing.append(key)
    return out, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--scale", required=True, choices=["n", "s", "m", "l"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    src_model = (ckpt.get("ema") or ckpt["model"]).float()
    src_sd = src_model.state_dict()

    cfg = yaml_model_load(str(CFG))
    cfg["scale"] = args.scale
    cfg["nc"] = src_model.nc
    dst = DetectionModel(cfg, ch=3, nc=src_model.nc, verbose=False)
    dst_sd = dst.state_dict()

    transferred, missing = remap(src_sd, dst_sd)
    dst.load_state_dict(transferred, strict=False)
    dst.names = src_model.names
    dst.args = getattr(src_model, "args", None)
    dst.eval()

    dropped = [k for k in src_sd if k not in transferred and not k.startswith(f"model.{STOCK_DETECT}.")]
    print(f"transferred {len(transferred)}/{len(dst_sd)} tensors; "
          f"{len(missing)} left at init; {len(dropped)} source tensors discarded")
    if missing:
        print("  not transferred:", missing[:8], "..." if len(missing) > 8 else "")
    print(f"params {get_num_params(src_model):,} -> {get_num_params(dst):,} "
          f"({100 * get_num_params(dst) / get_num_params(src_model):.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": dst.half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "epoch": -1,
        "best_fitness": None,
        "train_args": ckpt.get("train_args", {}),
        "train_metrics": {},
        "train_results": None,
        "date": ckpt.get("date"),
        "version": ckpt.get("version"),
        "license": ckpt.get("license"),
        "docs": ckpt.get("docs"),
        "pruned_from": str(args.src),
    }, out)
    print(f"wrote {out} ({out.stat().st_size:,} B = {out.stat().st_size / 1024**2:.1f} MiB)")
    print(json.dumps({"src": args.src, "out": str(out),
                      "params": get_num_params(dst),
                      "bytes": out.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
