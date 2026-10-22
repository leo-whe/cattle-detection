"""Time the compaction shortlist on an idle GPU.

The inference milliseconds recorded in runs/summary_*.json were measured while up
to ten training jobs shared the machine, so they must not be compared across runs.
This times warmed-up forward passes at each model's own imgsz, which for yolo26
includes the end-to-end head's postprocess (Detect does it inside forward when not
training).

Two numbers, because they measure different things at this model scale. Batch-1
latency on an A100 is largely kernel-launch bound for a 2-7M parameter detector,
so it is sensitive to CPU load and compresses the differences between models;
batch-8 throughput is GPU bound and reflects the actual compute. Run with nothing
else on the machine -- CPU contention alone was enough to make yolo26m look faster
than yolo26s at batch 1.

    python bench_latency.py --device 0 --iters 200
    python bench_latency.py --models mine=runs/x/weights/best.pt:1920
"""

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

from common import RUNS, WEIGHTS

# The shipped checkpoints, each at the input size it was trained and scored at.
SHIPPED = {
    "cattle_nadir_m": ("cattle_nadir_m.pt", 1920),
    "yolo26m_cowuav_2560": ("yolo26m_cowuav_2560.pt", 2560),
    "yolo26m_p5drop_cowuav_2560": ("yolo26m_p5drop_cowuav_2560.pt", 2560),
    "yolo26s_kd_cowuav_1280": ("yolo26s_kd_cowuav_1280.pt", 1280),
    "yolo26n_p5drop_cowuav_3200": ("yolo26n_p5drop_cowuav_3200.pt", 3200),
}


def time_forward(model, imgsz: int, dev: str, batch: int, iters: int, warmup: int) -> float:
    """Mean wall-clock milliseconds per forward pass."""
    x = torch.randn(batch, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def bench(weights: Path, imgsz: int, device: str, iters: int, warmup: int) -> dict:
    dev = f"cuda:{device}" if device.isdigit() else device
    model = YOLO(str(weights)).model.float().to(dev).eval()
    ms1 = time_forward(model, imgsz, dev, 1, iters, warmup)
    ms8 = time_forward(model, imgsz, dev, 8, max(iters // 4, 10), max(warmup // 2, 5))
    return {"params": get_num_params(model), "imgsz": imgsz,
            "checkpoint_MiB": round(weights.stat().st_size / 1024**2, 2),
            "batch1_ms": round(ms1, 2), "batch1_fps": round(1000 / ms1, 1),
            "batch8_img_ms": round(ms8 / 8, 2), "batch8_fps": round(8000 / ms8, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0", help="CUDA index or a torch device string")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--models", nargs="+", default=None,
                   help="name=path/to.pt:imgsz entries; default is every shipped checkpoint")
    p.add_argument("--out", type=Path, default=RUNS / "latency_idle_gpu.json")
    args = p.parse_args()

    if args.models:
        shortlist = {}
        for spec in args.models:
            name, _, rest = spec.partition("=")
            path, _, size = rest.rpartition(":")
            shortlist[name] = (Path(path), int(size))
    else:
        shortlist = {name: (WEIGHTS / f, size) for name, (f, size) in SHIPPED.items()}

    out = {}
    for name, (w, imgsz) in shortlist.items():
        row = bench(w, imgsz, args.device, args.iters, args.warmup)
        out[name] = row
        print(f"{name:28s} {row['checkpoint_MiB']:6.2f} MiB  {row['params']:>10,d} p  "
              f"imgsz={imgsz:5d}  b1 {row['batch1_ms']:6.2f} ms  "
              f"b8 {row['batch8_img_ms']:6.2f} ms/img  {row['batch8_fps']:6.1f} fps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
