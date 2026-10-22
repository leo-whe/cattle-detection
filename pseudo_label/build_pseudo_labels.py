"""Label video frames with LocateAnything-3B to cover the detector's blind spot.

The COCO `cow` class, an open-vocabulary detector (YOLOE) and the repository's
`cow_uav` checkpoint all return ~0 on low-altitude straight-down frames, where an
animal is seen as a bare back with no legs and no side profile. LocateAnything-3B
does fire there: on one frame the three fast detectors gave 3, 0 and 0 while
LocateAnything gave 46, and the boxes were visually confirmed to sit on real cattle.

It is far too slow to deploy (2-8 s/frame), so it is used here the way the methods
plan proposes: as a teacher that writes pseudo-labels for finetuning a fast detector.

Frames are sampled in strata by how many boxes the deployed detector already finds,
so the training set is weighted towards the failure regime without discarding the
regime that already works.

Writes a YOLO detection dataset (single class `cattle`) plus a manifest holding the
teacher's raw answer per frame, so any label can be audited later.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-counts", type=Path, default=None,
                        help="Existing *_frame_counts.csv used to stratify sampling")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--n-fail", type=int, default=200, help="Frames where the deployed detector finds <5")
    parser.add_argument("--n-mid", type=int, default=80, help="Frames with 5-19")
    parser.add_argument("--n-ok", type=int, default=120, help="Frames with >=20")
    parser.add_argument("--frame-range", type=int, nargs=2, default=None,
                        help="Restrict sampling to [start, end); use to hold out a val block")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "LocateAnything-3B",
                        help="LocateAnything-3B checkpoint directory")
    parser.add_argument("--eagle-dir", type=Path, default=ROOT / "third_party" / "Eagle" / "Embodied",
                        help="NVlabs/Eagle Embodied/ checkout providing locateanything_worker")
    parser.add_argument("--prompt", default="cattle")
    parser.add_argument("--mode", default="hybrid", choices=["fast", "hybrid", "slow"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-box-px", type=int, default=12, help="Drop teacher boxes smaller than this")
    parser.add_argument("--tag", default="", help="Prefix for output file names")
    return parser.parse_args()


def stratified_frames(counts_csv: Path | None, total_frames: int, args) -> list[int]:
    if counts_csv is None or not counts_csv.is_file():
        n = args.n_fail + args.n_mid + args.n_ok
        return list(np.linspace(0, total_frames - 1, n, dtype=int))
    active: dict[int, int] = {}
    with counts_csv.open() as handle:
        for row in csv.DictReader(handle):
            active[int(row["frame"])] = int(row["active_tracks"])
    lo, hi = args.frame_range if args.frame_range else (0, total_frames)
    pool = {f: v for f, v in active.items() if lo <= f < hi}
    strata = {
        "fail": ([f for f, v in pool.items() if v < 5], args.n_fail),
        "mid": ([f for f, v in pool.items() if 5 <= v < 20], args.n_mid),
        "ok": ([f for f, v in pool.items() if v >= 20], args.n_ok),
    }
    chosen: list[int] = []
    for name, (frames, want) in strata.items():
        frames.sort()
        if not frames:
            print(f"  stratum {name}: empty")
            continue
        take = min(want, len(frames))
        # even spacing keeps consecutive near-duplicate frames out of the sample
        chosen += [frames[i] for i in np.linspace(0, len(frames) - 1, take, dtype=int)]
        print(f"  stratum {name}: {len(frames)} available, taking {take}")
    return sorted(set(chosen))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.eagle_dir))
    sys.path.insert(0, str(args.model))
    from locateanything_worker import LocateAnythingWorker

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video.name}: {total} frames")
    frames = stratified_frames(args.frame_counts, total, args)
    print(f"sampling {len(frames)} frames")

    img_dir = args.out_dir / "images" / args.split
    lbl_dir = args.out_dir / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    worker = LocateAnythingWorker(
        str(args.model), device=args.device, use_batch_runtime=True,
        attn="la_flash", vision_attn="flash_attention_2", scheduler="pipeline", strict_attn=True,
    )

    tag = args.tag or args.video.stem[:24]
    manifest = []
    started = time.perf_counter()
    for i, frame_index in enumerate(frames, 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            continue
        height, width = bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = worker.detect(image, [args.prompt], generation_mode=args.mode,
                               max_new_tokens=args.max_new_tokens, verbose=False)
        answer = result["answer"]
        boxes = [
            b for b in worker.parse_boxes(answer, width, height)
            if 0 <= b["x1"] < b["x2"] <= width and 0 <= b["y1"] < b["y2"] <= height
            and (b["x2"] - b["x1"]) >= args.min_box_px and (b["y2"] - b["y1"]) >= args.min_box_px
        ]
        stem = f"{tag}_{frame_index:07d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        with (lbl_dir / f"{stem}.txt").open("w") as handle:
            for b in boxes:
                cx = (b["x1"] + b["x2"]) / 2 / width
                cy = (b["y1"] + b["y2"]) / 2 / height
                bw = (b["x2"] - b["x1"]) / width
                bh = (b["y2"] - b["y1"]) / height
                handle.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        manifest.append({"frame": frame_index, "stem": stem, "n_boxes": len(boxes),
                         "raw_answer": answer[:4000]})
        if i % 20 == 0:
            rate = (time.perf_counter() - started) / i
            left = rate * (len(frames) - i) / 60
            print(f"  {i}/{len(frames)} frames, {rate:.2f}s/frame, ~{left:.0f} min left", flush=True)
    capture.release()

    out = args.out_dir / f"manifest_{tag}_{args.split}.json"
    out.write_text(json.dumps({
        "video": str(args.video), "split": args.split, "prompt": args.prompt,
        "teacher": "LocateAnything-3B", "frames": len(manifest),
        "boxes_total": sum(m["n_boxes"] for m in manifest),
        "seconds": time.perf_counter() - started, "items": manifest,
    }, indent=2) + "\n")
    print(f"done: {len(manifest)} frames, {sum(m['n_boxes'] for m in manifest)} boxes -> {out}")


if __name__ == "__main__":
    main()
