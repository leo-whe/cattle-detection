"""Split the teacher-labelled frames into train and val by contiguous time blocks.

A random split would be dishonest here. The source is 59.94 fps, so frames a fraction
of a second apart are nearly identical images; scattering them across train and val
puts near-duplicates of training frames into validation and inflates every metric.

Frames are therefore grouped into blocks of consecutive video time, and whole blocks
are assigned to one split or the other, so any val frame is separated from every train
frame by at least a block boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--block-frames", type=int, default=900, help="Block size, 900 = 15 s at 60 fps")
    parser.add_argument("--val-every", type=int, default=6, help="Assign one block in N to val")
    parser.add_argument("--class-name", default="cattle")
    parser.add_argument("--min-spacing", type=int, default=0,
                        help="Drop train frames closer than this many frames to the previous kept "
                             "one. The failure regions are short, so sampling them densely is the "
                             "only way to get many examples; set this only to trade count for "
                             "diversity. Val is never thinned, so the metric stays comparable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_root = args.dataset / "images"
    lbl_root = args.dataset / "labels"
    staged = sorted((img_root / "train").glob("*.jpg"))
    if not staged:
        raise SystemExit(f"No images under {img_root/'train'}")

    # file names are "<tag>_<zero-padded frame>.jpg"
    per_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in staged:
        m = re.match(r"(.+)_(\d{7})$", path.stem)
        if not m:
            continue
        per_video[m.group(1)].append((int(m.group(2)), path))

    (img_root / "val").mkdir(parents=True, exist_ok=True)
    (lbl_root / "val").mkdir(parents=True, exist_ok=True)

    moved = kept = 0
    blocks_val: list[str] = []
    for tag, items in per_video.items():
        items.sort()
        blocks = sorted({frame // args.block_frames for frame, _ in items})
        val_blocks = {b for i, b in enumerate(blocks) if i % args.val_every == args.val_every - 1}
        blocks_val += [f"{tag}:{b}" for b in sorted(val_blocks)]
        for frame, path in items:
            if frame // args.block_frames in val_blocks:
                shutil.move(str(path), img_root / "val" / path.name)
                lab = lbl_root / "train" / f"{path.stem}.txt"
                if lab.is_file():
                    shutil.move(str(lab), lbl_root / "val" / lab.name)
                moved += 1
            else:
                kept += 1

    thinned = 0
    if args.min_spacing > 0:
        drop_dir = args.dataset / "thinned"
        (drop_dir / "images").mkdir(parents=True, exist_ok=True)
        (drop_dir / "labels").mkdir(parents=True, exist_ok=True)
        for tag, items in per_video.items():
            last = -10**9
            for frame, path in sorted(items):
                live = img_root / "train" / path.name
                if not live.is_file():
                    continue  # already moved to val
                if frame - last < args.min_spacing:
                    shutil.move(str(live), drop_dir / "images" / path.name)
                    lab = lbl_root / "train" / f"{path.stem}.txt"
                    if lab.is_file():
                        shutil.move(str(lab), drop_dir / "labels" / lab.name)
                    thinned += 1
                else:
                    last = frame
        kept -= thinned

    def count_boxes(split: str) -> int:
        return sum(sum(1 for _ in p.open()) for p in (lbl_root / split).glob("*.txt"))

    yaml_path = args.dataset / "data.yaml"
    yaml_path.write_text(
        f"# Teacher-labelled cattle frames (LocateAnything-3B), split by contiguous time blocks.\n"
        f"path: {args.dataset.resolve()}\n"
        f"train: images/train\nval: images/val\n\nnc: 1\nnames: ['{args.class_name}']\n"
    )
    summary = {
        "train_images": kept, "val_images": moved, "train_images_thinned_out": thinned,
        "min_spacing": args.min_spacing,
        "train_boxes": count_boxes("train"), "val_boxes": count_boxes("val"),
        "block_frames": args.block_frames, "val_every": args.val_every,
        "val_blocks": blocks_val, "data_yaml": str(yaml_path),
    }
    (args.dataset / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "val_blocks"}, indent=2))


if __name__ == "__main__":
    main()
