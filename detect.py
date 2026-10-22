"""Detect cattle in images, a folder of images, or a video.

Every shipped checkpoint is reduced to one ``cattle`` class: the cow_uav models
carry six Roboflow classes, of which ``0``, ``cattle``, ``cow`` and
``cowrotation`` are all cattle, and the rest are dropped. Each checkpoint defaults
to the input size it was trained and scored at.

Writes one YOLO-format label file per image (``0 cx cy w h conf``, normalised),
a ``counts.csv`` with the number of cattle per image or video frame, and with
``--save-vis`` annotated images or an annotated video.

    python detect.py --source frames/ --out runs/detect/frames --device 0
    python detect.py --source flight.MP4 --weights weights/cattle_nadir_m.pt --save-vis
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from common import RUNS, WEIGHTS

CATTLE_NAMES = {"0", "cattle", "cow", "cowrotation"}
DEFAULT_IMGSZ = {
    "cattle_nadir_m.pt": 1920,
    "yolo26m_cowuav_2560.pt": 2560,
    "yolo26m_p5drop_cowuav_2560.pt": 2560,
    "yolo26s_kd_cowuav_1280.pt": 1280,
    "yolo26n_p5drop_cowuav_3200.pt": 3200,
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Image, directory of images, or video")
    parser.add_argument("--weights", type=Path, default=WEIGHTS / "cattle_nadir_m.pt")
    parser.add_argument("--imgsz", type=int, default=None, help="Default: the checkpoint's own size, else 1920")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names to keep; default is every cattle class")
    parser.add_argument("--device", default=None, help="For example 0 or cpu")
    parser.add_argument("--out", type=Path, default=RUNS / "detect")
    parser.add_argument("--save-vis", action="store_true", help="Also write annotated images / video")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth video frame")
    return parser.parse_args()


def draw(image, boxes) -> None:
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
        cv2.putText(image, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"cattle: {len(boxes)}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    model = YOLO(str(args.weights))
    imgsz = args.imgsz or DEFAULT_IMGSZ.get(args.weights.name, 1920)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    wanted = ({c.strip() for c in args.classes.split(",")} if args.classes else CATTLE_NAMES)
    class_ids = [i for i, n in names.items() if n in wanted]
    if not class_ids:
        raise ValueError(f"No class of {sorted(wanted)} in model classes {names}")

    is_video = args.source.is_file() and args.source.suffix.lower() in VIDEO_SUFFIXES
    label_dir = args.out / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out / "vis"
    if args.save_vis and not is_video:
        vis_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = []
    results = model.predict(source=str(args.source), imgsz=imgsz, conf=args.conf, iou=args.iou,
                            classes=class_ids, device=args.device, stream=True, verbose=False,
                            vid_stride=args.vid_stride)
    for index, result in enumerate(results):
        h, w = result.orig_shape
        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        boxes = [(*b, float(c)) for b, c in zip(xyxy, confs)]

        key = f"frame_{index * args.vid_stride:07d}" if is_video else Path(result.path).stem
        with (label_dir / f"{key}.txt").open("w") as handle:
            for x1, y1, x2, y2, conf in boxes:
                handle.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {conf:.4f}\n")
        rows.append((key, len(boxes)))

        if args.save_vis:
            image = result.orig_img.copy()
            draw(image, boxes)
            if is_video:
                if writer is None:
                    fps = cv2.VideoCapture(str(args.source)).get(cv2.CAP_PROP_FPS) / args.vid_stride
                    writer = cv2.VideoWriter(str(args.out / f"{args.source.stem}_detect.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(image)
            else:
                cv2.imwrite(str(vis_dir / f"{key}.jpg"), image)

    if writer is not None:
        writer.release()
    with (args.out / "counts.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows([("frame" if is_video else "image", "cattle"), *rows])
    total = sum(n for _, n in rows)
    print(f"{len(rows)} {'frames' if is_video else 'images'}, {total} detections "
          f"(imgsz {imgsz}, conf {args.conf}) -> {args.out}")


if __name__ == "__main__":
    main()
