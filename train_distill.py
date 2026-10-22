"""Distil a large cow detector into a compact one without adding parameters.

Both p5drop_yolo26m (teacher) and the p34keep students end at Detect(P3, P4)
with nc=6 and reg_max=1, so their raw head outputs are shape-identical --
boxes (B, 4, A) and scores (B, 6, A) over the same anchor grid. Predictions can
therefore be distilled directly, with no projection layers, which keeps the
student's parameter count and checkpoint size exactly as they were.

    # weights/yolo26s_kd_cowuav_1280.pt: two-teacher ensemble, w_cls 8
    python train_distill.py --model configs/yolo26s-p34keep.yaml \
        --teacher runs/p5drop_yolo26m/weights/best.pt,runs/p5drop_yolo26l/weights/best.pt \
        --init runs/p5drop_yolo26s/weights/best.pt \
        --name kd_s_ens_w8 --device 0 --w-cls 8 --w-box 4 --epochs 200 --lr0 0.0005
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import get_num_params

from common import DEFAULT_DATA, RUNS, resolve_data

BRANCHES = ("one2many", "one2one")


def merge_teachers(preds: list[dict], branch: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Average an ensemble's targets: probabilities directly, boxes by confidence.

    Every teacher ends at Detect(P3, P4) with the same nc, reg_max and strides,
    so all of them score the identical anchor grid and can be pooled elementwise.
    """
    probs = [p[branch]["scores"].detach().sigmoid() for p in preds]
    boxes = [p[branch]["boxes"].detach() for p in preds]
    prob = torch.stack(probs).mean(0)
    if len(preds) == 1:
        return prob, boxes[0]
    # A teacher that sees nothing at an anchor should not drag the box target.
    w = torch.stack([p.amax(1, keepdim=True) for p in probs])
    box = (torch.stack(boxes) * w).sum(0) / w.sum(0).clamp(min=1e-6)
    return prob, box


def align(s_scores: torch.Tensor, s_boxes: torch.Tensor, n_teacher: int):
    """Trim a student with extra detection levels down to the teacher's anchors.

    Detect concatenates levels finest-first, so a Detect(P2, P3, P4) student and a
    Detect(P3, P4) teacher share the trailing anchors exactly. The student's P2
    predictions then learn from the detection loss alone.
    """
    n = s_scores.shape[-1]
    if n == n_teacher:
        return s_scores, s_boxes
    if n < n_teacher:
        raise ValueError(f"student has fewer anchors ({n}) than the teacher ({n_teacher})")
    return s_scores[..., n - n_teacher:], s_boxes[..., n - n_teacher:]


def kd_loss(student: dict, teachers: list[dict], w_cls: float, w_box: float) -> torch.Tensor:
    """Confidence-weighted prediction distillation summed over both head branches."""
    total = student[BRANCHES[0]]["scores"].sum() * 0.0
    for branch in BRANCHES:
        s = student[branch]
        t_prob, t_box = merge_teachers(teachers, branch)
        s_scores, s_boxes = align(s["scores"], s["boxes"], t_prob.shape[-1])
        s = {"scores": s_scores, "boxes": s_boxes}
        cls = F.binary_cross_entropy_with_logits(s["scores"], t_prob, reduction="mean")
        # Only mimic geometry where the teacher actually believes there is an object.
        conf = t_prob.amax(1, keepdim=True)
        box = ((s["boxes"] - t_box).abs() * conf).sum() / conf.sum().clamp(min=1e-6) / 4
        total = total + w_cls * cls + w_box * box
    return total


def make_trainer(teacher_paths: list[str], w_cls: float, w_box: float, init: str | None):
    class DistillTrainer(DetectionTrainer):
        """DetectionTrainer that adds a frozen teacher's predictions to the loss."""

        def _setup_train(self):
            super()._setup_train()
            self.teachers = [YOLO(t).model.float().to(self.device).eval() for t in teacher_paths]
            for teacher in self.teachers:
                for p in teacher.parameters():
                    p.requires_grad_(False)
            self.kd_running = 0.0
            self.kd_batches = 0

            model = self.model
            base_loss = model.loss

            def loss_with_kd(batch, preds=None):
                preds = model.forward(batch["img"]) if preds is None else preds
                loss, items = base_loss(batch, preds)
                with torch.no_grad():
                    # Detect in eval mode returns (decoded, raw dict); the dict is what matches.
                    t_preds = [teacher(batch["img"])[1] for teacher in self.teachers]
                kd = kd_loss(preds, t_preds, w_cls, w_box) * batch["img"].shape[0]
                self.kd_running += float(kd.detach())
                self.kd_batches += 1
                return loss + kd, items

            model.loss = loss_with_kd

        def save_metrics(self, metrics):
            # Logged apart from results.csv, whose column count ultralytics' plotter assumes.
            if self.kd_batches:
                log = Path(self.save_dir) / "kd_loss.csv"
                if not log.exists():
                    log.write_text("epoch,kd_loss\n")
                with log.open("a") as f:
                    f.write(f"{self.epoch + 1},{self.kd_running / self.kd_batches:.6f}\n")
                self.kd_running, self.kd_batches = 0.0, 0
            super().save_metrics(metrics)

    return DistillTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--teacher", required=True, help="one .pt, or several comma-separated to ensemble")
    p.add_argument("--init", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--w-cls", type=float, default=4.0)
    p.add_argument("--w-box", type=float, default=2.0)
    args = p.parse_args()

    data = resolve_data(args.data)
    teachers = [t for t in args.teacher.split(",") if t]
    model = YOLO(args.model)
    if args.init:
        model.load(args.init)
    print(f"{args.name}: {get_num_params(model.model):,} parameters, "
          f"teachers={teachers}, w_cls={args.w_cls}, w_box={args.w_box}")

    kwargs = dict(
        data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(RUNS),
        name=args.name, exist_ok=True, cache="ram", patience=args.patience,
        seed=0, deterministic=False, plots=True, val=True,
        trainer=make_trainer(teachers, args.w_cls, args.w_box, args.init),
    )
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0
    model.train(**kwargs)

    weights = RUNS / args.name / "weights" / "best.pt"
    best = YOLO(str(weights))
    metrics = best.val(data=data, split="test", imgsz=args.imgsz, batch=args.batch,
                       device=args.device, project=str(RUNS),
                       name=f"{args.name}_test", exist_ok=True, plots=True)
    names = best.model.names
    size = weights.stat().st_size
    summary = {
        "name": args.name, "model": args.model, "init": args.init,
        "teacher": teachers, "w_cls": args.w_cls, "w_box": args.w_box,
        "imgsz": args.imgsz, "batch": args.batch, "epochs": args.epochs,
        "params": get_num_params(best.model),
        "checkpoint_bytes": size, "checkpoint_MiB": round(size / 1024**2, 2),
        "test": {
            "mAP50-95": float(metrics.box.map), "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75), "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "test_per_class": {names[c]: {"mAP50-95": float(metrics.box.maps[c])}
                           for c in metrics.box.ap_class_index.tolist()},
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
    out = RUNS / f"summary_{args.name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.name} test summary written to {out} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
