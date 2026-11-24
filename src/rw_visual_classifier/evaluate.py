"""Evaluate PyTorch and/or ONNX models on the test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from rw_visual_classifier.config import (
    BEST_CHECKPOINT,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    IMAGES_DIR,
    METRICS_DIR,
    MODEL_NAME,
    ONNX_PATH,
)
from rw_visual_classifier.data import PreviewDataset, load_split_frames
from rw_visual_classifier.predict import load_session, predict_image
from rw_visual_classifier.train import create_model, pick_device, run_eval
from rw_visual_classifier.transforms import build_eval_transform



def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "per_class_f1": {
            name: float(score) for name, score in zip(CLASS_NAMES, per_class)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(CLASS_NAMES),
            digits=4,
            zero_division=0,
        ),
    }


@torch.no_grad()
def eval_pytorch(
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    device = pick_device()
    frames = load_split_frames()
    test_ds = PreviewDataset(frames["test"], transform=build_eval_transform())
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", MODEL_NAME)
    class_names = ckpt.get("class_names", list(CLASS_NAMES))
    if model_name != MODEL_NAME:
        import timm

        model = timm.create_model(
            model_name, pretrained=False, num_classes=len(class_names)
        )
    else:
        model = create_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    summary = run_eval(model, loader, device)

    all_preds: list[int] = []
    all_targets: list[int] = []
    for images, targets in tqdm(loader, desc="pytorch-test", leave=False):
        images = images.to(device)
        logits = model(images)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.tolist())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    metrics = _metrics_from_preds(y_true, y_pred)
    metrics["loss"] = summary["loss"]
    metrics["device"] = str(device)
    return metrics, y_true, y_pred


def eval_onnx(
    model_path: Path,
    max_samples: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    frames = load_split_frames()
    test = frames["test"]
    if max_samples is not None:
        test = test.head(max_samples)

    session = load_session(model_path)
    preds: list[int] = []
    targets: list[int] = []
    latencies: list[float] = []

    for row in tqdm(test.itertuples(index=False), total=len(test), desc="onnx-test"):
        path = IMAGES_DIR / row.rel_path
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        result = predict_image(session, image)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(int(result["index"]))
        targets.append(CLASS_TO_IDX[str(row.label)])

    y_true = np.asarray(targets)
    y_pred = np.asarray(preds)
    metrics = _metrics_from_preds(y_true, y_pred)
    latency = {
        "count": len(latencies),
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mean_ms": float(np.mean(latencies)) if latencies else 0.0,
    }
    metrics["latency"] = latency
    return metrics, y_true, y_pred, latency


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate RW visual classifier")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_CHECKPOINT,
        help="PyTorch checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ONNX_PATH,
        help="ONNX model path",
    )
    parser.add_argument(
        "--backend",
        choices=("pytorch", "onnx", "both"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--onnx-max-samples",
        type=int,
        default=None,
        help="Optional cap for ONNX latency/eval pass",
    )
    args = parser.parse_args(argv)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    pt_pred = None
    onnx_pred = None
    y_true = None

    if args.backend in ("pytorch", "both"):
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        pt_metrics, y_true, pt_pred = eval_pytorch(
            args.checkpoint, args.batch_size, args.num_workers
        )
        out["pytorch"] = pt_metrics
        print("=== PyTorch test ===")
        print(pt_metrics["classification_report"])
        print("confusion_matrix:", pt_metrics["confusion_matrix"])

    if args.backend in ("onnx", "both"):
        if not args.onnx.exists():
            raise SystemExit(f"ONNX model not found: {args.onnx}")
        onnx_metrics, y_true_onnx, onnx_pred, latency = eval_onnx(
            args.onnx, max_samples=args.onnx_max_samples
        )
        out["onnx"] = onnx_metrics
        print("=== ONNX CPU test ===")
        print(onnx_metrics["classification_report"])
        print("confusion_matrix:", onnx_metrics["confusion_matrix"])
        print(
            f"latency ms: p50={latency['p50_ms']:.2f} "
            f"p95={latency['p95_ms']:.2f} mean={latency['mean_ms']:.2f}"
        )
        if y_true is None:
            y_true = y_true_onnx

    if (
        args.backend == "both"
        and pt_pred is not None
        and onnx_pred is not None
        and len(pt_pred) == len(onnx_pred)
    ):
        agree = float((pt_pred == onnx_pred).mean())
        out["pytorch_onnx_label_agreement"] = agree
        print(f"PyTorch vs ONNX label agreement: {agree:.6f}")

    out_path = METRICS_DIR / "test_metrics.json"
    # Drop bulky text report from JSON duplicate if needed — keep it
    serializable = json.loads(json.dumps(out))
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
