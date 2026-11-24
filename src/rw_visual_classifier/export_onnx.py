"""Export the best PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rw_visual_classifier.config import (
    BEST_CHECKPOINT,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_NAME,
    ONNX_PATH,
)
from rw_visual_classifier.train import create_model


def export_onnx(
    checkpoint: Path = BEST_CHECKPOINT,
    output: Path = ONNX_PATH,
    opset: int = 18,
) -> Path:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", MODEL_NAME)
    class_names = ckpt.get("class_names", list(CLASS_NAMES))
    model = create_model(num_classes=len(class_names), pretrained=False)
    # Recreate with the stored architecture name if needed
    if model_name != MODEL_NAME:
        import timm

        model = timm.create_model(
            model_name, pretrained=False, num_classes=len(class_names)
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Torch 2.15+ prefers the dynamo exporter; keep a fixed batch-1 graph for
    # simple CPU ORT serving (batch can still be >1 if ORT accepts it).
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=True,
    )
    print(f"Exported ONNX → {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_CHECKPOINT,
        help="Path to best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ONNX_PATH,
        help="Output .onnx path",
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    export_onnx(checkpoint=args.checkpoint, output=args.output, opset=args.opset)


if __name__ == "__main__":
    main()
