"""Single-image ONNX Runtime CPU prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from rw_visual_classifier.config import CLASS_NAMES, ONNX_PATH
from rw_visual_classifier.transforms import preprocess_for_onnx


def load_session(
    model_path: Path = ONNX_PATH,
    intra_op_threads: int = 2,
) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def predict_image(
    session: ort.InferenceSession,
    image: Image.Image,
) -> dict:
    batch = preprocess_for_onnx(image)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    logits = np.asarray(logits)[0]
    # Stable softmax
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "index": idx,
        "probabilities": {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probs)
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict with ONNX CPU model")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--model",
        type=Path,
        default=ONNX_PATH,
        help="Path to model.onnx",
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    session = load_session(args.model, intra_op_threads=args.threads)
    image = Image.open(args.image).convert("RGB")
    result = predict_image(session, image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
