# ReliefWeb visual classifier

Lightweight image classifier that categorizes ReliefWeb document **first-page previews** into **Map**, **Infographic**, or **Other**.

The model is fine-tuned from a compact convolutional neural network (`mobilenetv4_conv_small`), exported to ONNX, and served on CPU by `ocha_ai_helper` at `POST /visual/classify`, allowing ReliefWeb to identify maps and infographics without calling a multimodal LLM.

```
PDF page preview (RGB) ──► Letterbox 224×224 ──► MobileNetV4 CNN ──► Map / Infographic / Other
```

---

## Key terms and files

- **Catalog (`data/catalog.csv`)** — Metadata table of all ReliefWeb reports with a PDF preview (`report_id`, content format, publishing organization, country, creation date, and preview URL). Built once from the ReliefWeb API and reused across sampling runs. See [Dataset pipeline details](#dataset-pipeline-details).
- **Manifest (`data/manifest.csv`)** — Stratified sample selected from the catalog for training and evaluation. Each report is assigned by `report_id` to `train` (80%, used to fit weights), `val` (10%, used to pick the best epoch), or `test` (10%, held out for final evaluation). See [Train from scratch](#train-from-scratch) and [Dataset pipeline details](#dataset-pipeline-details).
- **Checkpoints (`artifacts/checkpoints/`)** — PyTorch training snapshots saved after each epoch (`epoch_NNN.pt`) alongside `best.pt`, which records the epoch with the highest validation macro-F1. Used for resuming training and ONNX export. See [Train from scratch](#train-from-scratch).
- **CNN / `mobilenetv4_conv_small`** — Compact convolutional neural network backbone (`mobilenetv4_conv_small.e2400_r224_in1k` from `timm`, ~3.8M parameters) fine-tuned on 224×224 letterboxed previews to separate geographic maps, multi-element infographics, and text-heavy documents under tight CPU and RAM limits. See [Train from scratch](#train-from-scratch).
- **ONNX (`artifacts/model.onnx` + `artifacts/model.onnx.data`)** — Portable inference export of `best.pt` run via `onnxruntime` on CPU without PyTorch. The graph is stored in `model.onnx` (~243 KB) and external weight tensors in `model.onnx.data` (~9.5 MB); both files must be kept in the same directory. See [Train from scratch](#train-from-scratch) and [Predict (CPU ONNX)](#predict-cpu-onnx).
- **Macro-F1** — Unweighted average of per-class F1 scores across `Map`, `Infographic`, and `Other`. Used alongside per-class F1 and the confusion matrix so weaker performance on `Infographic` is not masked by overall accuracy. See [Train from scratch](#train-from-scratch) and [Pitfalls and operational notes](#pitfalls-and-operational-notes).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Copy [`.env.example`](.env.example) to `.env` and configure:

- `RELIEFWEB_API_APPNAME` — approved ReliefWeb API app name
- `RELIEFWEB_API_URL` — defaults to `https://api.reliefweb.int/v2`

Optional: set `HF_TOKEN` if Hugging Face rate-limits pretrained `timm` weight downloads.

Run unit tests:

```bash
pytest
```

**Git-ignored vs committed files:** everything under `data/` (catalog, manifest, downloaded preview images) and `notes/` is gitignored. Under `artifacts/`, only `artifacts/metrics/**` and the production ONNX export (`artifacts/model.onnx` and `artifacts/model.onnx.data`) are tracked. Training checkpoints (`artifacts/checkpoints/`), interactive review pages (`artifacts/review/`), and any other `.pt`/`.onnx` files remain local.

---

## Train from scratch

Full pipeline from an empty checkout to an evaluated ONNX export ready for `ocha_ai_helper`:

```bash
source .venv/bin/activate

# 1) Build catalog, draw stratified sample (default 5000/class), and download previews
python scripts/build_dataset.py --limit-per-class 5000

# Optional smaller smoke run:
# python scripts/build_dataset.py --limit-per-class 500

# 2) Fine-tune MobileNetV4-small (uses MPS on Apple Silicon, CUDA, or CPU)
python scripts/train.py --epochs 12 --batch-size 64

# 3) Export best.pt to ONNX (writes both model.onnx and model.onnx.data)
python scripts/export_onnx.py

# 4) Evaluate PyTorch and ONNX Runtime CPU on the held-out test split
python scripts/evaluate.py --backend both
```

Generated artifacts:

| Path | Role |
|------|------|
| `artifacts/checkpoints/best.pt` | Best validation macro-F1 PyTorch checkpoint (local only) |
| `artifacts/model.onnx` | Exported ONNX graph (~243 KB) |
| `artifacts/model.onnx.data` | Exported ONNX weights (~9.5 MB); must sit next to `model.onnx` |
| `artifacts/metrics/train_history.json` | Per-epoch training loss and validation metrics |
| `artifacts/metrics/test_metrics.json` | Test accuracy, per-class F1, confusion matrix, CPU latency, and PyTorch/ONNX agreement |

### Deploy to `ocha_ai_helper`

Copy **both** ONNX files into `ocha_ai_helper` so the Docker image bundles them into `/opt/visual/`:

```bash
cp artifacts/model.onnx artifacts/model.onnx.data ../ocha_ai_helper/html/models/visual/
```

Rebuild and restart `ocha_ai_helper`, then smoke-test `POST /visual/classify` with a base64-encoded preview image.

---

## Retrain

When newer ReliefWeb reports are available, refresh the catalog, draw a new stratified sample, download only missing previews, and re-run training, export, and evaluation:

```bash
source .venv/bin/activate

# Refresh catalog from the API, resample, and download missing previews
python scripts/build_dataset.py --refresh-catalog --limit-per-class 5000

# Or run stages individually to experiment with sample sizes:
# python scripts/build_dataset.py --catalog-only --refresh-catalog
# python scripts/build_dataset.py --sample-only --limit-per-class 5000
# python scripts/build_dataset.py --download-only

python scripts/train.py --epochs 12 --batch-size 64
python scripts/export_onnx.py
python scripts/evaluate.py --backend both
```

- `--refresh-catalog` is required to fetch newer ReliefWeb documents; `--sample-only` alone reuses the existing `data/catalog.csv`.
- Image downloads are resume-safe and skip non-empty files already cached under `data/images/`.
- Resampling changes `train`/`val`/`test` membership; compare `artifacts/metrics/test_metrics.json` (and review mismatches) against the previous run before copying updated ONNX files into `ocha_ai_helper`.
- Raise `--limit-per-class` to train on more samples, or pass `--limit-per-class 0` to keep all `Map` and `Infographic` reports plus up to 75,000 stratified `Other` reports.

---

## Dataset pipeline details

`scripts/build_dataset.py` executes three stages: **catalog** (all preview-bearing report metadata) -> **stratified sample** -> **download** (`file.preview.url-large`, 680px width, first attachment with a preview).

```bash
# Default: stratified 5000 per class
python scripts/build_dataset.py

# Stage control
python scripts/build_dataset.py --catalog-only --refresh-catalog
python scripts/build_dataset.py --sample-only --limit-per-class 5000
python scripts/build_dataset.py --download-only
```

Stratification keys:

- **`Map` / `Infographic`:** `source_id|year` (year from `date.created`)
- **`Other`:** `format_name|source_id`

Outputs:

- `data/catalog.csv` — full preview metadata
- `data/manifest.csv` — sampled rows with 80/10/10 `split` column
- `data/images/{Map,Infographic,Other}/{report_id}.jpg` — cached preview images

### Labels

| Class | ReliefWeb `format.id` |
|-------|------------------------|
| `Map` | `12` |
| `Infographic` | `12570` |
| `Other` | All other formats with a file preview |

---

## Predict (CPU ONNX)

Inference requires only `onnxruntime`, `Pillow`, and `numpy` (no PyTorch or `timm`):

```bash
pip install -r requirements-infer.txt
python scripts/predict.py path/to/preview.jpg
```

Requires both `artifacts/model.onnx` and `artifacts/model.onnx.data` in the same directory. Outputs predicted `label`, `index`, and per-class `probabilities` as JSON.

---

## Review predictions

Generate standalone HTML review pages to inspect test-set predictions, assign human labels (`Map`, `Infographic`, `Other`), view live counts, and export JSON:

```bash
# All test-set mismatches
python scripts/review_failures.py --mode failures --serve

# Stratified sample where model prediction matches ReliefWeb label (default 150; ~50/class)
python scripts/review_failures.py --mode agreements --sample 150 --serve
```

Generated pages:

- `artifacts/review/index.html` — failures
- `artifacts/review/agreements.html` — agreement sample

Without `--serve`, start a local server from the repository root (`python -m http.server 8765`) and open `http://127.0.0.1:8765/artifacts/review/index.html` or `.../agreements.html`.

---

## Pitfalls and operational notes

- **Ship both ONNX files together.** PyTorch's ONNX exporter splits weights into `artifacts/model.onnx.data`; copying only `model.onnx` breaks `onnxruntime` loading.
- **First-page preview ceiling.** Classification uses only the first attachment's page-1 preview. Multi-page situation reports or analyses with infographic-style covers are inherently ambiguous from page 1 alone.
- **`Infographic` is the noisiest boundary.** ReliefWeb's definitions allow infographics to contain maps and situation reports to contain charts; monitor per-class F1 and the confusion matrix rather than accuracy alone.
- **ReliefWeb API quota.** The API allows roughly 10,000 requests/day (`limit=1000`); a full catalog build needs a few hundred requests. Preview image downloads hit the public CDN and do not count against API quota.
