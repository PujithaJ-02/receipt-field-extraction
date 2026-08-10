# Document Intelligence Pipeline for Scanned Receipts

An end-to-end document-intelligence pipeline that extracts structured fields
(total, date, vendor) from scanned receipt images.

**Stack:** PaddleOCR (GPU) + PySpark + PyTorch (LayoutLMv3)

## Result

> Fill this in once measured, e.g. "Entity-level F1: 0.89 total / 0.94 date / 0.87 vendor on CORD test set."
> This number is only useful if you can explain how it was measured and where it fails.

## Pipeline

Raw scan -> OCR (words + positions) -> Spark (clean, structure at scale) -> LayoutLMv3 (tag fields) -> Evaluate (F1)

See `docs/architecture.md` for the full diagram and the justification for each tool.

## Dataset

CORD (or SROIE). Scanned receipts with ground-truth labels. Download via `src/ingest/download_data.py`.
Data and model weights are not committed (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python src/ingest/download_data.py     # 1. get data
python src/ocr/run_ocr.py              # 2. OCR on GPU
python src/spark/process.py            # 3. Spark processing
python src/model/train.py              # 4. train
python src/evaluate/evaluate.py        # 5. score
```

## Repo layout

```
config/     settings (yaml)
data/       raw / interim / processed   (gitignored)
src/        pipeline code, one folder per step
models/     trained weights             (gitignored)
notebooks/  exploration
reports/    figures and results
tests/      smoke test
docs/       architecture and notes
```
