# Document Intelligence Pipeline for Scanned Receipts

An end-to-end pipeline that extracts the **grand total** from photographed
receipts. It reads the images with OCR, processes the results at scale with
PySpark, and fine-tunes a layout-aware transformer (LayoutLMv3) to locate the
total on the page.

**Stack:** EasyOCR (GPU) · PySpark · PyTorch (LayoutLMv3) · CORD dataset

---

## Results

Measured on a held-out test set of 99 receipts the model never saw during training:

| Metric | Value |
|---|---|
| Precision (TOTAL) | 0.81 |
| Recall (TOTAL) | 0.96 |
| F1 (TOTAL) | ~0.88 |

**Read this honestly, end to end:** the OCR captures the grand total as a clean,
correctly-read token on about **64%** of receipts (photographed thermal receipts
are hard, and ~33% of all OCR tokens come back low-confidence). On the receipts
where the total survives OCR, the model then locates it with ~0.88 F1. The two
numbers chain: OCR capture is the ceiling, the model works within it.

Known failure mode: false positives on subtotal, tax, and cash-paid lines that
look like the total. The 99-receipt test set is small, so treat the F1 as
approximate.

---

## Pipeline

```
Scanned receipt (photo)
   -> EasyOCR on GPU        : words + box positions + confidence
   -> PySpark               : explode to ~15k word-rows, clean, normalize, join labels
   -> LayoutLMv3 (PyTorch)  : tag each word TOTAL vs OTHER, using text + position
   -> Evaluate              : precision / recall / F1 on held-out receipts
```

## Dataset

[CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2): 800 train / 100
val / 100 test photographed restaurant receipts with structured field labels.
Confirmed to be genuine photographed scans (skewed, uneven lighting), which is
what justifies an OCR step.

## Key numbers (from the analysis pass)

- 800 receipts, ~14,700 OCR word-tokens
- Mean OCR confidence 0.65, median 0.73
- 32.8% of tokens are low-confidence (<0.5)
- Average 18 words per receipt (range 5–86)

## Why these tools

- **OCR** is justified because the input is genuine photographed scans, not
  digital text.
- **PySpark** processes the OCR output at the word/token level. On this dataset
  the volume is modest; the Spark jobs are written to scale and validated at
  ~15k rows, not claimed to be strictly required at this size.
- **LayoutLMv3 (PyTorch)** is used because locating a field on a receipt depends
  on position, not just text, and naive string matching caps out at the OCR
  ceiling. A layout-aware model is the right tool.

## Scope

This project extracts the grand total. It was deliberately scoped to one field
on one dataset: the public mirrors of a second dataset (SROIE) proved
unreliable, and depth on one well-understood pipeline was judged more valuable
than breadth across two.

## Repo layout

```
config/     settings
src/        pipeline code (ingest, ocr, spark, model, evaluate)
notebooks/  exploratory work, in pipeline order (01..04)
data/       raw / interim / processed   (gitignored)
models/     trained weights             (gitignored)
reports/    figures and results
docs/       architecture notes
```

## Run order

```bash
python src/ocr/run_ocr.py        # OCR all receipts -> data/interim
# spark processing + model training: see notebooks 03 and 04
```
