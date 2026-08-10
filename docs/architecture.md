# Architecture

Raw scanned receipt (image)
        |
        v
[ Step 2: PaddleOCR on GPU ]  -> words + box positions + confidence (JSON)
        |
        v
[ Step 3: Spark ]  -> clean, normalise, explode to token level, join labels -> parquet
        |
        v
[ Step 4: LayoutLMv3 in PyTorch ]  -> tag each word: total / date / vendor / other
        |
        v
[ Step 5: Evaluate ]  -> entity-level F1 per field  (the resume number)

## Why each tool
- OCR: input is genuine scanned images, so text must be recovered. GPU engine because an A100 is available and Tesseract cannot use it.
- Spark: preprocessing written to scale, validated on a large token-level table.
- PyTorch / LayoutLMv3: layout-aware extraction is the state of the practice, and the target job descriptions ask for PyTorch.
