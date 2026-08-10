"""Step 2 - OCR (GPU).
Run PaddleOCR on each raw image. Tesseract is NOT used: it is CPU-only
and cannot use the A100. Output per image: words + their box positions
+ confidence, saved as JSON in data/interim/.
The box positions matter: the model reads words AND where they sit.
"""
