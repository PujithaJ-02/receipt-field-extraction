"""Step 2 - OCR (GPU).

Run EasyOCR on every CORD receipt and save the raw OCR output (text +
box position + confidence) to data/interim/ as one JSON file per receipt.

Why EasyOCR: it runs in-process on the GPU (the L40). Newer Surya required
Docker, which the JupyterHub does not allow. Tesseract is CPU-only and would
waste the GPU. EasyOCR is the pragmatic choice that actually runs here.

Design notes:
- Checkpointing: if a receipt already has an output file, it is skipped. So
  if the run is interrupted (kernel restart, session timeout), rerun and it
  picks up where it stopped instead of starting over.
- numpy types (int32/float64) from EasyOCR are converted to plain Python
  numbers, otherwise json.dump crashes.
"""

import os
import json
import numpy as np
import easyocr
from datasets import load_dataset
from tqdm import tqdm

# ---- config ----
SPLIT = "train"                     # run per split: train, validation, test
OUT_DIR = f"data/interim/{SPLIT}"   # where OCR json goes (gitignored)
os.makedirs(OUT_DIR, exist_ok=True)


def to_plain(obj):
    """Convert numpy numbers/arrays to plain Python so json can save them."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    return obj


def run():
    ds = load_dataset("naver-clova-ix/cord-v2")[SPLIT]
    reader = easyocr.Reader(["en"], gpu=True)

    for idx in tqdm(range(len(ds)), desc=f"OCR {SPLIT}"):
        out_path = os.path.join(OUT_DIR, f"{idx:04d}.json")
        if os.path.exists(out_path):
            continue  # checkpoint: already done, skip

        image = ds[idx]["image"]
        raw = reader.readtext(np.array(image))

        # keep the ground-truth label alongside the OCR, so the next step
        # can line up what the OCR read against the correct answer.
        record = {
            "idx": idx,
            "ground_truth": ds[idx]["ground_truth"],
            "ocr": [
                {
                    "box": to_plain(box),      # 4 corner points [x,y]
                    "text": text,
                    "conf": to_plain(conf),
                }
                for box, text, conf in raw
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)


if __name__ == "__main__":
    run()
