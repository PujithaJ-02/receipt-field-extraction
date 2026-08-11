"""Run the trained model on a single receipt and print the predicted total.

Loads the fine-tuned LayoutLMv3 from models/, runs it on one CORD receipt
(by index), and prints which OCR tokens it tagged as TOTAL.

Run:  python src/model/predict.py 5      # predict on receipt index 5
"""

import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTokenClassification

import easyocr

SAVE_DIR = "models/layoutlmv3-total"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID2LABEL = {0: "OTHER", 1: "TOTAL"}


def predict(idx: int):
    # load the receipt image
    images = load_dataset("naver-clova-ix/cord-v2")["train"]
    image = images[idx]["image"].convert("RGB")
    W, H = image.size

    # OCR it (same engine as the pipeline)
    reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
    raw = reader.readtext(np.array(image))
    words, boxes = [], []
    for box, text, _conf in raw:
        pts = np.array(box, dtype=float)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        def c(v, m):
            return max(0, min(1000, int(1000 * v / m)))
        words.append(text)
        boxes.append([c(x0, W), c(y0, H), c(x1, W), c(y1, H)])

    if not words:
        print("no text found on this receipt")
        return

    # load the trained model
    processor = AutoProcessor.from_pretrained(SAVE_DIR, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    model.eval()

    enc = processor(image, words, boxes=boxes, return_tensors="pt",
                    truncation=True, padding="max_length")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    enc["bbox"] = enc["bbox"].clamp(0, 1000)

    with torch.no_grad():
        preds = model(**enc).logits.argmax(-1)[0].cpu().tolist()

    # map token predictions back to words (first sub-token per word)
    word_ids = processor.tokenizer(words, boxes=boxes, is_split_into_words=False)
    # simple readout: show any word whose first token was tagged TOTAL
    tagged = []
    token_preds = preds[: len(words)]  # approximate: first tokens
    for w, p in zip(words, token_preds):
        if ID2LABEL.get(p) == "TOTAL":
            tagged.append(w)

    print(f"receipt {idx}")
    print("predicted TOTAL tokens:", tagged if tagged else "(none found)")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    predict(idx)