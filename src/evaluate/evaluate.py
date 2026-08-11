"""Step 5 - Evaluate.

Loads the trained model and reports precision / recall / F1 for the TOTAL class
on the held-out test split. Uses the same seed=42 split as training so the test
receipts are the ones the model never saw.

Reported honestly: this is token-level F1 on the subset of receipts where OCR
captured the total. OCR captures the total on ~64% of all receipts; this metric
is conditioned on that.

Run:  python src/evaluate/evaluate.py
"""

import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTokenClassification

# reuse the dataset class + loader from train.py
from src.model.train import ReceiptDataset, load_examples, SAVE_DIR

DEVICE = "cuda"


def main():
    random.seed(42)
    examples = load_examples()
    random.shuffle(examples)
    n_test = int(0.2 * len(examples))
    test_ex = examples[:n_test]

    images = load_dataset("naver-clova-ix/cord-v2")["train"]
    processor = AutoProcessor.from_pretrained(SAVE_DIR, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    model.eval()

    test_ds = ReceiptDataset(test_ex, processor, images)
    loader = DataLoader(test_ds, batch_size=2)

    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            preds = model(**batch).logits.argmax(-1)
            mask = labels != -100
            p, t = preds[mask], labels[mask]
            tp += ((p == 1) & (t == 1)).sum().item()
            fp += ((p == 1) & (t == 0)).sum().item()
            fn += ((p == 0) & (t == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    print(f"TOTAL  precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}")
    print(f"(tp={tp}, fp={fp}, fn={fn}, test receipts={len(test_ex)})")


if __name__ == "__main__":
    main()
