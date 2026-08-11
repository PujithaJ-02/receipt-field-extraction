"""Step 4 - Train (PyTorch / LayoutLMv3).

Loads the labeled parquet from data/processed/, groups it back into per-receipt
examples, fine-tunes LayoutLMv3 to tag each token TOTAL vs OTHER (using token
text + position), and saves the trained model to models/.

Run:  python src/model/train.py
"""

import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTokenClassification

PROCESSED = "data/processed/labeled_words.parquet"
MODEL_NAME = "microsoft/layoutlmv3-base"
SAVE_DIR = "models/layoutlmv3-total"
LABEL2ID = {"OTHER": 0, "TOTAL": 1}
DEVICE = "cuda"


def load_examples():
    """Read parquet, group flat word-rows into per-receipt examples."""
    parts = glob.glob(f"{PROCESSED}/*.parquet")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    def box_to_xyxy(box):
        pts = np.vstack([np.asarray(p, dtype=float) for p in box])
        return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max())]

    examples = []
    for idx, g in df.groupby("idx"):
        examples.append({
            "idx": int(idx),
            "words": g["text"].astype(str).tolist(),
            "boxes": [box_to_xyxy(b) for b in g["box"]],
            "labels": g["label"].tolist(),
        })
    # keep only receipts that have a TOTAL to learn from
    return [e for e in examples if "TOTAL" in e["labels"]]


class ReceiptDataset(Dataset):
    def __init__(self, examples, processor, images):
        self.examples = examples
        self.processor = processor
        self.images = images

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        image = self.images[ex["idx"]]["image"].convert("RGB")
        W, H = image.size

        def scale(b):
            def c(v):
                return max(0, min(1000, int(v)))
            return [c(1000*b[0]/W), c(1000*b[1]/H), c(1000*b[2]/W), c(1000*b[3]/H)]

        boxes = [scale(b) for b in ex["boxes"]]
        labels = [LABEL2ID[l] for l in ex["labels"]]
        enc = self.processor(image, ex["words"], boxes=boxes, word_labels=labels,
                             return_tensors="pt", truncation=True, padding="max_length")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["bbox"] = enc["bbox"].clamp(0, 1000)     # keep boxes in LayoutLMv3 range
        return enc


def main():
    random.seed(42)
    examples = load_examples()
    random.shuffle(examples)
    n_test = int(0.2 * len(examples))
    train_ex, test_ex = examples[n_test:], examples[:n_test]
    print(f"train {len(train_ex)}  test {len(test_ex)}")

    images = load_dataset("naver-clova-ix/cord-v2")["train"]
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = ReceiptDataset(train_ex, processor, images)
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # TOTAL is rare (~1:15), so weight it higher in the loss
    weights = torch.tensor([1.0, 15.0]).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(4):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits.reshape(-1, 2), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"epoch {epoch+1}/4  avg loss {total/len(loader):.4f}")

    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print("saved to", SAVE_DIR)


if __name__ == "__main__":
    main()
