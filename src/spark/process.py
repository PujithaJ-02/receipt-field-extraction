"""Step 3 - Spark processing.

Loads the per-receipt OCR JSON from data/interim/, explodes it to one row per
word, cleans and normalizes, parses the ground-truth grand total from the CORD
label, labels each word TOTAL vs OTHER, and writes a labeled parquet dataset to
data/processed/.

Run:  python src/spark/process.py
(Set JAVA_HOME first if Java is not on PATH; see the os.environ line below.)
"""

import os
import re
import json

# --- point Spark at the local JDK if JAVA_HOME is not already set ---
# (on the JupyterHub used for this project, Java was installed via install-jdk)
os.environ.setdefault("JAVA_HOME", os.path.expanduser("~/.jdk/jdk-17.0.20+8"))
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType

SPLIT = "train"
INTERIM = f"data/interim/{SPLIT}/*.json"
OUT = "data/processed/labeled_words.parquet"


def raw_total(gt_string):
    """Pull total_price out of the CORD label, coping with str, list, or missing."""
    try:
        t = json.loads(gt_string)["gt_parse"].get("total")
    except Exception:
        return None
    if not t:
        return None
    val = t.get("total_price")
    if isinstance(val, list):
        val = val[0] if val else None
    return str(val) if val is not None else None


def norm_total(gt_string):
    """Normalize the total to a plain integer string of whole currency units.

    Handles thousands separators and a trailing .dd / ,dd cents group, and
    strips currency letters like 'Rp'. Whole-unit prices only (safe for this data).
    """
    s = raw_total(gt_string)
    if s is None:
        return None
    s = re.sub(r"[^\d.,]", "", s.strip())          # drop 'Rp', spaces, symbols
    m = re.match(r"^(.*)[.,](\d{2})$", s)           # trailing 2-digit cents?
    core = m.group(1) if m else s
    digits = re.sub(r"[.,]", "", core)              # remaining separators -> gone
    return digits if digits.isdigit() else None


def main():
    spark = (SparkSession.builder
             .appName("receipt-pipeline")
             .master("local[*]")
             .getOrCreate())

    # 1. load all per-receipt OCR json
    df = spark.read.json(INTERIM)

    # 2. explode to one row per OCR word
    words = (df
        .select("idx", "ground_truth", F.explode("ocr").alias("w"))
        .select("idx", "ground_truth",
                F.col("w.text").alias("text"),
                F.col("w.conf").alias("conf"),
                F.col("w.box").alias("box")))

    # 3. clean: trim, drop empties, flag low confidence
    clean = (words
        .withColumn("text", F.trim("text"))
        .filter(F.length("text") > 0)
        .withColumn("low_conf", F.col("conf") < 0.5)
        .withColumn("ocr_digits", F.regexp_replace("text", r"[^\d]", "")))

    # 4. parse + normalize the ground-truth total (per receipt)
    norm_udf = F.udf(norm_total, StringType())
    clean = clean.withColumn("true_total", norm_udf("ground_truth"))

    # 5. label each word: TOTAL if its digits equal the receipt's true total
    labeled = clean.withColumn(
        "label",
        F.when((F.col("ocr_digits") != "") &
               (F.col("ocr_digits") == F.col("true_total")), "TOTAL")
         .otherwise("OTHER"))

    out = labeled.select("idx", "text", "conf", "box", "label")
    out.write.mode("overwrite").parquet(OUT)

    print("wrote:", OUT)
    out.groupBy("label").count().show()
    spark.stop()


if __name__ == "__main__":
    main()
