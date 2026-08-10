"""Step 3 - Spark.
Load all OCR JSON from data/interim/, clean and normalise at scale,
explode to word/token level so the table is genuinely large, join with
ground-truth labels, and write parquet to data/processed/.
Interview note: be ready to say Spark was written to scale and validated
at volume, NOT that it was strictly required on this dataset size.
"""
