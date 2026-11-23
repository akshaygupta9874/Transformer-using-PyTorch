# Dataset datasheets

Keep raw corpora outside version control. For every prepared TSV, generate and
commit its adjacent `.datasheet.json` using `prepare_data.py`. Record source,
license, pull timestamp, direction, split, filters, pair count, deduplication,
and contamination-check result.

`IN22-Gen`, `IN22-Conv`, and `FLORES-200 devtest` are frozen evaluation sets.
Do not pass them to tokenizer training, quality filtering, synthetic-data
generation, or model fitting.
