# Validation evidence and optional inference

## Recorded results — 16 October 2026

These are retained results from the reviewed checkpoint archive, not inference executed during this finalization.

| Check | Recorded outcome |
|---|---|
| ViT + T5 Freeze, 507 original PNGs and historical facts | **507/507 captions identical** to the archived outputs |
| Freeze metrics | BLEU-4 **0.4062**, CIDEr **2.3827**, NutAcc **96.37%**, matching the historical evaluation |
| CNN, 507 PNGs, best checkpoint at epoch 45 | Calorie MAE **55.67 kcal**, compared with **53.59 kcal** historically |
| Fresh CNN facts → Freeze, eight dishes | Completed; **100.00% NutAcc** against fresh facts; **0/8** captions identical to historical text |

The original training function returned the last epoch while the checkpoint stored the best epoch. This is a plausible contributor to the CNN discrepancy, not an isolated cause. Keep the historical tables separate from checkpoint results. NutAcc concerns supplied facts, not measured nutrition accuracy; template references and metadata ingredients also constrain interpretation. Other checkpoint variants and full 507-dish integration were not tested.

The [evidence folder](evidence/) contains three compact per-dish tables and one source-linked JSON record of the earlier runs, checksums, configurations, and metric comparison. It excludes weights, raw images, tokenizer files, and the audit history.

## Verify bundled evidence — no ML dependencies

From the repository root, using Python 3.12 and Node.js 22 or later:

```bash
python -B validation/verify.py
python -B -m unittest discover -s tests -v
node --check docs/assets/js/archive-data.js
node --check docs/assets/js/explorer.js
node --test tests/test_explorer.cjs
```

The verifier checks hashes, 507 photos, 3,042 captions, historical numerical metrics, the saved CNN discrepancy, reproduced Freeze text, eight integration samples, notebook syntax, and local links. It reads saved outputs only. Optional `python -B validation/verify.py --language` recomputes BLEU-4/CIDEr using `nltk` and `pycocoevalcap` from the existing requirements; no model download is needed. Neural unit tests skip explicitly when optional ML libraries are absent.

## Optional checkpoint inference — external inputs required

Install the notebook dependencies in [requirements.txt](../requirements.txt) in a separate environment. The earlier CPU run used torch 2.11.0, torchvision 0.26.0, timm 1.0.27, transformers 5.12.1, tokenizers 0.22.2, sentencepiece 0.2.2, numpy 2.5.2, and Pillow 12.3.0.

Obtain trusted original checkpoints from the author's private research archive. From the reviewed ZIP, extract only `validation/archive/` and `validation/vit/`, `validation/t5/` into an external workspace. Use this layout:

```text
external/
  inputs/
    cnn_best.pth
    captioner_freeze.pth
    vit/config.json
    t5/{config.json,generation_config.json,spiece.model,tokenizer_config.json,tokenizer.json}
    images/<dish_id>/rgb.png
  archive/
    dish_ids_test.txt
    metadata_clean.csv
    predictions.csv
    fact_strings.csv
    target_stats_log.json
    test_captions_freeze.csv
```

Use original Nutrition5k overhead PNGs for the 507 listed IDs, not the demo's WebP adaptations. Copy each dataset `imagery/realsense_overhead/<dish_id>/rgb.png` into `inputs/images/<dish_id>/rgb.png`. Configurations and tokenizer assets retain their upstream terms and notices from the reviewed archive. They are not redistributed here.

```bash
python -B validation/validate_checkpoints.py --stage freeze --inputs external/inputs --archive external/archive --check-inputs
python -B validation/validate_checkpoints.py --stage cnn --inputs external/inputs --archive external/archive --output runs/cnn-review
python -B validation/validate_checkpoints.py --stage freeze --inputs external/inputs --archive external/archive --output runs/freeze-review
python -B validation/validate_checkpoints.py --stage freeze --limit 8 --inputs external/inputs --archive external/archive --facts-csv runs/cnn-review/cnn_checkpoint_facts.csv --output runs/integration-review
```

Preflight compares input SHA-256 values with [checkpoint-evidence.json](evidence/checkpoint-evidence.json), including an aggregate of original PNG hashes. A mismatch stops execution. The inference-only script uses class definitions from the included notebook, strict weight loading, evaluation mode, and fresh output folders. `--notebook` optionally selects a separately supplied historical notebook. New summaries use the execution date and record the notebook hash; no historical file is overwritten. Neither verification nor CI trains models or deploys a website.
