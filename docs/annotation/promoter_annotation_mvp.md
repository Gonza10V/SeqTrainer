# Promoter Annotation MVP

SeqTrainer's first promoter annotation MVP adds model-predicted promoter
features to GenBank plasmid records while preserving existing annotations.

## Exact Hits Versus Predicted Promoters

SeqImprove-style exact-hit annotation compares an input DNA sequence against a
library of known parts. It can confidently annotate known sequences that match a
library entry, but it cannot find a novel promoter-like region that is absent
from the library.

SeqTrainer model annotation is different. It scans the sequence with fixed-size
windows and assigns each window a promoter score from a selected model. These
features are probabilistic computational predictions. They must be labeled as
`predicted_promoter`, not as confirmed biological part names such as
`BBa_J23101`.

## Benchmark Provenance

The annotation command can read a benchmark `manifest.json` to reuse the same
model sequence length and validation-selected threshold. This keeps annotation
connected to the Issue 3 benchmark policy:

- do not tune on the test set
- use validation-selected thresholds when available
- preserve model family, checkpoint, and threshold provenance in the manifest

Real DNABERT2 annotation requires the matching benchmark manifest. Dummy mode
may use explicit threshold and window settings for smoke tests.

## Model Families

- `dummy`: deterministic smoke-test predictor. It does not make biological
  claims and should only be used to verify file writing and pipeline behavior.
- `dnabert2`: primary real predictor. It requires a compatible checkpoint,
  matching benchmark manifest, and torch/transformers.

The public annotation interface currently supports only these two modes.

## Basic Smoke Run

Install the annotation extra first:

```bash
pip install -e ".[annotation]"
```

For the canonical local workflow, see `docs/annotation/README.md`.

Then run the dummy smoke test:

```bash
seqtrainer annotate promoters pAN1717_cyan.gb \
  --model-family dummy \
  --threshold 0.80 \
  --window-size 300 \
  --step-size 25 \
  --scan-both-strands \
  --output outputs/annotations/pAN1717_cyan_dummy_annotated.gb \
  --predictions-csv outputs/annotations/pAN1717_cyan_dummy_predictions.csv \
  --manifest outputs/annotations/pAN1717_cyan_dummy_manifest.json
```

## Outputs

The annotated GenBank file preserves existing features and appends additional
`promoter` features with qualifiers including:

- `label=predicted_promoter`
- model family
- score
- threshold
- window and step size
- note saying this is a computational prediction only

The predictions CSV includes every scanned window, its score, threshold result,
strand, coordinates, circular-boundary flag, overlap status, merged region ID,
and window sequence.

The manifest records input/output paths, model metadata, threshold source,
window settings, circular topology, number of scanned windows, number of
predicted promoters added, overlap counts, warnings, timestamp, and git SHA.

## Current Limitations

- Dummy mode is not biologically meaningful.
- DNABERT2 annotation requires a compatible checkpoint and benchmark manifest.
- The MVP adds promoter-like regions by sliding windows, not by experimentally
  validating promoter activity.
- Predicted promoters remain separate from exact library-hit annotations.
# Labelled external evaluation

The MVP now has an opt-in labelled-plasmid evaluation path. See `addgene_labeled_promoter_evaluation.md` for the Addgene article 18115 collection workflow. Existing `seqtrainer annotate promoters ...` runs remain compatible; add `--evaluation-dir` only when ground-truth extraction and evaluation artifacts are desired.
