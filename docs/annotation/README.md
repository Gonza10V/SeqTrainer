# Promoter Annotation

SeqTrainer scans a GenBank record with a trained DNABERT2 classifier, applies
the threshold selected on the validation split, merges nearby positive windows,
and writes computational `predicted_promoter` features without altering source
features. Predictions are not validated biological annotations.

## Install

```powershell
python -m pip install -e ".[annotation,torch]"
git lfs install
git lfs pull
```

The bundled Kaggle DNABERT2 model is stored at:

```text
outputs/models/dnabert2_kaggle_best/
  manifest.json
  checkpoints/best_model.pt
```

`manifest.json` supplies the validation-selected threshold and preprocessing
contract. `best_model.pt` supplies the trained weights. Keep them together.
If LFS is unavailable, use `scripts/prepare_dnabert2_annotation_bundle.ps1`
with the supplied Kaggle archive.

## Run DNABERT2

Keep a GenBank file outside the repository if preferred, then pass its full
path. This PowerShell command writes all artifacts into one disposable run
folder:

```powershell
seqtrainer annotate promoters "$env:USERPROFILE\Downloads\my_plasmid.gb" `
  --model-family dnabert2 `
  --model-bundle outputs\models\dnabert2_kaggle_best `
  --step-size 25 `
  --scan-both-strands `
  --output outputs\annotations\my_plasmid\annotated.gb `
  --predictions-csv outputs\annotations\my_plasmid\predictions.csv `
  --manifest outputs\annotations\my_plasmid\manifest.json `
  --clean-output `
  --open-output-folder
```

Use `--checkpoint` and `--benchmark-manifest` instead of `--model-bundle` only
when the two paths must be supplied explicitly. A real DNABERT2 run requires
the matching benchmark manifest; it will not silently substitute a default
threshold or window length. Do not tune the threshold on the plasmid being
annotated.

For a quick file-writing smoke test, replace the model arguments with:

```powershell
--model-family dummy --threshold 0.80 --window-size 300
```

The dummy predictor is deterministic and is not biological evidence.

## Outputs

| Artifact | Purpose |
| --- | --- |
| `annotated.gb` | Original GenBank record plus `predicted_promoter` features. |
| `predictions.csv` | Every scored strand-aware window and its threshold result. |
| `manifest.json` | Input, checkpoint, manifest SHA-256 values, threshold, settings, and counts. |
| `evaluation/` | Optional labelled-plasmid gold labels, window scores, promoter matches, and metrics. |
| `annotated.nt` | Optional SBOL3 machine-exchange export. |
| `annotated.rdf` | Optional Canvas-compatible SBOL2 RDF/XML export. |

`--clean-output` removes only the named primary artifacts and the explicitly
named evaluation directory before regeneration. Existing prediction CSVs are
never treated as reusable inference results.

## Labelled Plasmid Evaluation

Add `--evaluation-dir`, `--sbol-output`, or `--sbol2-output` to evaluate a
record containing deposited promoter labels. See
[Addgene labelled evaluation](addgene_labeled_promoter_evaluation.md) for the
collection workflow and [SBOL export](sbol3_export.md) for SBOLCanvas import.
An unlabelled plasmid demonstrates inference and export only; it does not
provide biological precision or recall.
