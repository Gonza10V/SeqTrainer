# Final T4 Training Benchmarks

These are the final, resumable Colab notebooks for the shared SeqTrainer
promoter benchmark. They run on a free or standard NVIDIA T4 and use the same
predefined files as CNN-v2.

## Notebooks

- [DNABERT2 staged fine-tuning](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/final_training/dnabert2_final_training_t4_colab.ipynb)
- [iPro-MP five-fold inference and stacker](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/final_training/ipromp_final_training_t4_colab.ipynb)

The notebooks are stored on branch issue-3-all-model-baselines. Open the
matching link, select a T4 GPU, and run cells from the top. The DNABERT2
notebook deliberately follows the same environment, Drive discovery, and
diagnostic sequence as the working T4 benchmark notebook, while calling the
stronger final-training helper.

## Fixed scientific contract

- Dataset: GSE144621 EP_DNA_BERT2_genomic_order
- Drive data directory:
  /content/drive/MyDrive/AIxBio/Promoter Classification/Data
- Required files:
  - train_EP_DNA_BERT2_genomic_order.csv (136,484 rows)
  - eval_EP_DNA_BERT2_genomic_order.csv (19,498 rows)
  - test_EP_DNA_BERT2_genomic_order.csv (38,996 rows)
- Required columns: sequence, label
- Labels: 0 = non-promoter, 1 = promoter
- Seed: 42
- Model selection: validation MCC, then validation AUPRC
- Threshold: exact search over unique validation probabilities
- Test: evaluated once only after the final candidate is selected

The notebooks stop before model setup if row counts, labels, columns, or
duplicate-sequence checks do not match. They do not silently use the older
incorrect split sizes.

## Local and persistent storage

Data is audited from Drive and copied atomically to:

/content/seqtrainer_final_training/data/

Training and inference read only this local copy. Persistent results are written
to:

- /content/drive/MyDrive/SeqTrainer/final_training/dnabert2_seed42/
- /content/drive/MyDrive/SeqTrainer/final_training/ipromp_seed42/

Each completed run contains config.json, input_split_audit.json,
environment.json, history.csv, metrics.csv, metrics.json,
predictions.csv, manifest.json, checkpoints/, plots/, and logs/.
DNABERT2 additionally writes token audit files and staged checkpoints.
iPro-MP additionally writes fold_predictions/, ensemble_candidates.csv,
and selected_ensemble.json.

## Resume after a disconnect

Both notebooks use RESUME_MODE:

- latest: default; resume the most recent completed epoch or inference stage
- best: restore the best validation-MCC state
- none: start a new run

Rerun the notebook from the top after reconnecting. The data audit and model
cache are restored, completed checkpoints or fold logits are verified by row
count and input SHA-256, and only missing work is repeated. Checkpoints are
written locally first and copied to Drive through a temporary filename before
the final name is replaced.

## DNABERT2 profile

The notebook uses zhihan1996/DNABERT-2-117M at revision
7bce263b15377fc15361f52cfab88f8b586abda0. It uses the stable SeqTrainer
loader with trust_remote_code=True, low_cpu_mem_usage=False,
device_map=None, pad-token repair, meta-device fallback, and standard
PyTorch attention instead of Triton FlashAttention.

The primary candidate uses maximum token length 104, physical batch size 2,
gradient accumulation 16 (effective batch size 32), six maximum epochs,
patience 2, warmup ratio 0.08, AdamW, weight decay 0.01, FP16 training, and
gradient checkpointing. Epoch 1 trains the classifier head. Epochs 2-6
unfreeze only the top four encoder layers, with learning rates 1e-4 for
the head and 1e-5 for the encoder. The head uses attention-mask-aware mean
and maximum pooling, LayerNorm, 256 hidden units, GELU, dropout 0.20, and one
logit. Validation and test probabilities are converted to FP32 before
metrics. The runner also applies the final partial gradient-accumulation
group at each epoch, rather than silently dropping it.

## DNABERT2 profile controls

Cell 2 exposes the final profile explicitly. Keep the default profile for the
first run. Each changed profile is a new validation candidate, not a
continuation of an earlier run: use RESUME_MODE = "none" or a new Drive output
directory. The runner checks checkpoint metadata and refuses to mix
incompatible profiles.

| Setting | Default | Purpose |
| --- | ---: | --- |
| MAX_EPOCHS | 6 | Maximum staged fine-tuning epochs; early stopping may end earlier. |
| PATIENCE | 2 | Consecutive non-improving validation-MCC epochs allowed. |
| HEAD_ONLY_EPOCHS | 1 | Stabilizes the new classifier before encoder layers are unfrozen. |
| UNFREEZE_TOP_LAYERS | 4 | Limits adaptation to the encoder layers closest to the classifier. |
| HEAD_LEARNING_RATE | 1e-4 | Faster learning for newly initialized classifier layers. |
| ENCODER_LEARNING_RATE | 1e-5 | Conservative learning rate for pretrained DNABERT2 layers. |
| DROPOUT | 0.20 | Regularization for the classifier head. |
| PHYSICAL_BATCH_SIZE | 2 | T4-safe microbatch size. |
| GRADIENT_ACCUMULATION | 16 | Gives effective batch size 32 without T4 out-of-memory failures. |
| WARMUP_RATIO | 0.08 | Prevents abrupt early optimizer updates. |

Changing epochs or learning rate can improve validation MCC, but does not
guarantee a test gain. Choose a candidate solely by validation MCC, breaking
ties with validation AUPRC; run the test set only for that selected candidate.

The notebook creates a token-length audit for limits 70, 104, and 128. The
optional RUN_MAX_LENGTH_128_CANDIDATE switch is disabled by default because
it requires a second full training candidate. When enabled, the run is
recorded as a separate 128-token candidate and must be compared using
validation MCC/AUPRC only before any test result is considered.

## iPro-MP profile

iPro-MP is inference-only here, so epochs do not apply. The notebook uses the
official E. coli species ID 10 model, DNABERT-6, overlapping 6-mer
tokenization, maximum length 300, five official fold checkpoints, and
sequential T4 inference with batch size 4.

Each fold and split gets its own cached logits file. The notebook reports the
official five-fold arithmetic mean and evaluates median, trimmed mean, and
train-fitted logistic stackers with C = [0.01, 0.1, 1.0, 10.0]. The
stacker is fitted on train only. Validation MCC and AUPRC choose the
ensemble and threshold; the selected ensemble is then evaluated on test once.
The stacked result is not described as unchanged official iPro-MP inference.

## Runtime expectations

- DNABERT2: model download and token audit can take several minutes; six
  staged epochs on the full dataset can take hours on a T4. Keep the tab open.
- iPro-MP: first model-cache download can take several minutes; five folds
  over three splits can take hours depending on Drive and T4 throughput.
- A disconnect does not invalidate completed work. Rerun from the top and
  leave RESUME_MODE = latest.

## Reading the results

Use test MCC as the primary held-out comparison and test AUPRC as the
secondary measure. Accuracy alone is not sufficient. The manifest records the
selected threshold, candidate, seed, data hashes, model revision, runtime,
memory, and resume source. No historical score is copied into a new result
field: new result fields stay empty until a notebook run produces them.

## Known limitations

- Free Colab sessions can disconnect before a full run; resume support reduces
  repeated work but cannot make a disconnected process continue in the
  background.
- The official iPro-MP weights and DNABERT-6 files must be available in Drive
  or downloaded once from their documented sources.
- The optional 128-token DNABERT2 candidate increases runtime and is disabled
  by default.
- No full training was executed by Codex; this commit contains runnable
  benchmark code and does not fabricate performance results.
