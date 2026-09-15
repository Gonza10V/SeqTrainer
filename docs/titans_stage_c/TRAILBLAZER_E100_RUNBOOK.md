# Adaptive E25-to-E100 trailblazer runbook

This workflow finishes the existing adaptive C19 E25 run in Colab, gates all
new spending on notebook 03m, runs the matched no-memory E25 control on Google
Cloud, and continues only the adaptive model through E100 on CURC Alpine.
Notebook 03l and the frozen v3 protocol are never edited.

The optional Google Cloud Research Credits extension protects this trajectory:
GCP either resumes the primary seed after a CURC failure or trains independent
seed `20260752`. It never performs both roles under one run identity.

## 0. Apply for Google Cloud Research Credits

Use `docs/titans_stage_c/GOOGLE_CLOUD_RESEARCH_CREDITS_APPLICATION.md` as the
copy-ready application. Complete the private Billing Account ID and shareable
Pricing Calculator URL only in Google's form. Submission is manual because the
applicant must personally review the program representations and terms.

Production adaptive GCP preflight requires an untracked activation record:

```json
{
  "active": true,
  "project": "divine-tempo-502518-j4",
  "award_usd": 5000,
  "award_id": "<award-id>",
  "billing_account_id": "<billing-account-id>",
  "gpu_hour_allocations": {
    "gcp-failover": 450,
    "gcp-replica": 582
  }
}
```

Store this outside the repository as `GOOGLE_CLOUD_CREDIT_ACTIVATION.json`.
Do not create it until the award is visible on the intended billing account.
The adaptive role allocations total 1,032 hours and cannot exceed the campaign's
1,250-hour calculator cap. The remaining allowance covers the independently
capped no-memory control and operational headroom.

## 1. Finish and gate E25 in Colab

1. Run `03l_stage_c_v3_medium_adaptive_e25.ipynb` until its final architecture
   and resume-verification cells pass.
2. Confirm `run_manifest.json` reports `scheduler_exhausted: true` and
   `stop_reason: panel_exhausted`, and `LIVE_STATUS.json` reports `completed`.
3. Run `03m_stage_c_v3_medium_e25_analysis_and_gate.ipynb`.
4. Stop if `scale_analysis_v1/gate/scale_gate.json` does not contain
   `"proceed": true`.
5. Open `03r_stage_c_v3_medium_adaptive_e100_curc_handoff.ipynb`. Its export
   cell creates and uploads the immutable post-gate bundle to project
   `divine-tempo-502518-j4`, bucket `ecoeus`.

The new amendment is prospective and additive. E50 and E75 are observational
snapshots; the only discretionary training gate is E25.

## 2. Request CURC Ascent

The handoff notebook prints `CURC_ASCENT_ALLOCATION_BRIEF.md`. Use it in the
Ascent request, adding the PI, funding, field, collaborator Identikeys, and
account name requested by the form. The estimate is one A100-40GB GPU for
approximately 450 hours, eight CPU cores per job, and about 53,000 service
units. The standard 350,000-SU Ascent allocation leaves ample queue and future
replication margin.

Do not use the Trailhead auto-allocation for production. It is suitable only
for setup and the one-hour `gpu-testing` pilot.

## 3. Transfer input through the laptop

Run the commands printed by notebook 03r. The route is:

```text
Drive → Colab verified bundle → gs://ecoeus → laptop → CURC project storage
```

Do not install a service-account key or permanent Google credential on CURC.
Use `gcloud storage cp --recursive` on the laptop, followed by `rsync -av
--partial` to CURC. Keep the immutable input under `/projects`; copy the active
dataset and run directory to Alpine scratch if the project filesystem is too
slow. The launcher rejects insufficient free space.

## 4. Prepare and pilot on CURC

On CURC, set explicit paths without repurposing system variables:

```bash
TRAIL_REPO=/projects/<identikey>/SeqTrainer
TRAIL_INPUT=/projects/<identikey>/stage-c/input
TRAIL_WORK=/scratch/alpine/<identikey>/stage-c
TRAIL_IMAGE=/projects/<identikey>/containers/seqtrainer-pytorch.sif
TRAIL_VENV=/projects/<identikey>/stage-c/venv
TRAIL_LAUNCHER=$TRAIL_REPO/scripts/stage_c_c20_curc.py
```

Checkout the cloud-support revision containing this workflow. Build or select
an Apptainer NVIDIA/PyTorch image whose PyTorch numeric stack matches
`TRAILBLAZER_INPUT_MANIFEST.json`; the launcher refuses version drift.

```bash
python3 "$TRAIL_LAUNCHER" prepare-runtime \
  --bundle "$TRAIL_INPUT" --repo "$TRAIL_REPO" \
  --image "$TRAIL_IMAGE" --venv "$TRAIL_VENV"

python3 "$TRAIL_LAUNCHER" preflight \
  --bundle "$TRAIL_INPUT" --repo "$TRAIL_REPO" \
  --image "$TRAIL_IMAGE" --python-bin "$TRAIL_VENV/bin/python" \
  --work-root "$TRAIL_WORK"
```

Submit the `pilot` command through `aa100`, `gpu-testing`, one
`a100-40gb`, and a one-hour limit. Inside that allocation, add `--on-compute`
to `preflight`, then run:

```bash
python3 "$TRAIL_LAUNCHER" pilot \
  --bundle "$TRAIL_INPUT" --repo "$TRAIL_REPO" \
  --image "$TRAIL_IMAGE" --python-bin "$TRAIL_VENV/bin/python" \
  --output "$TRAIL_WORK/pilot"
```

The pilot must pass exact E25 resume verification and advance exactly one
optimizer step. Do not submit production without `CURC_PILOT_REPORT.json`.

## 5. Submit the adaptive continuation

```bash
python3 "$TRAIL_LAUNCHER" submit \
  --bundle "$TRAIL_INPUT" --repo "$TRAIL_REPO" \
  --image "$TRAIL_IMAGE" --python-bin "$TRAIL_VENV/bin/python" \
  --account <ascent-account> \
  --pilot-report "$TRAIL_WORK/pilot/CURC_PILOT_REPORT.json" \
  --run-dir "$TRAIL_WORK/c20_v3_medium_adaptive_e100_increment" \
  --output "$TRAIL_WORK/control"
```

The launcher calculates safe chunks from the measured A100 throughput and
submits them with `afterok` dependencies under `gpu-long`. Each job requests
one A100-40GB, eight CPUs, 64 GB RAM, and at most 160 hours. Unexpected failure
stops the chain. E50, E75, and E100 produce immutable bundles in the control
output; internal wall-time chunks do not become scientific milestones.

Check progress with:

```bash
python3 "$TRAIL_LAUNCHER" status \
  --run-dir "$TRAIL_WORK/c20_v3_medium_adaptive_e100_increment" \
  --output "$TRAIL_WORK/control"
```

## 6. Return and evaluate milestones

For each generated `e50-step-*`, `e75-step-*`, or `e100-step-*` directory:

1. Download it from CURC to the laptop with `rsync -av --partial`.
2. Upload it from the laptop to `gs://ecoeus/stage-c/trailblazer/returns/`.
3. Paste that exact `gs://` directory into notebook 03r's import cell.
4. The notebook verifies every SHA-256, publishes through a `.partial`
   directory, writes `MODEL_POINTER.json`, and prints the Drive `model.pt`
   path.
5. Open `03s_stage_c_v3_medium_adaptive_milestone_evaluation.ipynb` and select
   the milestone. E50/E75 default to bounded diagnostics; E25/E100 use the full
   tier.

CURC training does not wait for E50/E75 evaluation. Only non-finite training,
configuration drift, corrupted state, checkpoint regression, or a failed
dependent job stops the trailblazer.

## 7. Run the GCP no-memory E25 control

After E25 passes, notebook 03r prints the GCP commands. First provision the
`pilot` phase. Download its `GCP_PILOT_REPORT.json`, then run the `production`
preflight. Production is refused if the measured standard-A100 projection plus
storage exceeds the $250 working cap. If refused, do not switch silently to
Spot; defer the control to CURC after adaptive E100.

The control VM uses one private `a2-highgpu-1g`, a non-auto-deleted persistent
SSD, no Jupyter server, and a 36-hour `STOP` runtime limit. A later session
resumes only its own verified no-memory checkpoint. Run `repatriate` only after
Cloud Storage contains a completed, panel-exhausted bundle.

Budget alerts do not stop spending. Persistent disks and Cloud Storage remain
billable after a VM stops. Cleanup is manual and only occurs after Drive hashes
have been checked; the workflow never deletes them automatically.

## 8. Use the grant for failover or seed 2

The adaptive launcher is `scripts/stage_c_adaptive_e100_gcp.py`. First run a
two-hour pilot. Production additionally requires the private activation JSON.

If CURC has not started within 14 days, or the primary run has a non-recoverable
failure, export its newest state on CURC:

```bash
python scripts/stage_c_c20_curc.py export-recovery \
  --run-dir "$TRAIL_WORK/c20_v3_medium_adaptive_e100_increment" \
  --output "$TRAIL_WORK/recovery"
```

Transfer the immutable recovery bundle CURC → laptop → GCS/Colab, then run:

Create an untracked `FAILOVER_AUTHORIZATION.json`. For a missed start window:

```json
{
  "authorized": true,
  "reason": "curc_not_started_14_days",
  "authorized_by": "Gonzalo Vidal Peña",
  "curc_planned_start_at": "2026-10-15T00:00:00-06:00",
  "authorized_at": "2026-10-29T00:00:00-06:00"
}
```

For a hard failure, use reason `curc_nonrecoverable_failure` and add a
`failure_evidence` path or job identifier. The launcher validates these triggers.

```bash
python scripts/stage_c_adaptive_e100_gcp.py preflight \
  --project divine-tempo-502518-j4 --bucket ecoeus \
  --bundle <trailblazer-input> --role gcp-failover \
  --recovery <curc-recovery> --region auto --phase production \
  --credit-activation <private-activation-json> \
  --failover-authorization <private-failover-authorization-json>
```

If CURC is healthy, omit `--recovery` and select the independent trajectory:

```bash
python scripts/stage_c_adaptive_e100_gcp.py preflight \
  --project divine-tempo-502518-j4 --bucket ecoeus \
  --bundle <trailblazer-input> --role gcp-replica \
  --region auto --phase production \
  --credit-activation <private-activation-json>
```

Replace `preflight` with `provision` only after it passes. The worker uses a
private standard `a2-highgpu-1g`, a non-auto-deleted SSD, a 36-hour STOP limit,
and immutable checkpoint backups. Each restart continues the current stage.
E25, E50, E75, and E100 create step-and-hash-named milestone bundles containing
both `latest.pt` and evaluation-stable `model.pt`.

Failover has priority. If seed 2 is already running when CURC becomes
unrecoverable, stop its VM normally after a verified checkpoint, preserve its
disk, complete seed 1 using a separate failover VM/disk/prefix, and resume seed
2 only if credits remain.
