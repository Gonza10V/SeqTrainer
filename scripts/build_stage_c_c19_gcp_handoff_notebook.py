"""Generate the Colab companion for the frozen 03l-to-Google-Cloud handoff."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks/titans_stage_c/03l_stage_c_v3_medium_adaptive_e25_gcp_handoff.ipynb"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    payload: dict[str, object] = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


cells = [
    cell(
        "# Stage C 03l — C19 Google Cloud handoff\n\n"
        "This companion leaves notebook 03l unchanged. Stop its training process before "
        "export. The cutover lock forbids concurrent Colab training; the final cell only "
        "publishes a fully verified, completed E25 run. The VM stops after 36 hours or "
        "successful completion. Budget alerts do not stop spending; the SSD and Cloud "
        "Storage remain billable and must be reviewed for manual cleanup after return. "
        "Cleanup is never automatic: use `gcloud compute instances delete stage-c-c19 "
        "--zone ZONE --keep-disks=data`, then (only after verified Drive return) "
        "`gcloud compute disks delete stage-c-c19 --zone ZONE` and `gcloud storage rm "
        "--recursive gs://BUCKET/stage-c-c19`.\n",
        "markdown",
    ),
    cell(
        "# REQUIRED OPERATOR INPUTS (the first executable cell)\n"
        "GIT_REF='main'  # cloud-support branch; C19 training itself is pinned separately to ae72fae…\n"
        "PROJECT_ID=input('Google Cloud project ID: ').strip()\n"
        "BUCKET=input('Cloud Storage bucket name (without gs://): ').strip()\n"
        "REGION=input('Preferred region [auto]: ').strip() or 'auto'\n"
        "DRIVE_ROOT=input('Drive root [/content/drive/MyDrive/SeqTrainerStageC]: ').strip() or '/content/drive/MyDrive/SeqTrainerStageC'\n"
        "if not PROJECT_ID or not BUCKET: raise ValueError('Project ID and bucket are required')\n"
    ),
    cell(
        "from pathlib import Path\n"
        "from google.colab import auth, drive\n"
        "import json, subprocess, sys\n"
        "mount=Path('/content/drive')\n"
        "if not (mount/'MyDrive').is_dir(): drive.mount(str(mount),timeout_ms=120000)\n"
        "auth.authenticate_user()\n"
        "subprocess.run(['gcloud','config','set','project',PROJECT_ID],check=True)\n"
        "repo=Path('/content/SeqTrainer-cloud-support')\n"
        "if not repo.exists(): subprocess.run(['git','clone','https://github.com/Gonza10V/SeqTrainer.git',str(repo)],check=True)\n"
        "subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)\n"
        "subprocess.run(['git','-C',str(repo),'checkout','--detach',f'origin/{GIT_REF}'],check=True)\n"
        "launcher=repo/'scripts/stage_c_c19_gcp.py'\n"
        "if not launcher.is_file(): raise FileNotFoundError('Cloud-support launcher is not present on the selected support branch')\n"
        "# Cloud systemd logs replace seqtrainer-titans-stage-c-colab-run for this unattended handoff.\n"
        "print('Training must be stopped in 03l. Export will refuse LIVE_STATUS=running.')\n"
    ),
    cell(
        "# Stable-copy only the declared dataset, panels, C18 result, study records, and full C19 run.\n"
        "exports=Path('/content/c19_gcp_cutovers')\n"
        "command=[sys.executable,str(launcher),'export','--drive-root',DRIVE_ROOT,'--output',str(exports),'--bucket',BUCKET]\n"
        "subprocess.run(command,check=True)\n"
        "bundles=sorted(exports.glob('step-*'))\n"
        "if not bundles: raise RuntimeError('No cutover was produced')\n"
        "BUNDLE=bundles[-1]\n"
        "CUTOVER=json.loads((BUNDLE/'CUTOVER_MANIFEST.json').read_text())\n"
        "print('Immutable cutover:',CUTOVER['cutover_id'])\n"
        "print('Resume step:',CUTOVER['checkpoint']['optimizer_step'])\n"
        "print('Post-checkpoint telemetry (archived, not resumable):',len(CUTOVER['post_checkpoint_telemetry']))\n\n"
        "# Validate billing, both GPU quotas, bucket, checksums, disk sizing, and the $250 working cap.\n"
        "preflight=[sys.executable,str(launcher),'preflight','--project',PROJECT_ID,'--bucket',BUCKET,'--region',REGION,'--bundle',str(BUNDLE)]\n"
        "subprocess.run(preflight,check=True)\n"
        "print('Preflight passed. Provisioning is deliberately a separate operator action.')\n"
        "print('Run:',' '.join([sys.executable,str(launcher),'provision','--project',PROJECT_ID,'--bucket',BUCKET,'--region',REGION,'--bundle',str(BUNDLE)]))\n"
    ),
    cell(
        "# FINAL RETURN — run only after status reports completion. Publication is atomic and guarded.\n"
        "if 'BUNDLE' in globals() and (BUNDLE/'CUTOVER_MANIFEST.json').is_file():\n"
        "    manifest=BUNDLE/'CUTOVER_MANIFEST.json'\n"
        "else:\n"
        "    lock=json.loads((Path(DRIVE_ROOT)/'runs/c19_v3_medium_adaptive_e25/GCP_CUTOVER_LOCK.json').read_text())\n"
        "    manifest=Path('/content/CUTOVER_MANIFEST.json')\n"
        "    subprocess.run(['gcloud','storage','cp',f\"gs://{BUCKET}/stage-c-c19/cutovers/{lock['cutover_id']}/CUTOVER_MANIFEST.json\",str(manifest)],check=True)\n"
        "subprocess.run([sys.executable,str(launcher),'repatriate','--bucket',BUCKET,'--drive-root',DRIVE_ROOT,'--cutover-manifest',str(manifest)],check=True)\n"
        "print('Verified completed C19 bundle published to Drive. The pre-cutover checkpoint is archived.')\n"
    ),
]

payload = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": OUTPUT.name},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
