"""Build the portable Stage C 03q context-anomaly and needle Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks/titans_stage_c/03q_stage_c_c19_context_anomaly_and_needle.ipynb"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    result: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


INTRO = """# Stage C 03q — context anomaly and DNA needle evaluation

The first code cell asks for an experiment name and Drive root, mounts Drive,
and creates the complete experiment folder structure. Run that cell first,
copy your checkpoint to the exact printed `inbox/latest.pt` path, and then
continue with the bootstrap cell.

The notebook evaluates that immutable Stage C checkpoint without writing to a
live training directory. It stages a verified copy in the experiment registry,
runs the two-host smoke benchmark, and exposes opt-in full validation and an
explicitly locked test run.

Validation calibrates score thresholds. The test cell copies its completed
validation configuration and thresholds unchanged. Memory conditions here are
causal inference interventions on one checkpoint, not separately trained controls.
"""


SETUP = """# @title 1. Create the experiment folders
EXPERIMENT_NAME = "c19_context_eval" # @param {type:"string"}
ROOT_FOLDER = "/content/drive/MyDrive/SeqTrainerStageC" # @param {type:"string"}

from pathlib import Path
from google.colab import drive

mount=Path('/content/drive')
if not (mount/'MyDrive').is_dir():
    drive.mount(str(mount),timeout_ms=120000)

experiment_name=EXPERIMENT_NAME.strip()
if not experiment_name or experiment_name in {'.','..'} or '/' in experiment_name or '\\\\' in experiment_name:
    raise ValueError('EXPERIMENT_NAME must be one folder name without / or \\\\ characters.')
root=Path(ROOT_FOLDER).expanduser()
if not root.is_absolute():
    raise ValueError('ROOT_FOLDER must be an absolute Colab path such as /content/drive/MyDrive/SeqTrainerStageC')

# DRIVE_ROOT is the shared Stage C data/study root. Each named experiment gets
# its own immutable registry and catalog beneath it.
DRIVE_ROOT=str(root)
EXPERIMENT_DIR=root/experiment_name
REGISTRY=EXPERIMENT_DIR/'evaluation_registry'
CHECKPOINT_SOURCE=REGISTRY/'inbox'/'latest.pt'
for folder in (
    EXPERIMENT_DIR,
    REGISTRY/'inbox',
    REGISTRY/'models',
    REGISTRY/'evaluations'/'context_anomaly_v1',
    REGISTRY/'notebook_runs',
):
    folder.mkdir(parents=True,exist_ok=True)

print('Experiment directory:',EXPERIMENT_DIR)
print('Created/verified the evaluation folder structure.')
print('\\nNEXT ACTION: copy your checkpoint to exactly:')
print(CHECKPOINT_SOURCE)
if CHECKPOINT_SOURCE.is_file():
    print('\\nA latest.pt already exists and was left unchanged. You may continue to cell 2.')
else:
    print('\\nAfter latest.pt finishes uploading, run cell 2. Do not upload a checkpoint still being replaced by training.')
"""


BOOTSTRAP = """# @title 2. Verify inputs and bootstrap the evaluator
# Shared dataset/panel locations. Edit these only if your Stage C data lives elsewhere.
DATASET_DIR=f'{DRIVE_ROOT}/stage_c_dataset/ordered_streams/nonoverlap_6mer_v1'
VALIDATION_PANEL=f'{DRIVE_ROOT}/study/stage_c_ecoli_medium_deep_memory_v3/panels/validation.json'
TEST_PANEL=f'{DRIVE_ROOT}/study/stage_c_ecoli_medium_deep_memory_v3/panels/test.json'

REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='ab2a63432ebda70da046c567775de6963969a5a4'  # immutable 03q evaluator implementation
VENV_DIR='/content/seqtrainer-context-eval-v1'
TRUST_OWNED_CHECKPOINT=True
RUN_SMOKE=True
RUN_FULL=False                 # opt in only after smoke COMPLETE.json exists
RUN_LOCKED_TEST=False          # explicit final holdout switch
VALIDATION_BUNDLE=''           # completed full validation directory for locked test

from pathlib import Path
import json, os, subprocess, sys

if not Path(CHECKPOINT_SOURCE).is_file():
    raise FileNotFoundError(f'Checkpoint not found. Copy latest.pt to {CHECKPOINT_SOURCE}, wait for the upload to finish, and rerun this cell.')
repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout',GIT_REF],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()

venv=Path(VENV_DIR)
if not (venv/'bin'/'python').is_file():
    subprocess.run([sys.executable,'-m','pip','install','--quiet','virtualenv>=20.26'],check=True)
    subprocess.run([sys.executable,'-m','virtualenv','--system-site-packages',str(venv)],check=True)
python=str(venv/'bin'/'python')
subprocess.run([python,'-m','pip','install','--quiet','--upgrade','numpy==1.26.4','pandas==2.2.2','pyarrow==18.1.0'],check=True)
subprocess.run([python,'-m','pip','install','--no-deps','-e',str(repo)],check=True)
# Fail here with an actionable message if GIT_REF predates the 03q evaluator.
module_probe=subprocess.run(
    [python,'-c','from seqtrainer.torch.titans_paper_mac_stage_c.context_eval_cli import main; from seqtrainer.torch.titans_paper_mac_stage_c.colab_cli import main as wrapped_main'],
    text=True,capture_output=True,
)
if module_probe.returncode:
    raise RuntimeError(
        f'GIT_REF={GIT_REF!r} does not contain the Stage C 03q evaluator. '
        'Set GIT_REF to the pushed branch or commit containing context_eval.py and context_eval_cli.py, then rerun this cell. '
        f'Import error: {module_probe.stderr.strip()}'
    )
# Invoke modules through the isolated interpreter rather than relying on pip to
# refresh console-script shims when this Colab virtualenv already exists.
# These are the module equivalents of seqtrainer-titans-stage-c-context-eval
# and seqtrainer-titans-stage-c-colab-run.
evaluator=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.context_eval_cli']
stage_c_runner=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.colab_cli']
for path in (Path(CHECKPOINT_SOURCE),Path(DATASET_DIR),Path(VALIDATION_PANEL)):
    if not path.exists(): raise FileNotFoundError(path)
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state checkpoint loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
print('Evaluation commit:',commit)
"""


STAGE = """# Verified copy-through-partial staging; the source is hashed before and after.
inspect=r'''import json,sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
t=p.get('trainer_state',{})
print(json.dumps({'optimizer_step':int(t.get('optimizer_step',0)),
 'processed_bases':int(t.get('processed_bases',0)),'model_config':p.get('model_config'),
 'code_commit':p.get('code_commit'),'dataset_fingerprint':p.get('dataset_fingerprint'),
 'copy_time_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}))'''
metadata=json.loads(subprocess.check_output([python,'-c',inspect,CHECKPOINT_SOURCE],text=True,env=os.environ))
metadata_path=Path('/content/context_checkpoint_metadata.json')
metadata_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\\n')
import hashlib
checkpoint_sha=hashlib.sha256(Path(CHECKPOINT_SOURCE).read_bytes()).hexdigest()
MODEL_DIR=str(Path(REGISTRY)/'models'/f"{metadata['optimizer_step']}_{checkpoint_sha[:12]}")
NOTEBOOK_RUN_DIR=str(Path(REGISTRY)/'notebook_runs'/Path(MODEL_DIR).name)
def wrapped(label,command):
    subprocess.run([*stage_c_runner,'--run-dir',NOTEBOOK_RUN_DIR,'--label',label,
      '--repo',str(repo),'--',*command],check=True,env=os.environ)
wrapped('stage_checkpoint',[*evaluator,'stage','--source',CHECKPOINT_SOURCE,'--registry',REGISTRY,
 '--metadata-json',str(metadata_path),'--trust-owned-checkpoint'])
print('Immutable model:',MODEL_DIR)
if RUN_SMOKE:
    command=[*evaluator,'run','--dataset-dir',DATASET_DIR,'--panel-manifest',VALIDATION_PANEL,
      '--model-dir',MODEL_DIR,'--registry',REGISTRY,'--split','val','--mode','smoke',
      '--code-commit',commit,'--device','auto','--trust-owned-checkpoint']
    wrapped('smoke_validation',command)
else: print('Smoke disabled.')
"""


FULL = """if RUN_FULL:
    model_id=Path(MODEL_DIR).name
    smoke=list((Path(REGISTRY)/'evaluations'/'context_anomaly_v1'/model_id/'val').glob('*/COMPLETE.json'))
    if not smoke: raise RuntimeError('A completed smoke bundle is required before full validation.')
    command=[*evaluator,'run','--dataset-dir',DATASET_DIR,'--panel-manifest',VALIDATION_PANEL,
      '--model-dir',MODEL_DIR,'--registry',REGISTRY,'--split','val','--mode','full',
      '--code-commit',commit,'--device','auto','--trust-owned-checkpoint']
    wrapped('full_validation',command)
else: print('Full validation is opt-in (RUN_FULL=False).')

if RUN_LOCKED_TEST:
    validation=Path(VALIDATION_BUNDLE)
    if not validation.is_dir() or not (validation/'COMPLETE.json').is_file():
        raise FileNotFoundError('VALIDATION_BUNDLE must name a completed validation bundle.')
    command=[*evaluator,'run','--dataset-dir',DATASET_DIR,'--panel-manifest',TEST_PANEL,
      '--model-dir',MODEL_DIR,'--registry',REGISTRY,'--split','test',
      '--validation-bundle',str(validation),'--run-locked-test',
      '--code-commit',commit,'--device','auto','--trust-owned-checkpoint']
    wrapped('locked_test',command)
else: print('Locked test disabled. This is the safe default.')

# Scan immutable completed bundles and rebuild the catalog from manifests.
wrapped('catalog',[*evaluator,'catalog','--registry',REGISTRY])
model_id=Path(MODEL_DIR).name
bundles=sorted((Path(REGISTRY)/'evaluations'/'context_anomaly_v1'/model_id).glob('*/*/COMPLETE.json'))
for complete in bundles:
    report=complete.parent/'REPORT.md'
    print('\\nBUNDLE:',complete.parent)
    if report.is_file(): print(report.read_text())
"""


notebook = {
    "cells": [
        cell(INTRO, "markdown"), cell(SETUP), cell(BOOTSTRAP), cell(STAGE), cell(FULL),
    ],
    "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
