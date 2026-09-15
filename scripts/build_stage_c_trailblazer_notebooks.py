#!/usr/bin/env python3
"""Generate the Colab handoff and milestone-evaluation companions for C20."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOKS = ROOT / "notebooks/titans_stage_c"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    payload: dict[str, object] = {
        "cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


def write(name: str, cells: list[dict[str, object]], *, gpu: bool = False) -> None:
    metadata: dict[str, object] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": name},
    }
    if gpu:
        metadata.update(accelerator="GPU")
    path = NOTEBOOKS / name
    path.write_text(json.dumps({"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n", encoding="utf-8")
    print(path)


HANDOFF_INTRO = """# Stage C 03r — adaptive E100 trailblazer handoff

Run this companion only after notebook 03m reports that the adaptive E25 scale
gate passed. It leaves the frozen 03l notebook and C19 run unchanged.

The notebook creates an immutable E25-to-CURC input bundle, uploads it to the
`ecoeus` bucket, generates the CURC allocation brief, and prints the laptop
commands used to transfer the bundle to Alpine. Its import cell performs the
reverse path for E50/E75/E100 milestone bundles: CURC → laptop → GCS → Colab →
Drive. No permanent Google credential is stored on CURC.
"""

HANDOFF_CONFIG = """# @title 1. Project and Drive configuration
PROJECT_ID='divine-tempo-502518-j4' # @param {type:"string"}
BUCKET='ecoeus' # @param {type:"string"}
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC' # @param {type:"string"}
GIT_REF='main' # @param {type:"string"}
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
STAGE_C_RUNNER='seqtrainer-titans-stage-c-colab-run'  # handoff operations themselves do not train

from pathlib import Path
from google.colab import auth, drive
import json, os, shutil, subprocess, sys

mount=Path('/content/drive')
if not (mount/'MyDrive').is_dir(): drive.mount(str(mount),timeout_ms=120000)
auth.authenticate_user()
subprocess.run(['gcloud','config','set','project',PROJECT_ID],check=True)
repo=Path('/content/SeqTrainer-trailblazer-support')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',f'origin/{GIT_REF}'],check=True)
launcher=repo/'scripts/stage_c_c20_curc.py'
control=repo/'scripts/stage_c_e25_control_gcp.py'
adaptive_gcp=repo/'scripts/stage_c_adaptive_e100_gcp.py'
for path in (launcher,control,adaptive_gcp):
    if not path.is_file(): raise FileNotFoundError(path)
print('Support checkout:',subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
"""

HANDOFF_EXPORT = """# @title 2. Export the passed E25 parent and upload the immutable CURC input
exports=Path('/content/stage_c_trailblazer_inputs')
subprocess.run([sys.executable,str(launcher),'export','--drive-root',DRIVE_ROOT,
 '--repo-root',str(repo),'--output',str(exports)],check=True)
bundles=sorted(path.parent for path in exports.glob('*/TRAILBLAZER_INPUT_MANIFEST.json'))
if not bundles: raise RuntimeError('No input bundle was produced')
INPUT_BUNDLE=bundles[-1]
INPUT_MANIFEST=json.loads((INPUT_BUNDLE/'TRAILBLAZER_INPUT_MANIFEST.json').read_text())
INPUT_URI=f"gs://{BUCKET}/stage-c/trailblazer/inputs/{INPUT_MANIFEST['bundle_id']}"
published=f"{INPUT_URI}/PUBLISHED_TRAILBLAZER_INPUT_MANIFEST.json"
probe=subprocess.run(['gcloud','storage','ls',published],text=True,capture_output=True)
if probe.returncode==0:
    raise RuntimeError(f'Immutable GCS input already exists: {published}')
subprocess.run(['gcloud','storage','rsync','--recursive',str(INPUT_BUNDLE),INPUT_URI],check=True)
subprocess.run(['gcloud','storage','cp','--if-generation-match=0',
 str(INPUT_BUNDLE/'TRAILBLAZER_INPUT_MANIFEST.json'),published],check=True)
brief=Path('/content/CURC_ASCENT_ALLOCATION_BRIEF.md')
subprocess.run([sys.executable,str(launcher),'allocation-brief','--output',str(brief)],check=True)
print(brief.read_text())
print('\\nINPUT URI:',INPUT_URI)
print('\\nON YOUR LAPTOP (not CURC):')
print(f"gcloud storage cp --recursive {INPUT_URI} ./stage-c-curc-input")
print("rsync -av --partial ./stage-c-curc-input/ <identikey>@login.rc.colorado.edu:/projects/<identikey>/stage-c/input/")
print('\\nThe CURC preflight/pilot/submit commands are documented in docs/titans_stage_c/TRAILBLAZER_E100_RUNBOOK.md.')
"""

HANDOFF_CONTROL = """# @title 3. GCP no-memory E25 commands (only after the same gate passed)
print('Pilot preflight:')
print(' '.join([sys.executable,str(control),'preflight','--project',PROJECT_ID,'--bucket',BUCKET,
 '--bundle',str(INPUT_BUNDLE),'--region','auto','--phase','pilot']))
print('\\nAfter downloading GCP_PILOT_REPORT.json, production preflight:')
print(' '.join([sys.executable,str(control),'preflight','--project',PROJECT_ID,'--bucket',BUCKET,
 '--bundle',str(INPUT_BUNDLE),'--region','auto','--phase','production',
 '--pilot-report','/content/GCP_PILOT_REPORT.json']))
print('Provision uses the same arguments with `provision` in place of `preflight`.')
"""


HANDOFF_ADAPTIVE = """# @title 4. Prepare adaptive GCP replica or CURC failover
ADAPTIVE_ROLE='gcp-replica' # @param ["gcp-replica", "gcp-failover"]
print('Production requires a private GOOGLE_CLOUD_CREDIT_ACTIVATION.json created only after the award is visible.')
command=[sys.executable,str(adaptive_gcp),'preflight','--project',PROJECT_ID,'--bucket',BUCKET,
 '--bundle',str(INPUT_BUNDLE),'--role',ADAPTIVE_ROLE,'--region','auto','--phase','production',
 '--credit-activation','/content/GOOGLE_CLOUD_CREDIT_ACTIVATION.json']
if ADAPTIVE_ROLE=='gcp-failover':
    print('First export CURC recovery:')
    print('python scripts/stage_c_c20_curc.py export-recovery --run-dir <curc-run> --output <recovery-output>')
    print('Transfer CURC -> laptop -> GCS -> Colab and add --recovery /content/curc-recovery below.')
    print('Production also requires --failover-authorization /content/FAILOVER_AUTHORIZATION.json.')
print('\\nAdaptive production preflight:')
print(' '.join(command))
print('Use `provision` in place of `preflight` only after this command passes.')
"""


HANDOFF_IMPORT = """# @title 5. Import one returned CURC or GCP milestone into Drive
MILESTONE='e50' # @param ["e50", "e75", "e100"]
MILESTONE_URI='' # @param {type:"string"}
if not MILESTONE_URI.startswith('gs://'):
    raise ValueError('Upload the CURC milestone bundle from your laptop to GCS, then paste its gs:// URI.')
download=Path('/content/curc-return')/MILESTONE
if download.exists(): shutil.rmtree(download)
download.mkdir(parents=True)
subprocess.run(['gcloud','storage','rsync','--recursive',MILESTONE_URI,str(download)],check=True)
verified=json.loads(subprocess.check_output([sys.executable,str(launcher),'verify-milestone','--bundle',str(download)],text=True))
if verified['milestone'] != MILESTONE: raise ValueError('Returned bundle has the wrong milestone label')
trajectory=verified.get('trajectory_id','adaptive-seed-20260751')
role=verified.get('execution_role','curc-primary')
if trajectory=='adaptive-seed-20260751' and role=='curc-primary':
    target=Path(DRIVE_ROOT)/'runs/c20_v3_medium_adaptive_e100_increment/milestones'/MILESTONE
else:
    target=Path(DRIVE_ROOT)/'runs/stage_c_adaptive_trajectories'/trajectory/role/'milestones'/MILESTONE
if target.exists():
    existing=json.loads((target/'MILESTONE_MANIFEST.json').read_text())
    if existing['checkpoint']['sha256'] != verified['checkpoint']['sha256']:
        raise RuntimeError(f'Refusing to replace a different Drive milestone: {target}')
    print('Identical milestone is already published:',target)
else:
    partial=target.with_name(target.name+'.partial')
    if partial.exists(): raise RuntimeError(f'Remove or inspect stale partial publication: {partial}')
    partial.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(download,partial)
    import hashlib
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        return h.hexdigest()
    if sha(partial/'latest.pt') != verified['checkpoint']['sha256']:
        raise RuntimeError('Drive copy checksum mismatch')
    if sha(partial/'model.pt') != verified['checkpoint']['sha256']:
        raise RuntimeError('Drive model.pt checksum mismatch')
    os.replace(partial,target)
pointer={
 'format_version':1,'milestone':MILESTONE,'checkpoint_path':str(target/'latest.pt'),
 'model_path':str(target/'model.pt'),'trajectory_id':trajectory,'execution_role':role,
 'scientific_seed':verified.get('scientific_seed'),
 'sha256':verified['checkpoint']['sha256'],
 'optimizer_step':verified['checkpoint']['optimizer_step'],
 'processed_bases':verified['checkpoint']['processed_bases'],
}
(target/'MODEL_POINTER.json').write_text(json.dumps(pointer,indent=2,sort_keys=True)+'\\n')
print('\\nPASTE THIS CHECKPOINT PATH INTO AN EVALUATION NOTEBOOK:')
print(target/'model.pt')
print(json.dumps(pointer,indent=2))
"""


EVAL_INTRO = """# Stage C 03s — E25/E50/E75/E100 milestone evaluation

This notebook evaluates an immutable adaptive trailblazer checkpoint. E25 and
E100 use the full analysis tier. E50 and E75 default to bounded held-out,
generation, memory, and context-anomaly diagnostics. Evaluations never write to
the training directory and do not pause the CURC continuation.
"""

EVAL_CONFIG = """# @title 1. Select the milestone
MILESTONE='e50' # @param ["e25", "e50", "e75", "e100"]
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC' # @param {type:"string"}
MODEL_PATH='' # @param {type:"string"}
RUN_FULL_INTERMEDIATE=False # @param {type:"boolean"}
GIT_REF='main' # @param {type:"string"}
TRUST_OWNED_CHECKPOINT=True

from pathlib import Path
from google.colab import drive
import hashlib, json, os, shutil, subprocess, sys
mount=Path('/content/drive')
if not (mount/'MyDrive').is_dir(): drive.mount(str(mount),timeout_ms=120000)
root=Path(DRIVE_ROOT)
if MODEL_PATH:
    checkpoint=Path(MODEL_PATH)
elif MILESTONE=='e25':
    checkpoint=root/'runs/c19_v3_medium_adaptive_e25/latest.pt'
else:
    checkpoint=root/'runs/c20_v3_medium_adaptive_e100_increment/milestones'/MILESTONE/'model.pt'
if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
dataset=root/'stage_c_dataset/ordered_streams/nonoverlap_6mer_v1'
panels=root/'study/stage_c_ecoli_medium_deep_memory_v3/panels'
protocol=root/'study/stage_c_ecoli_medium_deep_memory_v3/protocol.json'
taxonomy=root/'stage_c_dataset/manifests/accession_manifest.parquet'
output=root/'runs/c20_v3_medium_adaptive_e100_increment/milestone_evaluations'/MILESTONE
output.mkdir(parents=True,exist_ok=True)
for path in (dataset,panels/'validation.json',protocol,taxonomy):
    if not path.exists(): raise FileNotFoundError(path)
print('Checkpoint:',checkpoint)
"""

EVAL_BOOTSTRAP = """# @title 2. Bootstrap the pinned evaluator
repo=Path('/content/SeqTrainer-milestone-evaluation')
if not repo.exists(): subprocess.run(['git','clone','https://github.com/Gonza10V/SeqTrainer.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',f'origin/{GIT_REF}'],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
venv=Path('/content/seqtrainer-milestone-venv')
if not (venv/'bin/python').is_file():
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
python=venv/'bin/python'
subprocess.run([str(python),'-m','pip','install','--quiet','--no-deps','-e',str(repo)],check=True)
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state Stage C checkpoint requires explicit trust')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
def tool(name): return str(venv/'bin'/name)
stage_c_runner=tool('seqtrainer-titans-stage-c-colab-run')
def run(label,command):
    notebook_run=output/'notebook_runs'
    result=subprocess.run([stage_c_runner,'--run-dir',str(notebook_run),'--label',label,
      '--repo',str(repo),'--',*command],text=True,env=os.environ)
    if result.returncode:
        log=notebook_run/'logs'/f'{label}.log'
        raise RuntimeError(f'{label} failed; inspect {log}')
"""

EVAL_RUN = """# @title 3. Run the milestone tier
full=MILESTONE in {'e25','e100'} or RUN_FULL_INTERMEDIATE
limits=[] if full else ['--max-streams','4','--max-segments','256','--max-segments-per-accession','256']
run_id={'e25':'medium_adaptive_e25_analysis_v1','e50':'medium_adaptive_e50_analysis_v1',
 'e75':'medium_adaptive_e75_analysis_v1','e100':'medium_adaptive_e100_analysis_v1'}[MILESTONE]
amendment=repo/'studies/stage_c_ecoli_medium_deep_memory_v3/amendments/adaptive_trailblazer_e100_v1.json'
amendment_args=[] if MILESTONE in {'e25','e100'} else ['--protocol-amendment',str(amendment)]
run('heldout',[tool('seqtrainer-titans-stage-c-evaluate'),'--dataset-dir',str(dataset),
 '--panel-manifest',str(panels/'validation.json'),'--run',f'adaptive={checkpoint}',
 '--output-dir',str(output/'evaluation'),'--split','val','--comparison-mode','partial',
 '--device','cuda','--resume','--checkpoint-every-segments','512','--progress-every-segments','16',
 '--protocol',str(protocol),*amendment_args,'--run-id',run_id,*limits])
run('memory_behavior',[tool('seqtrainer-titans-stage-c-memory-behavior'),'--checkpoint',str(checkpoint),
 '--output',str(output/'memory_behavior.json'),'--pairs','64' if full else '16','--device','cuda'])
trace_segments='512' if full else '128'
run('memory_trace',[tool('seqtrainer-titans-stage-c-memory-trace'),'--dataset-dir',str(dataset),
 '--panel-manifest',str(panels/'validation.json'),'--checkpoint',str(checkpoint),
 '--output-dir',str(output/'memory_trace'),'--split','val','--memory-mode','adaptive',
 '--max-streams','12' if full else '4','--max-segments',trace_segments,'--device','cuda'])
if not shutil.which('prodigal'):
    subprocess.run(['apt-get','update'],check=True); subprocess.run(['apt-get','install','-y','prodigal'],check=True)
run('generation',[tool('seqtrainer-titans-stage-c-generate'),'--dataset-dir',str(dataset),
 '--panel-manifest',str(panels/'validation.json'),'--checkpoint',str(checkpoint),
 '--output-dir',str(output/'generation_t0p6'),'--split','val','--species','Escherichia coli',
 '--taxonomy-manifest',str(taxonomy),
 '--prompts','4' if full else '2','--prompt-tokens','128','--new-tokens','1024' if full else '256',
 '--temperatures','0.6','--top-k','1024','--top-p','0.99','--device','cuda',
 '--memory-mode','adaptive','--prodigal',shutil.which('prodigal')])
print('Core milestone evaluation complete:',output)
"""

EVAL_CONTEXT = """# @title 4. Run context anomaly and DNA-needle evaluation
registry=output/'context_registry'; inbox=registry/'inbox'; inbox.mkdir(parents=True,exist_ok=True)
source=inbox/'latest.pt'
if source.exists():
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        return h.hexdigest()
    if sha(source)!=sha(checkpoint):
        raise RuntimeError(f'Immutable context inbox already contains a different checkpoint: {source}')
else:
    partial=source.with_suffix('.pt.partial'); shutil.copy2(checkpoint,partial)
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        return h.hexdigest()
    if sha(partial)!=sha(checkpoint): raise RuntimeError('context checkpoint staging checksum failed')
    os.replace(partial,source)
inspect=r'''import json,sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); t=p.get('trainer_state',{})
print(json.dumps({'optimizer_step':int(t.get('optimizer_step',-1)),
 'processed_bases':int(t.get('processed_bases',-1)),'milestone':sys.argv[2],
 'code_commit':p.get('code_commit'),'dataset_fingerprint':p.get('dataset_fingerprint')}))'''
metadata_payload=json.loads(subprocess.check_output([str(python),'-c',inspect,str(source),MILESTONE],text=True,env=os.environ))
metadata=output/'context_metadata.json'
metadata.write_text(json.dumps(metadata_payload,indent=2,sort_keys=True)+'\\n')
context=tool('seqtrainer-titans-stage-c-context-eval')
run('context_stage',[context,'stage','--source',str(source),'--registry',str(registry),
 '--metadata-json',str(metadata),'--trust-owned-checkpoint'])
models=sorted((registry/'models').iterdir())
if not models: raise RuntimeError('context model registry is empty')
base_context=[context,'run','--dataset-dir',str(dataset),'--panel-manifest',str(panels/'validation.json'),
 '--model-dir',str(models[-1]),'--registry',str(registry),'--split','val',
 '--code-commit',commit,'--device','cuda','--trust-owned-checkpoint']
run('context_smoke',[*base_context,'--mode','smoke'])
if full: run('context_full',[*base_context,'--mode','full'])
run('context_catalog',[context,'catalog','--registry',str(registry)])
print('Context/anomaly evaluation complete:',registry)
"""


write(
    "03r_stage_c_v3_medium_adaptive_e100_curc_handoff.ipynb",
    [cell(HANDOFF_INTRO, "markdown"), cell(HANDOFF_CONFIG), cell(HANDOFF_EXPORT),
     cell(HANDOFF_CONTROL + "\n" + HANDOFF_ADAPTIVE), cell(HANDOFF_IMPORT)],
)
write(
    "03s_stage_c_v3_medium_adaptive_milestone_evaluation.ipynb",
    [cell(EVAL_INTRO, "markdown"), cell(EVAL_CONFIG), cell(EVAL_BOOTSTRAP),
     cell(EVAL_RUN), cell(EVAL_CONTEXT)], gpu=True,
)
