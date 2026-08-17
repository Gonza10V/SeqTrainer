"""Build the five-cell, sub-24-hour C19 anomaly-and-needle notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/titans_stage_c/03q_stage_c_c19_context_anomaly_and_needle.ipynb"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    value: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


RATIONALE = r"""# Stage C 03q — bounded C19 anomaly detection and DNA-needle validation

This validation-only study freezes its synthetic tests before loading a model, qualifies the planned C19 work against a 22-hour A100 budget, and then evaluates the immutable final C19 checkpoint. C16 is disabled by default and can be added later, in a separate resumable session, on the byte-identical frozen panel.

The bounded contract uses eight held-out hosts, near/far E25 donors, 1/16/64-segment replacements at depth 16, and needle distances 3/16/64 with zero or sixteen near-key distractors. Primary anomaly endpoints are leave-one-host-out AUPRC, TPR at 1% FPR, false positives/Mb, localization IoU, and boundary error. Primary needle endpoints are carried-minus-reset target log probability, Recall@1, exact recovery, and long-distance performance. Hosts are the inferential units.

Memory telemetry, predictive performance, and mechanistic diagnostics are reported separately. The protected test panel is unavailable to this workflow, and no horizontal-transfer, function, pathogenicity, or causal adaptive-memory claim is made.
"""


CONFIG = r'''# @title 2. Immutable inputs and execution switches
RUN_C19=True
RUN_C16_COMPARISON=False  # rerun later to add C16 on the completed C19 panel
ROOT_FOLDER='/content/drive/MyDrive/SeqTrainerStageC'
DRIVE_ROOT=ROOT_FOLDER
EXPERIMENT_NAME='c19_bounded_anomaly_needle_v1'
MAX_C19_HOURS=22.0

C19_CHECKPOINT=f'{ROOT_FOLDER}/runs/c19_v3_medium_adaptive_e25/latest.pt'
C16_CHECKPOINT=f'{ROOT_FOLDER}/runs/c16_v3_medium_adaptive_5m/latest.pt'
EXPECTED_CHECKPOINT_SHA256={
 'C19':'07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a',
 'C16':'21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541',
}
DATASET_DIR=f'{ROOT_FOLDER}/stage_c_dataset/ordered_streams/nonoverlap_6mer_v1'
VALIDATION_PANEL=f'{ROOT_FOLDER}/study/stage_c_ecoli_medium_deep_memory_v3/panels/validation.json'
E25_TRAINING_PANEL=f'{ROOT_FOLDER}/study/stage_c_ecoli_medium_deep_memory_v3/panels/e25.json'
ANI_PAIRS=f'{ROOT_FOLDER}/stage_c_dataset/manifests/ecoli_skani_triangle_extended.tsv'
ANI_MEMBERSHIP=f'{ROOT_FOLDER}/stage_c_dataset/manifests/ecoli_ani_membership.parquet'

REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='2e869da44c2fb00c93101f72dfe7a88074fb4e2c'
TRUST_OWNED_CHECKPOINT=True
'''


PREFLIGHT = r'''# @title 3. Mount Drive, test the evaluator, freeze cases, and qualify runtime
from pathlib import Path
from google.colab import drive
import errno,hashlib,json,os,subprocess,sys,torch

mount=Path('/content/drive')
def initialize_drive():
 """Return a write-qualified registry, remounting once after Drive FUSE EIO."""
 for attempt in range(2):
  try:
   if attempt or not (mount/'MyDrive').is_dir():
    if attempt:
     print('Drive returned Errno 5; forcing one clean remount...')
     try: drive.flush_and_unmount()
     except Exception as error: print('Unmount warning:',repr(error))
   drive.mount(str(mount),force_remount=bool(attempt),timeout_ms=120000)
   root=Path(ROOT_FOLDER)
   if not root.is_dir():
    if not attempt: raise OSError(errno.EIO,'Drive root is not readable',str(root))
    raise FileNotFoundError(root)
   probe=root/'.03q_drive_write_probe'
   probe.write_text('03q-drive-ok\n',encoding='utf-8')
   if probe.read_text(encoding='utf-8')!='03q-drive-ok\n':
    raise OSError(errno.EIO,'Drive write/read probe mismatch',str(probe))
   probe.unlink()
   experiment=root/EXPERIMENT_NAME
   registry=experiment/'validation_registry'
   registry.mkdir(parents=True,exist_ok=True)
   return root,experiment,registry
  except OSError as error:
   if attempt or error.errno not in {errno.EIO,errno.ESTALE,errno.ENOTCONN}:
    raise
 raise RuntimeError('Google Drive remained unavailable after a forced remount.')

root,experiment,registry=initialize_drive()
bundle=registry/'bounded'; panel=bundle/'frozen_panel'
repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',GIT_REF],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if commit != GIT_REF: raise RuntimeError('Evaluator commit did not resolve exactly.')

venv=Path('/content/seqtrainer-03q-bounded-v1')
if not (venv/'bin/python').is_file():
 subprocess.run([sys.executable,'-m','pip','install','--quiet','virtualenv>=20.26'],check=True)
 subprocess.run([sys.executable,'-m','virtualenv','--system-site-packages',str(venv)],check=True)
python=str(venv/'bin/python')
subprocess.run([python,'-m','pip','install','--quiet','--upgrade',
 'numpy==1.26.4','pandas==2.2.2','pyarrow==18.1.0','scipy>=1.11,<2',
 'scikit-learn>=1.3,<2','matplotlib>=3.7,<4','pytest>=8,<9'],check=True)
subprocess.run([python,'-m','pip','install','--no-deps','-e',str(repo)],check=True)
subprocess.run([python,'-m','pytest','-q',
 str(repo/'tests/test_titans_paper_mac_stage_c_anomaly_study.py'),
 str(repo/'tests/test_titans_paper_mac_stage_c_context_eval.py'),
 str(repo/'tests/test_titans_paper_mac_stage_c_model.py')],check=True)
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
 raise RuntimeError('03q bounded execution requires an NVIDIA A100 runtime.')

def sha256_file(path):
 digest=hashlib.sha256()
 with open(path,'rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
 return digest.hexdigest()

required_models={'C19':C19_CHECKPOINT}
if RUN_C16_COMPARISON: required_models['C16']=C16_CHECKPOINT
for model,path in required_models.items():
 if not Path(path).is_file(): raise FileNotFoundError(path)
 if sha256_file(path) != EXPECTED_CHECKPOINT_SHA256[model]:
  raise RuntimeError(f'{model} checkpoint hash mismatch')
for path in (DATASET_DIR,VALIDATION_PANEL,E25_TRAINING_PANEL,ANI_PAIRS,ANI_MEMBERSHIP):
 if not Path(path).exists(): raise FileNotFoundError(path)
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

runner=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study_cli']
stage_c_runner=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.colab_cli'] # seqtrainer-titans-stage-c-colab-run
def run_checked(label,command):
 wrapped=[*stage_c_runner,'--run-dir',str(registry/'notebook_runs'),'--label',label,
          '--repo',str(repo),'--',*command]
 result=subprocess.run(wrapped)
 if result.returncode:
  log=registry/'notebook_runs'/'logs'/f'{label}.log'
  raise RuntimeError(f'{label} failed; log={log}\n{log.read_text(errors="replace")[-20000:]}')

run_checked('bounded_freeze',[*runner,'freeze','--dataset-dir',DATASET_DIR,
 '--validation-panel',VALIDATION_PANEL,'--e25-panel',E25_TRAINING_PANEL,
 '--ani-pairs',ANI_PAIRS,'--ani-membership',ANI_MEMBERSHIP,
 '--mode','bounded','--output',str(panel)])
panel_manifest=json.loads((panel/'frozen_panel_manifest.json').read_text())
if panel_manifest['mode']!='bounded' or panel_manifest['planned_workload']['total_segment_forwards']!=25344:
 raise RuntimeError('Frozen panel is not the exact bounded 03q contract.')

projection=registry/'runtime_projection.json'
if not projection.is_file() or json.loads(projection.read_text()).get('panel_contract_sha256')!=panel_manifest['panel_contract_sha256']:
 run_checked('bounded_runtime_projection',[*runner,'estimate-runtime',
  '--checkpoint',C19_CHECKPOINT,'--frozen-panel',str(panel),'--output',str(projection),
  '--device','cuda','--probe-forwards','64','--max-hours',str(MAX_C19_HOURS),
  '--trust-owned-checkpoint'])
runtime=json.loads(projection.read_text())
if not runtime['accepted'] or runtime['projected_hours']>MAX_C19_HOURS:
 raise RuntimeError(f"C19 projection {runtime['projected_hours']:.2f} h exceeds budget")
print({'commit':commit,'gpu':torch.cuda.get_device_name(0),'panel':panel_manifest['panel_contract_sha256'],
       'planned_forwards':panel_manifest['planned_workload']['total_segment_forwards'],
       'projected_hours':runtime['projected_hours']})
'''


EXECUTION = r'''# @title 4. Run C19 first; optionally add C16 and paired comparison later
def run_model(model,checkpoint,max_hours=None):
 command=[*runner,'run-model','--model',model,'--checkpoint',checkpoint,
  '--dataset-dir',DATASET_DIR,'--validation-panel',VALIDATION_PANEL,
  '--frozen-panel',str(panel),'--output',str(bundle/model),'--device','cuda',
  '--trust-owned-checkpoint']
 if max_hours is not None: command.extend(['--max-runtime-hours',str(max_hours)])
 run_checked(f'bounded_{model}',command)
 return (bundle/model/'COMPLETE.json').is_file()

c19_complete=(bundle/'C19'/'COMPLETE.json').is_file()
if RUN_C19 and not c19_complete: c19_complete=run_model('C19',C19_CHECKPOINT,MAX_C19_HOURS)
if not c19_complete:
 paused=bundle/'C19'/'PAUSED.json'
 print('C19 paused safely; rerun this cell unchanged.',paused.read_text() if paused.is_file() else '')
else:
 print('C19 complete:',bundle/'C19')

c16_complete=(bundle/'C16'/'COMPLETE.json').is_file()
if c19_complete and RUN_C16_COMPARISON and not c16_complete:
 c16_complete=run_model('C16',C16_CHECKPOINT,MAX_C19_HOURS)
if RUN_C16_COMPARISON and not c16_complete:
 print('C16 comparison remains incomplete; rerun later without changing the frozen panel.')

analysis=bundle/'analysis'
if c19_complete:
 run_checked('bounded_analyze',[*runner,'compare','--input',str(bundle),'--output',str(analysis)])
print({'c19_complete':c19_complete,'c16_complete':c16_complete,'analysis':str(analysis)})
'''


ANALYSIS = r'''# @title 5. Validate, display, and content-address the available report
from IPython.display import Markdown,display
if not (bundle/'C19'/'COMPLETE.json').is_file():
 print('No report yet: rerun cell 4 to resume C19.')
else:
 analysis=bundle/'analysis'
 for required in ('COMPLETE.json','analysis_contract.json','SCIENTIFIC_REPORT.md',
                  'segment_metrics.parquet','token_metrics.parquet','block_metrics.parquet',
                  'needle_metrics.parquet','statistical_tests.parquet','out_of_fold_predictions.parquet'):
  if not (analysis/required).is_file(): raise FileNotFoundError(analysis/required)
 for model in ('C19','C16'):
  if not (bundle/model/'COMPLETE.json').is_file(): continue
  run_manifest=json.loads((bundle/model/'model_run_manifest.json').read_text())
  if run_manifest['panel_contract_sha256'] != panel_manifest['panel_contract_sha256']:
   raise RuntimeError(f'Refusing protocol-mismatched {model} result.')
 complete=json.loads((analysis/'COMPLETE.json').read_text())
 models='_'.join(complete['models'])
 archive=Path(str(analysis)+'.zip')
 archive_sha=sha256_file(archive)
 retained=analysis.parent/f"bounded_{models}_{panel_manifest['panel_contract_sha256'][:12]}_{archive_sha[:12]}.zip"
 if retained.exists():
  if sha256_file(retained)!=archive_sha: raise RuntimeError('Archive name collision')
  archive.unlink()
 else: archive.rename(retained)
 display(Markdown((analysis/'SCIENTIFIC_REPORT.md').read_text()))
 print('Content-addressed archive:',retained)
'''


notebook = {
    "cells": [cell(RATIONALE, "markdown"), cell(CONFIG), cell(PREFLIGHT), cell(EXECUTION), cell(ANALYSIS)],
    "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
