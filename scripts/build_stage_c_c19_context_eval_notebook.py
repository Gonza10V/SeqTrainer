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

This validation-only study freezes its synthetic tests before loading a model, qualifies C19 against a 22-hour A100 budget, and evaluates the immutable final C19 checkpoint. C16 is disabled by default and can be added later on the byte-identical frozen panel.

Drive usage is deliberately small: mount once, copy immutable inputs to Colab-local storage, and replace one resumable results ZIP after each four-hour compute chunk. The evaluator never reads or writes Drive directly. This avoids both the separate Drive-API credential propagation flow and high-frequency Drive filesystem I/O.

The bounded contract uses eight held-out hosts, near/far E25 donors, 1/16/64-segment replacements at depth 16, and needle distances 3/16/64 with zero or sixteen near-key distractors. Hosts are the inferential units. Memory telemetry, predictive performance, and mechanistic diagnostics are reported separately; the protected test panel is unavailable to this workflow.
"""


CONFIG = r'''# @title 2. Inputs and switches
RUN_C19=True
RUN_C16_COMPARISON=False  # enable in a later fresh session after C19 completes
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC'
EXPERIMENT_NAME='c19_bounded_anomaly_needle_v1'
LOCAL_INPUT_ROOT='/content/seqtrainer-03q-inputs'
LOCAL_WORK_ROOT='/content/seqtrainer-03q-work'
MAX_C19_HOURS=22.0
SYNC_CHUNK_HOURS=4.0

EXPECTED_CHECKPOINT_SHA256={
 'C19':'07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a',
 'C16':'21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541',
}
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='2e869da44c2fb00c93101f72dfe7a88074fb4e2c'
TRUST_OWNED_CHECKPOINT=True
'''


PREFLIGHT = r'''# @title 3. Mount once, stage locally, test, freeze, and qualify runtime
from pathlib import Path
from google.colab import drive
import errno,hashlib,json,os,shutil,subprocess,sys,time,torch

mount=Path('/content/drive')
drive.mount(str(mount),timeout_ms=120000)
root=Path(DRIVE_ROOT)
transient={errno.EIO,errno.ESTALE,errno.ENOTCONN,errno.ETIMEDOUT}
def drive_retry(label,operation):
 last_error=None
 for delay in (0,2,5,10,20,30):
  if delay: time.sleep(delay)
  try: return operation()
  except OSError as error:
   last_error=error
   if error.errno not in transient: raise
 raise RuntimeError(f'Drive operation {label!r} failed after retries: {last_error!r}')

drive_retry('read Stage C root',lambda:root.stat())
experiment=root/EXPERIMENT_NAME
drive_retry('create experiment folder',lambda:experiment.mkdir(parents=True,exist_ok=True))
resume_zip=experiment/'03q_resume.zip'
registry=Path(LOCAL_WORK_ROOT)/'validation_registry'
registry.mkdir(parents=True,exist_ok=True)

# Restore only one durable object; evaluation itself stays under /content.
if drive_retry('locate resume ZIP',lambda:resume_zip.is_file()):
 local_resume=Path('/content/03q_resume_restore.zip')
 drive_retry('restore resume ZIP',lambda:shutil.copyfile(resume_zip,local_resume))
 shutil.unpack_archive(local_resume,registry)
 print('Restored prior 03q progress.')
bundle=registry/'bounded'; panel=bundle/'frozen_panel'

repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',GIT_REF],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if commit!=GIT_REF: raise RuntimeError('Evaluator commit did not resolve exactly.')

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

local_inputs=Path(LOCAL_INPUT_ROOT); local_inputs.mkdir(parents=True,exist_ok=True)
sources={
 'C19':root/'runs/c19_v3_medium_adaptive_e25/latest.pt',
 'C16':root/'runs/c16_deep_adaptive_5m_paper_exact/latest.pt',
 'dataset':root/'stage_c_dataset/ordered_streams/nonoverlap_6mer_v1',
 'validation':root/'study/stage_c_ecoli_medium_deep_memory_v3/panels/validation.json',
 'e25':root/'study/stage_c_ecoli_medium_deep_memory_v3/panels/e25.json',
 'ani_pairs':root/'inputs/ecoli_skani_triangle.tsv',
 'ani_membership':root/'stage_c_dataset/manifests/ani99_membership.parquet',
}

def stage_file(label,source,target,expected_sha=None):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_file() and (expected_sha is None or sha256_file(target)==expected_sha): return target
 if not drive_retry(f'locate {label}',lambda:source.is_file()): raise FileNotFoundError(source)
 partial=target.with_name(target.name+'.partial')
 drive_retry(f'copy {label}',lambda:shutil.copyfile(source,partial))
 os.replace(partial,target)
 if expected_sha and sha256_file(target)!=expected_sha: raise RuntimeError(f'{label} checksum mismatch')
 return target

def stage_dataset(source,target):
 sentinel=target/'token_stream_manifest.json'
 if sentinel.is_file(): return target
 if not drive_retry('locate dataset',lambda:source.is_dir()): raise FileNotFoundError(source)
 target.mkdir(parents=True,exist_ok=True)
 drive_retry('copy dataset',lambda:shutil.copytree(source,target,dirs_exist_ok=True))
 return target

print('Local free space before staging:',round(shutil.disk_usage('/content').free/2**30,1),'GiB')
C19_CHECKPOINT=str(stage_file('C19 checkpoint',sources['C19'],local_inputs/'C19.pt',EXPECTED_CHECKPOINT_SHA256['C19']))
C16_CHECKPOINT=str(local_inputs/'C16.pt')
if RUN_C16_COMPARISON:
 C16_CHECKPOINT=str(stage_file('C16 checkpoint',sources['C16'],Path(C16_CHECKPOINT),EXPECTED_CHECKPOINT_SHA256['C16']))
DATASET_DIR=str(stage_dataset(sources['dataset'],local_inputs/'dataset'))
VALIDATION_PANEL=str(stage_file('validation panel',sources['validation'],local_inputs/'validation.json'))
E25_TRAINING_PANEL=str(stage_file('E25 panel',sources['e25'],local_inputs/'e25.json'))
ANI_PAIRS=str(stage_file('ANI pairs',sources['ani_pairs'],local_inputs/'ani_pairs.tsv'))
ANI_MEMBERSHIP=str(stage_file('ANI membership',sources['ani_membership'],local_inputs/'ani_membership.parquet'))
print('Local free space after staging:',round(shutil.disk_usage('/content').free/2**30,1),'GiB')
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

def sync_results():
 local_zip=Path(shutil.make_archive('/content/03q_resume','zip',root_dir=registry))
 partial=experiment/'03q_resume.partial.zip'
 drive_retry('upload resume ZIP',lambda:shutil.copyfile(local_zip,partial))
 drive_retry('commit resume ZIP',lambda:os.replace(partial,resume_zip))
 print('Saved resumable ZIP:',resume_zip,'bytes=',local_zip.stat().st_size)

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
sync_results()
print({'commit':commit,'gpu':torch.cuda.get_device_name(0),'panel':panel_manifest['panel_contract_sha256'],
       'planned_forwards':panel_manifest['planned_workload']['total_segment_forwards'],
       'projected_hours':runtime['projected_hours'],'local_registry':str(registry)})
'''


EXECUTION = r'''# @title 4. Run C19; optionally add C16 later
def run_model(model,checkpoint,max_hours):
 deadline=time.monotonic()+max_hours*3600
 while not (bundle/model/'COMPLETE.json').is_file():
  remaining=(deadline-time.monotonic())/3600
  if remaining<=0: break
  command=[*runner,'run-model','--model',model,'--checkpoint',checkpoint,
   '--dataset-dir',DATASET_DIR,'--validation-panel',VALIDATION_PANEL,
   '--frozen-panel',str(panel),'--output',str(bundle/model),'--device','cuda',
   '--max-runtime-hours',str(min(SYNC_CHUNK_HOURS,remaining)),'--trust-owned-checkpoint']
  run_checked(f'bounded_{model}',command)
  sync_results()
 return (bundle/model/'COMPLETE.json').is_file()

c19_complete=(bundle/'C19'/'COMPLETE.json').is_file()
if RUN_C19 and not c19_complete: c19_complete=run_model('C19',C19_CHECKPOINT,MAX_C19_HOURS)
if not c19_complete: print('C19 paused safely; rerun this cell unchanged.')
else: print('C19 complete:',bundle/'C19')

c16_complete=(bundle/'C16'/'COMPLETE.json').is_file()
if c19_complete and RUN_C16_COMPARISON and not c16_complete:
 c16_complete=run_model('C16',C16_CHECKPOINT,MAX_C19_HOURS)
if RUN_C16_COMPARISON and not c16_complete: print('C16 remains incomplete; rerun later unchanged.')

analysis=bundle/'analysis'
if c19_complete:
 run_checked('bounded_analyze',[*runner,'compare','--input',str(bundle),'--output',str(analysis)])
 sync_results()
print({'c19_complete':c19_complete,'c16_complete':c16_complete,'analysis':str(analysis)})
'''


ANALYSIS = r'''# @title 5. Validate and display the report
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
  manifest=json.loads((bundle/model/'model_run_manifest.json').read_text())
  if manifest['panel_contract_sha256']!=panel_manifest['panel_contract_sha256']:
   raise RuntimeError(f'Refusing protocol-mismatched {model} result.')
 sync_results()
 display(Markdown((analysis/'SCIENTIFIC_REPORT.md').read_text()))
 print('Durable resumable results:',resume_zip)
'''


notebook = {
    "cells": [cell(RATIONALE, "markdown"), cell(CONFIG), cell(PREFLIGHT), cell(EXECUTION), cell(ANALYSIS)],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
