"""Build the resumable, sub-24-hour C19 anomaly-and-needle notebook."""

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


RATIONALE = r"""# Stage C 03q — resumable C19 anomaly and DNA-needle validation (v2)

This is a clean v2 experiment. It first freezes eight canonical held-out hosts and their synthetic anomaly/needle cases, then qualifies and evaluates the immutable final C19 checkpoint within a 22-hour A100 budget. C16 is disabled by default and can be added later on the byte-identical frozen panel.

All active work stays under `/content`. At every significant stage and after each one-hour, case-safe model session, the notebook validates and atomically publishes one current Drive ZIP while retaining the previous generation. A new Colab runtime restores the newest valid generation and skips completed work automatically. Run the notebook from the first cell after every reconnect; do not skip cells manually.
"""


CONFIG = r'''# @title 2. Immutable contract and execution switches
RUN_C19=True
RUN_C16_COMPARISON=False
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC'
EXPERIMENT_NAME='c19_bounded_anomaly_needle_v2'
STUDY_VERSION='c16_c19_anomaly_needle_v2'
LOCAL_INPUT_ROOT='/content/seqtrainer-03q-inputs-v2'
LOCAL_WORK_ROOT='/content/seqtrainer-03q-work-v2'
MAX_C19_HOURS=22.0
MODEL_SYNC_HOURS=1.0

EXPECTED_CHECKPOINT_SHA256={
 'C19':'07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a',
 'C16':'21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541',
}
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='a065ef25aa65393af02e79e7de58157270df9c0f'
TRUST_OWNED_CHECKPOINT=True
'''


BOOTSTRAP = r'''# @title 3. Mount, bootstrap, and restore the newest valid generation
from pathlib import Path
from datetime import datetime,timezone
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
drive_retry('create v2 experiment folder',lambda:experiment.mkdir(parents=True,exist_ok=True))
registry=Path(LOCAL_WORK_ROOT)/'validation_registry'
registry.mkdir(parents=True,exist_ok=True)

repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',GIT_REF],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if commit!=GIT_REF: raise RuntimeError('Evaluator commit did not resolve exactly.')

venv=Path('/content/seqtrainer-03q-bounded-v2')
if not (venv/'bin/python').is_file():
 subprocess.run([sys.executable,'-m','pip','install','--quiet','virtualenv>=20.26'],check=True)
 subprocess.run([sys.executable,'-m','virtualenv','--system-site-packages',str(venv)],check=True)
python=str(venv/'bin/python')
subprocess.run([python,'-m','pip','install','--quiet','--upgrade',
 'numpy==1.26.4','pandas==2.2.2','pyarrow==18.1.0','scipy>=1.11,<2',
 'scikit-learn>=1.3,<2','matplotlib>=3.7,<4',
 'rdflib>=6.3.2','requests>=2.31','sbol2>=1.4'],check=True)
subprocess.run([python,'-m','pip','install','--no-deps','-e',str(repo)],check=True)
smoke=subprocess.run([python,'-c',
 'import numpy,pandas,pyarrow,rdflib,requests,sbol2,scipy,sklearn,torch; '
 'import seqtrainer; import seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study_cli; '
 'import seqtrainer.torch.titans_paper_mac_stage_c.resume_archive; '
 'print("03q v2 imports OK",numpy.__version__,pandas.__version__,torch.__version__)'],
 text=True,capture_output=True)
print(smoke.stdout,end='')
if smoke.returncode: raise RuntimeError('03q environment import failed:\n'+smoke.stderr[-20000:])

import importlib.util
resume_spec=importlib.util.spec_from_file_location(
 'stage_c_resume_archive',repo/'src/seqtrainer/torch/titans_paper_mac_stage_c/resume_archive.py'
)
resume_module=importlib.util.module_from_spec(resume_spec); resume_spec.loader.exec_module(resume_module)
ResumeArchiveManager=resume_module.ResumeArchiveManager
write_resume_state=resume_module.write_resume_state

BASE_CONTRACT={
 'experiment':EXPERIMENT_NAME,'study_version':STUDY_VERSION,
 'evaluator_commit':GIT_REF,'expected_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,
}
archive_manager=ResumeArchiveManager(registry,experiment)
restored=drive_retry('restore resume archive',lambda:archive_manager.restore(expected_contract=BASE_CONTRACT))
if restored:
 restored_archive,resume_state=restored
 print('Restored:',restored_archive,'stage=',resume_state.get('stage'),'status=',resume_state.get('status'))
else:
 resume_state={
  'format_version':2,'immutable_contract':dict(BASE_CONTRACT),'stage':'new','status':'ready',
  'stage_history':[],'c19_accumulated_hours':0.0,'c16_accumulated_hours':0.0,
 }

STAGE_ORDER={name:index for index,name in enumerate((
 'new','environment_ready','inputs_ready','panel_frozen','runtime_accepted',
 'c19_running','c19_complete','analysis_complete','c16_running','c16_complete','comparison_complete'
))}
def completed_case_counts():
 result={}
 for model in ('C19','C16'):
  case_root=registry/'bounded'/model/'resume'
  files=[] if not case_root.is_dir() else [path for path in case_root.glob('*.json') if path.name!='contract.json']
  result[model]={
   'total':len(files),'anomaly':sum(path.name.startswith('anomaly_') for path in files),
   'native':sum(path.name.startswith('native:') for path in files),
   'needle':sum(path.name.startswith('needle_') for path in files),
  }
 return result

def persist(stage=None,status='complete',details=None):
 current=str(resume_state.get('stage','new'))
 if stage is not None and STAGE_ORDER[stage]>=STAGE_ORDER.get(current,0): resume_state['stage']=stage
 resume_state['status']=status
 resume_state['completed_cases']=completed_case_counts()
 resume_state['updated_at']=datetime.now(timezone.utc).isoformat()
 if details: resume_state.setdefault('details',{}).update(details)
 resume_state.setdefault('stage_history',[]).append({
  'stage':resume_state['stage'],'status':status,'at':resume_state['updated_at']
 })
 write_resume_state(registry,resume_state)
 destination,digest=drive_retry(
  f"publish {resume_state['stage']}",lambda:archive_manager.save('/content/03q_resume_v2')
 )
 print('Drive checkpoint:',destination,'sha256=',digest,'cases=',resume_state['completed_cases'])
 return digest

bundle=registry/'bounded'; panel=bundle/'frozen_panel'
runner=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.anomaly_study_cli']
stage_c_runner=[python,'-m','seqtrainer.torch.titans_paper_mac_stage_c.colab_cli'] # seqtrainer-titans-stage-c-colab-run
def run_checked(label,command):
 wrapped=[*stage_c_runner,'--run-dir',str(registry/'notebook_runs'),'--label',label,
          '--repo',str(repo),'--',*command]
 result=subprocess.run(wrapped)
 if result.returncode:
  log=registry/'notebook_runs'/'logs'/f'{label}.log'
  try: persist(status='failed',details={'failed_label':label,'failed_log':str(log)})
  except Exception as sync_error: print('Failure checkpoint warning:',repr(sync_error))
  tail=log.read_text(errors='replace')[-20000:] if log.is_file() else 'log missing'
  raise RuntimeError(f'{label} failed; log={log}\n{tail}')

if not restored: persist('environment_ready')
print({'commit':commit,'resume_stage':resume_state['stage'],'drive_experiment':str(experiment)})
'''


FREEZE = r'''# @title 4. Stage immutable data, freeze the canonical panel, and checkpoint it
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
 source_manifest=source/'token_stream_manifest.json'
 manifest_sha=drive_retry('hash dataset manifest',lambda:sha256_file(source_manifest))
 sentinel=target/'.03q_dataset_complete.json'
 if sentinel.is_file() and json.loads(sentinel.read_text()).get('manifest_sha256')==manifest_sha:
  return target
 target.mkdir(parents=True,exist_ok=True)
 drive_retry('copy dataset',lambda:shutil.copytree(source,target,dirs_exist_ok=True))
 if sha256_file(target/'token_stream_manifest.json')!=manifest_sha: raise RuntimeError('Dataset manifest copy mismatch')
 sentinel.write_text(json.dumps({'manifest_sha256':manifest_sha})+'\n')
 return target

print('Local free space before staging:',round(shutil.disk_usage('/content').free/2**30,1),'GiB')
DATASET_DIR=str(stage_dataset(sources['dataset'],local_inputs/'dataset'))
VALIDATION_PANEL=str(stage_file('validation panel',sources['validation'],local_inputs/'validation.json'))
E25_TRAINING_PANEL=str(stage_file('E25 panel',sources['e25'],local_inputs/'e25.json'))
ANI_PAIRS=str(stage_file('ANI pairs',sources['ani_pairs'],local_inputs/'ani_pairs.tsv'))
ANI_MEMBERSHIP=str(stage_file('ANI membership',sources['ani_membership'],local_inputs/'ani_membership.parquet'))
input_hashes={
 'dataset_manifest':sha256_file(Path(DATASET_DIR)/'token_stream_manifest.json'),
 'validation_panel':sha256_file(VALIDATION_PANEL),'e25_panel':sha256_file(E25_TRAINING_PANEL),
 'ani_pairs':sha256_file(ANI_PAIRS),'ani_membership':sha256_file(ANI_MEMBERSHIP),
}
prior_inputs=resume_state['immutable_contract'].get('input_sha256')
if prior_inputs is not None and prior_inputs!=input_hashes: raise RuntimeError('Restored input contract differs')
resume_state['immutable_contract']['input_sha256']=input_hashes
persist('inputs_ready',details={'local_free_gib':round(shutil.disk_usage('/content').free/2**30,1)})

run_checked('bounded_freeze',[*runner,'freeze','--dataset-dir',DATASET_DIR,
 '--validation-panel',VALIDATION_PANEL,'--e25-panel',E25_TRAINING_PANEL,
 '--ani-pairs',ANI_PAIRS,'--ani-membership',ANI_MEMBERSHIP,
 '--mode','bounded','--output',str(panel)])
panel_manifest=json.loads((panel/'frozen_panel_manifest.json').read_text())
anomaly_count=sum(1 for line in (panel/'anomaly_cases.jsonl').read_text().splitlines() if line.strip())
needle_count=sum(1 for line in (panel/'needle_cases.jsonl').read_text().splitlines() if line.strip())
if (
 panel_manifest['study_version']!=STUDY_VERSION or panel_manifest['mode']!='bounded'
 or len(panel_manifest['hosts'])!=8 or anomaly_count!=48 or needle_count!=48
 or panel_manifest['planned_workload']['total_segment_forwards']!=25344
 or panel_manifest.get('canonical_selection',{}).get('alphabet')!='ACGT'
): raise RuntimeError('Frozen panel is not the exact canonical bounded 03q v2 contract.')
prior_panel=resume_state['immutable_contract'].get('panel_contract_sha256')
if prior_panel is not None and prior_panel!=panel_manifest['panel_contract_sha256']:
 raise RuntimeError('Restored frozen panel contract differs')
resume_state['immutable_contract']['panel_contract_sha256']=panel_manifest['panel_contract_sha256']
persist('panel_frozen',details={'anomaly_cases':anomaly_count,'needle_cases':needle_count})
print({'panel':panel_manifest['panel_contract_sha256'],'hosts':8,
       'anomaly_cases':anomaly_count,'needle_cases':needle_count,'planned_forwards':25344})
'''


QUALIFY = r'''# @title 5. Require A100, stage C19, qualify runtime, and checkpoint
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
 raise RuntimeError('The frozen panel is safely checkpointed. Select an A100 runtime and Run all again.')
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

print('Local free space before checkpoint:',round(shutil.disk_usage('/content').free/2**30,1),'GiB')
C19_CHECKPOINT=str(stage_file('C19 checkpoint',sources['C19'],local_inputs/'C19.pt',EXPECTED_CHECKPOINT_SHA256['C19']))
C16_CHECKPOINT=str(local_inputs/'C16.pt')
if RUN_C16_COMPARISON:
 C16_CHECKPOINT=str(stage_file('C16 checkpoint',sources['C16'],Path(C16_CHECKPOINT),EXPECTED_CHECKPOINT_SHA256['C16']))
projection=registry/'runtime_projection.json'
if not projection.is_file() or json.loads(projection.read_text()).get('panel_contract_sha256')!=panel_manifest['panel_contract_sha256']:
 run_checked('bounded_runtime_projection',[*runner,'estimate-runtime',
  '--checkpoint',C19_CHECKPOINT,'--frozen-panel',str(panel),'--output',str(projection),
  '--device','cuda','--probe-forwards','64','--max-hours',str(MAX_C19_HOURS),
  '--trust-owned-checkpoint'])
runtime=json.loads(projection.read_text())
if not runtime['accepted'] or runtime['projected_hours']>MAX_C19_HOURS:
 raise RuntimeError(f"C19 projection {runtime['projected_hours']:.2f} h exceeds budget")
persist('runtime_accepted',details={
 'gpu':torch.cuda.get_device_name(0),'projected_hours':runtime['projected_hours']
})
print({'gpu':torch.cuda.get_device_name(0),'projected_hours':runtime['projected_hours'],
       'remaining_c19_budget':MAX_C19_HOURS-float(resume_state.get('c19_accumulated_hours',0.0))})
'''


EXECUTION = r'''# @title 6. Resume C19 in one-hour safe sessions; optionally add C16 later
def run_model(model,checkpoint,max_hours):
 budget_key=f'{model.lower()}_accumulated_hours'
 used=float(resume_state.get(budget_key,0.0))
 while not (bundle/model/'COMPLETE.json').is_file() and used<max_hours:
  session_hours=min(MODEL_SYNC_HOURS,max_hours-used)
  started=time.monotonic()
  command=[*runner,'run-model','--model',model,'--checkpoint',checkpoint,
   '--dataset-dir',DATASET_DIR,'--validation-panel',VALIDATION_PANEL,
   '--frozen-panel',str(panel),'--output',str(bundle/model),'--device','cuda',
   '--max-runtime-hours',str(session_hours),'--trust-owned-checkpoint']
  run_checked(f'bounded_{model}',command)
  used+=(time.monotonic()-started)/3600
  resume_state[budget_key]=used
  complete=(bundle/model/'COMPLETE.json').is_file()
  persist(f'{model.lower()}_complete' if complete else f'{model.lower()}_running',
          status='complete' if complete else 'paused',
          details={f'{model.lower()}_remaining_hours':max(0.0,max_hours-used)})
 if not (bundle/model/'COMPLETE.json').is_file() and used>=max_hours:
  print(model,'stopped at the cumulative safety budget; completed cases are preserved.')
 return (bundle/model/'COMPLETE.json').is_file()

c19_complete=(bundle/'C19'/'COMPLETE.json').is_file()
if RUN_C19 and not c19_complete: c19_complete=run_model('C19',C19_CHECKPOINT,MAX_C19_HOURS)
if not c19_complete: print('C19 incomplete; Run all again to restore and continue if budget remains.')
else: print('C19 complete:',bundle/'C19')

c16_complete=(bundle/'C16'/'COMPLETE.json').is_file()
if c19_complete and RUN_C16_COMPARISON and not c16_complete:
 c16_complete=run_model('C16',C16_CHECKPOINT,MAX_C19_HOURS)
if RUN_C16_COMPARISON and not c16_complete: print('C16 remains incomplete and resumable.')

analysis=bundle/'analysis'
if c19_complete:
 run_checked('bounded_analyze',[*runner,'compare','--input',str(bundle),'--output',str(analysis)])
 persist('comparison_complete' if c16_complete else 'analysis_complete')
print({'c19_complete':c19_complete,'c16_complete':c16_complete,
       'completed_cases':completed_case_counts(),'analysis':str(analysis)})
'''


ANALYSIS = r'''# @title 7. Validate, display, and checkpoint the available report
from IPython.display import Markdown,display
if not (bundle/'C19'/'COMPLETE.json').is_file():
 print('No report yet. Run all again; restored case results will be skipped.')
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
 persist('comparison_complete' if (bundle/'C16'/'COMPLETE.json').is_file() else 'analysis_complete')
 display(Markdown((analysis/'SCIENTIFIC_REPORT.md').read_text()))
 print('Durable current archive:',archive_manager.current)
 print('Durable fallback archive:',archive_manager.previous)
'''


notebook = {
    "cells": [
        cell(RATIONALE, "markdown"), cell(CONFIG), cell(BOOTSTRAP),
        cell(FREEZE + "\n\n" + QUALIFY), cell(EXECUTION + "\n\n" + ANALYSIS),
    ],
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
