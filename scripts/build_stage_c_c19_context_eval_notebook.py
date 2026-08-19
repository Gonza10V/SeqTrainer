"""Build the CPU-first, resumable C19 anomaly-and-needle notebook."""

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


RATIONALE = r"""# Stage C 03q — query-specific resumable anomaly and DNA-needle validation (v4)

## Runtime workflow

1. Open this notebook with a **CPU runtime** and run cells 2–3. Cell 3 builds the exact 342-stream cache one source shard at a time, freezes the canonical panel, and saves both to Drive.
2. After cell 3 reports `PANEL_FROZEN`, disconnect the CPU runtime and select an **NVIDIA A100** runtime.
3. On the A100, rerun cell 2. You may skip cell 3, or rerun it for a fast verification-only no-op. Continue with cell 4.
4. Cell 4 requires the A100 and resumes C19 in one-hour, case-safe sessions. C16 remains optional; cell 5 displays the report.

The approximately 66 MiB compact cache stays outside the resume ZIP. Drive receives each atomic source-shard cache chunk and validated current/previous ZIP generations at stage boundaries. The A100 phase never opens either token-stream dataset. Never start it unless bootstrap reports a restored `panel_frozen` stage.
"""


CONFIG = r'''# @title 2. Configuration — safe on CPU or A100
RUN_C19=True
RUN_C16_COMPARISON=False
DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC'
EXPERIMENT_NAME='c19_bounded_anomaly_needle_v4'
STUDY_VERSION='c16_c19_anomaly_needle_v4'
PARENT_EXPERIMENT='c19_bounded_anomaly_needle_v3'
PARENT_EVALUATOR_COMMIT='e685d3de9312f1ba803e06a6283e4b7132c69681'
LOCAL_INPUT_ROOT='/content/seqtrainer-03q-inputs-v4'
LOCAL_WORK_ROOT='/content/seqtrainer-03q-work-v4'
MAX_C19_HOURS=22.0
MODEL_SYNC_HOURS=1.0

EXPECTED_CHECKPOINT_SHA256={
 'C19':'07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a',
 'C16':'21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541',
}
EXPECTED_INPUT_SHA256={
 'dataset_manifest':'2fbdb870606e6fb1ce0f6750524726d041474be5081da2cb2316948bc12a2ee9',
 'validation_panel':'2b1450dc69beb839724e5e5207a2fbc132cd43297ffd5e839050e403b1bdf316',
 'e25_panel':'0e55b96b1840adba501ff5b4edcb38ee545d987ec122d8662f116b261a012bf9',
 'ani_pairs':'d29034649b2b1ff03d013fcde667cf6632bd25475f6e52b8afb9af7c8365abbc',
 'ani_membership':'e8c8faba9a0310e20165eeab913e65dfa0eb9e58e6a34a9d2eb89225012d9e80',
}
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='2b16cf75a923140c097ac2ff5e3280494980540d'
TRUST_OWNED_CHECKPOINT=True
'''


BOOTSTRAP = r'''# @title 3. Common bootstrap and v4 restore — safe on CPU or A100
from pathlib import Path
from datetime import datetime,timezone
from google.colab import drive
import errno,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,torch

runtime_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print('03q runtime:',runtime_name)
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
drive_retry('create v4 experiment folder',lambda:experiment.mkdir(parents=True,exist_ok=True))
registry=Path(LOCAL_WORK_ROOT)/'validation_registry'
registry.mkdir(parents=True,exist_ok=True)
bundle=registry/'bounded'; panel=bundle/'frozen_panel'

repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',GIT_REF],check=True)
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if commit!=GIT_REF: raise RuntimeError('Evaluator commit did not resolve exactly.')

venv=Path('/content/seqtrainer-03q-bounded-v4')
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
 'print("03q v4 imports OK",numpy.__version__,pandas.__version__,torch.__version__)'],
 text=True,capture_output=True)
print(smoke.stdout,end='')
if smoke.returncode: raise RuntimeError('03q environment import failed:\n'+smoke.stderr[-20000:])

resume_spec=importlib.util.spec_from_file_location(
 'stage_c_resume_archive',repo/'src/seqtrainer/torch/titans_paper_mac_stage_c/resume_archive.py'
)
resume_module=importlib.util.module_from_spec(resume_spec); resume_spec.loader.exec_module(resume_module)
ResumeArchiveManager=resume_module.ResumeArchiveManager
write_resume_state=resume_module.write_resume_state

BASE_CONTRACT={
 'experiment':EXPERIMENT_NAME,'study_version':STUDY_VERSION,
 'evaluator_commit':GIT_REF,'expected_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,
 'input_sha256':EXPECTED_INPUT_SHA256,
 'parent':{'experiment':PARENT_EXPERIMENT,'evaluator_commit':PARENT_EVALUATOR_COMMIT},
}
archive_manager=ResumeArchiveManager(registry,experiment)
restored=drive_retry('restore v4 resume archive',lambda:archive_manager.restore(expected_contract=BASE_CONTRACT))
if restored:
 restored_archive,resume_state=restored
 print('Restored:',restored_archive,'stage=',resume_state.get('stage'),'status=',resume_state.get('status'))
else:
 resume_state={
  'format_version':3,'immutable_contract':dict(BASE_CONTRACT),'stage':'new','status':'ready',
  'stage_history':[],'c19_accumulated_hours':0.0,'c16_accumulated_hours':0.0,
 }

STAGE_ORDER={name:index for index,name in enumerate((
 'new','environment_ready','inputs_ready','cache_building','cache_complete','panel_frozen','runtime_accepted',
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
  f"publish {resume_state['stage']}",lambda:archive_manager.save('/content/03q_resume_v4')
 )
 print('Drive checkpoint:',destination,'sha256=',digest,'cases=',resume_state['completed_cases'])
 return digest

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

def stage_small_inputs():
 source_manifest=sources['dataset']/'token_stream_manifest.json'
 manifest=stage_file('dataset manifest',source_manifest,local_inputs/'source_token_stream_manifest.json',EXPECTED_INPUT_SHA256['dataset_manifest'])
 manifest_payload=json.loads(manifest.read_text())
 paths={
  'dataset_manifest':manifest,
  'dataset_index':stage_file('dataset index',sources['dataset']/manifest_payload['index'],local_inputs/'token_stream_index.jsonl',manifest_payload['index_sha256']),
  'validation':stage_file('validation panel',sources['validation'],local_inputs/'validation.json'),
  'e25':stage_file('E25 panel',sources['e25'],local_inputs/'e25.json'),
  'ani_pairs':stage_file('ANI pairs',sources['ani_pairs'],local_inputs/'ani_pairs.tsv'),
  'ani_membership':stage_file('ANI membership',sources['ani_membership'],local_inputs/'ani_membership.parquet'),
 }
 hashes={
  'dataset_manifest':sha256_file(paths['dataset_manifest']),
  'validation_panel':sha256_file(paths['validation']),'e25_panel':sha256_file(paths['e25']),
  'ani_pairs':sha256_file(paths['ani_pairs']),'ani_membership':sha256_file(paths['ani_membership']),
 }
 if hashes!=EXPECTED_INPUT_SHA256: raise RuntimeError(f'v4 inputs differ from the immutable v3 parent contract: {hashes}')
 return paths

def stage_compact_cache(source,target):
 from seqtrainer.data.bacteria_titan import validate_panel_stream_cache
 manifest=validate_panel_stream_cache(source)
 target.mkdir(parents=True,exist_ok=True)
 names=('cache_contract.json','token_stream_index.jsonl','token_stream_manifest.json','COMPLETE.json')
 for name in names: stage_file(f'compact cache {name}',source/name,target/name,sha256_file(source/name))
 for shard in manifest['shards']:
  source_chunk=source/shard['chunk']; target_chunk=target/shard['chunk']
  for name in ('tokens.npy','base_lengths.npy','index.jsonl','chunk_manifest.json','COMPLETE.json'):
   stage_file(f'compact {source_chunk.name}/{name}',source_chunk/name,target_chunk/name,sha256_file(source_chunk/name))
 validate_panel_stream_cache(target)
 return target

def validate_frozen_panel():
 manifest_path=panel/'frozen_panel_manifest.json'
 if not manifest_path.is_file(): return None
 manifest=json.loads(manifest_path.read_text())
 expected_artifacts={
  'anomaly_cases.jsonl','needle_cases.jsonl','anomaly_cases.parquet',
  'needle_cases.parquet','token_arrays.npz','retained_sequences.fasta.gz',
 }
 artifacts=manifest.get('artifact_sha256',{})
 if set(artifacts)!=expected_artifacts: raise RuntimeError('Frozen panel artifact contract is incomplete.')
 for name,digest in artifacts.items():
  path=panel/name
  if not path.is_file() or sha256_file(path)!=digest: raise RuntimeError(f'Frozen panel artifact changed: {name}')
 anomaly_count=sum(1 for line in (panel/'anomaly_cases.jsonl').read_text().splitlines() if line.strip())
 needle_count=sum(1 for line in (panel/'needle_cases.jsonl').read_text().splitlines() if line.strip())
 if (
  manifest.get('study_version')!=STUDY_VERSION or manifest.get('mode')!='bounded'
  or len(manifest.get('hosts',[]))!=8 or anomaly_count!=48 or needle_count!=48
  or manifest.get('planned_workload',{}).get('total_segment_forwards')!=25344
  or manifest.get('canonical_selection',{}).get('alphabet')!='ACGT'
  or manifest.get('canonical_selection',{}).get('policy_version')!=2
  or manifest.get('parent_dataset_fingerprint')!=EXPECTED_INPUT_SHA256['dataset_manifest']
  or not manifest.get('compact_cache_manifest_sha256')
  or manifest.get('compact_cache_manifest_sha256')!=manifest.get('dataset_manifest_sha256')
  or not manifest.get('cache_contract_sha256')
  or manifest.get('validation_panel_sha256')!=EXPECTED_INPUT_SHA256['validation_panel']
  or manifest.get('e25_training_panel_sha256')!=EXPECTED_INPUT_SHA256['e25_panel']
  or manifest.get('ani_pairs_sha256')!=EXPECTED_INPUT_SHA256['ani_pairs']
  or manifest.get('ani_membership_sha256')!=EXPECTED_INPUT_SHA256['ani_membership']
 ):
  raise RuntimeError('Frozen panel is not the exact canonical bounded 03q v4 contract.')
 return manifest

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
panel_manifest=validate_frozen_panel()
print({'commit':commit,'runtime':runtime_name,'resume_stage':resume_state['stage'],
       'panel_frozen':panel_manifest is not None,'drive_experiment':str(experiment)})
'''


CPU_PANEL = r'''# @title 4. CPU PHASE — build and save the canonical panel (A100 not required)
panel_manifest=validate_frozen_panel()
if panel_manifest is not None:
 print('PANEL_FROZEN already restored and verified. CPU generation is skipped.')
 print('On A100, continue with cell 4. It is safe to skip this cell after rerunning cell 2.')
else:
 if torch.cuda.is_available():
  raise RuntimeError(
   'Panel generation must use a CPU runtime to preserve A100 time. '
   'Switch Colab to CPU, rerun cells 2-4, and return to A100 only after PANEL_FROZEN.'
  )
 print('CPU PANEL GENERATION — staging small immutable metadata locally.')
 print('Local free space:',round(shutil.disk_usage('/content').free/2**30,1),'GiB')
 paths=stage_small_inputs()
 VALIDATION_PANEL=str(paths['validation'])
 E25_TRAINING_PANEL=str(paths['e25']); ANI_PAIRS=str(paths['ani_pairs'])
 ANI_MEMBERSHIP=str(paths['ani_membership'])
 persist('inputs_ready',details={'cpu_runtime':runtime_name,'input_sha256':EXPECTED_INPUT_SHA256})
 cache_drive=experiment/'panel_stream_cache_v1'
 cache_progress=registry/'cache_building.json'
 cache_command=[*runner,'build-cache','--source-dataset',str(sources['dataset']),
  '--source-manifest',str(paths['dataset_manifest']),'--source-index',str(paths['dataset_index']),
  '--validation-panel',VALIDATION_PANEL,'--e25-panel',E25_TRAINING_PANEL,
  '--output',str(cache_drive),'--scratch-dir',str(Path(LOCAL_WORK_ROOT)/'cache_scratch'),
  '--progress-json',str(cache_progress),'--expected-streams','342',
  '--expected-tokens','11546327','--expected-bases','67680627']
 cache_process=subprocess.Popen(cache_command)
 last_completed=tuple(resume_state.get('details',{}).get('completed_source_shards',[]))
 while cache_process.poll() is None:
  time.sleep(5)
  if cache_progress.is_file():
   progress=json.loads(cache_progress.read_text())
   completed=tuple(progress.get('completed_source_shards',[]))
   if completed!=last_completed:
    last_completed=completed
    persist('cache_building',status='paused',details=progress)
 if cache_process.returncode: raise RuntimeError(f'compact cache builder failed with {cache_process.returncode}')
 cache_manifest=json.loads((cache_drive/'token_stream_manifest.json').read_text())
 if len(cache_manifest['source_shard_indices'])!=7: raise RuntimeError('compact cache did not use exactly seven source shards')
 DATASET_DIR=str(stage_compact_cache(cache_drive,local_inputs/'compact_dataset'))
 resume_state['immutable_contract']['cache_contract_sha256']=cache_manifest['cache_contract_sha256']
 persist('cache_complete',details={
  'completed_source_shards':cache_manifest['source_shard_indices'],
  'cache_contract_sha256':cache_manifest['cache_contract_sha256'],
  'compact_streams':cache_manifest['streams'],'compact_tokens':cache_manifest['tokens'],
  'compact_bases':cache_manifest['bases'],
 })
 run_checked('bounded_freeze',[*runner,'freeze','--dataset-dir',DATASET_DIR,
  '--validation-panel',VALIDATION_PANEL,'--e25-panel',E25_TRAINING_PANEL,
  '--ani-pairs',ANI_PAIRS,'--ani-membership',ANI_MEMBERSHIP,
  '--mode','bounded','--output',str(panel)])
 panel_manifest=validate_frozen_panel()
 resume_state['immutable_contract']['panel_contract_sha256']=panel_manifest['panel_contract_sha256']
 persist('panel_frozen',details={
  'anomaly_cases':48,'needle_cases':48,
  'panel_contract_sha256':panel_manifest['panel_contract_sha256'],
 })
 print('PANEL_FROZEN:',panel_manifest['panel_contract_sha256'])
 print('CPU work is safely saved to Drive. Disconnect this runtime and select an NVIDIA A100.')
 print('On A100 rerun cell 2, skip or verify cell 3, then continue with cell 4.')
'''


A100_QUALIFY = r'''# @title 5. A100 PHASE — restore inputs, stage C19, and qualify runtime
print('A100 REQUIRED FROM THIS CELL FORWARD.')
panel_manifest=validate_frozen_panel()
if panel_manifest is None:
 raise RuntimeError('No valid PANEL_FROZEN archive. Return to a CPU runtime and complete cell 4 first.')
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
 raise RuntimeError('Select an NVIDIA A100 runtime, rerun cell 2, then continue with cell 4.')
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

print('A100 verified:',torch.cuda.get_device_name(0))
print('The frozen panel is self-contained; no token-stream dataset will be staged or opened.')
C19_CHECKPOINT=str(stage_file(
 'C19 checkpoint',sources['C19'],local_inputs/'C19.pt',EXPECTED_CHECKPOINT_SHA256['C19']
))
C16_CHECKPOINT=str(local_inputs/'C16.pt')
if RUN_C16_COMPARISON:
 C16_CHECKPOINT=str(stage_file(
  'C16 checkpoint',sources['C16'],Path(C16_CHECKPOINT),EXPECTED_CHECKPOINT_SHA256['C16']
 ))
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


EXECUTION = r'''# @title 6. A100 PHASE — resumable C19; optionally add C16 later
if 'C19_CHECKPOINT' not in globals():
 raise RuntimeError('Restart cell 4 so A100 qualification completes before model evaluation.')
def run_model(model,checkpoint,max_hours):
 budget_key=f'{model.lower()}_accumulated_hours'
 used=float(resume_state.get(budget_key,0.0))
 while not (bundle/model/'COMPLETE.json').is_file() and used<max_hours:
  session_hours=min(MODEL_SYNC_HOURS,max_hours-used)
  started=time.monotonic()
  command=[*runner,'run-model','--model',model,'--checkpoint',checkpoint,
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
if not c19_complete: print('C19 incomplete; rerun cells 2 and 4 on A100 to continue.')
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


ANALYSIS = r'''# @title 7. Validate and display the available scientific report
from IPython.display import Markdown,display
if not (bundle/'C19'/'COMPLETE.json').is_file():
 print('No scientific report yet. Resume the A100 execution cells; completed cases will be skipped.')
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
        cell(RATIONALE, "markdown"), cell(CONFIG + "\n\n" + BOOTSTRAP), cell(CPU_PANEL),
        cell(A100_QUALIFY + "\n\n" + EXECUTION), cell(ANALYSIS),
    ],
    "metadata": {
        "colab": {},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
