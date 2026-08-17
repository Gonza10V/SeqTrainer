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

Google Drive is accessed by folder ID through its API, not through the Colab Drive filesystem. Immutable inputs are downloaded once to Colab-local storage, all high-frequency evaluation I/O stays local, and only changed resumable artifacts are uploaded at four-hour chunk boundaries. This avoids repeated FUSE failures caused by resolving a My Drive root containing tens of thousands of items.

The bounded contract uses eight held-out hosts, near/far E25 donors, 1/16/64-segment replacements at depth 16, and needle distances 3/16/64 with zero or sixteen near-key distractors. Primary anomaly endpoints are leave-one-host-out AUPRC, TPR at 1% FPR, false positives/Mb, localization IoU, and boundary error. Primary needle endpoints are carried-minus-reset target log probability, Recall@1, exact recovery, and long-distance performance. Hosts are the inferential units.

Memory telemetry, predictive performance, and mechanistic diagnostics are reported separately. The protected test panel is unavailable to this workflow, and no horizontal-transfer, function, pathogenicity, or causal adaptive-memory claim is made.
"""


CONFIG = r'''# @title 2. Immutable inputs and execution switches
RUN_C19=True
RUN_C16_COMPARISON=False  # rerun later to add C16 on the completed C19 panel
STAGE_C_FOLDER_ID='1vygqdWpiV7KDkTZLz4H_Gzc4tnz333GR'
DRIVE_ROOT=f'drive-id:{STAGE_C_FOLDER_ID}'  # API identity; never a mounted filesystem path
LOCAL_INPUT_ROOT='/content/seqtrainer-03q-inputs'
LOCAL_WORK_ROOT='/content/seqtrainer-03q-work'
EXPERIMENT_NAME='c19_bounded_anomaly_needle_v1'
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


PREFLIGHT = r'''# @title 3. Stage through the Drive API, test, freeze cases, and qualify runtime
from pathlib import Path
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload,MediaIoBaseDownload
import hashlib,json,os,shutil,subprocess,sys,time,torch

# Authenticate once. Do not mount Drive: MyDrive FUSE lookup is the source of Errno 5.
auth.authenticate_user()
drive_api=build('drive','v3',cache_discovery=False)
def drive_retry(label,operation):
 last_error=None
 for delay in (0,2,5,10,20,30):
  if delay: time.sleep(delay)
  try: return operation()
  except (HttpError,OSError) as error:
   last_error=error
 raise RuntimeError(f'Drive operation {label!r} failed after retries: {last_error!r}')

def list_drive_children(parent_id):
 items=[]; token=None
 while True:
  response=drive_retry('list children',lambda:drive_api.files().list(
   q=f"'{parent_id}' in parents and trashed=false",spaces='drive',
   fields='nextPageToken,files(id,name,mimeType,md5Checksum,modifiedTime,size)',
   pageSize=1000,pageToken=token).execute())
  items.extend(response.get('files',[])); token=response.get('nextPageToken')
  if not token: return items

def drive_child(parent_id,name,mime_type=None,create_folder=False):
 if create_folder: mime_type='application/vnd.google-apps.folder'
 matches=[item for item in list_drive_children(parent_id)
          if item['name']==name and (mime_type is None or item['mimeType']==mime_type)]
 if len(matches)>1: raise RuntimeError(f'Ambiguous Drive child {name!r} under {parent_id}')
 if matches: return matches[0]
 if not create_folder: raise FileNotFoundError(f'Drive child {name!r} under {parent_id}')
 body={'name':name,'mimeType':'application/vnd.google-apps.folder','parents':[parent_id]}
 return drive_retry(f'create folder {name}',lambda:drive_api.files().create(
  body=body,fields='id,name,mimeType,modifiedTime').execute())

def resolve_drive(relative):
 item={'id':STAGE_C_FOLDER_ID,'name':'SeqTrainerStageC','mimeType':'application/vnd.google-apps.folder'}
 for part in Path(relative).parts: item=drive_child(item['id'],part)
 return item

stage_root=drive_retry('verify Stage C folder ID',lambda:drive_api.files().get(
 fileId=STAGE_C_FOLDER_ID,fields='id,name,mimeType').execute())
if stage_root['mimeType']!='application/vnd.google-apps.folder': raise RuntimeError('STAGE_C_FOLDER_ID is not a folder')
experiment_drive=drive_child(STAGE_C_FOLDER_ID,EXPERIMENT_NAME,create_folder=True)
drive_registry=drive_child(experiment_drive['id'],'validation_registry',create_folder=True)
registry=Path(LOCAL_WORK_ROOT)/'validation_registry'
registry.mkdir(parents=True,exist_ok=True)

def download_drive_file(item,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 partial=target.with_name(target.name+'.partial')
 request=drive_api.files().get_media(fileId=item['id'])
 with open(partial,'wb') as stream:
  downloader=MediaIoBaseDownload(stream,request,chunksize=8*1024*1024); done=False
  while not done: _,done=drive_retry(f"download {item['name']}",downloader.next_chunk)
 os.replace(partial,target)

# Restore durable state once. All active evaluation I/O remains on local disk.
def restore_drive_tree(folder_id,target):
 for item in list_drive_children(folder_id):
  child=target/item['name']
  if item['mimeType']=='application/vnd.google-apps.folder':
   child.mkdir(parents=True,exist_ok=True); restore_drive_tree(item['id'],child)
  else: download_drive_file(item,child)
restore_drive_tree(drive_registry['id'],registry)
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

local_inputs=Path(LOCAL_INPUT_ROOT); local_inputs.mkdir(parents=True,exist_ok=True)
source_paths={
 'C19':'runs/c19_v3_medium_adaptive_e25/latest.pt',
 'C16':'runs/c16_deep_adaptive_5m_paper_exact/latest.pt',
 'dataset':'stage_c_dataset/ordered_streams/nonoverlap_6mer_v1',
 'validation':'study/stage_c_ecoli_medium_deep_memory_v3/panels/validation.json',
 'e25':'study/stage_c_ecoli_medium_deep_memory_v3/panels/e25.json',
 'ani_pairs':'stage_c_dataset/manifests/ecoli_skani_triangle_extended.tsv',
 'ani_membership':'stage_c_dataset/manifests/ecoli_ani_membership.parquet',
}

def stage_file(label,source_relative,target,expected_sha=None):
 item=resolve_drive(source_relative)
 marker=target.with_name(target.name+'.drive.json')
 fingerprint={key:item.get(key) for key in ('id','md5Checksum','modifiedTime','size')}
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_file() and marker.is_file() and json.loads(marker.read_text())==fingerprint:
  if expected_sha is None or sha256_file(target)==expected_sha: return target
 download_drive_file(item,target)
 if expected_sha is not None and sha256_file(target)!=expected_sha:
  raise RuntimeError(f'{label} staged checksum mismatch')
 marker.write_text(json.dumps(fingerprint,sort_keys=True)+'\n')
 return target

def stage_dataset(source_relative,target):
 source=resolve_drive(source_relative)
 manifest=drive_child(source['id'],'token_stream_manifest.json')
 fingerprint={key:manifest.get(key) for key in ('id','md5Checksum','modifiedTime','size')}
 sentinel=target/'.03q_local_stage.json'
 if sentinel.is_file() and json.loads(sentinel.read_text())==fingerprint:
  return target
 target.mkdir(parents=True,exist_ok=True)
 restore_drive_tree(source['id'],target)
 sentinel.write_text(json.dumps(fingerprint,sort_keys=True)+'\n')
 return target

print('Local free space before staging:',shutil.disk_usage('/content').free/2**30,'GiB')
C19_CHECKPOINT=str(stage_file('C19 checkpoint',source_paths['C19'],local_inputs/'checkpoints/C19.pt',EXPECTED_CHECKPOINT_SHA256['C19']))
C16_CHECKPOINT=str(local_inputs/'checkpoints/C16.pt')
if RUN_C16_COMPARISON:
 C16_CHECKPOINT=str(stage_file('C16 checkpoint',source_paths['C16'],Path(C16_CHECKPOINT),EXPECTED_CHECKPOINT_SHA256['C16']))
DATASET_DIR=str(stage_dataset(source_paths['dataset'],local_inputs/'dataset'))
VALIDATION_PANEL=str(stage_file('validation panel',source_paths['validation'],local_inputs/'panels/validation.json'))
E25_TRAINING_PANEL=str(stage_file('E25 panel',source_paths['e25'],local_inputs/'panels/e25.json'))
ANI_PAIRS=str(stage_file('ANI pairs',source_paths['ani_pairs'],local_inputs/'manifests/ani_pairs.tsv'))
ANI_MEMBERSHIP=str(stage_file('ANI membership',source_paths['ani_membership'],local_inputs/'manifests/ani_membership.parquet'))
print('Local free space after staging:',shutil.disk_usage('/content').free/2**30,'GiB')
if not TRUST_OWNED_CHECKPOINT: raise ValueError('Full-state loading requires explicit trust.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

# Incrementally publish only files changed since the restore/stage boundary.
synced={}
for local_path in registry.rglob('*'):
 if local_path.is_file(): synced[str(local_path.relative_to(registry))]=(local_path.stat().st_size,local_path.stat().st_mtime_ns)

drive_folder_cache={'.':drive_registry['id']}
def ensure_drive_folder(relative):
 key=str(relative)
 if key in drive_folder_cache: return drive_folder_cache[key]
 parent_id=ensure_drive_folder(relative.parent)
 item=drive_child(parent_id,relative.name,mime_type='application/vnd.google-apps.folder',create_folder=True)
 drive_folder_cache[key]=item['id']; return item['id']

def upload_drive_file(local_path,relative):
 parent_id=ensure_drive_folder(relative.parent)
 existing=[item for item in list_drive_children(parent_id)
           if item['name']==relative.name and item['mimeType']!='application/vnd.google-apps.folder']
 if len(existing)>1: raise RuntimeError(f'Ambiguous Drive output {relative}')
 media=MediaFileUpload(str(local_path),resumable=True,chunksize=8*1024*1024)
 if existing: request=drive_api.files().update(fileId=existing[0]['id'],media_body=media,fields='id')
 else: request=drive_api.files().create(body={'name':relative.name,'parents':[parent_id]},media_body=media,fields='id')
 response=None
 while response is None: _,response=drive_retry(f'upload {relative}',request.next_chunk)

def sync_registry_to_drive():
 copied=0
 for local_path in sorted(registry.rglob('*')):
  if not local_path.is_file(): continue
  relative=local_path.relative_to(registry); key=str(relative)
  signature=(local_path.stat().st_size,local_path.stat().st_mtime_ns)
  if synced.get(key)==signature: continue
  upload_drive_file(local_path,relative)
  synced[key]=signature; copied+=1
 print('Drive sync complete; changed files copied:',copied)

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
sync_registry_to_drive()
print({'commit':commit,'gpu':torch.cuda.get_device_name(0),'panel':panel_manifest['panel_contract_sha256'],
       'planned_forwards':panel_manifest['planned_workload']['total_segment_forwards'],
       'projected_hours':runtime['projected_hours'],'live_work_root':str(registry),
       'durable_registry':f"https://drive.google.com/drive/folders/{drive_registry['id']}"})
'''


EXECUTION = r'''# @title 4. Run C19 first; optionally add C16 and paired comparison later
def run_model(model,checkpoint,max_hours=None):
 deadline=time.monotonic()+(max_hours or SYNC_CHUNK_HOURS)*3600
 while not (bundle/model/'COMPLETE.json').is_file():
  remaining_hours=(deadline-time.monotonic())/3600
  if remaining_hours<=0: break
  chunk_hours=min(SYNC_CHUNK_HOURS,remaining_hours)
  command=[*runner,'run-model','--model',model,'--checkpoint',checkpoint,
   '--dataset-dir',DATASET_DIR,'--validation-panel',VALIDATION_PANEL,
   '--frozen-panel',str(panel),'--output',str(bundle/model),'--device','cuda',
   '--max-runtime-hours',str(chunk_hours),'--trust-owned-checkpoint']
  run_checked(f'bounded_{model}',command)
  sync_registry_to_drive()
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
 sync_registry_to_drive()
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
 sync_registry_to_drive()
 display(Markdown((analysis/'SCIENTIFIC_REPORT.md').read_text()))
 print('Local content-addressed archive:',retained)
 print('Durable Drive registry:',f"https://drive.google.com/drive/folders/{drive_registry['id']}")
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
