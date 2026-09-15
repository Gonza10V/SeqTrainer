"""Build the single-Colab exact C16/C19 generation comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks/titans_stage_c/03p_stage_c_c19_checkpoint_generation_status.ipynb"


def cell(source: str, kind: str = "code") -> dict[str, object]:
    payload: dict[str, object] = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


INTRO = r"""# Stage C 03p — exact C16–C19 generation comparison

This single-GPU notebook applies the **exact original C16 generation protocol**
to the immutable final C19 E25 checkpoint, then compares the two reports only
after enforcing their complete protocol identity. It generates four held-out
prompts at temperatures 0.8, 1.0, and 1.2 (12 continuations total).

The separate temperature-0.6 E25 gate arm remains in notebook 03m and is not
run here. This assessment is exploratory: it measures conditional-generation
calibration, cannot independently authorize E100, and cannot establish
biological function, expression, fitness, viability, or safety. Expected A100
runtime is approximately 2–4 hours.
"""


CONFIG = r"""# USER CONFIGURATION — normally this is the only cell you edit.
REPO_URL='https://github.com/Gonza10V/SeqTrainer.git'
GIT_REF='ae72fae21ff9a0b50e4fe1d9d642c38c643b4923'  # C19 training code commit

DRIVE_ROOT='/content/drive/MyDrive/SeqTrainerStageC'
INPUT_FOLDER=f'{DRIVE_ROOT}/runs/c19_v3_medium_adaptive_e25'
CHECKPOINT_FILE='latest.pt'
EXPECTED_CHECKPOINT_SHA256='07fb2069b1f29a76898a90d8dfb899c5ca46cb90608fac45bc0ddff9876dbd1a'
C16_BASELINE_REPORT=f'{DRIVE_ROOT}/runs/c16_deep_adaptive_5m_paper_exact/generation_diagnostics_v1/generation_evaluation.json'

DATASET_DIR=f'{DRIVE_ROOT}/stage_c_dataset/ordered_streams/nonoverlap_6mer_v1'
TAXONOMY_MANIFEST=f'{DRIVE_ROOT}/stage_c_dataset/manifests/accession_manifest.parquet'
OUTPUT_FOLDER=f'{INPUT_FOLDER}/exact_c16_protocol_comparison_v1'
VENV_DIR='/content/seqtrainer-c19-exact-c16-v1'

# Immutable C16 protocol. The validation cell also checks these against C16.
PROMPTS=4
PROMPT_TOKENS=32
NEW_TOKENS=1024
TEMPERATURES='0.8,1.0,1.2'
TOP_K=128
TOP_P=0.95
SEED=20260781
MEMORY_MODE='adaptive'
INSTALL_PRODIGAL=True
TRUST_OWNED_CHECKPOINT=True  # checkpoints include owned optimizer/RNG metadata
"""


BOOTSTRAP = r"""from pathlib import Path
from google.colab import drive
import csv, hashlib, json, math, os, shutil, subprocess, sys

mount=Path('/content/drive')
if not (mount/'MyDrive').is_dir():
    drive.mount(str(mount),timeout_ms=120000)

repo=Path('/content/SeqTrainer')
if not repo.exists(): subprocess.run(['git','clone',REPO_URL,str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(repo),'checkout',GIT_REF],check=True)

input_dir=Path(INPUT_FOLDER)
bootstrap_log=input_dir/'results'/'exact_c16_comparison_bootstrap.log'
bootstrap_log.parent.mkdir(parents=True,exist_ok=True)
def run_bootstrap(command):
    with bootstrap_log.open('a',encoding='utf-8') as log:
        log.write('command='+repr(command)+'\n')
        try: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True,check=True)
        except subprocess.CalledProcessError:
            print(bootstrap_log.read_text(encoding='utf-8')[-12000:]); raise

# Isolate the numeric stack from the live Colab kernel.
venv=Path(VENV_DIR)
if not (venv/'bin'/'python').is_file():
    run_bootstrap([sys.executable,'-m','pip','install','--quiet','virtualenv>=20.26'])
    run_bootstrap([sys.executable,'-m','virtualenv','--system-site-packages',str(venv)])
stage_c_python=str(venv/'bin'/'python')
run_bootstrap([stage_c_python,'-m','pip','install','--quiet','--upgrade',
    'numpy==1.26.4','pandas==2.2.2','pyarrow==18.1.0','matplotlib==3.9.2',
    'rdflib>=6.3.2','requests>=2.31','sbol2>=1.4'])
run_bootstrap([stage_c_python,'-m','pip','install','--no-deps','-e',str(repo)])
run_bootstrap([stage_c_python,'-c',
    "import json,numpy,pandas,pyarrow,matplotlib,torch,seqtrainer; print(json.dumps({'numpy':numpy.__version__,'pandas':pandas.__version__,'pyarrow':pyarrow.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__},sort_keys=True))"])
stage_c_generate=str(venv/'bin'/'seqtrainer-titans-stage-c-generate')
stage_c_runner=str(venv/'bin'/'seqtrainer-titans-stage-c-colab-run')
for executable in (stage_c_generate,stage_c_runner):
    if not Path(executable).is_file(): raise FileNotFoundError(executable)

import torch
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime before continuing.')
if not TRUST_OWNED_CHECKPOINT: raise ValueError('TRUST_OWNED_CHECKPOINT must be true for this owned full-state artifact.')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
if INSTALL_PRODIGAL and not shutil.which('prodigal'):
    try:
        subprocess.run(['apt-get','update'],check=True)
        subprocess.run(['apt-get','install','-y','prodigal'],check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError('Prodigal is required for the complete comparison.') from error
if not shutil.which('prodigal'): raise FileNotFoundError('Prodigal is required for this comparison.')

checkpoint=input_dir/CHECKPOINT_FILE
dataset=Path(DATASET_DIR)
taxonomy=Path(TAXONOMY_MANIFEST)
baseline_path=Path(C16_BASELINE_REPORT)
for label,path in {'C19 checkpoint':checkpoint,'dataset':dataset,'taxonomy manifest':taxonomy,'original C16 report':baseline_path}.items():
    if not path.exists(): raise FileNotFoundError(f'Missing {label}: {path}')

def sha256_file(path,chunk_size=8*1024*1024):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(chunk_size),b''): digest.update(chunk)
    return digest.hexdigest()
checkpoint_sha256=sha256_file(checkpoint)
if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
    raise RuntimeError(f'Refusing non-final C19 checkpoint: expected {EXPECTED_CHECKPOINT_SHA256}, found {checkpoint_sha256}')

inspect_checkpoint=r'''import json,sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if not isinstance(p,dict): raise ValueError('Checkpoint payload is not a mapping.')
t=p.get('trainer_state',{}); c=p.get('model_config',{})
print(json.dumps({'optimizer_step':int(t.get('optimizer_step',0)),'processed_bases':int(t.get('processed_bases',0)),'code_commit':p.get('code_commit'),'format_version':p.get('format_version'),'tokenizer_name':c.get('tokenizer_name'),'tokenizer_checksum':c.get('tokenizer_checksum'),'block_count':c.get('block_count'),'d_model':c.get('d_model')},sort_keys=True))'''
checkpoint_metadata=json.loads(subprocess.check_output([stage_c_python,'-c',inspect_checkpoint,str(checkpoint)],text=True,env={**os.environ,'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'}))
run_tag=f"step_{checkpoint_metadata['optimizer_step']:08d}_{checkpoint_sha256[:12]}"
output_root=Path(OUTPUT_FOLDER)/run_tag
generation_dir=output_root/'c19_exact_c16_protocol'
output_root.mkdir(parents=True,exist_ok=True)
preflight={'format_version':1,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha256,**checkpoint_metadata,'dataset':str(dataset),'taxonomy_manifest':str(taxonomy),'c16_baseline_report':str(baseline_path),'c16_baseline_sha256':sha256_file(baseline_path),'git_ref':GIT_REF,'gpu':torch.cuda.get_device_name(0),'isolated_python':stage_c_python}
(output_root/'checkpoint_preflight.json').write_text(json.dumps(preflight,indent=2,sort_keys=True)+'\n')
print(json.dumps(preflight,indent=2,sort_keys=True))
print('Output directory:',output_root)
"""


RUN = r"""def load_report(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError(f'Report is not an object: {path}')
    return value

EXPECTED_PROTOCOL={'split':'val','species':'Escherichia coli','memory_mode':MEMORY_MODE,
    'prompt_tokens':PROMPT_TOKENS,'new_tokens':NEW_TOKENS,
    'temperatures':[0.8,1.0,1.2],'top_k':TOP_K,'top_p':TOP_P,'seed':SEED}
def prompt_identity(report):
    return sorted({(str(row.get('prompt_id')),str(row.get('source_stream_id')),str(row.get('source_accession'))) for row in report.get('generation_rows',[])})
def report_is_complete(path):
    try:
        report=load_report(path)
        rows=report.get('generation_rows',[])
        return (all(report.get(k)==v or (k=='temperatures' and list(report.get(k,[]))==v) for k,v in EXPECTED_PROTOCOL.items())
            and len(rows)==12 and len(prompt_identity(report))==4
            and all(int(row['tokens'])==NEW_TOKENS and int(row['bases'])==NEW_TOKENS*6 for row in rows)
            and report.get('prodigal',{}).get('status')=='completed')
    except (KeyError,TypeError,ValueError,json.JSONDecodeError): return False

def run_logged(root,label,command):
    root=Path(root); (root/'logs').mkdir(parents=True,exist_ok=True)
    drive_log=root/'logs'/f'{label}.log'; local_log=Path('/content')/f'{run_tag}_{label}.log'
    with local_log.open('w',encoding='utf-8',buffering=1) as log:
        log.write('command='+json.dumps(command)+'\n')
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=repo,env={**os.environ,'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'})
        assert process.stdout is not None
        for line in process.stdout: print(line,end='',flush=True); log.write(line)
        return_code=process.wait()
    shutil.copyfile(local_log,drive_log)
    execution={'format_version':1,'label':label,'command':command,'return_code':return_code,'local_log':str(local_log),'drive_log':str(drive_log)}
    (root/f'{label}_execution.json').write_text(json.dumps(execution,indent=2,sort_keys=True)+'\n')
    if return_code:
        (root/'FAILED.txt').write_text(f'generation failed: {label}\nreturn_code={return_code}\nlog={drive_log}\n')
        print(local_log.read_text(encoding='utf-8',errors='replace')[-20000:])
        raise subprocess.CalledProcessError(return_code,command)
    if (root/'FAILED.txt').exists(): (root/'FAILED.txt').unlink()

baseline=load_report(baseline_path)
baseline_protocol_mismatches={k:{'expected':v,'baseline':baseline.get(k)} for k,v in EXPECTED_PROTOCOL.items() if not (baseline.get(k)==v or (k=='temperatures' and list(baseline.get(k,[]))==v))}
if baseline_protocol_mismatches: raise ValueError('Original C16 report does not encode the immutable protocol: '+json.dumps(baseline_protocol_mismatches,sort_keys=True))
if len(baseline.get('generation_rows',[]))!=12 or len(prompt_identity(baseline))!=4:
    raise ValueError('Original C16 report must contain exactly 12 continuations from four prompts.')

report_path=generation_dir/'generation_evaluation.json'
if report_is_complete(report_path):
    print('Validated complete C19 report exists; generation skipped:',report_path)
else:
    command=[stage_c_generate,'--dataset-dir',str(dataset),'--taxonomy-manifest',str(taxonomy),
        '--checkpoint',str(checkpoint),'--output-dir',str(generation_dir),'--split','val',
        '--species','Escherichia coli','--prompts',str(PROMPTS),'--prompt-tokens',str(PROMPT_TOKENS),
        '--new-tokens',str(NEW_TOKENS),'--temperatures',TEMPERATURES,'--top-k',str(TOP_K),
        '--top-p',str(TOP_P),'--seed',str(SEED),'--device','cuda','--memory-mode',MEMORY_MODE,
        '--prodigal',shutil.which('prodigal')]
    run_logged(generation_dir,'generation_exact_c16_protocol',command)
    if not report_is_complete(report_path): raise RuntimeError('Generation finished without a complete valid 12-sequence report.')
print('Generation workflow complete. Run the comparison cell.')
"""


ANALYZE = r"""# Exact protocol validation and analysis. Any mismatch stops before comparison.
import matplotlib.pyplot as plt
import numpy as np

candidate=load_report(report_path)
baseline=load_report(baseline_path)
def finite_numbers(value):
    if isinstance(value,dict): return all(finite_numbers(v) for v in value.values())
    if isinstance(value,list): return all(finite_numbers(v) for v in value)
    if isinstance(value,(float,np.floating)): return math.isfinite(float(value))
    return True
def fasta_records(path):
    records={}; name=None
    for line in Path(path).read_text(encoding='ascii').splitlines():
        if line.startswith('>'): name=line[1:].split()[0]; records[name]=''
        elif name is not None: records[name]+=line.strip()
    return records
def fasta_lengths(path): return {key:len(value) for key,value in fasta_records(path).items()}

protocol_fields=('dataset_fingerprint','taxonomy_manifest_sha256','split','species','memory_mode','prompt_tokens','new_tokens','temperatures','top_k','top_p','seed')
mismatches={key:{'c16':baseline.get(key),'c19':candidate.get(key)} for key in protocol_fields if baseline.get(key)!=candidate.get(key)}
for model,report in (('c16',baseline),('c19',candidate)):
    config=report.get('model_config',{})
    if config.get('tokenizer_name')!='nonoverlap_6mer_v1': mismatches[f'{model}.tokenizer_name']={'expected':'nonoverlap_6mer_v1','actual':config.get('tokenizer_name')}
if baseline.get('model_config',{}).get('tokenizer_checksum')!=candidate.get('model_config',{}).get('tokenizer_checksum'):
    mismatches['tokenizer_checksum']={'c16':baseline.get('model_config',{}).get('tokenizer_checksum'),'c19':candidate.get('model_config',{}).get('tokenizer_checksum')}
if prompt_identity(baseline)!=prompt_identity(candidate): mismatches['prompt_identities']={'c16':prompt_identity(baseline),'c19':prompt_identity(candidate)}
for model,report in (('c16',baseline),('c19',candidate)):
    rows=report.get('generation_rows',[])
    if len(rows)!=12: mismatches[f'{model}.generated_sequence_count']={'expected':12,'actual':len(rows)}
    bad=[row.get('sequence_id') for row in rows if row.get('tokens')!=1024 or row.get('bases')!=6144]
    if bad: mismatches[f'{model}.sequence_lengths']={'expected_tokens':1024,'expected_bases':6144,'invalid_sequence_ids':bad}
    if report.get('prodigal',{}).get('status')!='completed': mismatches[f'{model}.prodigal']={'expected':'completed','actual':report.get('prodigal',{}).get('status')}
if not finite_numbers(baseline) or not finite_numbers(candidate): mismatches['numeric_finiteness']={'c16':finite_numbers(baseline),'c19':finite_numbers(candidate)}
baseline_dir=baseline_path.parent
for label,path in {'c16 generated FASTA':baseline_dir/'generated_sequences.fasta','c16 reference FASTA':baseline_dir/'heldout_reference_continuations.fasta','c19 generated FASTA':generation_dir/'generated_sequences.fasta','c19 reference FASTA':generation_dir/'heldout_reference_continuations.fasta'}.items():
    if not path.is_file(): mismatches[label]={'missing':str(path)}
    else:
        records=fasta_records(path); expected_count=12 if 'generated' in label else 4
        invalid={key:{'length':len(sequence),'non_dna':sorted(set(sequence.upper())-set('ACGT'))} for key,sequence in records.items() if len(sequence)!=6144 or not set(sequence.upper()).issubset(set('ACGT'))}
        if len(records)!=expected_count or invalid: mismatches[label]={'expected_records':expected_count,'actual_records':len(records),'invalid_sequences':invalid}
c16_prompts=baseline_dir/'heldout_prompts.fasta'; c19_prompts=generation_dir/'heldout_prompts.fasta'
if not c16_prompts.is_file() or not c19_prompts.is_file(): mismatches['heldout_prompt_fastas']={'c16':str(c16_prompts),'c19':str(c19_prompts)}
else:
    c16_prompt_records=fasta_records(c16_prompts); c19_prompt_records=fasta_records(c19_prompts)
    if c16_prompt_records!=c19_prompt_records or len(c19_prompt_records)!=4 or any(len(sequence)!=PROMPT_TOKENS*6 for sequence in c19_prompt_records.values()):
        mismatches['heldout_prompt_sequences']={'identical':c16_prompt_records==c19_prompt_records,'c16_lengths':{k:len(v) for k,v in c16_prompt_records.items()},'c19_lengths':{k:len(v) for k,v in c19_prompt_records.items()}}
if mismatches: raise ValueError('REFUSING COMPARISON — protocol mismatch: '+json.dumps(mismatches,sort_keys=True))

groups=('temperature_0.8','temperature_1','temperature_1.2')
models={'c16':baseline,'c19':candidate}
def summary(report,group): return report['distribution_summary'][group]
def absolute_error(report,group,key): return abs(float(summary(report,group)[key])-float(summary(report,'reference')[key]))
def prodigal_error(report,group,key): return abs(float(report['prodigal']['groups'][group][key])-float(report['prodigal']['groups']['reference'][key]))

def prodigal_distributions(report,report_dir):
    fasta_sizes=fasta_lengths(Path(report_dir)/'generated_sequences.fasta')
    calls={sequence_id:[] for sequence_id in fasta_sizes}
    gff=Path(report_dir)/'generated_sequences.prodigal.gff'
    if not gff.is_file(): raise FileNotFoundError(f'Missing Prodigal GFF required for distribution comparison: {gff}')
    for line in gff.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'): continue
        fields=line.split('\t')
        if len(fields)==9 and fields[2]=='CDS': calls.setdefault(fields[0],[]).append((int(fields[3]),int(fields[4])))
    policies={str(row['sequence_id']):str(row['policy']) for row in report['generation_rows']}
    output={group:{'gene_lengths':[],'intergenic_lengths':[]} for group in groups}
    for sequence_id,length in fasta_sizes.items():
        group=policies[sequence_id]; merged=[]
        for start,end in sorted(calls.get(sequence_id,[])):
            output[group]['gene_lengths'].append(end-start+1)
            if merged and start<=merged[-1][1]+1: merged[-1][1]=max(merged[-1][1],end)
            else: merged.append([start,end])
        cursor=1
        for start,end in merged:
            if start>cursor: output[group]['intergenic_lengths'].append(start-cursor)
            cursor=end+1
        if cursor<=length: output[group]['intergenic_lengths'].append(length-cursor+1)
    return output

raw_prodigal={
    'c16':prodigal_distributions(baseline,baseline_dir),
    'c19':prodigal_distributions(candidate,generation_dir),
}
def distribution_summary(values):
    array=np.asarray(values,dtype=float)
    return {'count':int(array.size),'minimum':float(array.min()) if array.size else 0.0,
        'q25':float(np.quantile(array,.25)) if array.size else 0.0,'median':float(np.median(array)) if array.size else 0.0,
        'q75':float(np.quantile(array,.75)) if array.size else 0.0,'maximum':float(array.max()) if array.size else 0.0}

comparison={'format_version':1,'classification':'exploratory_exact_c16_protocol_comparison','protocol_match':True,
    'checkpoint_sha256':checkpoint_sha256,'source_report_sha256':{'c16':sha256_file(baseline_path),'c19':sha256_file(report_path)},
    'protocol':{key:candidate.get(key) for key in protocol_fields},'prompt_identities':[{'prompt_id':p,'source_stream_id':s,'source_accession':a} for p,s,a in prompt_identity(candidate)],
    'generated_sequence_count':12,'all_numeric_metrics_finite':True,'temperatures':{},
    'limits':['Exploratory conditional-generation quality comparison only.','This result cannot independently authorize E100.','Sequence, ORF, and Prodigal similarity cannot establish biological function, expression, fitness, viability, or safety.']}
metric_rows=[]
for group in groups:
    temperature=float(group.removeprefix('temperature_'))
    entry={'kmer_jsd':{},'sequence_calibration':{},'prodigal_calibration':{},'memory_diagnostics':{}}
    for model,report in models.items():
        entry['kmer_jsd'][model]={str(k):float(report['kmer_jsd_to_heldout_reference'][group][str(k)]) for k in range(1,7)}
        s=summary(report,group); ref=summary(report,'reference')
        entry['sequence_calibration'][model]={
            'gc_fraction':float(s['gc_fraction']),'gc_absolute_error':absolute_error(report,group,'gc_fraction'),
            'base_entropy_bits':float(s['base_entropy_bits']),'entropy_absolute_error':absolute_error(report,group,'base_entropy_bits'),
            'overlapping_unique_6mer_fraction':float(s['unique_6mer_fraction']),'overlapping_diversity_ratio':float(s['unique_6mer_fraction'])/max(float(ref['unique_6mer_fraction']),1e-12),
            'aligned_unique_6mer_fraction':float(s['aligned_unique_6mer_fraction']),'aligned_diversity_ratio':float(s['aligned_unique_6mer_fraction'])/max(float(ref['aligned_unique_6mer_fraction']),1e-12),
            'max_homopolymer':float(s['max_homopolymer']),'max_homopolymer_absolute_error':absolute_error(report,group,'max_homopolymer'),
            'heuristic_orf_count':float(s['heuristic_orfs_at_least_90bp']),'orf_count_absolute_error':absolute_error(report,group,'heuristic_orfs_at_least_90bp'),
            'heuristic_longest_orf_bases':float(s['heuristic_longest_orf_bases']),'longest_orf_absolute_error':absolute_error(report,group,'heuristic_longest_orf_bases')}
        pg=report['prodigal']['groups'][group]
        entry['prodigal_calibration'][model]={key:float(pg[key]) for key in ('coding_density','genes','genes_per_10kb','median_gene_bases','median_intergenic_bases')}
        entry['prodigal_calibration'][model].update({f'{key}_absolute_error':prodigal_error(report,group,key) for key in ('coding_density','genes_per_10kb','median_gene_bases','median_intergenic_bases')})
        entry['prodigal_calibration'][model]['gene_length_distribution']=distribution_summary(raw_prodigal[model][group]['gene_lengths'])
        entry['prodigal_calibration'][model]['intergenic_length_distribution']=distribution_summary(raw_prodigal[model][group]['intergenic_lengths'])
        selected=[row for row in report['generation_rows'] if row['policy']==group]
        diagnostic_keys=sorted(key for key in selected[0] if key.startswith('mean_'))
        entry['memory_diagnostics'][model]={key:float(np.mean([float(row[key]) for row in selected])) for key in diagnostic_keys}
        for k,value in entry['kmer_jsd'][model].items(): metric_rows.append({'temperature':temperature,'model':model,'family':'kmer_jsd','metric':f'{k}-mer_jsd_bits','value':value})
        for family in ('sequence_calibration','prodigal_calibration','memory_diagnostics'):
            for key,value in entry[family][model].items():
                if isinstance(value,dict):
                    for statistic,number in value.items(): metric_rows.append({'temperature':temperature,'model':model,'family':family,'metric':f'{key}.{statistic}','value':number})
                else: metric_rows.append({'temperature':temperature,'model':model,'family':family,'metric':key,'value':value})
    comparison['temperatures'][group]=entry
if not finite_numbers(comparison): raise RuntimeError('Comparison produced a non-finite numeric value.')

colors={'c16':'#7251A3','c19':'#138A8A'}
def finish(fig,path,title): fig.suptitle(title,fontsize=15,fontweight='bold'); fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(path,dpi=220,facecolor='white'); plt.close(fig)
# k=1..6 curves, separately at every temperature.
fig,axes=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for axis,group in zip(axes,groups):
    for model in models: axis.plot(range(1,7),[comparison['temperatures'][group]['kmer_jsd'][model][str(k)] for k in range(1,7)],marker='o',label=model.upper(),color=colors[model])
    axis.set(title=f"T={group.split('_')[-1]}",xlabel='k',ylabel='JSD (bits)'); axis.set_xticks(range(1,7)); axis.grid(alpha=.25)
axes[0].legend(frameon=False); finish(fig,output_root/'c16_c19_kmer_jsd_k1_to_k6.png','Exact C16 protocol: k-mer divergence to held-out reference')
fig,axis=plt.subplots(figsize=(8,5)); x=np.arange(3); width=.34
for offset,model in ((-width/2,'c16'),(width/2,'c19')): axis.bar(x+offset,[comparison['temperatures'][g]['kmer_jsd'][model]['6'] for g in groups],width,label=model.upper(),color=colors[model])
axis.set(xticks=x,xticklabels=['0.8','1.0','1.2'],xlabel='temperature',ylabel='6-mer JSD (bits)'); axis.legend(frameon=False); axis.grid(axis='y',alpha=.25); finish(fig,output_root/'c16_c19_temperature_6mer_jsd.png','Temperature-specific 6-mer divergence')

def calibration_plot(filename,title,items,family='sequence_calibration'):
    fig,axes=plt.subplots(1,len(items),figsize=(5.2*len(items),4.8),squeeze=False)
    for axis,(key,label,ideal) in zip(axes[0],items):
        x=np.arange(3); width=.34
        for offset,model in ((-width/2,'c16'),(width/2,'c19')): axis.bar(x+offset,[comparison['temperatures'][g][family][model][key] for g in groups],width,label=model.upper(),color=colors[model])
        if ideal is not None: axis.axhline(ideal,color='#C64B45',ls='--',lw=1)
        axis.set(xticks=x,xticklabels=['0.8','1.0','1.2'],xlabel='temperature',ylabel=label); axis.grid(axis='y',alpha=.25)
    axes[0,0].legend(frameon=False); finish(fig,output_root/filename,title)
calibration_plot('c16_c19_gc_entropy_calibration.png','GC and entropy calibration',(('gc_absolute_error','GC absolute error',0),('entropy_absolute_error','entropy absolute error (bits)',0)))
calibration_plot('c16_c19_diversity_calibration.png','6-mer diversity calibration',(('overlapping_diversity_ratio','overlapping diversity ratio',1),('aligned_diversity_ratio','tokenizer-aligned diversity ratio',1)))
calibration_plot('c16_c19_orf_calibration.png','Heuristic ORF calibration',(('orf_count_absolute_error','ORF-count absolute error',0),('longest_orf_absolute_error','longest-ORF absolute error (bases)',0)))
calibration_plot('c16_c19_prodigal_calibration.png','Prodigal calibration',(('coding_density_absolute_error','coding-density absolute error',0),('genes_per_10kb_absolute_error','gene-density absolute error',0),('median_gene_bases_absolute_error','median gene-length error (bases)',0),('median_intergenic_bases_absolute_error','median intergenic error (bases)',0)),family='prodigal_calibration')

fig,axes=plt.subplots(2,1,figsize=(14,9))
positions=np.arange(6); labels=[]
for group in groups:
    for model in ('c16','c19'): labels.append(f"T={group.split('_')[-1]}\n{model.upper()}")
for axis,key,ylabel in ((axes[0],'gene_lengths','gene length (bases)'),(axes[1],'intergenic_lengths','intergenic length (bases)')):
    values=[raw_prodigal[model][group][key] for group in groups for model in ('c16','c19')]
    boxes=axis.boxplot(values,positions=positions,showfliers=False,patch_artist=True)
    for patch,model in zip(boxes['boxes'],('c16','c19')*3): patch.set_facecolor(colors[model]); patch.set_alpha(.75)
    axis.set(xticks=positions,xticklabels=labels,ylabel=ylabel); axis.grid(axis='y',alpha=.25)
finish(fig,output_root/'c16_c19_prodigal_length_distributions.png','Prodigal gene-length and intergenic distributions')

# Persist the complete table/report before archiving.
(output_root/'c16_c19_exact_comparison.json').write_text(json.dumps(comparison,indent=2,sort_keys=True)+'\n')
with (output_root/'c16_c19_comparison_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=('temperature','model','family','metric','value')); writer.writeheader(); writer.writerows(metric_rows)
lines=['# Exact C16–C19 generation comparison','',f"- Classification: `{comparison['classification']}`",f"- Protocol identity verified: `{comparison['protocol_match']}`",f"- Final C19 checkpoint SHA-256: `{checkpoint_sha256}`",'- Generated sequences: `12` (four identical held-out prompts × three temperatures)','- All numeric metrics finite: `True`','',
    '## 6-mer JSD to held-out reference','', '| Temperature | C16 | C19 | C19 − C16 |','|---:|---:|---:|---:|']
for group in groups:
    a=comparison['temperatures'][group]['kmer_jsd']['c16']['6']; b=comparison['temperatures'][group]['kmer_jsd']['c19']['6']; lines.append(f"| {group.split('_')[-1]} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
lines.extend(['','## Artifacts','', '- k-mer JSD curves for k=1…6 and temperature-specific 6-mer comparison', '- GC/entropy, overlapping/aligned diversity, heuristic ORF, and Prodigal calibration plots', '- Machine-readable JSON and long-form metrics CSV', '- Generated/reference FASTA, sequence metrics, Prodigal calls, and execution logs in `c19_exact_c16_protocol/`','', '## Interpretation limits','',*[f'- {item}' for item in comparison['limits']],''])
(output_root/'EXACT_C16_C19_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
archive=shutil.make_archive(str(output_root),'zip',root_dir=output_root)
print('Machine-readable comparison:',output_root/'c16_c19_exact_comparison.json')
print('Metrics CSV:',output_root/'c16_c19_comparison_metrics.csv')
print('Generated FASTA:',generation_dir/'generated_sequences.fasta')
print('Reference FASTA:',generation_dir/'heldout_reference_continuations.fasta')
print('Logs:',generation_dir/'logs')
print('ZIP archive:',archive)
"""


notebook = {
    "cells": [cell(INTRO, "markdown"), cell(CONFIG), cell(BOOTSTRAP), cell(RUN), cell(ANALYZE)],
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
