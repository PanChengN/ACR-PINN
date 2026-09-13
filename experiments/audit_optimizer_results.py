#!/usr/bin/env python
"""Fail-closed audit for the representative task-gradient comparison multitask optimizer matrix."""
import argparse, csv, json
from pathlib import Path

def folder(root, problem, tag):
    return root/f'results_{tag}' if problem=='burgers' else root/f'results_helmholtz_a1_4.0_a2_4.0_{tag}'

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    manifest=json.loads(a.manifest.read_text()); failures=[]; checks=[]; hashes={}
    for job in manifest['jobs']:
        d=folder(a.root,job['problem'],job['result_tag']); runs=sorted((d/'runs').glob('run_*'))
        summary=d/'aggregate/logs/summary.csv'; rows=[]
        if summary.exists():
            with summary.open(newline='') as h: rows=[r for r in csv.DictReader(h) if r.get('run','').isdigit()]
        if len(runs)!=5 or len(rows)!=5: failures.append(f"{job['result_tag']}: runs={len(runs)}, summary={len(rows)}")
        for run in runs:
            mp=run/'logs/run_metadata.json'; cp=run/'checkpoints/model_final.pt'
            if not mp.exists() or not cp.exists(): failures.append(f'{run}: missing metadata/checkpoint'); continue
            m=json.loads(mp.read_text()); c=m['config']; s=m['seeds']
            if c.get('network')!=job['architecture'] or c.get('mode')!=job['optimizer'] or c.get('dynamic_sampling') is not True: failures.append(f'{run}: protocol mismatch')
            key=(job['problem'],s['master_seed'],s['sample_seed'],s['init_seed'])
            prior=hashes.setdefault(key,m['dataset_sha256'])
            if prior!=m['dataset_sha256']: failures.append(f'{run}: unequal paired training-data hash')
            checks.append({'tag':job['result_tag'],'run':run.name,**s,'dataset_sha256':m['dataset_sha256']})
    payload={'passed':not failures,'expected_runs':5*len(manifest['jobs']),'checks':checks,'failures':failures}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'passed':payload['passed'],'checks':len(checks),'failures':failures},indent=2)); raise SystemExit(1 if failures else 0)
if __name__=='__main__': main()
