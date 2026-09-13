#!/usr/bin/env python
"""Audit and summarize the exclusive-GPU controlled-cost controlled cost matrix."""
import argparse, csv, json
from pathlib import Path
import numpy as np

def result_dir(root, problem, tag):
    if problem=='burgers': return root/f'results_{tag}'
    if problem=='helmholtz_1_4': return root/f'results_helmholtz_a1_1.0_a2_4.0_{tag}'
    if problem=='helmholtz_4_4': return root/f'results_helmholtz_a1_4.0_a2_4.0_{tag}'
    if problem=='klein_gordon': return root/f'results_klein_gordon_{tag}'
    if problem=='lid_driven_cavity': return root/f'results_lid_driven_cavity_{tag}'
    if problem=='poisson_5d': return root/f'results_poisson_5d_{tag}'
    if problem=='schrodinger': return root/f'results_schrodinger_{tag}'
    raise ValueError(problem)

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text()); failures=[]; rows=[]
    for job in m['jobs']:
        d=result_dir(a.root,job['problem'],job['result_tag']); files=list(d.glob('runs/run_*/logs/cost_measurement.json'))
        if len(files)!=1: failures.append(f"{job['result_tag']}: expected one cost file, found {len(files)}");continue
        x=json.loads(files[0].read_text()); times=x.get('step_times_sec',[])
        if x.get('protocol')!='controlled-cost-v1' or x.get('warmup_steps')!=100 or x.get('timed_steps')!=200 or len(times)!=200: failures.append(f"{job['result_tag']}: invalid profiler protocol");continue
        if not x.get('device') or x.get('parameter_count',0)<=0 or x.get('peak_memory_allocated_mib',0)<=0: failures.append(f"{job['result_tag']}: missing hardware/cost fields");continue
        rows.append({'problem':job['problem'],'architecture':job['architecture'],'optimizer':job['optimizer'],'result_tag':job['result_tag'],'repetition':job['master_seed_base']-19017,'parameter_count':x['parameter_count'],'step_time_mean_sec':x['step_time_sec']['mean'],'step_time_median_sec':x['step_time_sec']['median'],'step_time_std_sec':x['step_time_sec']['std'],'peak_memory_allocated_mib':x['peak_memory_allocated_mib'],'device':x['device'],'torch_version':x['torch_version'],'cuda_version':x['cuda_version']})
    if len({(r['device'],r['torch_version'],r['cuda_version']) for r in rows}) > 1:
        failures.append('controlled-cost jobs were not measured in one identical hardware/software environment')
    if failures or len(rows)!=len(m['jobs']): raise SystemExit(json.dumps({'passed':False,'failures':failures,'rows':len(rows)},indent=2))
    summary=[]
    for key in sorted({(r['problem'],r['architecture'],r['optimizer']) for r in rows}):
        selected=[r for r in rows if (r['problem'],r['architecture'],r['optimizer'])==key]
        for metric in ('step_time_mean_sec','peak_memory_allocated_mib'):
            v=np.array([float(r[metric]) for r in selected]);summary.append({'problem':key[0],'architecture':key[1],'optimizer':key[2],'metric':metric,'n':len(v),'mean':float(v.mean()),'std':float(v.std(ddof=1))})
        if len({r['parameter_count'] for r in selected})!=1: raise ValueError(f'{key}: parameter count changed across repetitions')
    a.output_dir.mkdir(parents=True,exist_ok=True);write(a.output_dir/'run_level_cost.csv',rows);write(a.output_dir/'cost_summary.csv',summary);(a.output_dir/'audit.json').write_text(json.dumps({'passed':True,'jobs':len(rows),'conditions':len(summary)//2},indent=2)+'\n');print((a.output_dir/'audit.json').read_text())
if __name__=='__main__':main()
