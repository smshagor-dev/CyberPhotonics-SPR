"""Illustrative planar TM film continuation; never PCF or Au evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from planar_three_layer_tm import solve

def campaign(thicknesses=(.05, .075, .1, .15, .2, .3, .5), wavelength=.8,
             dielectric=1.77, metal=-24+1.5j):
    ts = [float(t) for t in thicknesses]
    if not ts or any(not np.isfinite(t) or t <= 0 for t in ts) or any(b <= a for a,b in zip(ts,ts[1:])):
        raise ValueError('strictly increasing positive thickness required')
    if not np.isfinite(wavelength) or wavelength <= 0 or not (dielectric > 0 and metal.imag > 0 and metal.real < -dielectric):
        raise ValueError('invalid passive illustrative material/wavelength')
    reference = np.sqrt(dielectric*metal/(dielectric+metal))
    seed = reference
    rows = []
    for t in ts:
        attempts = []
        solutions = []
        for guess in (seed, reference):
            try:
                result = solve(wavelength,t,dielectric,metal,dielectric,guess)
                n = complex(result['neff_real'],result['neff_imag'])
                if n.real <= 0 or not np.isfinite(n.real+n.imag):
                    raise ValueError('nonpassive root')
                attempts.append({'success':True,'residual':float(result['residual'])})
                solutions.append((abs(n-seed),n,result))
            except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
                attempts.append({'success':False,'error':type(exc).__name__})
        if solutions:
            _,seed,solution=min(solutions,key=lambda item:item[0])
            rows.append({'thickness_um':t,'status':'converged','neff':[seed.real,seed.imag],
                         'distance_to_single_interface':float(abs(seed-reference)),
                         'attenuation_db_cm':float(solution['attenuation_db_cm']),
                         'residual':float(solution['residual']),'attempts':attempts})
        else:
            rows.append({'thickness_um':t,'status':'failed','attempts':attempts})
    return {'source_kind':'illustrative_planar_maxwell_control','physics_qualified':False,
            'publication_ready':False,'comsol_verified':False,'wavelength_um':wavelength,
            'illustrative_eps_metal':[metal.real,metal.imag],
            'single_interface_neff':[reference.real,reference.imag],
            'converged':sum(r['status']=='converged' for r in rows),
            'failed':sum(r['status']=='failed' for r in rows),'rows':rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    payload=(json.dumps(campaign(),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:
        f.write(payload)
    print(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'output':str(args.output)}))

if __name__=='__main__':
    main()
