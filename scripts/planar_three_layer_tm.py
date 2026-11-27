"""Exploratory three-layer planar TM Maxwell film reference; NOT PCF or measured Au."""
import argparse
import json
import numpy as np
from scipy.optimize import root

def decay(beta, eps, k0):
    q = np.sqrt(beta*beta-k0*k0*eps+0j)
    return -q if q.real < 0 else q

def dispersion(neff, wavelength_um, thickness_um, eps_left, eps_film, eps_right):
    """Bound-film TM determinant for exp(i beta z - i omega t).

    Y_j=epsilon_j/q_j; q_j=sqrt(beta^2-k0^2 epsilon_j).
    D=(Y0+Y1)(Y1+Y2)+(Y0-Y1)(Y1-Y2)exp(-2q1*t).
    """
    if not np.isfinite(wavelength_um) or not np.isfinite(thickness_um) or wavelength_um <= 0 or thickness_um <= 0:
        raise ValueError("positive finite wavelength and thickness required")
    k0=2*np.pi/wavelength_um
    beta=k0*neff
    q=[decay(beta,e,k0) for e in (eps_left,eps_film,eps_right)]
    if min(abs(x) for x in q)<1e-14:
        raise ValueError("degenerate transverse decay")
    y0,y1,y2=[e/qi for e,qi in zip((eps_left,eps_film,eps_right),q)]
    return ((y0+y1)*(y1+y2)+(y0-y1)*(y1-y2)*np.exp(-2*q[1]*thickness_um))/(y1*y1)

def solve(wavelength_um, thickness_um, eps_left, eps_film, eps_right, guess):
    def f(v):
        z=dispersion(complex(*v),wavelength_um,thickness_um,eps_left,eps_film,eps_right)
        return [z.real,z.imag]
    result=root(f,[guess.real,guess.imag],tol=1e-11)
    n=complex(*result.x)
    residual=abs(dispersion(n,wavelength_um,thickness_um,eps_left,eps_film,eps_right))
    if not np.isfinite(residual) or residual>1e-8 or n.imag<0:
        raise RuntimeError(f"unqualified root residual={residual}, n={n}, solver={result.message}")
    return dict(neff_real=n.real,neff_imag=n.imag,residual=residual,
                attenuation_db_cm=(20/np.log(10))*(2*np.pi/wavelength_um)*n.imag*1e4,
                physics_qualified=False,publication_ready=False,comsol_verified=False,
                source_kind="illustrative_planar_three_layer_maxwell")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wavelength-um",type=float,default=.8)
    p.add_argument("--thickness-um",type=float,default=.05)
    args=p.parse_args()
    metal=-24+1.5j
    eps=1.77
    guess=np.sqrt(eps*metal/(eps+metal))
    out=solve(args.wavelength_um,args.thickness_um,eps,metal,eps,guess)
    out.update(wavelength_um=args.wavelength_um,thickness_um=args.thickness_um,
               illustrative_eps_metal=[metal.real,metal.imag])
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
