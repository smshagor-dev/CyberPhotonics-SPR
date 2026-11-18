"""Interpolation sensitivity diagnostic, NOT verified optical physics."""
import math

def interpolation_consistency(table, wavelength_nm):
    points = table["points"]
    if not isinstance(wavelength_nm, (int, float)) or not math.isfinite(wavelength_nm):
        raise ValueError("finite wavelength required")
    if wavelength_nm < points[0][0] or wavelength_nm > points[-1][0]:
        raise ValueError("no spectral extrapolation")
    for j in range(len(points)-1):
        a,n0,k0=points[j]
        b,n1,k1=points[j+1]
        if a <= wavelength_nm <= b:
            t=(wavelength_nm-a)/(b-a)
            eps_nk=complex(n0+t*(n1-n0),k0+t*(k1-k0))**2
            e0,e1=complex(n0,k0)**2,complex(n1,k1)**2
            eps_linear=(1-t)*e0+t*e1
            rel=abs(eps_nk-eps_linear)/max(abs(eps_nk),abs(eps_linear),1e-30)
            return {"wavelength_nm":float(wavelength_nm),
                    "epsilon_from_nk":eps_nk,"epsilon_linear":eps_linear,
                    "relative_discrepancy":rel,
                    "physics_qualified":False,"publication_ready":False,
                    "comsol_verified":False}
    raise ValueError("unbracketed wavelength")

def audit_interpolation(table,samples_per_interval=3):
    """Difference between interpolation choices is NOT an error bound."""
    if (not isinstance(samples_per_interval,int) or isinstance(samples_per_interval,bool)
            or not 1 <= samples_per_interval <= 100):
        raise ValueError("samples_per_interval must be integer 1..100")
    points=table["points"]
    if len(points)<3:
        raise ValueError("at least three nodes required")
    results=[]
    for j in range(len(points)-1):
        a,b=points[j][0],points[j+1][0]
        for i in range(1,samples_per_interval+1):
            w=a+(b-a)*i/(samples_per_interval+1)
            results.append(interpolation_consistency(table,w))
    return {"max_relative_discrepancy":max(r["relative_discrepancy"] for r in results),
            "samples":results,"source_sha256":table["sha256"],
            "source_verified":False,"physics_qualified":False,
            "publication_ready":False,"comsol_verified":False}
