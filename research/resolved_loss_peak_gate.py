"""Fail-closed numerical loss peak postprocessor. NOT physics qualification."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Peak:
    resonance_um: float
    fwhm_um: float
    peak_excess_db_cm: float

def analyze(wavelength_um, loss_db_cm, *, max_step_um, min_samples_per_width=5):
    x=list(map(float,wavelength_um)); y=list(map(float,loss_db_cm))
    if len(x)!=len(y) or len(x)<7 or not all(map(math.isfinite,x+y)):
        raise ValueError("invalid spectrum")
    if min(x)<=0 or min(y)<0 or not math.isfinite(max_step_um) or max_step_um<=0:
        raise ValueError("invalid physical units")
    if not math.isfinite(min_samples_per_width) or min_samples_per_width<3:
        raise ValueError("invalid resolution")
    steps=[b-a for a,b in zip(x,x[1:])]
    if min(steps)<=0 or max(steps)>max_step_um*(1+1e-12):
        raise ValueError("nonmonotone or insufficiently fine wavelength grid")
    span=x[-1]-x[0]
    z=[v-(y[0]+(y[-1]-y[0])*(w-x[0])/span) for w,v in zip(x,y)]
    k=max(range(len(z)),key=z.__getitem__)
    if k==0 or k==len(z)-1 or z[k]<=0:
        raise ValueError("no positive interior peak")
    if z[k]<=z[k-1] or z[k]<=z[k+1]:
        raise ValueError("flat or non-isolated peak")
    half=z[k]/2
    left=next((i for i in range(k-1,-1,-1) if z[i]<=half),None)
    right=next((i for i in range(k,len(z)-1) if z[i+1]<=half),None)
    if left is None or right is None:
        raise ValueError("half-maximum not bracketed")
    def crossing(i):
        return x[i]+(half-z[i])*(x[i+1]-x[i])/(z[i+1]-z[i])
    width=crossing(right)-crossing(left)
    if width<=0 or width/max(steps)<min_samples_per_width:
        raise ValueError("unresolved FWHM")
    for i in range(1,len(z)-1):
        if (i<left or i>right+1) and z[i]>z[i-1] and z[i]>=z[i+1] and z[i]>=half:
            raise ValueError("ambiguous multiple peaks")
    # Three-point quadratic vertex on the actual (possibly nonuniform) grid.
    # This is sub-grid interpolation, not a claim of spectral convergence.
    l=x[k-1]-x[k]; r=x[k+1]-x[k]
    sl=(z[k-1]-z[k])/l; sr=(z[k+1]-z[k])/r
    curvature=(sr-sl)/(r-l)
    if not math.isfinite(curvature) or curvature>=0:
        raise ValueError("unresolved peak curvature")
    slope=sl-curvature*l
    offset=-slope/(2*curvature)
    if not math.isfinite(offset) or not l<=offset<=r:
        raise ValueError("peak vertex outside adjacent samples")
    return Peak(x[k]+offset,width,z[k]-slope*slope/(4*curvature))

def sensitivity_nm_per_riu(a,b,ri_a,ri_b):
    if not all(map(math.isfinite,(ri_a,ri_b))) or ri_a==ri_b:
        raise ValueError("invalid RI pair")
    return 1000*(b.resonance_um-a.resonance_um)/(ri_b-ri_a)
