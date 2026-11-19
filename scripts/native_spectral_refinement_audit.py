"""Software-only spectral refinement audit; never implies Maxwell convergence."""
import math

def peak_metrics(x, y):
    if len(x) != len(y) or len(x) < 7:
        raise ValueError("at least seven paired samples required")
    x, y = list(map(float,x)), list(map(float,y))
    if not all(map(math.isfinite,x+y)) or any(v < 0 for v in y):
        raise ValueError("nonfinite or negative sample")
    if any(b <= a for a,b in zip(x,x[1:])):
        raise ValueError("wavelength order")
    peaks = [i for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
    if len(peaks)!=1:
        raise ValueError("ambiguous peak")
    p=peaks[0]
    baseline=max(min(y[:p+1]),min(y[p:]))
    height=y[p]-baseline
    if height<=0:
        raise ValueError("no prominence")
    half=baseline+height/2
    def cross(pairs):
        for a,b in pairs:
            if (y[a]-half)*(y[b]-half)<=0 and y[a]!=y[b]:
                return x[a]+(half-y[a])*(x[b]-x[a])/(y[b]-y[a])
        raise ValueError("half-prominence not bracketed")
    left=cross((i,i+1) for i in range(p-1,-1,-1))
    right=cross((i,i+1) for i in range(p,len(x)-1))
    return {"peak_nm":x[p],"fwhm_nm":right-left,"prominence":height}

def compare_refinement(coarse_x,coarse_y,fine_x,fine_y,*,peak_tol=0.05,width_tol=0.05):
    """Requires independent solver provenance and fixed-geometry identity upstream."""
    if not (0<peak_tol<1 and 0<width_tol<1):
        raise ValueError("invalid tolerance")
    coarse=peak_metrics(coarse_x,coarse_y)
    fine=peak_metrics(fine_x,fine_y)
    if fine_x[0]!=coarse_x[0] or fine_x[-1]!=coarse_x[-1]:
        raise ValueError("different spectral domains")
    if len(fine_x)<=len(coarse_x) or not set(coarse_x).issubset(set(fine_x)):
        raise ValueError("not a strict nested refinement")
    shift=abs(fine["peak_nm"]-coarse["peak_nm"])
    width=abs(fine["fwhm_nm"]-coarse["fwhm_nm"])/fine["fwhm_nm"]
    return {"coarse":coarse,"fine":fine,"peak_shift_nm":shift,
            "relative_width_change":width,
            "spectral_grid_converged":shift<=peak_tol*fine["fwhm_nm"] and width<=width_tol,
            "physics_qualified":False,"publication_ready":False,"comsol_verified":False}
