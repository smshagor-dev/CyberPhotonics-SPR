"""Diagnostic only: same-mesh complex E/H overlap; never physics qualification."""
import numpy as np
from itertools import permutations

def overlap(e1,h1,e2,h2,weights):
    fields=[np.asarray(x,dtype=np.complex128) for x in (e1,h1,e2,h2)]
    w=np.asarray(weights,dtype=float)
    if any(x.ndim!=2 or x.shape[1]!=3 for x in fields):
        raise ValueError("expected vector fields (N,3)")
    n=len(fields[0])
    if not n or any(len(x)!=n for x in fields) or w.shape!=(n,):
        raise ValueError("same mesh and quadrature required")
    if not np.all(np.isfinite(w)) or np.any(w<=0) or any(not np.all(np.isfinite(x)) for x in fields):
        raise ValueError("invalid field/weight")
    def score(a,b):
        ab=np.sum(w[:,None]*np.conj(a)*b)
        aa=np.sum(w[:,None]*abs(a)**2)
        bb=np.sum(w[:,None]*abs(b)**2)
        if aa<=0 or bb<=0:
            raise ValueError("zero norm")
        return float(np.clip(abs(ab)**2/(aa*bb),0,1))
    return (score(fields[0],fields[2])+score(fields[1],fields[3]))/2

def match_modes(reference,candidates,weights,minimum=0.8,margin=0.05):
    n=len(reference)
    if n<1 or n>8 or n!=len(candidates) or not (0<=minimum<=1 and 0<=margin<=1):
        raise ValueError("invalid mode count or thresholds")
    scores=np.array([[overlap(*r,*c,weights) for c in candidates] for r in reference])
    for row in scores:
        s=sorted(row,reverse=True)
        if s[0]<minimum or (n>1 and s[0]-s[1]<margin):
            raise ValueError("ambiguous or low-overlap modes")
    p=max(permutations(range(n)),key=lambda p:sum(scores[i,j] for i,j in enumerate(p)))
    if any(scores[i,j]<minimum for i,j in enumerate(p)):
        raise ValueError("no unique high-overlap assignment")
    return {"assignment":list(p),"overlaps":[float(scores[i,j]) for i,j in enumerate(p)],
            "physics_qualified":False,"publication_ready":False,"comsol_verified":False}
