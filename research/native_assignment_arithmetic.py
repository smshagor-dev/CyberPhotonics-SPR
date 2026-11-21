"""Fail-closed independent arithmetic verification of exploratory two-mode assignment edges.

This is NOT FEM-weighted tracking or qualified Maxwell evidence.
"""
import json
import math
import sys
from pathlib import Path

def verify(report):
    digest=report.get("source_sha256","")
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid source digest")
    edges=report.get("edges",[])
    if len(edges)!=12: raise ValueError("Expected 12 edges")
    seen=set(); assignments={}
    for e in edges:
        a,b=e["from"],e["to"]
        if not (isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==2 and
                all(type(v)==int and 0<=v<=2 for v in a+b)):
            raise ValueError("Invalid grid coordinate")
        delta=(b[0]-a[0],b[1]-a[1])
        if delta not in ((1,0),(0,1)): raise ValueError("Invalid neighbor")
        if e.get("axis")!=("analyte_ri" if delta[0] else "wavelength"):
            raise ValueError("Invalid axis")
        key=(tuple(a),tuple(b))
        if key in seen: raise ValueError("Duplicate edge")
        seen.add(key)
        s=e["similarity"]
        if not (isinstance(s,list) and len(s)==2 and all(isinstance(r,list) and len(r)==2 for r in s)):
            raise ValueError("Invalid matrix shape")
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for r in s for v in r):
            raise ValueError("Invalid similarity")
        same=s[0][0]+s[1][1]; swap=s[0][1]+s[1][0]
        margin=abs(same-swap)
        if margin<=1e-12: raise ValueError("Ambiguous assignment")
        permutation=[0,1] if same>swap else [1,0]
        if e["permutation"]!=permutation: raise ValueError("Permutation mismatch")
        expected={"assignment_margin":margin,
                  "matched_min":min(s[0][permutation[0]],s[1][permutation[1]]),
                  "matched_mean":(s[0][permutation[0]]+s[1][permutation[1]])/2}
        for field,value in expected.items():
            v=e.get(field)
            if type(v) not in (int,float) or not math.isfinite(v) or not math.isclose(v,value,rel_tol=1e-9,abs_tol=1e-10):
                raise ValueError(field+" mismatch")
        assignments[key]=permutation
    expected={((i,j),(i,j+1)) for i in range(3) for j in range(2)}
    expected|={((i,j),(i+1,j)) for i in range(2) for j in range(3)}
    if seen!=expected: raise ValueError("Incomplete grid")
    def compose(p,q): return [q[p[0]],q[p[1]]]
    squares=[]
    for i in range(2):
        for j in range(2):
            a=(i,j);b=(i,j+1);c=(i+1,j);d=(i+1,j+1)
            squares.append(compose(assignments[a,b],assignments[b,d])==
                           compose(assignments[a,c],assignments[c,d]))
    return {"schema":"native_assignment_arithmetic_v1","source_sha256":digest,
            "edges_checked":12,"plaquettes_checked":4,"consistent_plaquettes":sum(squares),
            "physics_qualified":False,"publication_ready":False,"comsol_verified":False,
            "physical_mode_identity_verified":False,
            "caveat":"Unweighted coefficient-space only; no PML, FEM mass matrix or physical core/SPP identification."}

if __name__=="__main__":
    output=verify(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
