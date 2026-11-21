"""Exploratory path consistency of native coefficient-space assignments; not physical mode tracking."""
import json
import sys
from pathlib import Path

def compose(p, q):
    if sorted(p) != [0, 1] or sorted(q) != [0, 1]:
        raise ValueError("Invalid 2-mode permutation")
    return [q[p[0]], q[p[1]]]

def plaquettes(report):
    edges = {}
    for edge in report["edges"]:
        key = (tuple(edge["from"]), tuple(edge["to"]))
        if key in edges:
            raise ValueError("Duplicate directed edge")
        edges[key] = edge
    output = []
    for i in range(2):
        for j in range(2):
            a, b, c, d = (i,j), (i,j+1), (i+1,j), (i+1,j+1)
            try:
                ab, bd, ac, cd = (edges[(a,b)],edges[(b,d)],edges[(a,c)],edges[(c,d)])
            except KeyError as exc:
                raise ValueError("Missing plaquette edge") from exc
            top = compose(ab["permutation"],bd["permutation"])
            bottom = compose(ac["permutation"],cd["permutation"])
            output.append({"origin":[i,j],"wavelength_then_ri":top,
                           "ri_then_wavelength":bottom,"consistent":top==bottom,
                           "minimum_edge_margin":min(float(e["assignment_margin"]) for e in (ab,bd,ac,cd))})
    return output

def evaluate(report):
    squares = plaquettes(report)
    return {"schema":"exploratory_native_plaquette_v1",
            "source_sha256":report["source_sha256"],"plaquettes":squares,
            "consistent_count":sum(s["consistent"] for s in squares),
            "inconsistent_count":sum(not s["consistent"] for s in squares),
            "physics_qualified":False,"publication_ready":False,
            "comsol_verified":False,"physical_mode_identity_verified":False,
            "caveat":"Unweighted coefficient-space assignments only; no FEM mass matrix, PML, or physical core/SPP identification."}

if __name__ == "__main__":
    source = json.loads(Path(sys.argv[1]).read_text())
    output = evaluate(source)
    Path(sys.argv[2]).write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"consistent":output["consistent_count"],"inconsistent":output["inconsistent_count"]}))
