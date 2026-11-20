"""Fail-closed overlap assignment graph audit. Integrity is NOT physical mode identity."""
import math
from mode_plaquette_audit import audit_mode_plaquettes

def audit_overlap_graph(edges, *, n_ri, n_wavelength, n_modes):
    if min(n_ri, n_wavelength, n_modes) < 1:
        raise ValueError("invalid grid")
    expected = {("wavelength", i, j-1, j) for i in range(n_ri) for j in range(1,n_wavelength)}
    expected |= {("analyte_ri", j, i-1, i) for j in range(n_wavelength) for i in range(1,n_ri)}
    grouped = {}
    ambiguous = 0
    for e in edges:
        axis = e.get("varying")
        if axis == "wavelength":
            key = (axis,e.get("ri_index"),e.get("previous_wl_index"),e.get("current_wl_index"))
        elif axis == "analyte_ri":
            key = (axis,e.get("wavelength_index"),e.get("previous_ri_index"),e.get("current_ri_index"))
        else:
            raise ValueError("unknown axis")
        if key not in expected:
            raise ValueError("unexpected/nonadjacent graph edge")
        a,b=e.get("previous_mode_index"),e.get("current_mode_index")
        if type(a) is not int or type(b) is not int or not (0<=a<n_modes and 0<=b<n_modes):
            raise ValueError("invalid mode indices")
        selected,row,col=(e.get(k) for k in ("vector_eh_poynting_overlap_abs","unselected_overlap_abs","reverse_competing_overlap_abs"))
        if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1+1e-9 for v in (selected,row,col)):
            raise ValueError("invalid overlap")
        fwd,rev=selected-row,selected-col
        for name,value in (("forward_overlap_margin",fwd),("reverse_overlap_margin",rev),("bidirectional_min_overlap_margin",min(fwd,rev))):
            actual=e.get(name)
            if type(actual) not in (float,int) or not math.isfinite(actual) or abs(actual-value)>1e-10:
                raise ValueError("incorrect overlap margin")
        flag=selected<=row or selected<=col
        if e.get("assignment_ambiguous_nonpositive_margin") is not flag:
            raise ValueError("incorrect ambiguity flag")
        if e.get("mode_identity_physically_verified") is not False:
            raise ValueError("unqualified physical mode identity")
        ambiguous+=flag
        grouped.setdefault(key,[]).append((a,b))
    if set(grouped)!=expected:
        raise ValueError("missing adjacency")
    for pairs in grouped.values():
        if len(pairs)!=n_modes or {a for a,b in pairs}!=set(range(n_modes)) or {b for a,b in pairs}!=set(range(n_modes)):
            raise ValueError("non-bijective mode assignment")
    plaquettes = audit_mode_plaquettes(edges, n_ri=n_ri, n_wavelength=n_wavelength, n_modes=n_modes)
    return {"plaquette_consistency":plaquettes, "adjacency_pairs":len(expected),"matched_edges":len(edges),"ambiguous_edges":ambiguous,
            "graph_integrity_verified":True,"mode_identity_physically_verified":False,"physics_qualified":False}
