"""Check modal-assignment path consistency on RI/wavelength plaquettes.

This is a discrete permutation diagnostic, NOT physical core/SPP identification.
"""

def audit_mode_plaquettes(edges, *, n_ri, n_wavelength, n_modes):
    if any(type(x) is not int or x < 1 for x in (n_ri, n_wavelength, n_modes)):
        raise ValueError("invalid grid dimensions")
    maps = {}
    for e in edges:
        axis = e.get("varying")
        if axis == "wavelength":
            key = (axis, e.get("ri_index"), e.get("previous_wl_index"), e.get("current_wl_index"))
        elif axis == "analyte_ri":
            key = (axis, e.get("wavelength_index"), e.get("previous_ri_index"), e.get("current_ri_index"))
        else:
            raise ValueError("unknown edge axis")
        src, dst = e.get("previous_mode_index"), e.get("current_mode_index")
        if type(src) is not int or type(dst) is not int or not 0 <= src < n_modes or not 0 <= dst < n_modes:
            raise ValueError("invalid mode index")
        mapping = maps.setdefault(key, {})
        if src in mapping:
            raise ValueError("duplicate source mode")
        mapping[src] = dst
    expected = {("wavelength", i, j, j+1) for i in range(n_ri) for j in range(n_wavelength-1)}
    expected |= {("analyte_ri", j, i, i+1) for j in range(n_wavelength) for i in range(n_ri-1)}
    if set(maps) != expected:
        raise ValueError("missing or unexpected adjacency")
    for mapping in maps.values():
        if set(mapping) != set(range(n_modes)) or set(mapping.values()) != set(range(n_modes)):
            raise ValueError("non-bijective assignment")
    defects = []
    for ri in range(n_ri-1):
        for wl in range(n_wavelength-1):
            bottom = maps["wavelength",ri,wl,wl+1]
            right = maps["analyte_ri",wl+1,ri,ri+1]
            left = maps["analyte_ri",wl,ri,ri+1]
            top = maps["wavelength",ri+1,wl,wl+1]
            mismatches = [mode for mode in range(n_modes) if right[bottom[mode]] != top[left[mode]]]
            if mismatches:
                defects.append({"ri_index":ri,"wavelength_index":wl,"mismatched_start_modes":mismatches})
    return {"plaquettes_checked":(n_ri-1)*(n_wavelength-1),
            "holonomy_defects":defects,
            "path_independent_assignment":not defects and (n_ri-1)*(n_wavelength-1)>0,
            "nondegenerate_grid":n_ri>1 and n_wavelength>1,\n            "mode_identity_physically_verified":False,"physics_qualified":False}
