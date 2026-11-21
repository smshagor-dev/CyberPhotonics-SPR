"""Exploratory native FEM coefficient continuity; not physical core/SPP tracking."""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def fields(archive, ri, wl):
    with np.load(io.BytesIO(archive.read(f"raw_vector_eh_ri{ri}_wl{wl}.npz")), allow_pickle=False) as data:
        vectors = []
        for mode in range(2):
            e, h = data[f"mode_{mode}_E"], data[f"mode_{mode}_H"]
            if e.ndim != 1 or e.shape != h.shape or not np.iscomplexobj(e) or not np.iscomplexobj(h):
                raise ValueError("Invalid native E/H coefficients")
            if not np.isfinite(e).all() or not np.isfinite(h).all():
                raise ValueError("Nonfinite native coefficients")
            vector = np.concatenate((e, h))
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Zero or invalid native coefficients")
            vectors.append(vector / norm)
        return vectors


def match(left, right):
    similarities = np.array([[abs(np.vdot(a, b)) for b in right] for a in left])
    if similarities.shape != (2, 2) or not np.isfinite(similarities).all() or np.any(similarities > 1 + 1e-8):
        raise ValueError("Invalid coefficient similarity")
    rows, cols = linear_sum_assignment(-similarities)
    permutation = [int(cols[np.where(rows == i)[0][0]]) for i in range(2)]
    return {
        "permutation": permutation,
        "similarity_matrix": similarities.tolist(),
        "min_matched_similarity": float(min(similarities[rows, cols])),
        "assignment_margin": float(np.sum(similarities[rows, cols]) - np.sum(similarities[rows, 1 - cols])),
    }


def audit(path):
    import hashlib
    path = Path(path)
    with path.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "schema": "native_coefficient_continuity_v1",
        "source_sha256": source_sha256,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "physical_mode_identity_verified": False,
        "metric": "unweighted E/H coefficient cosine; no FEM mass matrix or Poynting flux",
        "edges": [],
    }
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate archive members")
        grid = {(i, j): fields(archive, i, j) for i in range(3) for j in range(3)}
        for i in range(3):
            for j in range(2):
                result["edges"].append({"axis": "wavelength", "from": [i, j], "to": [i, j + 1], **match(grid[i, j], grid[i, j + 1])})
        for j in range(3):
            for i in range(2):
                result["edges"].append({"axis": "analyte_ri", "from": [i, j], "to": [i + 1, j], **match(grid[i, j], grid[i + 1, j])})
    result["min_matched_similarity"] = min(edge["min_matched_similarity"] for edge in result["edges"])
    result["nonidentity_permutations"] = sum(edge["permutation"] != [0, 1] for edge in result["edges"])
    return result


if __name__ == "__main__":
    report = audit(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"edges": len(report["edges"]), "nonidentity_permutations": report["nonidentity_permutations"]}))
