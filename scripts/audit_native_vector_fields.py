"""Fail-closed inspection of registered native FEM E/H coefficient arrays.

Integrity-only evidence: never qualifies physical modes, COMSOL or publication.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def audit_vector_fields(path, max_uncompressed_bytes=1_000_000_000):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or any(
            name.startswith("/") or ".." in Path(name).parts or "\\" in name
            for name in names
        ):
            raise ValueError("unsafe or duplicate archive paths")
        if sum(item.file_size for item in entries) > max_uncompressed_bytes:
            raise ValueError("oversized native archive")
        readiness = json.loads(archive.read("GRID_READINESS.json"))
        protocol = json.loads(archive.read("experiment_protocol.json"))
        hashes = readiness["artifacts_sha256"]
        if set(hashes) != set(names) - {"GRID_READINESS.json"}:
            raise ValueError("unregistered native artifact")
        dimensions = (len(protocol["analyte_ris"]), len(protocol["wavelengths_um"]), protocol["num_modes"])
        if not all(type(v) is int and 0 < v <= 100 for v in dimensions):
            raise ValueError("invalid grid dimensions")
        count = 0
        lengths = set()
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                name = f"raw_vector_eh_ri{i}_wl{j}.npz"
                if name not in hashes:
                    raise ValueError(f"missing native fields: {name}")
                payload = archive.read(name)
                if hashlib.sha256(payload).hexdigest() != hashes[name]:
                    raise ValueError(f"checksum mismatch: {name}")
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    expected = {f"mode_{k}_{component}" for k in range(dimensions[2]) for component in ("E", "H")}
                    if set(data.files) != expected:
                        raise ValueError(f"incomplete E/H fields: {name}")
                    for key in expected:
                        field = data[key]
                        if field.ndim != 1 or not field.size or field.dtype.kind != "c":
                            raise ValueError(f"invalid vector shape/type: {name}:{key}")
                        if not np.all(np.isfinite(field)):
                            raise ValueError(f"nonfinite native field: {name}:{key}")
                        scale = float(np.max(np.abs(field)))
                        if not np.isfinite(scale) or scale <= 0:
                            raise ValueError(f"zero/invalid native field: {name}:{key}")
                        norm = scale * float(np.linalg.norm(field / scale))
                        if not np.isfinite(norm) or norm <= 0:
                            raise ValueError(f"invalid norm: {name}:{key}")
                        lengths.add(field.size)
                        count += 1
        if len(lengths) != 1:
            raise ValueError("inconsistent coefficient lengths")
        return {
            "native_field_arrays_verified": count,
            "coefficient_length": next(iter(lengths)),
            "physics_qualified": False,
            "publication_ready": False,
            "comsol_verified": False,
        }


if __name__ == "__main__":
    print(json.dumps(audit_vector_fields(sys.argv[1]), indent=2))
