"""Fail-closed geometry-disjoint training gate; does not certify Maxwell physics."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SHA_KEYS = ("topology_sha256", "material_dataset_sha256", "mesh_sha256",
            "solver_config_sha256", "raw_field_archive_sha256")
SOURCES = {"native_femwell", "native_ngsolve", "native_fenicsx"}
SPLITS = {"train", "validation", "test"}


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def validate(document):
    errors = []
    if not isinstance(document, dict):
        return {"valid": False, "errors": ["manifest must be an object"],
                "physics_qualified": False, "publication_ready": False, "comsol_verified": False}
    records = document.get("records")
    if not isinstance(records, list) or not records:
        errors.append("records must be a nonempty list")
        records = []
    partitions = defaultdict(set)
    identities = set()
    sources = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index}: not an object")
            continue
        split = record.get("split")
        if split not in SPLITS:
            errors.append(f"record {index}: invalid split")
            continue
        geometry = record.get("geometry_sha256")
        artifact = record.get("solver_artifact_sha256")
        source = record.get("solver_source")
        if not is_sha(geometry) or not is_sha(artifact):
            errors.append(f"record {index}: invalid SHA256")
            continue
        provenance = record.get("evidence_registry")
        if not isinstance(provenance, dict) or any(
            not is_sha(provenance.get(key)) for key in SHA_KEYS
        ):
            errors.append(f"record {index}: incomplete evidence registry")
        if source not in SOURCES:
            errors.append(f"record {index}: unsupported source")
            continue
        if record.get("physics_qualified") is not True or record.get("source_verified") is not True:
            errors.append(f"record {index}: unqualified or unverified native physics")
        for key in ("wavelength_um", "analyte_ri", "loss_db_per_cm"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                errors.append(f"record {index}: invalid {key}")
            elif key != "loss_db_per_cm" and value <= 0:
                errors.append(f"record {index}: nonpositive {key}")
            elif key == "loss_db_per_cm" and value < 0:
                errors.append(f"record {index}: negative loss")
        identity = (geometry, record.get("wavelength_um"), record.get("analyte_ri"), source)
        if identity in identities:
            errors.append(f"record {index}: duplicated native solver observation")
        identities.add(identity)
        partitions[geometry].add(split)
        sources.add(source)
    for geometry, split_set in partitions.items():
        if len(split_set) > 1:
            errors.append(f"geometry leakage across splits: {geometry}")
    if not partitions or set().union(*partitions.values()) != SPLITS:
        errors.append("train, validation and test splits required")
    if document.get("comsol_verified") is True or document.get("publication_ready") is True:
        errors.append("unsafe top-level readiness claim")
    return {"valid": not errors, "errors": errors, "geometry_count": len(partitions),
            "solver_sources": sorted(sources), "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        result = validate(json.loads(args.manifest.read_text()))
    except (OSError, ValueError) as exc:
        result = {"valid": False, "errors": [str(exc)], "physics_qualified": False,
                  "publication_ready": False, "comsol_verified": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 2)


if __name__ == "__main__":
    main()
