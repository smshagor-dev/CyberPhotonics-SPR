"""Fail-closed attenuation calculation for source-labelled open-source FEM eigenmodes.

This does not establish mode identity, PML convergence, or physical qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def attenuation_db_per_cm(wavelength_um, neff_real, neff_imag):
    """Power attenuation, exp(+i beta z - i omega t), lambda in micrometres."""
    for name, value in (("wavelength_um", wavelength_um),
                        ("neff_real", neff_real), ("neff_imag", neff_imag)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number, not bool")
    if wavelength_um <= 0 or neff_imag < 0:
        raise ValueError("positive wavelength and nonnegative passive loss required")
    loss = (20.0 / math.log(10.0)) * (2.0 * math.pi / wavelength_um) * neff_imag * 10000.0
    if not math.isfinite(loss):
        raise ValueError("attenuation overflow: unrepresentable finite dB/cm required")
    return loss


def audit(payload):
    if not isinstance(payload, dict) or payload.get("source_type") != "open_source_fem_native":
        raise ValueError("explicit open_source_fem_native source_type required")
    if (payload.get("field_time_convention") != "exp(-i omega t)"
            or payload.get("propagation_convention") != "exp(+i beta z)"):
        raise ValueError("explicit supported field and propagation conventions required")
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError("nonempty records required")
    outputs = []
    for index, rec in enumerate(payload["records"]):
        if not isinstance(rec, dict):
            raise ValueError(f"record {index} must be an object")
        loss = attenuation_db_per_cm(rec.get("wavelength_um"),
                                     rec.get("neff_real"), rec.get("neff_imag"))
        outputs.append({"index": index, "attenuation_db_per_cm": loss})
    return {"source_type": "open_source_fem_native", "software_audit_only": True,
            "physics_qualified": False, "publication_ready": False,
            "comsol_verified": False, "record_count": len(outputs), "results": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = audit(json.loads(raw))
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite immutable audit output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
