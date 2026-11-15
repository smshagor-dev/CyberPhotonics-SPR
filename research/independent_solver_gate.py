"""Independent native Maxwell solver pair audit. Diagnostic only; never publication evidence."""
import json
import math
import re
import sys
from pathlib import Path

KEYS = ("solver", "implementation", "geometry_sha256", "au_data_sha256",
        "field_sha256", "run_sha256", "mode_id", "wavelength_nm",
        "analyte_ri", "neff_real", "neff_imag", "overlap_verified")
HASH = re.compile(r"^[0-9a-f]{64}$")


def compare(a, b, neff_tolerance=1e-3, loss_tolerance=0.05):
    result = {"comparison_pass": False, "publication_qualified": False,
              "comsol_verified": False, "reasons": [], "metrics": {}}
    errors = result["reasons"]
    for name, row in (("a", a), ("b", b)):
        if not isinstance(row, dict):
            errors.append(f"{name}: invalid record")
            return result
        missing = set(KEYS) - row.keys()
        if missing:
            errors.append(f"{name}: missing {sorted(missing)}")
            return result
        for key in ("geometry_sha256", "au_data_sha256", "field_sha256", "run_sha256"):
            if not isinstance(row[key], str) or not HASH.fullmatch(row[key]):
                errors.append(f"{name}: invalid {key}")
        for key in ("solver", "implementation", "mode_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                errors.append(f"{name}: invalid {key}")
        for key in ("wavelength_nm", "analyte_ri", "neff_real", "neff_imag"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{name}: invalid {key}")
        if row["overlap_verified"] is not True:
            errors.append(f"{name}: unverified E/H overlap")
        if str(row["solver"]).lower() == "comsol":
            errors.append(f"{name}: COMSOL needs separate native provenance audit")
    if errors:
        return result
    if not all(math.isfinite(x) and x > 0 for x in (neff_tolerance, loss_tolerance)):
        errors.append("invalid tolerance")
        return result
    if a["solver"] == b["solver"] or a["implementation"] == b["implementation"]:
        errors.append("not independent implementations")
    if a["run_sha256"] == b["run_sha256"] or a["field_sha256"] == b["field_sha256"]:
        errors.append("reused run or field archive")
    for key in ("geometry_sha256", "au_data_sha256", "mode_id", "wavelength_nm", "analyte_ri"):
        if a[key] != b[key]:
            errors.append(f"mismatched {key}")
    if a["wavelength_nm"] <= 0 or not 1 <= a["analyte_ri"] <= 2:
        errors.append("invalid sampling coordinates")
    if min(a["neff_imag"], b["neff_imag"]) < 0:
        errors.append("negative passive attenuation for exp(+i beta z - i omega t)")
    if errors:
        return result
    za = complex(a["neff_real"], a["neff_imag"])
    zb = complex(b["neff_real"], b["neff_imag"])
    dn = abs(za - zb) / max(abs(za), abs(zb), 1e-15)
    dl = abs(za.imag - zb.imag) / max(abs(za.imag), abs(zb.imag), 1e-15)
    result["metrics"] = {"relative_complex_neff_difference": dn,
                         "relative_attenuation_difference": dl}
    if dn > neff_tolerance:
        errors.append("complex neff disagreement")
    if dl > loss_tolerance:
        errors.append("attenuation disagreement")
    result["comparison_pass"] = not errors
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: independent_solver_gate.py A.json B.json")
    try:
        outcome = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:]))
    except (OSError, ValueError) as exc:
        outcome = {"comparison_pass": False, "publication_qualified": False,
                   "comsol_verified": False, "reasons": [str(exc)], "metrics": {}}
    print(json.dumps(outcome, indent=2, sort_keys=True))
    raise SystemExit(0 if outcome["comparison_pass"] else 1)
