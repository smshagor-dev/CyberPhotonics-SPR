"""Executed domain-truncation sensitivity for the *unqualified* planar 1D TM benchmark.

No PCF topology, PML, independently measured Au data, or COMSOL evidence.
"""
import argparse
import json
import math
from pathlib import Path

from research.planar_tm_fd_eigenmode import solve


def campaign(dx_um=0.02, domains_um=(1.6, 2.4, 3.2, 4.0)):
    if len(domains_um) < 3 or any(not math.isfinite(d) or d <= 0 for d in domains_um):
        raise ValueError("at least three positive finite domains required")
    if any(b <= a for a, b in zip(domains_um, domains_um[1:])):
        raise ValueError("domain sizes must increase strictly")
    rows = [solve(dx_um=dx_um, half_domain_um=domain) for domain in domains_um]
    for row in rows:
        if row["relative_matrix_residual"] >= 1e-8:
            raise RuntimeError("eigenpair residual too large")
    final = complex(*rows[-1]["beta_numeric_per_um"])
    penultimate = complex(*rows[-2]["beta_numeric_per_um"])
    drift = abs(final - penultimate) / abs(final)
    return {
        "records": rows,
        "last_pair_relative_beta_drift": float(drift),
        "domain_convergence_diagnostic_only": bool(drift < 1e-5),
        "method": "planar_1d_tm_dirichlet_domain_truncation",
        "metal_source": "illustrative_constant_complex_epsilon",
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "limitation": "Domain stability of illustrative planar TM benchmark does not establish PCF PML or Au convergence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = campaign()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"last_pair_relative_beta_drift": payload["last_pair_relative_beta_drift"],
                      "physics_qualified": False}))
    if not payload["domain_convergence_diagnostic_only"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
