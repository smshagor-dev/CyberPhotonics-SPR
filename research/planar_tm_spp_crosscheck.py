"""Independent planar TM SPP root solve; illustrative epsilon, NOT PCF FEM."""
import argparse
import json
from math import pi
from scipy.optimize import root


def solve(wavelength_um, analyte_ri, gold_epsilon):
    if wavelength_um <= 0 or analyte_ri <= 0 or gold_epsilon.imag <= 0:
        raise ValueError("passive interface requires positive wavelength, RI and Im(eps)")
    dielectric = analyte_ri**2
    k0 = 2*pi/wavelength_um
    analytic = k0 * (gold_epsilon*dielectric/(gold_epsilon+dielectric))**0.5
    if analytic.real < 0:
        analytic = -analytic

    def decay(beta, epsilon):
        q = (beta*beta - epsilon*k0*k0)**0.5
        return -q if q.real < 0 else q

    def residual(x):
        beta = complex(*x)
        f = decay(beta, dielectric)/dielectric + decay(beta, gold_epsilon)/gold_epsilon
        return [f.real, f.imag]

    # Numerical seeds depend only on the dielectric light line, not the closed form.
    seeds = [k0 * analyte_ri * complex(a, b) for a, b in
             ((1.02, 0.001), (1.05, 0.01), (1.10, 0.02))]
    candidates = [root(residual, [seed.real, seed.imag], tol=1e-12)
                  for seed in seeds]
    admissible = [(complex(*candidate.x), abs(complex(*residual(candidate.x))))
                  for candidate in candidates
                  if all(__import__("math").isfinite(v) for v in candidate.x)]
    admissible = [(beta, err) for beta, err in admissible
                  if beta.real > 0 and beta.imag > 0 and err < 1e-8]
    if len(admissible) != len(seeds):
        raise RuntimeError("independent TM root failed to converge")
    numerical, norm = min(admissible, key=lambda pair: pair[1])
    max_seed_disagreement = max(abs(beta - numerical) / abs(numerical)
                                for beta, _ in admissible)
    if max_seed_disagreement > 1e-8:
        raise RuntimeError("independent TM roots disagree")
    error = abs(numerical-analytic)/abs(analytic)
    return dict(wavelength_um=wavelength_um, analyte_ri=analyte_ri,
                epsilon_gold=[gold_epsilon.real, gold_epsilon.imag],
                beta_analytic_per_um=[analytic.real, analytic.imag],
                beta_numeric_per_um=[numerical.real, numerical.imag],
                relative_beta_error=error, tm_boundary_residual=norm,
                root_solver_success=bool(norm < 1e-10),
                independent_seed_count=len(seeds),
                max_seed_relative_disagreement=max_seed_disagreement,
                planar_crosscheck_passed=bool(error < 1e-8 and norm < 1e-8 and numerical.imag > 0),
                source_type="illustrative_planar_tm_crosscheck_not_pcf",
                physics_qualified=False, publication_ready=False, comsol_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [solve(wl, ri, complex(-24, 1.5))
               for wl in (0.75, 0.80, 0.85) for ri in (1.35, 1.36, 1.37)]
    data = dict(records=records, physics_qualified=False,
                publication_ready=False, comsol_verified=False,
                source_note="Constant illustrative Au epsilon -24+1.5j: not independently sourced dispersion or PCF physics.")
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(data, out, indent=2)
    print("planar points:", len(records), "passed:", sum(x["planar_crosscheck_passed"] for x in records))
    if not all(x["planar_crosscheck_passed"] for x in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
