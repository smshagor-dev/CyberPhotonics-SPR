# Stage 12 — COMSOL spectral data-quality and source-integrity controls

Stage 11 established that a nominal spectrum can be returned while an analyte RI parameter is disconnected. Stage 12 closes a lower-level extraction and reproducibility gap: a COMSOL evaluation containing multiple eigenmodes could previously be flattened into one artificial spectral trace, while a short, boundary-truncated or physically nonsensical spectrum could be reported as a normal solver success.

## Production correctness rules

These checks run on every COMSOL geometry solve (not just the preflight pilot). They also run again when accepting frozen Stage 7 sweeps or restoring Stage 8 checkpointed batches:

- Refuse genuine multi-column eigenmode or multi-solution matrices instead of flattening them as a single wavelength/loss spectrum. One-dimensional series and true singleton-column/row series are supported.
- Refuse complex-valued optical exports with nonzero imaginary parts, nonfinite values, non-monotonic wavelength samples and invalid unit conversion factors. Preserve explicit expected-wavelength range checks.
- Require **at least 16 matched finite samples** for a solver row to count as an optical spectrum; preserve the one-sample failure in the frozen denominator.
- Refuse confinement losses that are materially negative, a flat trace with no resolvable peak contrast, a resonance too close to the sweep boundaries, a nonfinite/negative FWHM, or a peak width narrower than three median grid steps.
- Preserve wavelength and loss values using sufficient precision for binary64 round-trip rather than rounding everything to six decimal places. Derived resonance and FWHM are independently rechecked against the raw arrays during checkpoint restoration.
- Reject `NaN` or infinite geometry parameters before assigning them to the COMSOL model. Range comparisons alone previously did not reject `NaN`.

Each invalid geometry solve is marked `failed: <reason>` and subsequent planned geometry calls proceed, preserving the original sample ID and failure accounting. This cannot repair an invalid model, but avoids silently counting a malformed spectrum as a successful physical result.

## Verification boundary

Run regression tests locally:

```bash
pytest -q -W error tests/test_solver_spectrum_quality.py tests/test_comsol_sweep.py tests/test_comsol_campaign.py tests/test_resumable_campaign.py tests/test_comsol_ri_response.py
```

The new tests create a **fake MPh client and synthetic Lorentzian spectra** and deliberately inject multi-mode columns, complex data, negative confinement loss, missing resonance contrast, boundary peaks, narrow under-resolved peaks, nonfinite geometry and a defective first solve followed by a successful next solve. This proves only the guards' software behavior, **NOT that the PCF-SPR model or sensor physics is valid**.

The threshold of three median wavelength steps is a conservative preliminary engineering data-quality check; it is not a substitute for formal wavelength-grid refinement, eigenmode tracking, FEM discretization convergence or qualified review of metal dispersion, PML placement, physical source logs and material data.

**Real evidence still absent:** The repository does not contain an authenticated COMSOL `.mph` model, native logs covering a complete verification campaign, independently mesh-converged raw spectra and an external physics review. Issue [#42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42) remains open. No synthetic spectrum from CI may appear in journal tables as an experimental observation.