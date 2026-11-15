# Independent real native-grid archive integrity inspection — 2026-11-15

Source: GitHub Actions run [43685290196](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/43685290196), artifact **14149464285**, head `7c7736dcf68dd652429a03aa1937f3822716498e`. Downloaded the actual 55,377,388-byte ZIP through the GitHub connector, not a test fixture.

ZIP SHA256: `30b28d83cef11faca79dae829c3a931edc73da2bcc9e69cd12af50b66f3cf4eb` (matches GitHub artifact digest).

Executed a local independent read-only Python/NumPy audit against the downloaded ZIP:
- ZIP CRC valid; 14/14 embedded artifact SHA256 values match manifest.
- Exactly 18 unique native eigenmode tuples across 3 RI × 3 wavelengths × 2 modes.
- All nine raw native NPZ field archives open with `allow_pickle=False`; all stored arrays finite.
- CSV complex eigenvalue and Au dielectric components finite.
- Seven nonpublication/physics-qualification flags remain explicitly FALSE.
- Zero integrity failures in this inspection.

**Interpretation:** This validates archive structure and declared hashes, not solver correctness, source-to-field physics, geometry adequacy, numerical convergence or sensor performance. The 3-point spectral grid cannot resolve FWHM or a trustworthy RI sensitivity. Au Johnson–Christy is a bulk reference, not a measured deposited film. No PML, independent PCF Maxwell cross-solver agreement, literature-frozen sensor topology, or physical core/SPP tracking has been verified. No COMSOL model exists; none is claimed. `scientific_submission_ready=false` must remain unchanged.

The committed reusable `scripts/audit_native_grid_zip.py` generalizes this audit and must be exercised by CI against the same downloaded artifact before merge. It must not be used to promote any publication flag.

Follow-up: execute measured-Au domain/PML and multi-mesh spectral campaigns, perform field-overlap identity validation and independent solver comparison; retain every failed solve and immutable field archive. Issue #42 was closed, but its original COMSOL-specific acceptance criteria have not been met by this exploratory FEM dataset.
