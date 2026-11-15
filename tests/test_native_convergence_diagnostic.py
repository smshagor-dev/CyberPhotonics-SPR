from sprpcf.validation.native_convergence_diagnostic import NativePoint, assess_convergence


def point(mesh, domain, pml, idx, mode="core"):
    return NativePoint(mesh, domain, pml, 1.45 + idx * 1e-7, 1e-6, f"{idx:064x}", mode)


def campaign():
    # Fixtures test gate behavior ONLY, never Maxwell physics.
    return [
        point(0.03, 8, 1, 1), point(0.02, 8, 1, 2), point(0.01, 8, 1, 3),
        point(0.01, 10, 1, 4), point(0.01, 12, 1, 5),
        point(0.01, 12, 1.5, 6), point(0.01, 12, 2, 7),
    ]


def test_three_independent_axes_diagnostic_only():
    result = assess_convergence(campaign())
    assert result["diagnostic_pass"]
    assert not result["publication_qualified"]
    assert set(result["axes"]) == {"mesh_um", "domain_um", "pml_um"}


def test_diagonal_sweep_rejected():
    result = assess_convergence([point(0.03, 8, 1, 1), point(0.02, 10, 1.5, 2), point(0.01, 12, 2, 3)])
    assert not result["diagnostic_pass"]
    assert len(result["reasons"]) == 3


def test_duplicate_archive_rejected():
    pts = campaign()
    pts[1] = point(0.02, 8, 1, 1)
    assert not assess_convergence(pts)["diagnostic_pass"]


def test_mode_switch_rejected():
    pts = campaign()
    pts[3] = point(0.01, 10, 1, 4, "spp")
    assert not assess_convergence(pts)["diagnostic_pass"]


def test_nonfinite_rejected():
    pts = campaign()
    pts[2] = NativePoint(0.01, 8, 1, float("nan"), 1e-6, f"{3:064x}", "core")
    assert not assess_convergence(pts)["diagnostic_pass"]


def test_missing_pml_sweep_rejected():
    assert not assess_convergence(campaign()[:5])["diagnostic_pass"]


def test_no_records_rejected():
    assert not assess_convergence([])["diagnostic_pass"]
