"""Regression checks for measured-reference planar Richardson convergence."""
import pytest
from research.planar_tm_spp_richardson import audit_point, run


@pytest.mark.parametrize("w", [0.75, 0.80, 0.85])
@pytest.mark.parametrize("n", [1.35, 1.36, 1.37])
def test_second_order_convergence(w, n):
    result = audit_point(w, n)
    assert len(result["successive_error_ratios"]) == 2
    assert result["richardson_relative_error"] < 1e-8
    assert result["publication_ready"] is False
    assert result["qualified_pcf_physics"] is False


def test_immutable_report(tmp_path):
    out = tmp_path / "richardson.json"
    result = run(out)
    assert len(result["points"]) == 9
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        run(out)
    assert out.read_bytes() == original


def test_fail_closed_when_convergence_is_corrupted(monkeypatch):
    import research.planar_tm_spp_richardson as module
    def invalid(w, n, h):
        return {
            "relative_error": 0.001,
            "finite_difference_dbeta_dn_real_um_inv_riu_inv": 1.0,
            "finite_difference_dbeta_dn_imag_um_inv_riu_inv": 0.0,
            "analytic_dbeta_dn_real_um_inv_riu_inv": 1.0,
            "analytic_dbeta_dn_imag_um_inv_riu_inv": 0.0,
        }
    monkeypatch.setattr(module, "planar_derivative_control", invalid)
    with pytest.raises(RuntimeError, match="second-order"):
        module.audit_point(0.8, 1.36)
