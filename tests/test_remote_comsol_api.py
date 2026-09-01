from __future__ import annotations

import math
import threading

import pandas as pd
import pytest

from sprpcf.simulation.comsol_api import ComsolApiContext, create_comsol_api_server
from sprpcf.simulation.remote_comsol import (
    RemoteComsolSettings,
    normalize_base_url,
    run_remote_comsol_geometries,
)
from sprpcf.simulation.schema import Geometry


def _geometry(ri: float = 1.33) -> Geometry:
    return Geometry(
        d_over_lambda=0.55,
        pitch_um=2.1,
        metal_thickness_nm=45.0,
        channel_radius_um=0.6,
        analyte_ri=ri,
    )


def _successful_runner(geometries):
    rows = []
    for sample_id, geometry in enumerate(geometries):
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                **geometry.__dict__,
                "lambda_res_nm": 700.0 + sample_id * 10.0,
                "peak_loss_db_per_cm": 12.0,
                "fwhm_nm": 20.0,
                "sensitivity_nm_per_riu": 1000.0,
                "fom_per_riu": 50.0,
                "wavelength_nm": "690,700,710",
                "loss_db_per_cm": "1,12,1",
            }
        )
    return pd.DataFrame(rows)


def _start_server(runner, token="test-token"):
    server = create_comsol_api_server(
        "127.0.0.1",
        0,
        ComsolApiContext(runner=runner, token=token, max_geometries=16),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_remote_client_round_trip_with_bearer_token():
    server, thread = _start_server(_successful_runner)
    try:
        host, port = server.server_address
        settings = RemoteComsolSettings(
            base_url=f"http://{host}:{port}",
            token="test-token",
            timeout_seconds=5,
        )
        frame = run_remote_comsol_geometries(settings, [_geometry(1.33), _geometry(1.34)])
        assert frame["sample_id"].tolist() == [0, 1]
        assert frame["status"].tolist() == ["ok", "ok"]
        assert frame["analyte_ri"].tolist() == pytest.approx([1.33, 1.34])
        assert frame["lambda_res_nm"].tolist() == pytest.approx([700.0, 710.0])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_api_serializes_failed_rows_with_null_metrics():
    def runner(geometries):
        geometry = geometries[0]
        return pd.DataFrame(
            [
                {
                    "sample_id": 0,
                    "status": "failed: solver error",
                    **geometry.__dict__,
                    "lambda_res_nm": math.nan,
                    "fwhm_nm": math.nan,
                }
            ]
        )

    server, thread = _start_server(runner)
    try:
        host, port = server.server_address
        frame = run_remote_comsol_geometries(
            RemoteComsolSettings(base_url=f"http://{host}:{port}", token="test-token", timeout_seconds=5),
            [_geometry()],
        )
        assert frame.loc[0, "status"].startswith("failed:")
        assert pd.isna(frame.loc[0, "lambda_res_nm"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_api_rejects_bad_token_without_exposing_secret():
    server, thread = _start_server(_successful_runner)
    try:
        host, port = server.server_address
        with pytest.raises(RuntimeError, match="HTTP 401") as exc_info:
            run_remote_comsol_geometries(
                RemoteComsolSettings(base_url=f"http://{host}:{port}", token="wrong-secret", timeout_seconds=5),
                [_geometry()],
            )
        assert "wrong-secret" not in str(exc_info.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_url_rejects_embedded_credentials_and_query_secrets():
    with pytest.raises(ValueError, match="embed credentials"):
        normalize_base_url("https://user:password@example.com")
    with pytest.raises(ValueError, match="query"):
        normalize_base_url("https://example.com?token=secret")
    assert normalize_base_url("https://example.com/api/") == "https://example.com/api"


def test_remote_api_rejects_noncontiguous_sample_ids():
    def bad_runner(geometries):
        rows = _successful_runner(geometries)
        rows.loc[0, "sample_id"] = 9
        return rows

    server, thread = _start_server(bad_runner)
    try:
        host, port = server.server_address
        with pytest.raises(RuntimeError, match="HTTP 500"):
            run_remote_comsol_geometries(
                RemoteComsolSettings(base_url=f"http://{host}:{port}", token="test-token", timeout_seconds=5),
                [_geometry()],
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
