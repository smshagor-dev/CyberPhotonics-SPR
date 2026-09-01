from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd

from sprpcf.simulation.schema import Geometry

PROTOCOL_VERSION = 1
DEFAULT_TOKEN_ENV = "SPR_COMSOL_API_TOKEN"
DEFAULT_URL_ENV = "SPR_COMSOL_API_URL"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RemoteComsolSettings:
    """Connection settings for the repository-owned remote COMSOL API."""

    base_url: str
    token: str | None = None
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str | None = None,
        token_env: str = DEFAULT_TOKEN_ENV,
        timeout_seconds: float = 300.0,
    ) -> "RemoteComsolSettings":
        resolved_url = base_url or os.getenv(DEFAULT_URL_ENV)
        if not resolved_url:
            raise ValueError(
                f"Remote COMSOL URL is required. Pass it explicitly or set {DEFAULT_URL_ENV}."
            )
        token = os.getenv(token_env) if token_env else None
        return cls(base_url=resolved_url, token=token, timeout_seconds=timeout_seconds)

    def validated(self) -> "RemoteComsolSettings":
        base_url = normalize_base_url(self.base_url)
        if self.timeout_seconds <= 0:
            raise ValueError("Remote COMSOL timeout must be > 0 seconds.")
        return RemoteComsolSettings(
            base_url=base_url,
            token=self.token.strip() if self.token else None,
            timeout_seconds=float(self.timeout_seconds),
        )


def normalize_base_url(value: str) -> str:
    """Normalize a remote endpoint while rejecting embedded credentials/query secrets."""
    raw = value.strip()
    if not raw:
        raise ValueError("Remote COMSOL base URL cannot be empty.")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Remote COMSOL URL must use http or https.")
    if not parsed.hostname:
        raise ValueError("Remote COMSOL URL must include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Do not embed credentials in the remote COMSOL URL; use a bearer-token environment variable.")
    if parsed.query or parsed.fragment:
        raise ValueError("Remote COMSOL base URL must not contain query parameters or fragments.")
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _request_payload(geometries: Sequence[Geometry]) -> dict[str, Any]:
    if not geometries:
        return {"schema_version": PROTOCOL_VERSION, "geometries": []}
    rows: list[dict[str, float | int]] = []
    for sample_id, geometry in enumerate(geometries):
        geometry.validate()
        rows.append({"sample_id": sample_id, **geometry.__dict__})
    return {"schema_version": PROTOCOL_VERSION, "geometries": rows}


def _read_limited(response: Any, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise RuntimeError("Remote COMSOL response is larger than the allowed limit.")
        except ValueError:
            pass
    body = response.read(limit + 1)
    if len(body) > limit:
        raise RuntimeError("Remote COMSOL response is larger than the allowed limit.")
    return body


def _parse_response(body: bytes, expected_samples: int) -> pd.DataFrame:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Remote COMSOL server returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Remote COMSOL response must be a JSON object.")
    if payload.get("schema_version") != PROTOCOL_VERSION:
        raise RuntimeError("Remote COMSOL protocol version mismatch.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Remote COMSOL response is missing the results array.")
    if len(results) != expected_samples:
        raise RuntimeError(
            f"Remote COMSOL returned {len(results)} rows for {expected_samples} requested samples."
        )
    frame = pd.DataFrame(results)
    if expected_samples == 0:
        return frame
    required = [
        "sample_id",
        "status",
        "pitch_um",
        "d_over_lambda",
        "metal_thickness_nm",
        "channel_radius_um",
        "analyte_ri",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Remote COMSOL results are missing required columns: {missing}")
    ids = frame["sample_id"].astype(int)
    if ids.duplicated().any():
        raise RuntimeError("Remote COMSOL results contain duplicate sample_id values.")
    expected_ids = set(range(expected_samples))
    if set(ids.tolist()) != expected_ids:
        raise RuntimeError("Remote COMSOL results do not match the requested sample IDs.")
    return frame.sort_values("sample_id").reset_index(drop=True)


def run_remote_comsol_geometries(
    settings: RemoteComsolSettings,
    geometries: Sequence[Geometry],
) -> pd.DataFrame:
    """Execute geometry rows on a remote licensed COMSOL host via the repository API."""
    validated = settings.validated()
    payload = _request_payload(geometries)
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "CyberPhotonics-SPR/remote-comsol-v1",
    }
    if validated.token:
        headers["Authorization"] = f"Bearer {validated.token}"
    request = Request(
        f"{validated.base_url}/v1/simulations/geometries",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=validated.timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError(f"Remote COMSOL server returned HTTP {response.status}.")
            response_body = _read_limited(response)
    except HTTPError as exc:
        message = f"Remote COMSOL server returned HTTP {exc.code}."
        try:
            error_payload = json.loads(exc.read(4096).decode("utf-8"))
            detail = error_payload.get("error") if isinstance(error_payload, dict) else None
            if isinstance(detail, str) and detail:
                message = f"{message} {detail}"
        except Exception:
            pass
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the remote COMSOL server: {exc.reason}") from exc
    return _parse_response(response_body, expected_samples=len(geometries))
