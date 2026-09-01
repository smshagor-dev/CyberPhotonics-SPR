from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from sprpcf.simulation.schema import Geometry

PROTOCOL_VERSION = 1
DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_GEOMETRIES = 256
GeometryRunner = Callable[[Sequence[Geometry]], pd.DataFrame]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ComsolApiContext:
    runner: GeometryRunner
    token: str | None
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_geometries: int = DEFAULT_MAX_GEOMETRIES
    model_sha256: str | None = None
    config_sha256: str | None = None

    def validate(self) -> None:
        if self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes must be at least 1024 bytes.")
        if self.max_geometries < 1:
            raise ValueError("max_geometries must be >= 1.")


def _parse_geometry_rows(payload: Any, max_geometries: int) -> list[Geometry]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    if payload.get("schema_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version.")
    rows = payload.get("geometries")
    if not isinstance(rows, list):
        raise ValueError("Request body must contain a geometries array.")
    if len(rows) > max_geometries:
        raise ValueError(f"Too many geometries; maximum per request is {max_geometries}.")

    geometries: list[Geometry] = []
    for expected_sample_id, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Each geometry row must be a JSON object.")
        if row.get("sample_id") != expected_sample_id:
            raise ValueError("sample_id values must be contiguous and start at zero.")
        try:
            geometry = Geometry(
                pitch_um=float(row["pitch_um"]),
                d_over_lambda=float(row["d_over_lambda"]),
                metal_thickness_nm=float(row["metal_thickness_nm"]),
                channel_radius_um=float(row["channel_radius_um"]),
                analyte_ri=float(row["analyte_ri"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid geometry row at sample_id={expected_sample_id}.") from exc
        geometry.validate()
        geometries.append(geometry)
    return geometries


def _validate_runner_results(frame: pd.DataFrame, expected_samples: int) -> pd.DataFrame:
    if expected_samples == 0:
        return pd.DataFrame(columns=["sample_id", "status"])
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("COMSOL runner must return a pandas DataFrame.")
    if len(frame) != expected_samples:
        raise RuntimeError(
            f"COMSOL runner returned {len(frame)} rows for {expected_samples} requested samples."
        )
    if "sample_id" not in frame.columns or "status" not in frame.columns:
        raise RuntimeError("COMSOL runner results must include sample_id and status columns.")
    ids = frame["sample_id"].astype(int)
    if ids.duplicated().any() or set(ids.tolist()) != set(range(expected_samples)):
        raise RuntimeError("COMSOL runner returned invalid sample_id values.")
    return frame.sort_values("sample_id").reset_index(drop=True)


def create_comsol_api_server(
    host: str,
    port: int,
    context: ComsolApiContext,
) -> ThreadingHTTPServer:
    """Create a small dependency-free HTTP API around a COMSOL geometry runner."""
    context.validate()

    class Handler(BaseHTTPRequestHandler):
        server_version = "CyberPhotonicsSPRComsolAPI/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            super().log_message(format, *args)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not context.token:
                return True
            header = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not header.startswith(prefix):
                return False
            return secrets.compare_digest(header[len(prefix) :], context.token)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "schema_version": PROTOCOL_VERSION,
                    "auth_required": bool(context.token),
                    "max_geometries": context.max_geometries,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/simulations/geometries":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._authorized():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing bearer token"})
                return

            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Content-Type must be application/json"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
                return
            if content_length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "request body is required"})
                return
            if content_length > context.max_request_bytes:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
                return

            try:
                raw = self.rfile.read(content_length)
                payload = json.loads(raw.decode("utf-8"))
                geometries = _parse_geometry_rows(payload, context.max_geometries)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            try:
                results = _validate_runner_results(context.runner(geometries), len(geometries))
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"COMSOL execution failed: {type(exc).__name__}: {exc}"},
                )
                return

            self._send_json(
                HTTPStatus.OK,
                {
                    "schema_version": PROTOCOL_VERSION,
                    "results": results.to_dict(orient="records"),
                    "provenance": {
                        "evidence_class": "comsol_physics",
                        "model_sha256": context.model_sha256,
                        "config_sha256": context.config_sha256,
                    },
                },
            )

    return ThreadingHTTPServer((host, port), Handler)
