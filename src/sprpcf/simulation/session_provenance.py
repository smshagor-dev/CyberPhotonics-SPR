"""Per-execution-session COMSOL-native log binding and audit.

A file hash can demonstrate that recorded bytes have not changed. It cannot
prove that supplied bytes really came from COMSOL, nor that a model solves
Maxwell's equations correctly. All provenance here remains unqualified.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from sprpcf.utils.reproducibility import sha256_file

SESSION_DIR = "solver_sessions"
LOG_DIR = "native_solver_logs"
SESSION_INDEX = "session_log_evidence.json"
SESSION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")


def _id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_ID_RE.fullmatch(value):
        raise ValueError("Invalid solver session ID")
    return value


def _file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "/" in relative or "\\" in relative:
        raise ValueError("Unsafe session evidence filename")
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file() or not candidate.stat().st_size:
        raise ValueError("Missing or symlinked solver session evidence: " + relative)
    return candidate


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected solver session JSON object")
    return value


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # x mode refuses overwrites and preserves the first recorded evidence.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def start_solver_session(root: Path, contract: dict[str, Any], first_sample: int) -> str:
    session_id = uuid.uuid4().hex
    _write_once(root / SESSION_DIR / (session_id + ".start.json"), {
        "schema_version": 1, "session_id": session_id,
        "kind": "solver_execution_session_start",
        "backend_kind": contract["backend_kind"],
        "checkpoint_plan_sha256": contract["plan_sha256"],
        "model_sha256": contract["model_sha256"],
        "config_sha256": contract["config_sha256"],
        "first_sample": first_sample,
        "started_at_utc": _now(),
        "source_authenticity_verified": False,
    })
    return session_id


def finish_solver_session(root: Path, session_id: str, status: str) -> None:
    _id(session_id)
    if status not in ("completed", "interrupted", "manually_closed_after_crash"):
        raise ValueError("Invalid solver session outcome")
    start = _file(root / SESSION_DIR, session_id + ".start.json")
    dest = root / SESSION_DIR / (session_id + ".end.json")
    if dest.exists():
        return
    _write_once(dest, {
        "schema_version": 1, "session_id": session_id,
        "kind": "solver_execution_session_end", "status": status,
        "start_sha256": sha256_file(start), "ended_at_utc": _now(),
        "human_review_required": True,
    })


def close_crashed_solver_session(root: str | Path, session_id: str) -> dict[str, Any]:
    """Explicit operator action after verifying the original worker is gone."""
    root = Path(root).resolve()
    _id(session_id)
    if (root / ".active-comsol-campaign.lock").exists():
        raise RuntimeError("Active worker lock exists; cannot close crashed session")
    _file(root / SESSION_DIR, session_id + ".start.json")
    if (root / SESSION_DIR / (session_id + ".end.json")).exists():
        raise ValueError("Session has already ended")
    finish_solver_session(root, session_id, "manually_closed_after_crash")
    return {"session_id": session_id, "status": "manually_closed_after_crash",
            "source_authenticity_verified": False}


def _sessions(root: Path, contract: dict[str, Any],
              chunks: list[dict[str, Any]], *, closed_only: bool = True) -> list[dict[str, Any]]:
    folder = root / SESSION_DIR
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Campaign lacks its per-invocation solver session directory")
    files = list(folder.iterdir())
    if any(p.is_symlink() or not p.is_file() or not (
        p.name.endswith(".start.json") or p.name.endswith(".end.json")
    ) for p in files):
        raise ValueError("Unexpected/incomplete solver session evidence file")
    starts = sorted(p.name[:-len(".start.json")] for p in files if p.name.endswith(".start.json"))
    ends = {p.name[:-len(".end.json")] for p in files if p.name.endswith(".end.json")}
    if len(starts) == 0 or len(starts) != len(set(starts)) or ends.difference(starts):
        raise ValueError("Missing or orphaned solver execution sessions")
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in starts}
    for batch in chunks:
        sid = batch.get("session_id")
        if sid not in groups:
            raise ValueError("Committed chunk has unknown/missing solver session ID")
        groups[sid].append(batch)
    records = []
    for sid in starts:
        _id(sid)
        start_file = _file(folder, sid + ".start.json")
        start = _read(start_file)
        for key, expected in {
            "schema_version": 1, "session_id": sid,
            "kind": "solver_execution_session_start",
            "backend_kind": contract["backend_kind"],
            "checkpoint_plan_sha256": contract["plan_sha256"],
            "model_sha256": contract["model_sha256"],
            "config_sha256": contract["config_sha256"],
        }.items():
            if start.get(key) != expected:
                raise ValueError("Solver session start mismatches frozen campaign: " + key)
        batches = groups[sid]
        if batches:
            if not isinstance(start.get("first_sample"), int) or (
                min(x["start"] for x in batches) != start["first_sample"]
            ):
                raise ValueError("Solver session first committed sample does not match start record")
            positions = sorted((x["start"], x["stop"]) for x in batches)
            if any(a[1] != b[0] for a, b in zip(positions, positions[1:])):
                raise ValueError("Solver session batches have an internal gap")
        end_file = folder / (sid + ".end.json")
        if not end_file.is_file():
            if closed_only:
                raise ValueError("Solver session is unfinished; operator must close crashed session: " + sid)
            status = "open"
            end_sha = None
        else:
            end = _read(_file(folder, sid + ".end.json"))
            if (end.get("session_id") != sid or end.get("schema_version") != 1
                    or end.get("kind") != "solver_execution_session_end"
                    or end.get("start_sha256") != sha256_file(start_file)
                    or end.get("status") not in (
                        "completed", "interrupted", "manually_closed_after_crash"
                    )):
                raise ValueError("Solver session termination evidence mismatch: " + sid)
            status = end["status"]
            end_sha = sha256_file(end_file)
        records.append({
            "session_id": sid, "status": status,
            "start_sha256": sha256_file(start_file),
            "end_sha256": end_sha,
            "first_sample": start["first_sample"],
            "chunk_ranges": [[b["start"], b["stop"]] for b in batches],
            "chunk_manifest_sha256": [b["manifest_sha256"] for b in batches],
        })
    return records


def inspect_session_ledger(run_dir: str | Path, contract: dict[str, Any],
                           chunks: list[dict[str, Any]], *, closed_only: bool = True
                           ) -> list[dict[str, Any]]:
    return _sessions(Path(run_dir).resolve(), contract, chunks, closed_only=closed_only)


def bind_native_session_log(
    run_dir: str | Path, session_id: str, log_path: str | Path,
    contract: dict[str, Any], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy actual user-provided native log bytes, bind them to one session.

    This cannot verify that the bytes came from COMSOL. Evidence stays review-only.
    """
    root = Path(run_dir).resolve()
    session_id = _id(session_id)
    records = _sessions(root, contract, chunks)
    selected = next((r for r in records if r["session_id"] == session_id), None)
    if selected is None:
        raise ValueError("Session ID not part of this campaign")
    source = Path(log_path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size < 1:
        raise ValueError("Native solver log must be nonempty and not symlinked")
    folder = root / LOG_DIR
    if folder.is_symlink():
        raise ValueError("Unsafe solver log directory")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (session_id + ".log")
    receipt = folder / (session_id + ".json")
    if destination.exists() or receipt.exists():
        raise FileExistsError("Native solver log already bound for this session")
    shutil.copyfile(source, destination)
    digest = sha256_file(destination)
    if digest != sha256_file(source):
        destination.unlink(missing_ok=True)
        raise ValueError("Native solver log copy changed unexpectedly")
    payload = {
        "schema_version": 1, "session_id": session_id,
        "kind": "native_solver_log_binding",
        "source_session_start_sha256": selected["start_sha256"],
        "source_session_end_sha256": selected["end_sha256"],
        "source_chunk_manifest_sha256": selected["chunk_manifest_sha256"],
        "source_chunk_ranges": selected["chunk_ranges"],
        "log_sha256": digest,
        "log_size_bytes": destination.stat().st_size,
        "source_authenticity_verified": False,
        "log_origin": "external_operator_supplied_not_cryptographically_authenticated",
        "bound_at_utc": _now(),
    }
    _write_once(receipt, payload)
    return payload


def audit_native_session_logs(
    run_dir: str | Path, contract: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate every session's actual log-copy bytes and frozen chunk ownership."""
    root = Path(run_dir).resolve()
    records = _sessions(root, contract, chunks)
    folder = root / LOG_DIR
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Native COMSOL logs are missing for execution sessions")
    known: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        sid = record["session_id"]
        receipt_path = _file(folder, sid + ".json")
        log_path = _file(folder, sid + ".log")
        data = _read(receipt_path)
        checks = {
            "schema_version": 1,
            "session_id": sid,
            "kind": "native_solver_log_binding",
            "source_session_start_sha256": record["start_sha256"],
            "source_session_end_sha256": record["end_sha256"],
            "source_chunk_manifest_sha256": record["chunk_manifest_sha256"],
            "source_chunk_ranges": record["chunk_ranges"],
            "log_sha256": sha256_file(log_path),
            "log_size_bytes": log_path.stat().st_size,
            "source_authenticity_verified": False,
        }
        for key, value in checks.items():
            if data.get(key) != value:
                raise ValueError("Native solver log receipt mismatch: " + sid + " " + key)
        known.update({sid + ".json", sid + ".log"})
        output.append({
            **record,
            "native_log_sha256": checks["log_sha256"],
            "native_log_size_bytes": checks["log_size_bytes"],
            "receipt_sha256": sha256_file(receipt_path),
            "source_authenticity_verified": False,
        })
    if {f.name for f in folder.iterdir()} != known:
        raise ValueError("Unregistered, missing or additional native solver log evidence files")
    return output



def audit_sealed_session_logs(archive_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recheck copied original logs, receipts and session boundaries in Stage 5.

    Checksum completeness is all this can establish; native-log source
    authenticity and physical model correctness remain independently unverified.
    """
    root = Path(archive_dir).resolve()
    if manifest.get("session_log_provenance_required") is not True:
        raise ValueError("Sealed checkpointed COMSOL archive lacks mandatory session log gate")
    if manifest.get("session_log_evidence") != SESSION_INDEX:
        raise ValueError("Sealed archive has unexpected session index path")
    index_path = root / SESSION_INDEX
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Missing or symlinked sealed session log evidence index")
    if manifest.get("session_log_evidence_sha256") != sha256_file(index_path):
        raise ValueError("Sealed native session log index SHA-256 mismatch")
    index = _read(index_path)
    if (index.get("schema_version") != 1
            or index.get("type") != "per_execution_session_log_provenance"
            or index.get("source_authenticity_verified") is not False):
        raise ValueError("Sealed session log index violates its evidence contract")
    records = index.get("session_records")
    file_map = index.get("files_sha256")
    if (not isinstance(records, list) or not records
            or len(records) != index.get("session_count")
            or not isinstance(file_map, dict)):
        raise ValueError("Sealed session log inventory is missing or empty")
    expected_paths: set[str] = set()
    seen: set[str] = set()
    native_hashes: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed sealed solver session record")
        sid = _id(record.get("session_id"))
        if sid in seen:
            raise ValueError("Duplicate sealed solver session IDs")
        seen.add(sid)
        paths = {
            "start": "session_logs/" + sid + ".start.json",
            "end": "session_logs/" + sid + ".end.json",
            "log": "session_logs/" + sid + ".log",
            "receipt": "session_logs/" + sid + ".json",
        }
        source_files = {}
        for name, relative in paths.items():
            expected_paths.add(relative)
            file = root / relative
            if (file.parent.is_symlink() or file.is_symlink() or not file.is_file()
                    or file.stat().st_size == 0
                    or sha256_file(file) != file_map.get(relative)):
                raise ValueError("Sealed native solver session evidence hash mismatch: " + relative)
            source_files[name] = file
        start = _read(source_files["start"])
        end = _read(source_files["end"])
        receipt = _read(source_files["receipt"])
        log_digest = sha256_file(source_files["log"])
        native_hashes.add(log_digest)
        if (start.get("session_id") != sid or end.get("session_id") != sid
                or receipt.get("session_id") != sid
                or end.get("status") != record.get("status")
                or record.get("start_sha256") != sha256_file(source_files["start"])
                or record.get("end_sha256") != sha256_file(source_files["end"])
                or end.get("start_sha256") != sha256_file(source_files["start"])
                or receipt.get("source_session_start_sha256") != record["start_sha256"]
                or receipt.get("source_session_end_sha256") != record["end_sha256"]
                or receipt.get("source_chunk_manifest_sha256") != record.get("chunk_manifest_sha256")
                or receipt.get("source_chunk_ranges") != record.get("chunk_ranges")
                or receipt.get("log_sha256") != log_digest
                or receipt.get("log_size_bytes") != source_files["log"].stat().st_size
                or record.get("native_log_sha256") != log_digest
                or record.get("receipt_sha256") != sha256_file(source_files["receipt"])
                or start.get("checkpoint_plan_sha256") != manifest.get("campaign_plan_sha256")):
            raise ValueError("Sealed solver session ledger does not match original log or chunk ownership")
        if end.get("status") not in ("completed", "interrupted", "manually_closed_after_crash"):
            raise ValueError("Unclosed execution session in sealed evidence")
    if set(file_map) != expected_paths:
        raise ValueError("Missing or extra sealed session evidence entries")
    session_dir = root / "session_logs"
    if (not session_dir.is_dir() or session_dir.is_symlink()
            or {p.name for p in session_dir.iterdir()} != {
                Path(name).name for name in expected_paths
            }):
        raise ValueError("Additional or missing sealed native session files")
    final_log = root / manifest.get("solver_log", "")
    if not final_log.is_file() or sha256_file(final_log) not in native_hashes:
        raise ValueError("Top-level solver log must match one bound native session log")
    return {
        "session_count": len(records),
        "recorded_native_logs_complete": True,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
    }
