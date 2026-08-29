from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


API_VERSION = "2026-03-10"
DEFAULT_REPOSITORY = "smshagor-dev/CyberPhotonics-SPR"
DEFAULT_HOMEPAGE = "https://smshagor-dev.github.io/CyberPhotonics-SPR/"
DEFAULT_TOPICS = [
    "photonics",
    "photonic-crystal-fiber",
    "surface-plasmon-resonance",
    "spr-sensor",
    "inverse-design",
    "physics-informed-ml",
    "edge-ai",
    "tflite",
    "onnx",
    "comsol",
    "scientific-computing",
    "cyber-physical-systems",
    "research-software",
    "python",
]


@dataclass(frozen=True)
class ApiResponse:
    status: int
    payload: Any | None


def _request(token: str, method: str, url: str, payload: dict[str, Any] | None = None) -> ApiResponse:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "CyberPhotonics-SPR-governance-bootstrap",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            decoded = json.loads(body.decode("utf-8")) if body else None
            return ApiResponse(int(response.status), decoded)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body) if body else None
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"GitHub API {method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _branch_protection_payload() -> dict[str, Any]:
    # A single-maintainer research repository still benefits from pull-request
    # discipline without requiring the author to approve their own PR.
    return {
        "required_status_checks": None,
        "enforce_admins": False,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "required_approving_review_count": 0,
            "require_last_push_approval": False,
        },
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": True,
    }


def _print_plan(repository: str, homepage: str, topics: list[str]) -> None:
    print("CyberPhotonics-SPR GitHub governance plan")
    print(f"  repository: {repository}")
    print("  branch protection: main -> PR workflow, linear history, no force-push/delete, conversations resolved")
    print(f"  topics ({len(topics)}): {', '.join(topics)}")
    print(f"  homepage: {homepage}")
    print("  GitHub Pages: workflow build enabled")


def _configure_pages(token: str, api_root: str, repository: str) -> None:
    pages_url = f"{api_root}/repos/{repository}/pages"
    try:
        _request(token, "GET", pages_url)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        _request(token, "POST", pages_url, {"build_type": "workflow"})
        print("Enabled GitHub Pages with GitHub Actions as the publishing source.")
        return

    _request(token, "PUT", pages_url, {"build_type": "workflow", "https_enforced": True})
    print("Updated GitHub Pages to use GitHub Actions and enforce HTTPS.")


def apply_governance(token: str, repository: str, homepage: str, topics: list[str]) -> None:
    api_root = "https://api.github.com"

    _request(token, "PATCH", f"{api_root}/repos/{repository}", {"homepage": homepage})
    print("Updated repository homepage.")

    _request(token, "PUT", f"{api_root}/repos/{repository}/topics", {"names": topics})
    print("Updated repository topics.")

    _request(
        token,
        "PUT",
        f"{api_root}/repos/{repository}/branches/main/protection",
        _branch_protection_payload(),
    )
    print("Enabled main-branch protection.")

    _configure_pages(token, api_root, repository)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply CyberPhotonics-SPR repository-level GitHub governance settings."
    )
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY, help="Repository in owner/name form.")
    parser.add_argument("--homepage", default=DEFAULT_HOMEPAGE)
    parser.add_argument("--apply", action="store_true", help="Actually mutate GitHub settings. Without this flag the command is a dry run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topics = list(DEFAULT_TOPICS)
    _print_plan(args.repo, args.homepage, topics)

    if not args.apply:
        print("\nDry run only. Re-run with --apply and an admin-scoped GH_TOKEN to make changes.")
        return 0

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        print(
            "GH_TOKEN is required for --apply. The token must be able to administer the repository and manage Pages.",
            file=sys.stderr,
        )
        return 2

    try:
        apply_governance(token, args.repo, args.homepage, topics)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("\nRepository governance settings applied successfully.")
    print("Next: run the 'Documentation Site' workflow manually once to deploy the Pages artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
