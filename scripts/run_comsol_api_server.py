from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.simulation.comsol_api import ComsolApiContext, create_comsol_api_server, file_sha256
from sprpcf.simulation.comsol_sweep import run_comsol_geometries


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expose a licensed local COMSOL runtime as the CyberPhotonics-SPR remote simulation API."
    )
    parser.add_argument("--model", type=Path, required=True, help="Server-local COMSOL .mph model path.")
    parser.add_argument("--config", type=Path, required=True, help="Server-local COMSOL sweep YAML path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="SPR_COMSOL_API_TOKEN")
    parser.add_argument("--allow-unauthenticated-localhost", action="store_true")
    parser.add_argument("--max-geometries", type=int, default=256)
    parser.add_argument("--max-request-bytes", type=int, default=2 * 1024 * 1024)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"COMSOL model does not exist: {args.model}")
    if not args.config.is_file():
        parser.error(f"COMSOL config does not exist: {args.config}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be within 1..65535")

    token = os.getenv(args.token_env) if args.token_env else None
    if token:
        token = token.strip()
    if not token:
        if not (args.allow_unauthenticated_localhost and _is_loopback(args.host)):
            parser.error(
                f"No API token found in {args.token_env}. Set it, or use "
                "--allow-unauthenticated-localhost only with a loopback host for local testing."
            )
        token = None

    def runner(geometries):
        return run_comsol_geometries(args.model, args.config, geometries)

    context = ComsolApiContext(
        runner=runner,
        token=token,
        max_request_bytes=args.max_request_bytes,
        max_geometries=args.max_geometries,
        model_sha256=file_sha256(args.model),
        config_sha256=file_sha256(args.config),
    )
    server = create_comsol_api_server(args.host, args.port, context)
    auth_mode = "bearer-token" if token else "unauthenticated-loopback-test"
    print(f"CyberPhotonics-SPR COMSOL API listening on http://{args.host}:{args.port} ({auth_mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
