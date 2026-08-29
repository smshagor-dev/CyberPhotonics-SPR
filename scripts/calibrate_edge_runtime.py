from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from sprpcf.edge.hardware import (
    EdgeCalibrationBundle,
    KerasModel,
    LiteRTModel,
    PredictionIntervalCalibration,
    SpectralOODDetector,
)
from sprpcf.edge.train_denoiser import normalize_spectra, parse_spectra
from sprpcf.ml.dataset import GEOMETRY_COLUMNS, geometry_group_labels, read_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(path: Path, runtime: str):
    if runtime == "litert":
        return LiteRTModel(path)
    if runtime == "keras":
        return KerasModel(path)
    raise ValueError("runtime must be 'litert' or 'keras'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build held-out prediction intervals and spectral OOD calibration for the edge sensor runtime."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--denoiser", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--runtime", choices=["litert", "keras"], default="litert")
    parser.add_argument("--out", type=Path, default=Path("models/edge_calibration.json"))
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--ood-coverage", type=float, default=0.99)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    if not 0.05 <= args.calibration_fraction <= 0.5:
        raise ValueError("--calibration-fraction must be within [0.05, 0.5].")

    required = ["loss_db_per_cm", "analyte_ri", "lambda_res_nm", *GEOMETRY_COLUMNS]
    frame = read_table(args.data).dropna(subset=required).reset_index(drop=True)
    if len(frame) < 6:
        raise ValueError("At least six labeled spectra are required for edge calibration.")

    spectra, _, _ = normalize_spectra(parse_spectra(frame))
    targets = frame[["analyte_ri", "lambda_res_nm"]].to_numpy(np.float32)
    groups = geometry_group_labels(frame)
    if np.unique(groups).size < 2:
        raise ValueError("At least two unique geometries are required for grouped calibration.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.calibration_fraction, random_state=args.seed)
    reference_idx, calibration_idx = next(splitter.split(frame, groups=groups))
    if calibration_idx.size < 3:
        raise ValueError("Held-out calibration split has fewer than three rows; use more data.")

    denoiser = _model(args.denoiser, args.runtime)
    predictor = _model(args.predictor, args.runtime)
    denoised = denoiser.predict(spectra[calibration_idx, :, None])
    predictions = predictor.predict(denoised)

    bundle = EdgeCalibrationBundle(
        spectral_ood=SpectralOODDetector.fit(spectra[reference_idx], coverage=args.ood_coverage),
        prediction_interval=PredictionIntervalCalibration.fit(
            targets[calibration_idx],
            predictions,
            coverage=args.coverage,
        ),
    )
    bundle.save(args.out)

    metadata = {
        "schema_version": 1,
        "evidence_class": "calibration_data_dependent",
        "runtime": args.runtime,
        "coverage": args.coverage,
        "ood_coverage": args.ood_coverage,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "rows": int(len(frame)),
        "reference_rows": int(reference_idx.size),
        "calibration_rows": int(calibration_idx.size),
        "data": str(args.data),
        "data_sha256": _sha256(args.data),
        "denoiser": str(args.denoiser),
        "denoiser_sha256": _sha256(args.denoiser),
        "predictor": str(args.predictor),
        "predictor_sha256": _sha256(args.predictor),
        "calibration": str(args.out),
        "calibration_sha256": _sha256(args.out),
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
