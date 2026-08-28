from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf import __version__
from sprpcf.publication.evidence import EvidenceSource, build_reviewer_package
from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.synthetic import DEFAULT_ANALYTE_RI, build_synthetic_dataset
from sprpcf.utils.reproducibility import create_reproducibility_bundle
from sprpcf.validation.benchmark import run_validation_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic software-only publication demo and reviewer package."
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/publication_demo"))
    parser.add_argument("--samples", type=int, default=24, help="Base geometries; each uses the fixed RI sweep.")
    parser.add_argument("--wavelengths", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def _write_demo_index(out: Path, dataset_rows: int, geometry_sweeps: int, reviewer_artifacts: int) -> None:
    figure_links = []
    for filename, label in (
        ("resonance_shift.png", "Resonance shift"),
        ("sensitivity_distribution.png", "Sensitivity distribution"),
    ):
        if (out / "validation" / filename).is_file():
            figure_links.append(
                f'<figure><img src="validation/{filename}" alt="{label}"><figcaption>{label} — synthetic demo</figcaption></figure>'
            )
    figures = "\n".join(figure_links) or "<p>No demo figures were generated.</p>"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyberPhotonics-SPR Publication Demo</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1080px;margin:0 auto;padding:32px;line-height:1.55;color:#1f2937}}
.hero{{padding:28px;border:1px solid #d1d5db;border-radius:16px;background:#f8fafc}}
.badge{{display:inline-block;padding:5px 10px;border-radius:999px;background:#fff7ed;border:1px solid #fdba74;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:22px 0}}
.card{{padding:16px;border:1px solid #e5e7eb;border-radius:12px}}
.present{{font-weight:700}} .missing{{font-weight:700}}
figure{{margin:28px 0}} img{{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:10px}}
code{{background:#f3f4f6;padding:2px 5px;border-radius:4px}}
a{{color:inherit}}
</style>
</head>
<body>
<section class="hero">
<span class="badge">SOFTWARE-ONLY DEMO</span>
<h1>CyberPhotonics-SPR Publication & Reviewer Evidence Demo</h1>
<p>This static page demonstrates the evidence packaging workflow. It does not claim COMSOL, laboratory, fabricated-sensor, or target-device performance.</p>
</section>
<div class="grid">
<div class="card"><strong>Dataset rows</strong><br>{dataset_rows}</div>
<div class="card"><strong>Fixed-geometry sweeps</strong><br>{geometry_sweeps}</div>
<div class="card"><strong>Reviewer artifacts</strong><br>{reviewer_artifacts}</div>
<div class="card"><strong>Package version</strong><br>{__version__}</div>
</div>
<h2>Evidence status</h2>
<div class="grid">
<div class="card"><span class="present">Software/synthetic evidence: present</span></div>
<div class="card"><span class="missing">COMSOL physics evidence: not supplied</span></div>
<div class="card"><span class="missing">Experimental sensor evidence: not supplied</span></div>
<div class="card"><span class="missing">Target-device benchmark: not supplied</span></div>
</div>
<h2>Reviewer entry points</h2>
<ul>
<li><a href="reviewer_package/REVIEWER_GUIDE.md">Reviewer guide</a></li>
<li><a href="reviewer_package/CLAIMS_MATRIX.md">Claims-to-evidence matrix</a></li>
<li><a href="reviewer_package/artifact_index.csv">Artifact index</a></li>
<li><a href="reviewer_package/manifest.json">Machine-readable manifest</a></li>
<li><a href="reviewer_package/checksums.sha256">SHA-256 checksums</a></li>
<li><a href="validation/validation_report.md">Scientific validation report</a></li>
</ul>
<h2>Demo figures</h2>
{figures}
<h2>Interpretation</h2>
<p>The demo is deliberately conservative. Synthetic and surrogate outputs validate software behavior and reporting methodology only. Physical performance requires separately supplied COMSOL or experimental evidence, and deployment performance requires measurements on the exact device.</p>
</body>
</html>
"""
    (out / "DEMO_INDEX.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 4:
        raise SystemExit("--samples must be >= 4.")
    if args.wavelengths < 32:
        raise SystemExit("--wavelengths must be >= 32.")
    if args.bootstrap_resamples < 100:
        raise SystemExit("--bootstrap-resamples must be >= 100.")

    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Demo output must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = out / "demo_synthetic.parquet"
    validation_dir = out / "validation"
    reproducibility_dir = out / "reproducibility"
    reviewer_dir = out / "reviewer_package"

    frame = build_synthetic_dataset(
        args.samples,
        wavelengths=args.wavelengths,
        seed=args.seed,
    )
    write_dataset(
        frame,
        dataset_path,
        metadata={
            "source": "synthetic",
            "evidence_class": "software_only",
            "purpose": "publication_demo",
            "seed": args.seed,
            "base_geometries": args.samples,
            "wavelength_samples": args.wavelengths,
            "analyte_ri_values": list(DEFAULT_ANALYTE_RI),
        },
    )

    summary = run_validation_pack(
        data_path=dataset_path,
        output_dir=validation_dir,
        checkpoint_path=None,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    create_reproducibility_bundle(
        reproducibility_dir,
        experiment_name="publication_demo_software_only",
        seed=args.seed,
        repo_root=args.repo_root,
        config={
            "sprpcf_version": __version__,
            "evidence_class": "software_only",
            "base_geometries": args.samples,
            "wavelength_samples": args.wavelengths,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        artifacts=(
            ("demo_dataset", dataset_path),
            ("validation_summary", validation_dir / "summary.json"),
            ("validation_provenance", validation_dir / "provenance.json"),
            ("validation_report", validation_dir / "validation_report.md"),
        ),
        notes=(
            "Deterministic publication demo using synthetic spectra only. "
            "It validates the software/reporting workflow and must not be cited as COMSOL or experimental performance."
        ),
    )

    manifest = build_reviewer_package(
        reviewer_dir,
        title="CyberPhotonics-SPR Publication Demo Reviewer Package",
        version=__version__,
        repo_root=args.repo_root,
        sources=(
            EvidenceSource(
                role="demo_dataset",
                path=dataset_path,
                evidence_class="software_only",
                label="Synthetic demo dataset",
            ),
            EvidenceSource(
                role="validation",
                path=validation_dir,
                evidence_class="software_only",
                label="Synthetic scientific-validation demo",
            ),
            EvidenceSource(
                role="reproducibility",
                path=reproducibility_dir,
                evidence_class="reproducibility",
                label="Demo reproducibility bundle",
            ),
        ),
    )

    geometry_sweeps = int(summary["fixed_geometry_sweeps"]["geometry_sweeps"])
    demo_summary = {
        "schema_version": 1,
        "sprpcf_version": __version__,
        "evidence_class": "software_only",
        "seed": args.seed,
        "base_geometries": args.samples,
        "dataset_rows": len(frame),
        "fixed_geometry_sweeps": geometry_sweeps,
        "reviewer_artifacts": len(manifest["artifacts"]),
        "reviewer_evidence_classes": manifest["evidence_classes"],
        "comsol_physics_supplied": False,
        "experimental_sensor_supplied": False,
        "device_benchmark_supplied": False,
    }
    (out / "demo_summary.json").write_text(
        json.dumps(demo_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    demo_readme = [
        "# CyberPhotonics-SPR Publication Demo",
        "",
        "This directory is a deterministic **software-only** demonstration of the publication/reviewer evidence workflow.",
        "",
        "Open `DEMO_INDEX.html` for the polished static demo, or start reviewer inspection at `reviewer_package/REVIEWER_GUIDE.md`.",
        "",
        "It demonstrates:",
        "",
        "- fixed-geometry validation tables,",
        "- bootstrap statistics,",
        "- manuscript-ready figures,",
        "- provenance and environment records,",
        "- SHA-256-bound reviewer artifacts,",
        "- a claims-to-evidence matrix.",
        "",
        "## Scientific boundary",
        "",
        (
            "The demo dataset is synthetic. The generated sensitivity/FOM/model-validation outputs demonstrate software "
            "execution and reporting only; they are not COMSOL validation, laboratory measurements, detection limits, "
            "fabrication results, or target-device benchmark evidence."
        ),
        "",
    ]
    (out / "README.md").write_text("\n".join(demo_readme), encoding="utf-8")
    _write_demo_index(
        out,
        dataset_rows=len(frame),
        geometry_sweeps=geometry_sweeps,
        reviewer_artifacts=len(manifest["artifacts"]),
    )

    print(json.dumps(demo_summary, indent=2))


if __name__ == "__main__":
    main()
