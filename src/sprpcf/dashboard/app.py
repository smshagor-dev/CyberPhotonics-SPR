from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sprpcf.dashboard.core import (
    evidence_sha256,
    geometry_figure,
    load_json_if_exists,
    research_report_markdown,
    spectrum_figure,
    target_frame,
    xai_feature_summary,
)
from sprpcf.ml.dataset import read_table
from sprpcf.ml.multiobjective import optimize_target_table
from sprpcf.validation.closed_loop import AcceptanceThresholds, run_closed_loop_iteration


def _existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _save_design_outputs(output_dir: Path, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.candidates.to_csv(output_dir / "pareto_candidates.csv", index=False)
    result.selected.to_csv(output_dir / "pareto_selected_designs.csv", index=False)
    (output_dir / "design_calibration.json").write_text(
        json.dumps(result.calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _metrics_strip(selected: pd.Series) -> None:
    columns = st.columns(5)
    columns[0].metric("Confidence", f"{float(selected.get('confidence_score', float('nan'))):.3f}")
    columns[1].metric("OOD score", f"{float(selected.get('ood_score', float('nan'))):.3f}")
    columns[2].metric("Pareto rank", int(selected.get("pareto_rank", -1)))
    columns[3].metric("Pitch (µm)", f"{float(selected['pitch_um']):.3f}")
    columns[4].metric("Metal (nm)", f"{float(selected['metal_thickness_nm']):.2f}")


def _design_tab(
    checkpoint_text: str,
    reference_text: str,
    output_root: Path,
    target: pd.DataFrame,
) -> None:
    st.subheader("AI inverse design")
    st.caption(
        "Generates a latent candidate population, applies fabrication projection, forward-ensemble/conformal "
        "evaluation, OOD scoring, and Pareto ranking."
    )
    col1, col2, col3 = st.columns(3)
    candidates = col1.number_input("Candidates", min_value=4, max_value=2048, value=128, step=4)
    confidence = col2.slider("Calibration confidence", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    latent_scale = col3.number_input("Latent scale", min_value=0.001, max_value=1.0, value=0.10, step=0.01)
    seed = st.number_input("Design seed", min_value=0, max_value=2_147_483_647, value=7, step=1)

    if st.button("Generate calibrated design", type="primary", use_container_width=True):
        try:
            checkpoint = _existing_path(checkpoint_text, "Checkpoint")
            reference = _existing_path(reference_text, "Reference dataset")
            result = optimize_target_table(
                checkpoint,
                target,
                reference,
                candidates_per_target=int(candidates),
                confidence=float(confidence),
                latent_scale=float(latent_scale),
                seed=int(seed),
                device="cpu",
            )
            design_dir = output_root / "design"
            _save_design_outputs(design_dir, result)
            st.session_state["design_result"] = result
            st.session_state["target_frame"] = target
            st.success(f"Selected 1 design from {len(result.candidates)} candidates.")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("design_result")
    if result is None:
        st.info("Generate a calibrated design to populate the geometry and Pareto evidence.")
        return

    selected = result.selected.iloc[0]
    _metrics_strip(selected)
    left, right = st.columns([1.0, 1.1])
    with left:
        st.pyplot(geometry_figure(selected), use_container_width=True)
        plt.close("all")
    with right:
        display_columns = [
            "candidate_id",
            "pareto_rank",
            "confidence_score",
            "ood_score",
            "in_calibration_domain",
            "fabrication_projection_distance",
            "pitch_um",
            "d_over_lambda",
            "metal_thickness_nm",
            "channel_radius_um",
            "predicted_sensitivity_nm_per_riu",
            "predicted_fom_per_riu",
            "predicted_lambda_res_nm",
        ]
        available = [column for column in display_columns if column in result.candidates.columns]
        st.dataframe(
            result.candidates.sort_values(["pareto_rank", "composite_score"])[available].head(32),
            use_container_width=True,
            hide_index=True,
        )
        st.json(result.calibration)


def _physics_tab(
    checkpoint_text: str,
    reference_text: str,
    output_root: Path,
    target: pd.DataFrame,
) -> None:
    st.subheader("Physics verification gate")
    result = st.session_state.get("design_result")
    if result is None:
        st.info("Generate a design first. The physics gate preserves the exact selected candidate.")
        return

    selected = result.selected.copy()
    backend = st.radio("Backend", ["synthetic", "comsol"], horizontal=True)
    ri_col, points_col = st.columns(2)
    ri_span = ri_col.number_input("RI sweep span", min_value=0.001, max_value=0.5, value=0.04, step=0.005)
    ri_points = points_col.number_input("RI points", min_value=3, max_value=21, value=5, step=2)

    with st.expander("Reviewer-facing acceptance thresholds"):
        a, b, c, d = st.columns(4)
        max_sensitivity = a.number_input("Sensitivity error", min_value=0.0, value=150.0)
        max_fom = b.number_input("FOM error", min_value=0.0, value=5.0)
        max_lambda = c.number_input("λ error (nm)", min_value=0.0, value=30.0)
        min_r2 = d.number_input("Min linearity R²", min_value=0.0, max_value=1.0, value=0.95)

    model_text = ""
    config_text = ""
    if backend == "comsol":
        model_text = st.text_input("COMSOL .mph model", value="path/to/pcf_spr.mph")
        config_text = st.text_input("COMSOL sweep YAML", value="sweep.example.yaml")

    if st.button("Verify exact selected design", type="primary", use_container_width=True):
        try:
            checkpoint = _existing_path(checkpoint_text, "Checkpoint")
            base_dataset = _existing_path(reference_text, "Base dataset")
            physics_dir = output_root / "physics"
            physics_dir.mkdir(parents=True, exist_ok=True)
            target_path = physics_dir / "dashboard_target.csv"
            target.to_csv(target_path, index=False)

            model_path = None
            config_path = None
            if backend == "comsol":
                model_path = _existing_path(model_text, "COMSOL model")
                config_path = _existing_path(config_text, "COMSOL config")

            thresholds = AcceptanceThresholds(
                max_sensitivity_error_nm_per_riu=float(max_sensitivity),
                max_fom_error_per_riu=float(max_fom),
                max_lambda_error_nm=float(max_lambda),
                min_linearity_r2=float(min_r2),
            )

            def fixed_designer(_: pd.DataFrame) -> pd.DataFrame:
                return selected.copy()

            artifacts = run_closed_loop_iteration(
                checkpoint_path=checkpoint,
                target_path=target_path,
                base_dataset_path=base_dataset,
                output_dir=physics_dir,
                backend=backend,
                model_path=model_path,
                config_path=config_path,
                ri_span=float(ri_span),
                ri_points=int(ri_points),
                thresholds=thresholds,
                device="cpu",
                seed=7,
                designer=fixed_designer,
                retrain=False,
            )
            verification = read_table(artifacts.verification_results)
            simulation = read_table(artifacts.simulation_results)
            st.session_state["physics_artifacts"] = artifacts
            st.session_state["verification"] = verification
            st.session_state["simulation"] = simulation
            st.session_state["physics_backend"] = backend
            if not verification.empty and bool(verification.iloc[0]["accepted"]):
                st.success("Physics gate accepted the selected design.")
            else:
                st.warning("Physics gate rejected the selected design. See evidence below.")
        except Exception as exc:
            st.exception(exc)

    verification = st.session_state.get("verification")
    simulation = st.session_state.get("simulation")
    if verification is None:
        st.info("Run the physics gate to produce RI-sweep evidence.")
        return

    st.dataframe(verification, use_container_width=True, hide_index=True)
    if simulation is not None and not simulation.empty:
        try:
            st.pyplot(spectrum_figure(simulation), use_container_width=True)
            plt.close("all")
        except ValueError as exc:
            st.warning(str(exc))


def _evidence_tab(output_root: Path) -> None:
    st.subheader("Edge, XAI, validation & provenance evidence")
    st.caption("Loads existing evidence files. It never converts synthetic evidence into physical claims.")

    defaults = {
        "Scientific validation JSON": output_root.parent / "validation" / "summary.json",
        "XAI attribution CSV": output_root.parent / "feature_attribution.csv",
        "Hardware benchmark JSON": output_root.parent / "hardware" / "benchmark.json",
    }
    evidence_hashes: dict[str, str] = {}
    for label, default in defaults.items():
        path_text = st.text_input(label, value=str(default), key=f"evidence_{label}")
        path = Path(path_text).expanduser()
        if not path.exists():
            st.caption(f"Not found: {path}")
            continue
        evidence_hashes[label] = evidence_sha256(path)
        if path.suffix.lower() == ".json":
            try:
                payload = load_json_if_exists(path)
                if payload is not None:
                    st.json(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                st.warning(str(exc))
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            st.dataframe(frame.head(64), use_container_width=True, hide_index=True)
            try:
                summary = xai_feature_summary(frame)
                st.bar_chart(summary.set_index("feature"))
            except ValueError:
                pass
    st.session_state["evidence_hashes"] = evidence_hashes


def _report_tab(output_root: Path, target: pd.DataFrame) -> None:
    st.subheader("Export reviewer-readable evidence report")
    result = st.session_state.get("design_result")
    verification = st.session_state.get("verification")
    backend = st.session_state.get("physics_backend")
    selected = result.selected.iloc[0] if result is not None else None
    verification_row = verification.iloc[0] if verification is not None and not verification.empty else None
    report = research_report_markdown(
        target.iloc[0],
        selected,
        verification_row,
        backend=backend,
        evidence=st.session_state.get("evidence_hashes", {}),
    )
    st.code(report, language="markdown")
    st.download_button(
        "Download Markdown evidence report",
        data=report,
        file_name="cyberphotonics_dashboard_evidence.md",
        mime="text/markdown",
        use_container_width=True,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if st.button("Write report to outputs", use_container_width=True):
        path = output_root / "dashboard_evidence_report.md"
        path.write_text(report, encoding="utf-8")
        st.success(f"Wrote {path}")


def main() -> None:
    st.set_page_config(
        page_title="CyberPhotonics-SPR Research Dashboard",
        page_icon="🔬",
        layout="wide",
    )
    st.title("CyberPhotonics-SPR")
    st.caption("AI inverse design → exact-candidate physics gate → spectra → edge/XAI evidence → reproducible report")
    st.warning(
        "Evidence rule: synthetic backend results validate software flow only. "
        "Physical simulation claims require a verified COMSOL model/configuration; experimental claims require real sensor data."
    )

    with st.sidebar:
        st.header("Research workspace")
        checkpoint_text = st.text_input("Tandem/ensemble checkpoint", value="models/tandem_ensemble.pt")
        reference_text = st.text_input("Reference/base dataset", value="data/processed/training.parquet")
        output_text = st.text_input("Dashboard output directory", value="outputs/dashboard")
        st.divider()
        st.subheader("Target")
        sensitivity = st.number_input("Sensitivity (nm/RIU)", value=800.0, step=10.0)
        fom = st.number_input("FOM (/RIU)", value=20.0, step=0.5)
        lambda_res = st.number_input("Resonance λ (nm)", value=750.0, step=1.0)
        analyte_ri = st.number_input("Analyte RI", min_value=1.000001, max_value=1.999999, value=1.37, step=0.001)

    try:
        target = target_frame(sensitivity, fom, lambda_res, analyte_ri)
    except ValueError as exc:
        st.error(str(exc))
        return
    output_root = Path(output_text).expanduser()

    design, physics, evidence, report = st.tabs(
        ["1 · Design Studio", "2 · Physics Gate", "3 · Edge / XAI Evidence", "4 · Report"]
    )
    with design:
        _design_tab(checkpoint_text, reference_text, output_root, target)
    with physics:
        _physics_tab(checkpoint_text, reference_text, output_root, target)
    with evidence:
        _evidence_tab(output_root)
    with report:
        _report_tab(output_root, target)


if __name__ == "__main__":
    main()
