from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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
from sprpcf.dashboard.operations import (
    PROJECT_ROOT,
    TaskResult,
    artifact_inventory,
    capability_inventory,
    human_bytes,
    run_cli_task,
)
from sprpcf.ml.dataset import read_table
from sprpcf.ml.multiobjective import optimize_target_table
from sprpcf.validation.closed_loop import AcceptanceThresholds, run_closed_loop_iteration


DASHBOARD_PAGES = [
    "Overview",
    "Data & Training",
    "Pipeline & Streaming",
    "HIL Lab",
    "Research Design",
    "Physics Gate",
    "Evidence & Report",
]


def _existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1500px; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
        .spr-hero {
            padding: 1.2rem 1.35rem;
            border: 1px solid rgba(49, 130, 206, .24);
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(17,94,89,.13), rgba(30,64,175,.09));
            margin-bottom: 1rem;
        }
        .spr-hero h1 { margin: 0; font-size: 1.85rem; }
        .spr-hero p { margin: .4rem 0 0; opacity: .78; }
        .spr-kicker { font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .66; }
        .spr-note {
            padding: .75rem .9rem;
            border-radius: 10px;
            background: rgba(128,128,128,.08);
            border: 1px solid rgba(128,128,128,.14);
        }
        div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.14); padding: .75rem; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        """
        <div class="spr-hero">
          <div class="spr-kicker">PCF-SPR · AI inverse design · edge deployment</div>
          <h1>CyberPhotonics-SPR Control Center</h1>
          <p>Generate data, train models, run the full pipeline, benchmark streaming/HIL hardware, and produce reviewer-facing research evidence from one interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _task_history() -> list[dict[str, object]]:
    return st.session_state.setdefault("operations_history", [])


def _remember_task(result: TaskResult) -> None:
    history = _task_history()
    history.insert(0, result.to_dict())
    del history[20:]


def _run_operation(name: str, subcommand: str, arguments: Iterable[str]) -> TaskResult:
    lines: list[str] = []
    with st.status(f"{name} is running", expanded=True) as status:
        st.caption("The dashboard uses the same Python environment and CLI backend as main.py. No shell command is interpolated.")
        output_box = st.empty()

        def on_output(line: str) -> None:
            lines.append(line)
            output_box.code("\n".join(lines[-80:]) or "Starting…", language="text")

        try:
            result = run_cli_task(name, subcommand, arguments, on_output=on_output)
        except Exception as exc:
            status.update(label=f"{name} failed to start", state="error", expanded=True)
            st.exception(exc)
            raise

        if result.output and not lines:
            output_box.code(result.output, language="text")
        if result.success:
            status.update(label=f"{name} completed in {result.elapsed_sec:.1f}s", state="complete", expanded=False)
        else:
            status.update(label=f"{name} failed with exit code {result.returncode}", state="error", expanded=True)
    _remember_task(result)
    if result.success:
        st.success(f"{name} completed successfully.")
    else:
        st.error(f"{name} failed. Review the captured output above.")
    return result


def _render_task_history() -> None:
    history = _task_history()
    if not history:
        return
    with st.expander("Recent dashboard operations", expanded=False):
        summary = pd.DataFrame(
            [
                {
                    "Task": row["name"],
                    "Status": "Success" if row["success"] else "Failed",
                    "Exit": row["returncode"],
                    "Seconds": round(float(row["elapsed_sec"]), 2),
                }
                for row in history
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        latest = history[0]
        if latest.get("output"):
            st.code(str(latest["output"])[-12000:], language="text")


def _overview_page(dataset_text: str, model_dir_text: str, hil_report_text: str) -> None:
    st.subheader("Workspace readiness")
    dataset = Path(dataset_text)
    model_dir = Path(model_dir_text)
    hil_report = Path(hil_report_text)
    inventory = artifact_inventory(dataset, model_dir, hil_report)
    ready = {item.label: item.exists for item in inventory}

    metrics = st.columns(4)
    metrics[0].metric("Dataset", "Ready" if ready.get("Dataset") else "Missing")
    metrics[1].metric("Inverse model", "Ready" if ready.get("Tandem checkpoint") else "Missing")
    edge_ready = ready.get("INT8 denoiser", False) and ready.get("INT8 RI predictor", False)
    metrics[2].metric("Edge INT8", "Ready" if edge_ready else "Missing")
    metrics[3].metric("HIL evidence", "Ready" if ready.get("HIL report") else "Not run")

    st.markdown("#### Artifact inventory")
    rows = []
    for item in inventory:
        rows.append(
            {
                "Artifact": item.label,
                "State": "Ready" if item.exists else "Missing",
                "Path": item.path,
                "Size": human_bytes(item.size_bytes) if item.exists else "—",
                "Modified (UTC)": item.modified_utc or "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Capability matrix")
    capabilities = pd.DataFrame(capability_inventory())
    capabilities["status"] = capabilities["available"].map({True: "Available", False: "Not installed"})
    st.dataframe(
        capabilities[["capability", "status", "module", "required_for_dashboard"]],
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns([1.15, 1.0])
    with left:
        st.markdown("#### Recommended workflow")
        st.markdown(
            """
            1. **Generate Data** — create the fixed-geometry RI sweep dataset.  
            2. **Train Inverse** — train the tandem model and export ONNX.  
            3. **Train Edge** — train denoiser/RI predictor and export TFLite.  
            4. **Pipeline & Streaming** — validate the complete A → B → C flow.  
            5. **HIL Lab** — benchmark mock, serial, or socket hardware.  
            6. **Research Design / Physics Gate** — create calibrated designs and verification evidence.
            """
        )
    with right:
        st.markdown("#### Execution model")
        st.markdown(
            '<div class="spr-note">Training and benchmark jobs run in isolated child Python processes. '
            "This keeps TensorFlow/PyTorch runtime state away from the Streamlit server while preserving the exact same virtual environment and project code.</div>",
            unsafe_allow_html=True,
        )
    _render_task_history()


def _data_training_page(dataset_text: str, model_dir_text: str) -> None:
    st.subheader("Data generation & model training")
    generate_tab, inverse_tab, edge_tab = st.tabs(["Generate Data", "Train Inverse", "Train Edge"])

    with generate_tab:
        st.caption("Generate synthetic PCF-SPR spectra with a fixed geometry and refractive-index sweep.")
        with st.form("generate_data_form"):
            a, b, c = st.columns(3)
            samples = a.number_input("Base geometries", min_value=1, max_value=100000, value=100, step=10)
            wavelengths = b.number_input("Wavelength samples", min_value=32, max_value=8192, value=256, step=32)
            seed = c.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=7, step=1)
            output = st.text_input("Output dataset", value=dataset_text)
            submitted = st.form_submit_button("Generate dataset", type="primary", use_container_width=True)
        if submitted:
            _run_operation(
                "Synthetic data generation",
                "generate-data",
                ["--samples", samples, "--wavelengths", wavelengths, "--seed", seed, "--out", output],
            )

    with inverse_tab:
        st.caption("Train the physics-conditioned tandem inverse model and export an ONNX design model.")
        with st.form("inverse_training_form"):
            data = st.text_input("Training dataset", value=dataset_text, key="inverse_data")
            a, b, c = st.columns(3)
            epochs = a.number_input("Epochs", min_value=1, max_value=10000, value=100, step=10)
            batch_size = b.number_input("Batch size", min_value=1, max_value=4096, value=64, step=16)
            device = c.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
            d, e, f = st.columns(3)
            lr = d.number_input("Learning rate", min_value=0.000001, max_value=1.0, value=0.001, format="%.6f")
            alpha = e.number_input("Alpha", min_value=0.0, value=1.0, step=0.1)
            beta = f.number_input("Beta", min_value=0.0, value=1.0, step=0.1)
            g, h = st.columns(2)
            dispersion_weight = g.number_input("Dispersion weight", min_value=0.0, value=0.0, step=0.01)
            train_seed = h.number_input("Training seed", min_value=0, max_value=2_147_483_647, value=7, step=1)
            advanced_schedule = st.checkbox("Use separate forward/inverse epoch schedule")
            forward_epochs = inverse_epochs = None
            if advanced_schedule:
                s1, s2 = st.columns(2)
                forward_epochs = s1.number_input("Forward epochs", min_value=1, max_value=10000, value=100, step=10)
                inverse_epochs = s2.number_input("Inverse epochs", min_value=1, max_value=10000, value=100, step=10)
            checkpoint = st.text_input("Checkpoint output", value=str(Path(model_dir_text) / "tandem.pt"))
            onnx = st.text_input("ONNX output", value=str(Path(model_dir_text) / "inverse_pcf_spr.onnx"))
            submitted = st.form_submit_button("Train inverse model", type="primary", use_container_width=True)
        if submitted:
            args: list[object] = [
                "--data", data,
                "--epochs", epochs,
                "--batch-size", batch_size,
                "--lr", lr,
                "--device", device,
                "--alpha", alpha,
                "--beta", beta,
                "--dispersion-weight", dispersion_weight,
                "--seed", train_seed,
                "--checkpoint", checkpoint,
                "--export-onnx", onnx,
            ]
            if advanced_schedule and forward_epochs is not None and inverse_epochs is not None:
                args.extend(["--forward-epochs", forward_epochs, "--inverse-epochs", inverse_epochs])
            _run_operation("Tandem inverse training", "train-inverse", [str(value) for value in args])

    with edge_tab:
        st.caption("Train the spectral denoiser and refractive-index predictor, with optional quantized TFLite export.")
        with st.form("edge_training_form"):
            data = st.text_input("Training dataset", value=dataset_text, key="edge_data")
            a, b, c = st.columns(3)
            epochs = a.number_input("Epochs", min_value=1, max_value=10000, value=50, step=10, key="edge_epochs")
            batch_size = b.number_input("Batch size", min_value=1, max_value=4096, value=64, step=16, key="edge_batch")
            device = c.selectbox("Device", ["auto", "cpu", "gpu"], index=0, key="edge_device")
            d, e = st.columns(2)
            seed = d.number_input("Seed", min_value=0, max_value=2_147_483_647, value=7, step=1, key="edge_seed")
            quantize = e.checkbox("Export quantized TFLite", value=True)
            export_dir = st.text_input("Model output directory", value=model_dir_text)
            submitted = st.form_submit_button("Train edge models", type="primary", use_container_width=True)
        if submitted:
            args = [
                "--data", data,
                "--epochs", epochs,
                "--batch-size", batch_size,
                "--device", device,
                "--seed", seed,
                "--export-dir", export_dir,
            ]
            if quantize:
                args.append("--quantize")
            _run_operation("Edge model training", "train-edge", [str(value) for value in args])

    _render_task_history()


def _pipeline_streaming_page(dataset_text: str, model_dir_text: str) -> None:
    st.subheader("End-to-end pipeline & streaming benchmark")
    pipeline_tab, streaming_tab = st.tabs(["Full Pipeline", "Streaming Benchmark"])

    with pipeline_tab:
        st.caption("Run dataset generation → tandem inverse model → edge models → quantized streaming benchmark.")
        with st.form("pipeline_form"):
            a, b, c = st.columns(3)
            samples = a.number_input("Base geometries", min_value=1, max_value=100000, value=100, step=10, key="pipe_samples")
            wavelengths = b.number_input("Wavelength samples", min_value=32, max_value=8192, value=256, step=32, key="pipe_wavelengths")
            seed = c.number_input("Seed", min_value=0, max_value=2_147_483_647, value=7, step=1, key="pipe_seed")
            d, e, f = st.columns(3)
            inverse_epochs = d.number_input("Inverse epochs", min_value=1, max_value=10000, value=100, step=10)
            edge_epochs = e.number_input("Edge epochs", min_value=1, max_value=10000, value=50, step=10)
            batch_size = f.number_input("Batch size", min_value=1, max_value=4096, value=64, step=16, key="pipe_batch")
            g, h = st.columns(2)
            device = g.selectbox("Inverse device", ["auto", "cpu", "cuda"], index=0)
            edge_device = h.selectbox("Edge device", ["auto", "cpu", "gpu"], index=0)
            data = st.text_input("Dataset", value=dataset_text, key="pipe_data")
            export_dir = st.text_input("Model output directory", value=model_dir_text, key="pipe_models")
            i, j, k = st.columns(3)
            duration = i.number_input("Streaming seconds", min_value=0.1, max_value=3600.0, value=10.0, step=1.0)
            noise = j.number_input("Noise std", min_value=0.0, max_value=2.0, value=0.08, step=0.01)
            drift = k.number_input("Drift std", min_value=0.0, max_value=2.0, value=0.03, step=0.01)
            dispersion_weight = st.number_input("Dispersion weight", min_value=0.0, value=0.0, step=0.01, key="pipe_dispersion")
            submitted = st.form_submit_button("Run complete pipeline", type="primary", use_container_width=True)
        if submitted:
            _run_operation(
                "Full A → B → C pipeline",
                "run-pipeline",
                [
                    "--samples", str(samples),
                    "--wavelengths", str(wavelengths),
                    "--seed", str(seed),
                    "--data", data,
                    "--export-dir", export_dir,
                    "--inverse-epochs", str(inverse_epochs),
                    "--edge-epochs", str(edge_epochs),
                    "--batch-size", str(batch_size),
                    "--device", device,
                    "--dispersion-weight", str(dispersion_weight),
                    "--edge-device", edge_device,
                    "--duration-sec", str(duration),
                    "--noise-std", str(noise),
                    "--drift-std", str(drift),
                ],
            )

    with streaming_tab:
        st.caption("Benchmark the exported quantized denoiser and RI predictor against a noisy spectrum stream.")
        with st.form("streaming_form"):
            data = st.text_input("Dataset", value=dataset_text, key="stream_data")
            tflite_dir = st.text_input("TFLite directory", value=model_dir_text)
            a, b, c = st.columns(3)
            duration = a.number_input("Duration (s)", min_value=0.1, max_value=3600.0, value=10.0, step=1.0, key="stream_duration")
            noise = b.number_input("Noise std", min_value=0.0, max_value=2.0, value=0.08, step=0.01, key="stream_noise")
            drift = c.number_input("Drift std", min_value=0.0, max_value=2.0, value=0.03, step=0.01, key="stream_drift")
            submitted = st.form_submit_button("Run streaming benchmark", type="primary", use_container_width=True)
        if submitted:
            _run_operation(
                "Quantized streaming benchmark",
                "simulate-stream",
                [
                    "--data", data,
                    "--tflite-dir", tflite_dir,
                    "--duration-sec", str(duration),
                    "--noise-std", str(noise),
                    "--drift-std", str(drift),
                ],
            )

    _render_task_history()


def _hil_page(model_dir_text: str, hil_report_text: str) -> None:
    st.subheader("Hardware-in-the-loop laboratory")
    st.caption("Benchmark quantized edge inference with mock, serial, or socket transport and optional thermal-drift injection.")
    with st.form("hil_form"):
        protocol = st.radio("Transport", ["mock", "serial", "socket"], horizontal=True)
        tflite_dir = st.text_input("TFLite directory", value=model_dir_text)
        report = st.text_input("Benchmark report", value=hil_report_text)
        a, b, c = st.columns(3)
        duration = a.number_input("Duration (s)", min_value=0.1, max_value=86400.0, value=30.0, step=5.0)
        fps = b.number_input("Target FPS", min_value=0.1, max_value=10000.0, value=30.0, step=1.0)
        buffer_size = c.number_input("Buffer size", min_value=1, max_value=65536, value=256, step=32)
        d, e, f = st.columns(3)
        samples = d.number_input("Synthetic geometries", min_value=1, max_value=100000, value=256, step=32)
        wavelengths = e.number_input("Wavelength samples", min_value=32, max_value=8192, value=256, step=32)
        seed = f.number_input("Seed", min_value=0, max_value=2_147_483_647, value=23, step=1)
        inject_drift = st.checkbox("Inject thermal drift", value=False)
        serial_port = ""
        baudrate = 115200
        socket_host = "127.0.0.1"
        socket_port = 9000
        if protocol == "serial":
            p1, p2 = st.columns(2)
            serial_port = p1.text_input("Serial port", value="COM3")
            baudrate = p2.number_input("Baudrate", min_value=1200, max_value=4_000_000, value=115200, step=9600)
        elif protocol == "socket":
            p1, p2 = st.columns(2)
            socket_host = p1.text_input("Socket host", value="127.0.0.1")
            socket_port = p2.number_input("Socket port", min_value=1, max_value=65535, value=9000, step=1)
        submitted = st.form_submit_button("Run HIL benchmark", type="primary", use_container_width=True)

    if submitted:
        args = [
            "--tflite-dir", tflite_dir,
            "--duration", str(duration),
            "--report", report,
            "--protocol", protocol,
            "--fps", str(fps),
            "--samples", str(samples),
            "--wavelengths", str(wavelengths),
            "--seed", str(seed),
            "--buffer-size", str(buffer_size),
        ]
        if inject_drift:
            args.append("--inject-thermal-drift")
        if protocol == "serial":
            args.extend(["--serial-port", serial_port, "--baudrate", str(baudrate)])
        elif protocol == "socket":
            args.extend(["--socket-host", socket_host, "--socket-port", str(socket_port)])
        _run_operation("HIL benchmark", "hil-benchmark", args)

    report_path = _project_path(report if "report" in locals() else hil_report_text)
    if report_path.exists():
        st.markdown("#### Latest HIL report")
        try:
            st.json(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read {report_path}: {exc}")
    _render_task_history()


def _design_tab(checkpoint_text: str, reference_text: str, output_root: Path, target: pd.DataFrame) -> None:
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


def _physics_tab(checkpoint_text: str, reference_text: str, output_root: Path, target: pd.DataFrame) -> None:
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
        "Hardware benchmark JSON": PROJECT_ROOT / "reports" / "phase4_hil_benchmark.json",
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
        page_title="CyberPhotonics-SPR Control Center",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    _hero()

    with st.sidebar:
        st.markdown("### Control Center")
        page = st.radio("Workspace", DASHBOARD_PAGES, label_visibility="collapsed", key="dashboard_page")
        st.divider()
        st.markdown("#### Project paths")
        dataset_text = st.text_input("Dataset", value="data/processed/synthetic.parquet")
        model_dir_text = st.text_input("Model directory", value="models")
        output_text = st.text_input("Research outputs", value="outputs/dashboard")
        hil_report_text = st.text_input("HIL report", value="reports/phase4_hil_benchmark.json")
        st.caption(f"Project root: {PROJECT_ROOT}")
        if page in {"Research Design", "Physics Gate", "Evidence & Report"}:
            st.divider()
            st.markdown("#### Research target")
            sensitivity = st.number_input("Sensitivity (nm/RIU)", value=800.0, step=10.0)
            fom = st.number_input("FOM (/RIU)", value=20.0, step=0.5)
            lambda_res = st.number_input("Resonance λ (nm)", value=750.0, step=1.0)
            analyte_ri = st.number_input(
                "Analyte RI",
                min_value=1.000001,
                max_value=1.999999,
                value=1.37,
                step=0.001,
            )
        else:
            sensitivity, fom, lambda_res, analyte_ri = 800.0, 20.0, 750.0, 1.37

    dataset_path = _project_path(dataset_text)
    model_dir_path = _project_path(model_dir_text)
    output_root = _project_path(output_text)
    hil_report_path = _project_path(hil_report_text)

    if page == "Overview":
        _overview_page(str(dataset_path), str(model_dir_path), str(hil_report_path))
    elif page == "Data & Training":
        _data_training_page(str(dataset_path), str(model_dir_path))
    elif page == "Pipeline & Streaming":
        _pipeline_streaming_page(str(dataset_path), str(model_dir_path))
    elif page == "HIL Lab":
        _hil_page(str(model_dir_path), str(hil_report_path))
    else:
        st.warning(
            "Evidence boundary: synthetic results validate the software and analytical flow only. "
            "Physical simulation claims require a verified COMSOL model/configuration, and experimental claims require real sensor data."
        )
        try:
            target = target_frame(sensitivity, fom, lambda_res, analyte_ri)
        except ValueError as exc:
            st.error(str(exc))
            return
        checkpoint_text = str(model_dir_path / "tandem.pt")
        reference_text = str(dataset_path)
        if page == "Research Design":
            with st.expander("Research model inputs", expanded=False):
                checkpoint_text = st.text_input("Tandem/ensemble checkpoint", value=checkpoint_text)
                reference_text = st.text_input("Reference dataset", value=reference_text)
            _design_tab(checkpoint_text, reference_text, output_root, target)
        elif page == "Physics Gate":
            with st.expander("Physics model inputs", expanded=False):
                checkpoint_text = st.text_input("Tandem/ensemble checkpoint", value=checkpoint_text, key="physics_checkpoint")
                reference_text = st.text_input("Base dataset", value=reference_text, key="physics_reference")
            _physics_tab(checkpoint_text, reference_text, output_root, target)
        elif page == "Evidence & Report":
            evidence_tab, report_tab = st.tabs(["Evidence Registry", "Research Report"])
            with evidence_tab:
                _evidence_tab(output_root)
            with report_tab:
                _report_tab(output_root, target)


if __name__ == "__main__":
    main()
