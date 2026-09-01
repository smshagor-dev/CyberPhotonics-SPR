# Remote COMSOL API

CyberPhotonics-SPR can run the existing COMSOL physics gate on another machine while the inverse-design and validation pipeline runs locally. The remote machine owns the licensed COMSOL installation, `.mph` model, and COMSOL configuration. The client sends only validated PCF-SPR geometry/RI rows and receives the normal simulation-result table.

## Architecture

```text
CyberPhotonics-SPR client
  -> HTTP API (Bearer token)
  -> remote/local COMSOL host
  -> mph Python bridge
  -> .mph model + configured study
  -> wavelength/loss spectra
  -> CyberPhotonics-SPR closed-loop validation
```

The API is a transport for real COMSOL numerical evidence. It does not convert synthetic data into COMSOL or experimental evidence.

## 1. Prepare the COMSOL host

Install the project with the COMSOL extra in a Python environment that can access the licensed COMSOL installation:

```powershell
python -m pip install -e ".[comsol]"
```

Verify the existing direct COMSOL path first:

```powershell
python -c "import mph; print(mph.__version__)"
```

The server uses the same `run_comsol_geometries()` implementation as the local backend, so the `.mph` model and YAML unit/study contract remain authoritative.

## 2. Start the API server

Recommended authenticated local/remote mode:

```powershell
$env:SPR_COMSOL_API_TOKEN = "replace-with-a-long-random-secret"

python scripts/run_comsol_api_server.py `
  --model "C:\research\pcf_spr.mph" `
  --config "configs\comsol_sweep.example.yaml" `
  --host 127.0.0.1 `
  --port 8765
```

For a same-machine smoke test only, authentication can be disabled explicitly on loopback:

```powershell
python scripts/run_comsol_api_server.py `
  --model "C:\research\pcf_spr.mph" `
  --config "configs\comsol_sweep.example.yaml" `
  --host 127.0.0.1 `
  --port 8765 `
  --allow-unauthenticated-localhost
```

Unauthenticated mode is rejected for non-loopback hosts.

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/healthz
```

Expected fields include `status=ok`, protocol version, authentication state, and the maximum geometry batch size. The health endpoint does not expose local model paths or tokens.

## 3. Run the normal closed loop through the API

In another terminal:

```powershell
$env:SPR_COMSOL_API_URL = "http://127.0.0.1:8765"
$env:SPR_COMSOL_API_TOKEN = "replace-with-a-long-random-secret"

python scripts/run_comsol_closed_loop.py `
  --backend remote-comsol `
  --checkpoint models\tandem.pt `
  --targets data\processed\design_targets.csv `
  --base-data data\processed\training.parquet `
  --out outputs\remote_closed_loop `
  --ri-span 0.04 `
  --ri-points 5
```

The URL can also be supplied with `--remote-comsol-url`. The bearer token is read from `SPR_COMSOL_API_TOKEN` by default and is never written into result artifacts or the iteration manifest.

## 4. Run the uncertainty/Pareto closed loop remotely

```powershell
python scripts/run_advanced_closed_loop.py `
  --backend remote-comsol `
  --checkpoint models\tandem_ensemble.pt `
  --targets data\processed\design_targets.csv `
  --base-data data\processed\training.parquet `
  --out outputs\remote_advanced_closed_loop `
  --candidates-per-target 128 `
  --confidence 0.95 `
  --ri-span 0.04 `
  --ri-points 5
```

The Pareto-selected geometry is sent to the remote host and the returned COMSOL spectra pass through the same fixed-geometry RI sensitivity/FOM/linearity acceptance gates as the local COMSOL backend.

## Security and deployment rules

- Do not commit API tokens, COMSOL credentials, license information, or private `.mph` paths.
- Do not put credentials in `--remote-comsol-url`; embedded URL credentials and query strings are rejected.
- The built-in server is HTTP. For a real remote/cloud deployment, put it behind a TLS reverse proxy/VPN/private network rather than exposing plaintext HTTP to the public internet.
- The API serializes COMSOL execution requests inside one server process to avoid concurrent access to the same runtime/model and unnecessary license pressure.
- Request size, response size, batch count, sample IDs, geometry bounds, and returned row contracts are validated.
- Server-side COMSOL exceptions are logged on the host but only a generic execution error is returned to clients.

## Evidence/provenance behavior

Remote COMSOL is recorded as `comsol_physics`, not `software_only`. The client manifest also records `execution_transport=remote_comsol_api`, the sanitized base URL, protocol version, and token environment-variable name. The token value is never persisted.

The API response includes hashes of the server-side `.mph` model and YAML configuration when the standard server launcher is used. These values can be retained with the research evidence package to bind remote numerical results to the exact physics model/configuration.

## Current API contract

### `GET /healthz`

Returns service readiness metadata.

### `POST /v1/simulations/geometries`

Request:

```json
{
  "schema_version": 1,
  "geometries": [
    {
      "sample_id": 0,
      "pitch_um": 2.1,
      "d_over_lambda": 0.55,
      "metal_thickness_nm": 45.0,
      "channel_radius_um": 0.6,
      "analyte_ri": 1.33
    }
  ]
}
```

Successful response:

```json
{
  "schema_version": 1,
  "results": [
    {
      "sample_id": 0,
      "status": "ok",
      "pitch_um": 2.1,
      "d_over_lambda": 0.55,
      "metal_thickness_nm": 45.0,
      "channel_radius_um": 0.6,
      "analyte_ri": 1.33,
      "lambda_res_nm": 710.4,
      "fwhm_nm": 21.2,
      "wavelength_nm": "...",
      "loss_db_per_cm": "..."
    }
  ],
  "provenance": {
    "evidence_class": "comsol_physics",
    "model_sha256": "...",
    "config_sha256": "..."
  }
}
```

A failed COMSOL sample remains a returned row with a failure `status`; unavailable numeric fields are serialized as JSON `null` rather than invalid `NaN` tokens.
