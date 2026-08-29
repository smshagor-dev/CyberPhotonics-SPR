from __future__ import annotations

import json

import pyarrow.parquet as pq

from sprpcf.simulation.bigdata import write_large_synthetic_dataset


def test_large_dataset_writer_streams_complete_ri_groups(tmp_path) -> None:
    output = tmp_path / "large.parquet"
    metadata = write_large_synthetic_dataset(
        samples=7,
        wavelengths=32,
        seed=11,
        output=output,
        chunk_size=3,
    )

    assert output.exists()
    assert not output.with_name(output.name + ".part").exists()
    assert pq.ParquetFile(output).metadata.num_rows == 35

    frame = pq.read_table(output).to_pandas()
    assert len(frame) == 35
    assert frame["sample_id"].tolist() == list(range(35))
    assert frame["sensitivity_nm_per_riu"].notna().all()
    assert frame["fom_per_riu"].notna().all()

    sidecar = json.loads(output.with_suffix(".parquet.meta.json").read_text(encoding="utf-8"))
    assert sidecar["rows"] == 35
    assert sidecar["base_geometries"] == 7
    assert sidecar["chunk_size"] == 3
    assert sidecar["generation_mode"] == "chunked"
    assert sidecar["sha256"] == metadata["sha256"]
    assert sidecar["size_bytes"] == output.stat().st_size
