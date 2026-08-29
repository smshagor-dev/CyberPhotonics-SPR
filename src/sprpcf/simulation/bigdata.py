from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sprpcf.simulation.synthetic import DEFAULT_ANALYTE_RI, build_synthetic_dataset


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_large_synthetic_dataset(
    *,
    samples: int,
    wavelengths: int,
    seed: int,
    output: Path,
    chunk_size: int = 500,
) -> dict[str, object]:
    """Generate a large synthetic dataset without holding the full table in RAM.

    ``samples`` is the number of base geometries. Each base geometry receives the
    configured five-point RI sweep, so the default row count is ``samples * 5``.
    Data is written to a temporary ``.part`` file and atomically promoted only
    after every chunk succeeds.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if wavelengths < 32:
        raise ValueError("wavelengths must be >= 32")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    part.unlink(missing_ok=True)

    total_rows = 0
    columns: list[str] = []
    writer: pq.ParquetWriter | None = None
    csv_header = True
    ri_count = len(DEFAULT_ANALYTE_RI)

    try:
        for chunk_index, start in enumerate(range(0, samples, chunk_size)):
            count = min(chunk_size, samples - start)
            chunk_seed = int(seed + chunk_index * 1_000_003)
            frame = build_synthetic_dataset(samples=count, wavelengths=wavelengths, seed=chunk_seed)
            frame["sample_id"] = frame["sample_id"].astype("int64") + start * ri_count
            columns = list(frame.columns)

            if output.suffix.lower() == ".parquet":
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        part,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table)
            else:
                frame.to_csv(part, mode="w" if csv_header else "a", header=csv_header, index=False)
                csv_header = False

            total_rows += len(frame)
            completed = start + count
            percent = completed / samples * 100.0
            print(
                f"DATASET_PROGRESS {completed}/{samples} base geometries "
                f"({percent:.1f}%) · {total_rows:,} rows",
                flush=True,
            )
            del frame

        if writer is not None:
            writer.close()
            writer = None

        part.replace(output)
        digest = _sha256_file(output)
        metadata: dict[str, object] = {
            "schema_version": 1,
            "source": "synthetic",
            "generation_mode": "chunked",
            "seed": seed,
            "base_geometries": samples,
            "rows": total_rows,
            "columns": columns,
            "wavelength_samples": wavelengths,
            "analyte_ri_values": list(DEFAULT_ANALYTE_RI),
            "chunk_size": chunk_size,
            "sha256": digest,
            "size_bytes": output.stat().st_size,
        }
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"Wrote {total_rows:,} rows ({samples:,} base geometries) to {output} "
            f"[{output.stat().st_size / (1024 * 1024):.1f} MiB]",
            flush=True,
        )
        return metadata
    except BaseException:
        if writer is not None:
            writer.close()
        part.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a RAM-safe large synthetic PCF-SPR dataset.")
    parser.add_argument("--samples", type=int, default=10_000, help="Number of base geometries (five RI rows each).")
    parser.add_argument("--wavelengths", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/processed/synthetic.parquet"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_large_synthetic_dataset(
        samples=args.samples,
        wavelengths=args.wavelengths,
        seed=args.seed,
        output=args.out,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
