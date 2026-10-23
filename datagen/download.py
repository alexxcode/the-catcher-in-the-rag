"""Downloader for external real-world datasets (CUAD v1 and OCDS).

Adheres to Non-negotiable Principle:
- Datasets are downloaded via reproducible scripts; raw corpora are not committed to git.
- Full provenance, licenses, and dates are recorded in DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

CUAD_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"

OCDS_SAMPLE_URLS = [
    (
        "ocds-full-record.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001.json",
    ),
    (
        "ocds-01-planning.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001-01-planning.json",
    ),
    (
        "ocds-02-tender.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001-02-tender.json",
    ),
    (
        "ocds-04-award.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001-04-award.json",
    ),
    (
        "ocds-05-contract.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001-05-contract.json",
    ),
    (
        "ocds-06-implementation.json",
        "https://raw.githubusercontent.com/open-contracting/sample-data/master/fictional-example/1.1/ocds-213czf-000-00001-06-implementation.json",
    ),
]


def download_cuad(output_dir: Path) -> Path:
    """Download and extract CUAD v1 dataset (CC BY 4.0)."""
    cuad_dir = output_dir / "cuad"
    cuad_dir.mkdir(parents=True, exist_ok=True)
    target_file = cuad_dir / "CUADv1.json"

    if target_file.exists():
        print(
            f"CUAD dataset already exists at {target_file} ({target_file.stat().st_size / (1024 * 1024):.1f} MB)"
        )
        return target_file

    print(f"Downloading CUAD v1 archive from {CUAD_URL}...")
    req = urllib.request.Request(
        CUAD_URL,
        headers={"User-Agent": "TheCatcherInTheRAG/0.4.0 (research-audit-tool)"},
    )
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    print(f"Downloaded {len(zip_bytes) / (1024 * 1024):.1f} MB archive. Extracting CUADv1.json...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extract("CUADv1.json", path=cuad_dir)

    print(f"Successfully extracted CUADv1.json to {target_file}")
    return target_file


def download_ocds(output_dir: Path) -> list[Path]:
    """Download OCDS public procurement samples (Apache 2.0 / CC0)."""
    ocds_dir = output_dir / "ocds"
    ocds_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[Path] = []

    print(
        "Downloading OCDS public procurement standard samples from Open Contracting Partnership..."
    )
    for filename, url in OCDS_SAMPLE_URLS:
        target_path = ocds_dir / filename
        if target_path.exists():
            print(f"  [cached] {filename}")
            downloaded_files.append(target_path)
            continue

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TheCatcherInTheRAG/0.4.0"},
            )
            with urllib.request.urlopen(req) as resp:
                content = resp.read()
            target_path.write_bytes(content)
            print(f"  [downloaded] {filename} ({len(content)} bytes)")
            downloaded_files.append(target_path)
        except Exception as err:
            print(f"  [warning] Failed to download {url}: {err}", file=sys.stderr)

    return downloaded_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download real-world external datasets (CUAD & OCDS)."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "cuad", "ocds"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real"),
        help="Directory to store downloaded datasets (default: data/real)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "cuad"):
        download_cuad(args.output_dir)

    if args.dataset in ("all", "ocds"):
        download_ocds(args.output_dir)

    print("\nDataset downloads completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
