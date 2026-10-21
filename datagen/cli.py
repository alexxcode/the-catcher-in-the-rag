from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datagen.generator import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic procurement packages with labeled anomalies."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory where packages will be written (default: data/synthetic)",
    )
    parser.add_argument(
        "--count-per-anomaly",
        type=int,
        default=30,
        help="Number of packages per anomaly class (default: 30, total 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )

    args = parser.parse_args()

    print(
        f"Generating {args.count_per_anomaly * 10} packages across 10 anomaly classes "
        f"into {args.output_dir} (seed={args.seed})..."
    )
    package_ids = generate_dataset(
        output_dir=args.output_dir,
        count_per_anomaly=args.count_per_anomaly,
        base_seed=args.seed,
    )
    print(f"Successfully generated {len(package_ids)} packages. Index saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
