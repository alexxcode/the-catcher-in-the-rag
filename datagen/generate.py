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
        "--count",
        type=int,
        default=300,
        help="Total number of packages across all 10 anomaly classes (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )

    args = parser.parse_args()
    count_per_anomaly = max(1, args.count // 10)

    print(
        f"Generating {count_per_anomaly * 10} packages ({count_per_anomaly} per anomaly class) "
        f"into {args.output_dir} (seed={args.seed})..."
    )
    package_ids = generate_dataset(
        output_dir=args.output_dir,
        count_per_anomaly=count_per_anomaly,
        base_seed=args.seed,
    )
    print(f"Successfully generated {len(package_ids)} packages. Index saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
