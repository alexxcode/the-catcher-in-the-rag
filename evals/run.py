from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catcher_in_the_rag.schemas import (
    PackageLabels,
    ProcurementPackage,
)
from datagen.generator import generate_dataset
from evals.extraction_eval import ExtractionBenchmarkRunner, print_extraction_summary
from evals.report_writer import print_summary_table, save_report
from evals.runner import EvaluationRunner


def load_dataset(dataset_dir: Path) -> list[tuple[ProcurementPackage, PackageLabels]]:
    """Load procurement packages and ground-truth labels from a directory."""
    packages: list[tuple[ProcurementPackage, PackageLabels]] = []

    # If dataset directory does not exist or has no packages, auto-generate a baseline set
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print(
            f"Dataset directory '{dataset_dir}' empty or missing. Generating 30 packages for evaluation..."
        )
        generate_dataset(output_dir=dataset_dir, count_per_anomaly=3, base_seed=42)

    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir():
            pkg_file = item / "package.json"
            lbl_file = item / "labels.json"
            if pkg_file.exists() and lbl_file.exists():
                try:
                    pkg_data = json.loads(pkg_file.read_text(encoding="utf-8"))
                    lbl_data = json.loads(lbl_file.read_text(encoding="utf-8"))
                    pkg = ProcurementPackage.model_validate(pkg_data)
                    lbl = PackageLabels.model_validate(lbl_data)
                    packages.append((pkg, lbl))
                except Exception as err:
                    print(f"Warning: Failed to load package in {item}: {err}")

    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description="Run The Catcher in the RAG evaluation harness.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/reports"),
        help="Directory where evaluation reports will be saved (default: evals/reports)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Path to synthetic dataset directory (default: data/synthetic)",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run ablation experiments (baseline vs no-RAG)",
    )
    parser.add_argument(
        "--eval-extraction",
        action="store_true",
        help="Run field-by-field contract extraction benchmark from raw text (Phase 2)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional maximum number of packages to evaluate",
    )

    args = parser.parse_args()

    print(f"Loading evaluation dataset from {args.dataset_dir}...")
    dataset = load_dataset(args.dataset_dir)
    if not dataset:
        print("Error: No valid packages could be loaded for evaluation.", file=sys.stderr)
        return 1

    if args.sample_size and args.sample_size < len(dataset):
        dataset = dataset[: args.sample_size]

    print(f"Loaded {len(dataset)} package(s). Running baseline evaluation...")
    runner = EvaluationRunner(mode="baseline")
    baseline_report = runner.run_eval(dataset)
    report_file = save_report(baseline_report, output_dir=args.output)
    print_summary_table(baseline_report)
    print(f"Baseline report saved to: {report_file}")

    if args.ablations:
        print("\nRunning Ablation: No-RAG (Disabling Contract Retrieval & Clause Application)...")
        no_rag_runner = EvaluationRunner(mode="no-rag")
        no_rag_report = no_rag_runner.run_eval(dataset)
        no_rag_file = save_report(no_rag_report, output_dir=args.output)
        print_summary_table(no_rag_report)
        print(f"Ablation (no-RAG) report saved to: {no_rag_file}")

        print("\n" + "=" * 78)
        print(" ABLATION COMPARISON: IMPACTO DE TÉRMINOS CONTRACTUALES")
        print("=" * 78)
        print(
            f" Baseline (Con Términos Contractuales):  {baseline_report.decision_accuracy * 100:.1f}%"
        )
        print(
            f" Ablación (Solo 3-Way Match, Sin Términos): {no_rag_report.decision_accuracy * 100:.1f}%"
        )
        delta = (baseline_report.decision_accuracy - no_rag_report.decision_accuracy) * 100
        print(f" Valor de los términos contractuales estructurados: +{delta:.1f}%")
        print(
            " [Nota: La ablación actual evalúa la aplicación de reglas sobre campos estructurados"
        )
        print("  del contrato vs cotejo tripartito básico. Ver StateOfCatcher.md para detalles.]")
        print("=" * 78)

    if args.eval_extraction:
        print("\nRunning Field-by-Field Unstructured Contract Extraction Benchmark (Phase 2)...")
        ext_runner = ExtractionBenchmarkRunner(extractor_type="rule")
        ext_report = ext_runner.evaluate_dataset(dataset)
        print_extraction_summary(ext_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
