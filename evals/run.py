from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from catcher_in_the_rag.schemas import (
    PackageLabels,
    ProcurementPackage,
)
from datagen.generator import generate_dataset, generate_diverse_contracts
from evals.bakeoff import (
    ExtractionBakeOff,
    print_bakeoff_summary,
    save_bakeoff_report,
)
from evals.calibrate_bartleby import (
    BartlebyCalibrator,
    apply_calibrated_thresholds,
    print_calibration_summary,
    save_calibration_report,
)
from evals.cuad_benchmark import (
    CUADBenchmarkRunner,
    print_cuad_summary,
    save_cuad_report,
)
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
        "--bakeoff",
        action="store_true",
        help="Run extraction bake-off: Flash vs Pro vs Rule-based baseline (Phase 3)",
    )
    parser.add_argument(
        "--calibrate-bartleby",
        action="store_true",
        help="Run empirical threshold sweep for Bartleby on A09 vs non-A09 packages (Phase 4)",
    )
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Automatically apply optimal calibrated thresholds to config/thresholds.yaml",
    )
    parser.add_argument(
        "--eval-cuad",
        action="store_true",
        help="Run real-world CUAD commercial contract retrieval and Bartleby abstention benchmark",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional maximum number of packages to evaluate",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        default=True,
        help="Use diverse contract phrasings for extraction bake-off (demonstrating LLM vs Regex)",
    )
    parser.add_argument(
        "--standard-style",
        dest="diverse",
        action="store_false",
        help="Use standard template contract phrasings instead of diverse styles",
    )

    args = parser.parse_args()

    has_subcommand = (
        args.eval_extraction or args.bakeoff or args.calibrate_bartleby or args.eval_cuad
    )

    dataset: list[Any] = []
    if not has_subcommand or args.ablations or args.calibrate_bartleby or args.eval_extraction:
        print(f"Loading evaluation dataset from {args.dataset_dir}...")
        dataset = load_dataset(args.dataset_dir)
        if not dataset:
            print("Error: No valid packages could be loaded for evaluation.", file=sys.stderr)
            return 1
        if args.sample_size and args.sample_size < len(dataset):
            dataset = dataset[: args.sample_size]

    if not has_subcommand or args.ablations:
        print(f"Loaded {len(dataset)} package(s). Running baseline evaluation...")
        runner = EvaluationRunner(mode="baseline")
        baseline_report = runner.run_eval(dataset)
        report_file = save_report(
            baseline_report,
            output_dir=args.output,
            canonical_name="canonical_baseline_report.json" if args.ablations else None,
        )
        print_summary_table(baseline_report)
        print(f"Baseline report saved to: {report_file}")
    else:
        # Load dataset if synthetic subcommands need it
        if args.eval_extraction or args.bakeoff or args.calibrate_bartleby:
            dataset = load_dataset(args.dataset_dir)
            if args.sample_size and args.sample_size < len(dataset):
                dataset = dataset[: args.sample_size]

    if args.ablations:
        print("\nRunning Ablation: No-RAG (Disabling Contract Retrieval & Clause Application)...")
        no_rag_runner = EvaluationRunner(mode="no-rag")
        no_rag_report = no_rag_runner.run_eval(dataset)
        no_rag_file = save_report(
            no_rag_report, output_dir=args.output, canonical_name="canonical_norag_report.json"
        )
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

    if args.bakeoff:
        sample_sz = args.sample_size or 50
        print(f"\nRunning Model Extraction Bake-Off (n={sample_sz}, diverse={args.diverse})...")
        if args.diverse:
            print(
                f"Generating {sample_sz} contracts with diverse phrasing (non-standard rate tables and terms)..."
            )
            contracts_to_evaluate: list[tuple[str, Any]] = generate_diverse_contracts(
                count=sample_sz
            )
        else:
            contracts_to_evaluate = [
                (pkg.contract.raw_text, pkg.contract)
                for pkg, _ in dataset
                if pkg.contract and pkg.contract.raw_text
            ]
        bakeoff_runner = ExtractionBakeOff(
            contracts=contracts_to_evaluate,
            sample_size=sample_sz,
        )
        bakeoff_results = bakeoff_runner.run_bakeoff()
        print_bakeoff_summary(bakeoff_results)
        bakeoff_file = save_bakeoff_report(bakeoff_results, output_dir=args.output)
        print(f"Bake-off report saved to: {bakeoff_file}")

    if args.calibrate_bartleby:
        print("\nRunning Empirical Calibration Sweep for Bartleby (Phase 4)...")
        calibrator = BartlebyCalibrator(dataset=dataset)
        cal_report = calibrator.run_sweep()
        print_calibration_summary(cal_report)
        cal_file = save_calibration_report(cal_report, output_dir=args.output)
        print(f"Calibration report saved to: {cal_file}")

        if args.apply_calibration:
            applied = apply_calibrated_thresholds(cal_report)
            if applied:
                print("Applied optimal calibrated thresholds to config/thresholds.yaml.")

    if args.eval_cuad:
        cuad_runner = CUADBenchmarkRunner(sample_size=args.sample_size or 25)
        cuad_report, ret_cases, abs_cases = cuad_runner.run_benchmark()
        print_cuad_summary(cuad_report)
        cuad_file = save_cuad_report(cuad_report, ret_cases, abs_cases, output_dir=args.output)
        print(f"CUAD benchmark report saved to: {cuad_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
