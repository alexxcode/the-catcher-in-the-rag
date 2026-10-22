from __future__ import annotations

from pathlib import Path

from evals.metrics import EvaluationReport


def save_report(report: EvaluationReport, output_dir: Path | str = "evals/reports") -> Path:
    """Save an evaluation report to a versioned JSON file and update latest.json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_clean = report.timestamp.replace(":", "-").replace(".", "-")
    file_name = f"eval_report_{report.config_name}_{timestamp_clean}.json"
    report_file = out_path / file_name

    json_content = report.model_dump_json(indent=2)
    report_file.write_text(json_content, encoding="utf-8")

    # Update latest.json
    latest_file = out_path / "latest.json"
    latest_file.write_text(json_content, encoding="utf-8")

    return report_file


def print_summary_table(report: EvaluationReport) -> None:
    """Print an executive summary of the evaluation results to standard output."""
    sep = "=" * 78
    print(sep)
    print(f" THE CATCHER IN THE RAG - EVALUATION REPORT ({report.config_name.upper()})")
    print(f" Timestamp: {report.timestamp} | Total packages evaluated: {report.total_packages}")
    print(sep)
    print(f" Overall Decision Accuracy:    {report.decision_accuracy * 100:.1f}%")
    print(f" Mean Latency:                 {report.mean_latency_ms:.1f} ms / package")
    print(f" Clean Package (A10) FP Rate:  {report.false_positive_rate_a10 * 100:.1f}%")
    print(
        f" Citation Validity Rate:       {report.citations.citation_validity_rate * 100:.1f}% "
        f"({report.citations.valid_citations}/{report.citations.total_citations})"
    )
    print(
        f" Bartleby Correct Abstention:  {report.bartleby.correct_abstention_rate * 100:.1f}% "
        f"({report.bartleby.correct_abstentions_a09}/{report.bartleby.total_a09} on A09)"
    )
    print(
        f" Bartleby Over-Abstention:     {report.bartleby.over_abstention_rate * 100:.1f}% "
        f"({report.bartleby.false_abstentions}/{report.bartleby.total_non_a09} on non-A09)"
    )
    print(
        f" Retrieval Recall@1:           {report.retrieval.recall_at_1 * 100:.1f}%  |  "
        f"Recall@5: {report.retrieval.recall_at_5 * 100:.1f}%  |  MRR: {report.retrieval.mrr:.4f}"
    )
    print("-" * 78)
    print(
        f" {'Anomaly':<8} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
    )
    print("-" * 78)

    anomaly_titles = {
        "A01": "Price > contract rate",
        "A02": "Qty billed > received",
        "A03": "Duplicate invoice",
        "A04": "Payment terms mismatch",
        "A05": "Late-delivery penalty",
        "A06": "Contract expired",
        "A07": "Unknown vendor / no PO",
        "A08": "Amount > $50k threshold",
        "A09": "Clause absent (abstain)",
        "A10": "Clean package",
    }

    for a_code, metric in sorted(report.per_class.items()):
        title = anomaly_titles.get(a_code, "")
        print(
            f" {a_code:<8} | {metric.support:<8} | {metric.precision * 100:>8.1f}% | {metric.recall * 100:>8.1f}% | {metric.f1:>8.3f}  ({title})"
        )

    print(sep)
    if report.failure_cases:
        print(f" [!] Failure cases ({len(report.failure_cases)}):")
        for fc in report.failure_cases[:5]:
            print(
                f"   * {fc.package_id}: expected {fc.expected_decision}, got {fc.actual_decision} | {fc.justification}"
            )
        if len(report.failure_cases) > 5:
            print(f"   ... and {len(report.failure_cases) - 5} more.")
    else:
        print(" [OK] Zero failure cases observed.")
    print(sep)
