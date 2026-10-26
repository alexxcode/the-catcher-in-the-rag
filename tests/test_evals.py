from __future__ import annotations

import json
from pathlib import Path

import pytest

from catcher_in_the_rag.schemas import AnomalyType
from datagen.generator import generate_package
from evals.metrics import ClassMetric, EvaluationReport
from evals.report_writer import save_report
from evals.runner import EvaluationRunner


def _pin_synthetic_tau(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic templates use the tau calibrated on them, not the CUAD tau in config."""
    import catcher_in_the_rag.bartleby.guard as guard_module

    report = Path("evals/reports/calibration_bartleby_report.json")
    synth_tau = json.loads(report.read_text(encoding="utf-8"))["optimal_threshold"]
    original = guard_module.load_bartleby_thresholds
    monkeypatch.setattr(
        guard_module, "load_bartleby_thresholds", lambda tau=None: original(tau=synth_tau)
    )


def test_class_metric_computation() -> None:
    metric = ClassMetric(anomaly="A01")
    metric.true_positives = 8
    metric.false_positives = 2
    metric.false_negatives = 0
    metric.support = 8
    metric.compute()

    assert metric.precision == 0.8  # 8 / (8 + 2)
    assert metric.recall == 1.0  # 8 / (8 + 0)
    assert metric.f1 == 0.8889


def test_evaluation_runner_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_synthetic_tau(monkeypatch)
    # Generate 1 package for each anomaly type (10 packages total)
    dataset = [
        generate_package(seed=500 + idx, anomaly=anomaly, package_id=f"PKG-EVAL-{anomaly.value}")
        for idx, anomaly in enumerate(AnomalyType)
    ]

    runner = EvaluationRunner(mode="baseline")
    report = runner.run_eval(dataset)

    assert report.total_packages == 10
    assert report.decision_accuracy == 1.0
    assert report.bartleby.correct_abstention_rate == 1.0
    assert report.citations.citation_validity_rate == 1.0
    assert len(report.failure_cases) == 0
    assert report.false_positive_rate_a10 == 0.0


def test_evaluation_runner_ablation_no_rag() -> None:
    # On no-rag mode, contract-dependent anomalies (A04, A05, A06, A09) should fail
    dataset = [
        generate_package(seed=600 + idx, anomaly=anomaly, package_id=f"PKG-ABL-{anomaly.value}")
        for idx, anomaly in enumerate(AnomalyType)
    ]

    baseline_runner = EvaluationRunner(mode="baseline")
    baseline_report = baseline_runner.run_eval(dataset)

    no_rag_runner = EvaluationRunner(mode="no-rag")
    no_rag_report = no_rag_runner.run_eval(dataset)

    # Baseline RAG must outperform no-RAG ablation
    assert baseline_report.decision_accuracy > no_rag_report.decision_accuracy
    assert len(no_rag_report.failure_cases) > 0


def test_report_save_and_reload(tmp_path: Path) -> None:
    dataset = [
        generate_package(seed=700, anomaly=AnomalyType.A10, package_id="PKG-SAVE"),
    ]
    runner = EvaluationRunner(mode="baseline")
    report = runner.run_eval(dataset)

    saved_file = save_report(report, output_dir=tmp_path)
    assert saved_file.exists()
    assert (tmp_path / "latest.json").exists()

    reloaded = EvaluationReport.model_validate_json(saved_file.read_text(encoding="utf-8"))
    assert reloaded.total_packages == 1
    assert reloaded.decision_accuracy == 1.0
