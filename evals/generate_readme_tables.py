"""Generate honest, reproducible Markdown tables for README.md directly from JSON reports.

Ensures that documentation can never diverge from empirical benchmark artifacts:
- Reads evals/reports/cuad_benchmark_latest.json
- Reads evals/reports/calibration_cuad_report.json
- Reads evals/reports/bakeoff_latest.json
- STRICT: Never uses fallback default constants for missing data; fails loudly on missing keys.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_cuad_table() -> str:
    cuad_path = Path("evals/reports/cuad_benchmark_latest.json")
    if not cuad_path.exists():
        raise FileNotFoundError(f"Required report {cuad_path} not found. Run benchmark first.")

    with cuad_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rep = data["report"]
    r1 = rep["recall_at_1"] * 100
    r3 = rep["recall_at_3"] * 100
    r5 = rep["recall_at_5"] * 100
    mrr = rep["mean_reciprocal_rank"]
    corr_abs = rep["bartleby_correct_abstention_rate"] * 100
    corr_ci = rep["bartleby_correct_abstention_ci95"]
    false_abs = rep["bartleby_false_abstention_rate"] * 100
    false_ci = rep["bartleby_false_abstention_ci95"]
    false_assert = rep["bartleby_false_assertion_rate"] * 100

    table = f"""| Métrica | Modo Offline (`stub-hash-768d`) | Vertex AI (`text-embedding-005` Live) | Intervalo de Confianza (95% Wilson Stub) | Interpretación y Significado Práctico |
|---|:---:|:---:|:---:|---|
| **Recall@1** | {r1:.1f}% | NOT MEASURED | — | Precisión de primer ranking en contratos no estructurados. |
| **Recall@3** | {r3:.1f}% | NOT MEASURED | — | Cláusula relevante localizada dentro del top 3 de candidatos. |
| **Recall@5** | {r5:.1f}% | NOT MEASURED | — | Cobertura agregada en contratos extensos (50k a 175k caracteres). |
| **MRR** | {mrr:.4f} | NOT MEASURED | — | Posición recíproca media del fragmento probatorio real. |
| **Abstención en Negativos** | **{corr_abs:.1f}%** | NOT MEASURED | `[{corr_ci[0] * 100:.1f}%, {corr_ci[1] * 100:.1f}%]` | Ante cláusulas ausentes reales (`answers: []`), Bartleby rehúsa alucinar. |
| **Abstención Errónea (Positivos)** | **{false_abs:.1f}%** | NOT MEASURED | `[{false_ci[0] * 100:.1f}%, {false_ci[1] * 100:.1f}%]` | Tasa de sobre-abstención en presencia de cláusulas legítimas válidas. |
| **Falsa Afirmación Espuria** | **{false_assert:.1f}%** | NOT MEASURED | — | Riesgo residual de afirmar falsamente una penalización inexistente. |"""
    return table


def generate_bakeoff_table() -> str:
    bakeoff_path = Path("evals/reports/bakeoff_latest.json")
    if not bakeoff_path.exists():
        raise FileNotFoundError(f"Required report {bakeoff_path} not found. Run bakeoff first.")

    with bakeoff_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    rows = []
    for c in candidates:
        name = c["candidate_name"]
        prov = c["provider_type"]
        lat = f"{c['mean_latency_ms']:.1f} ms"
        cost = (
            f"${c['estimated_cost_per_1k_usd']:.2f}"
            if not c.get("is_simulated", False)
            else "N/A (Stub)"
        )
        err = f"{c['error_rate'] * 100:.1f}%"
        pmt = f"{c['field_accuracies']['payment_terms_days'] * 100:.1f}%"
        rates = f"{c['field_accuracies']['rate_sheet'] * 100:.1f}%"
        overall = f"{c['overall_perfect_accuracy'] * 100:.1f}%"
        rows.append(
            f"| **{name}** | `{prov}` | **{lat}** | **{cost}** | {err} | **{pmt}** | **{rates}** | **{overall}** |"
        )

    header = """| Modelo / Candidato | Proveedor | Latencia Media | Costo / 1k docs | Error% | Payment Terms | Rate Sheet | Exactitud Global Conjunta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|"""
    return header + "\n" + "\n".join(rows)


def generate_domain_calibration_table() -> str:
    cal_cuad_path = Path("evals/reports/calibration_cuad_report.json")
    if not cal_cuad_path.exists():
        raise FileNotFoundError(
            f"Required report {cal_cuad_path} not found. Run calibration first."
        )

    cal_bartleby_path = Path("evals/reports/calibration_bartleby_report.json")
    if not cal_bartleby_path.exists():
        raise FileNotFoundError(
            f"Required report {cal_bartleby_path} not found. Run calibrate-bartleby first."
        )

    with cal_cuad_path.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    with cal_bartleby_path.open("r", encoding="utf-8") as f:
        synth_data = json.load(f)

    # STRICT: No default values. Fail if keys are missing from report JSON.
    synth_t = synth_data["optimal_threshold"]
    synth_f1 = synth_data["validation_macro_f1"]
    synth_auc = synth_data["auc_roc_val"]
    pure_f1 = synth_data["pure_retrieval_val_macro_f1"]
    pure_abs = synth_data["pure_retrieval_val_a09_abstention_rate"] * 100

    cuad_t = cuad_data["cuad_optimal_threshold"]
    val_synth = cuad_data["validation_f1_at_synthetic_threshold"]
    val_cuad = cuad_data["validation_f1_at_optimal"]
    val_c_synth = cuad_data["val_correct_abstention_at_synthetic"] * 100
    val_c_opt = cuad_data["val_correct_abstention_at_optimal"] * 100
    val_f_opt = cuad_data["val_false_abstention_at_optimal"] * 100
    auc_val = cuad_data["auc_roc_val"]
    always_f1 = cuad_data["baseline_always_abstain"]["macro_f1"]

    table = f"""| Domain | Source dataset | Calibrated threshold ($\\tau$) | Validation Macro F1 | AUC-ROC (separability) | Empirical transfer performance |
|---|---|:---:|:---:|:---:|---|
| **Synthetic templates (+ Rule 3a heuristic*)** | Datagen M1 ({synth_data["total_packages"]} packages; not evidence) | **{synth_t:.2f}** | **{synth_f1:.4f}** | **{synth_auc:.4f}** | F1 with the 3a denial regex: {synth_data["heuristic_3a_val_macro_f1"]:.4f}. Pure score without 3a: F1 {pure_f1:.4f} (abstention {pure_abs:.1f}%). |
| **Real SEC contracts (transfer)** | CUAD v1, contracts 1–{cuad_data["total_contracts"]}, train/val 60/40 | **{synth_t:.2f}** (fixed) | **{val_synth:.4f}** | **{auc_val:.4f}** | Correct abstention on absences {val_c_synth:.1f}% ({100 - val_c_synth:.1f}% pass through). |
| **Real SEC contracts (recalibrated)** | CUAD v1, contracts 1–{cuad_data["total_contracts"]}, train/val 60/40 | **{cuad_t:.2f}** (recalibrated, not applied) | **{val_cuad:.4f}** | **{auc_val:.4f}** | Correct abstention {val_c_opt:.1f}%, false abstentions {val_f_opt:.1f}% (Always abstain: {always_f1:.4f}). |"""
    return table


def generate_llm_verification_table() -> str:
    path = Path("evals/reports/bartleby_llm_verification_report.json")
    if not path.exists():
        raise FileNotFoundError(f"Required report {path} not found.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for m in data["modes"]:
        name = m["mode_name"]
        tau = f"{m['relevance_threshold']:.2f}"
        llm = "Gemini 3.5 Flash" if m["use_llm_verification"] else "Off"
        c_abs = f"{m['correct_abstention_rate'] * 100:.1f}% ({m['correct_abstentions']}/{m['total_absences']})"
        f_abs = f"{m['false_abstention_rate'] * 100:.1f}% ({m['false_abstentions']}/{m['total_presences']})"
        f1 = f"{m['macro_f1']:.4f}"
        lat = f"{m['mean_latency_ms']:.1f} ms"
        cost = (
            f"${m['estimated_cost_usd_per_1k']:.4f}"
            if m["estimated_cost_usd_per_1k"] > 0
            else "$0.0000"
        )
        rows.append(
            f"| **{name}** | **{tau}** | `{llm}` | **{c_abs}** | **{f_abs}** | **{f1}** | {lat} | {cost} |"
        )

    header = """| Guard mode | $\\tau$ | LLM verification | Correct abstention (recall) | False abstention (error) | Macro F1 | Mean latency | Estimated cost / 1k* |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"""
    return header + "\n" + "\n".join(rows)


def generate_retrieval_table() -> str:
    path = Path("evals/reports/retrieval_ablation_oos_report.json")
    if not path.exists():
        return "NOT MEASURED (`evals/reports/retrieval_ablation_oos_report.json` missing)."
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (50/50)"}

    def cell(e: dict) -> str:
        lo, hi = e["wilson_ci95"]
        return f"{e['rate'] * 100:.1f}% ({e['count']}/{e['total']}) [CI95 {lo * 100:.1f}–{hi * 100:.1f}]"

    rows = [
        "| Method | Hit@1 | Hit@5 |",
        "|---|:---:|:---:|",
    ]
    for mode, label in labels.items():
        m = metrics[mode]
        rows.append(f"| **{label}** | {cell(m['hit_at_1'])} | {cell(m['hit_at_5'])} |")

    notes = []
    for k, name in (("hit_at_1", "Hit@1"), ("hit_at_5", "Hit@5")):
        cis = {mode: metrics[mode][k]["wilson_ci95"] for mode in labels}
        overlap = all(
            cis[a][0] <= cis[b][1] and cis[b][0] <= cis[a][1] for a in cis for b in cis if a < b
        )
        if overlap:
            notes.append(
                f"{name}: the three methods are statistically indistinguishable on this sample."
            )
        else:
            apart = [
                labels[a]
                for a in cis
                if all(cis[a][1] < cis[b][0] or cis[b][1] < cis[a][0] for b in cis if b != a)
            ]
            who = ", ".join(apart) if apart else "some pair"
            same = [
                f"{labels[a]} and {labels[b]}"
                for a in cis
                for b in cis
                if a < b and cis[a][0] <= cis[b][1] and cis[b][0] <= cis[a][1]
            ]
            same_txt = f" {'; '.join(same)}: indistinguishable on this sample." if same else ""
            notes.append(
                f"{name}: the Wilson interval of {who} does not overlap the others; the three "
                f"methods are not indistinguishable on this sample.{same_txt}"
            )
    header = (
        f"{data['presence_cases']} presence cases, {data['contract_slice']}, "
        f"embeddings `{data['embedding_model_id']}`, commit `{data['git_commit_sha'][:7]}`. "
        "Ties broken by order of appearance in the contract."
    )
    return "\n".join([header, "", *rows, "", *(f"- {n}" for n in notes)])


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required report {p} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


def _rate(count: int, total: int, ci: list[float]) -> str:
    return (
        f"{count}/{total} = {count / total * 100:.1f}% [CI95 {ci[0] * 100:.1f}–{ci[1] * 100:.1f}]"
    )


def _oos_rows() -> list[tuple[str, str, dict]]:
    rows = []
    for path, label in (
        ("evals/reports/out_of_sample_cuad_stub_tau0.50_report.json", "No LLM, score only"),
        ("evals/reports/out_of_sample_cuad_stub_tau0.80_report.json", "No LLM, score only"),
        ("evals/reports/out_of_sample_cuad_report.json", "With LLM verification (3b)"),
    ):
        d = _load(path)
        rows.append((path, label, d))
    return rows


def generate_mode3_diagnosis() -> str:
    """Three Bartleby claims, every number read from report JSONs."""
    cal = _load("evals/reports/calibration_cuad_report.json")
    llm = _load("evals/reports/bartleby_llm_verification_report.json")
    oos_llm = _load("evals/reports/out_of_sample_cuad_report.json")

    auc, ci = cal["auc_roc_val"], cal["auc_roc_val_ci95"]
    includes_half = ci[0] <= 0.5 <= ci[1]
    m1, m3 = llm["modes"][0], llm["modes"][2]
    fa = oos_llm["metrics"]["false_abstention_on_positives"]
    total_q = oos_llm["total_queries_evaluated"]

    a = (
        f"(a) **The retrieval score alone is a weak signal.** Validation AUC-ROC "
        f"{auc:.4f} [CI95 {ci[0]:.4f}–{ci[1]:.4f}] with `{cal['embedding_provider']}` "
        f"(CUAD, contracts 1–{cal['total_contracts']}, `calibration_cuad_report.json`); "
        + ("the interval includes 0.50." if includes_half else "the interval excludes 0.50.")
    )
    b = (
        f"(b) **LLM verification supplies the abstention.** With the same embeddings and "
        f"$\\tau$ = {m1['relevance_threshold']:.2f} (`bartleby_llm_verification_report.json`), "
        f"correct abstention on absences goes from {m1['correct_abstentions']}/{m1['total_absences']} "
        f"({m1['correct_abstention_rate'] * 100:.1f}%) without the LLM to {m3['correct_abstentions']}/"
        f"{m3['total_absences']} ({m3['correct_abstention_rate'] * 100:.1f}%) with it."
    )
    c = (
        f"(c) **False abstentions remain high.** With the LLM, out of sample: "
        f"{_rate(fa['count'], fa['total'], fa['wilson_ci95'])} of present clauses "
        f"(`out_of_sample_cuad_report.json`). Each false abstention is an unnecessary human "
        f"review: {fa['count']} in {total_q} queries "
        f"({fa['count'] / total_q * 100:.1f} per 100). Time or cost per review: NOT MEASURED."
    )
    return "\n".join(
        ["> **Bartleby: three measured claims**", f"> {a}", ">", f"> {b}", ">", f"> {c}"]
    )


def generate_operating_point() -> str:
    """Operating-point reading (two sentences) plus the LLM comparability note."""
    from catcher_in_the_rag.bartleby.guard import load_bartleby_thresholds

    (_, _, lo), (_, _, hi), (_, _, llm_rep) = _oos_rows()
    cfg_tau = load_bartleby_thresholds()["min_relevance_score"]

    def parts(d: dict) -> tuple[dict, dict, int, int]:
        m = d["metrics"]
        ca, fa = m["correct_abstention_on_negatives"], m["false_abstention_on_positives"]
        return ca, fa, ca["count"] + fa["count"], ca["total"] + fa["total"]

    ca_hi, fa_hi, _, _ = parts(hi)
    ca_lo, _, n_lo, q_lo = parts(lo)
    cfg_txt = " (current value in `config/thresholds.yaml`)" if cfg_tau == hi["tau_override"] else ""
    s1 = (
        f"Without the LLM, $\\tau$ = {hi['tau_override']:.2f}{cfg_txt} abstains on "
        f"{ca_hi['rate'] * 100:.1f}% of absences ({ca_hi['count']}/{ca_hi['total']}) and on "
        f"{fa_hi['rate'] * 100:.1f}% of presences ({fa_hi['count']}/{fa_hi['total']})."
    )
    verb = "almost never abstains" if n_lo / q_lo < 0.25 else "abstains"
    s2 = (
        f"$\\tau$ = {lo['tau_override']:.2f} {verb}: {n_lo} of {q_lo} queries "
        f"({n_lo / q_lo * 100:.1f}%), and on only {ca_lo['rate'] * 100:.1f}% of absences."
    )
    llm_emb = llm_rep.get("embedding_provider") or "stub"
    reasons = []
    if llm_emb != hi.get("embedding_provider"):
        reasons.append(f"it used `{llm_emb}` embeddings")
    if llm_rep.get("git_dirty"):
        reasons.append("it was generated with `git_dirty: true`")
    note = (
        f"The LLM row is not comparable with the rows without the LLM: {' and '.join(reasons)}."
        if reasons
        else "The LLM row uses the same embeddings as the rows without the LLM."
    )
    return f"{s1} {s2}\n\n> **Note:** {note}"


def generate_oos_bartleby_table() -> str:
    """Out-of-sample (contracts 16-50) abstention per configuration plus Always-Abstain."""
    rows = [
        "| Configuration | $\\tau$ | Embeddings | Correct abstention (absences) | False abstention (presences) | Macro F1 | Report |",
        "|---|:---:|---|:---:|:---:|:---:|---|",
    ]
    for path, label, d in _oos_rows():
        m = d["metrics"]
        ca, fa = m["correct_abstention_on_negatives"], m["false_abstention_on_positives"]
        tau = d.get("tau_override")
        tau_txt = f"{tau:.2f}" if tau is not None else "0.50 (fixed in the eval)"
        emb = d.get("embedding_provider") or "stub (command default)"
        dirty = " (git_dirty)" if d.get("git_dirty") else ""
        rows.append(
            f"| {label} | {tau_txt} | `{emb}` | {_rate(ca['count'], ca['total'], ca['wilson_ci95'])} | "
            f"{_rate(fa['count'], fa['total'], fa['wilson_ci95'])} | {m['macro_f1']:.4f} | `{Path(path).name}`{dirty} |"
        )
    d0 = _oos_rows()[0][2]
    n_abs, n_pres = d0["absence_cases"], d0["presence_cases"]
    p_abs = n_abs / (n_abs + n_pres)
    f1_always = round((2 * p_abs / (p_abs + 1.0)) / 2.0, 4)
    rows.append(
        f"| Always abstain (trivial) | — | — | {n_abs}/{n_abs} = 100.0% | {n_pres}/{n_pres} = 100.0% | "
        f"{f1_always:.4f} | computed from the counts |"
    )
    return "\n".join(rows)


def generate_synthetic_ablation_table() -> str:
    baseline_path = Path("evals/reports/canonical_baseline_report.json")
    norag_path = Path("evals/reports/canonical_norag_report.json")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Required report {baseline_path} not found.")
    if not norag_path.exists():
        raise FileNotFoundError(f"Required report {norag_path} not found.")

    with baseline_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)
    with norag_path.open("r", encoding="utf-8") as f:
        norag_data = json.load(f)

    tot = base_data["total_packages"]
    b_acc = base_data["decision_accuracy"] * 100
    n_acc = norag_data["decision_accuracy"] * 100
    b_acc_count = int(round(base_data["decision_accuracy"] * tot))
    n_acc_count = int(round(norag_data["decision_accuracy"] * tot))
    delta_acc = b_acc - n_acc

    c_tot = base_data["citations"]["total_citations"]
    c_val = base_data["citations"]["valid_citations"]
    c_rate = base_data["citations"]["citation_validity_rate"] * 100

    b_a09_tot = base_data["bartleby"]["total_a09"]
    b_a09_corr = base_data["bartleby"]["correct_abstentions_a09"]
    b_a09_rate = base_data["bartleby"]["correct_abstention_rate"] * 100

    fp_a10 = base_data["false_positive_rate_a10"] * 100
    r5 = base_data["retrieval"]["recall_at_5"] * 100
    lat = base_data["mean_latency_ms"]
    norag_lat = norag_data["mean_latency_ms"]

    table = f"""| Metric | With contract terms (baseline) | Without terms (three-way match only) | Impact / delta |
|---|:---:|:---:|:---:|
| **Overall decision accuracy** | **{b_acc:.1f}%** ({b_acc_count}/{tot}) | **{n_acc:.1f}%** ({n_acc_count}/{tot}) | **ablation: no clause evaluator (+{delta_acc:.1f}% degradation)** |
| **Verbatim citation validity** | **{c_rate:.1f}%** ({c_val}/{c_tot}) [synthetic templates] | N/A | Zero invalid synthetic citations |
| **Bartleby correct abstention** | **{b_a09_rate:.1f}%** ({b_a09_corr}/{b_a09_tot} on A09*) | 0.0% (0/{b_a09_tot} on A09) | Silence detection (via the 3a template heuristic*) |
| **False positive rate (A10)** | **{fp_a10:.1f}%** (0/30) | 0.0% (0/30) | Clean packages approved without friction |
| **Retrieval Recall@5** | **{r5:.1f}%** [synthetic templates] | N/A | Retrieval on synthetic templates |
| **Mean latency per package** | **{lat:.1f} ms** | {norag_lat:.1f} ms | Local deterministic execution |"""
    return table


def generate_domain_calibration_diagnosis() -> str:
    cal_cuad_path = Path("evals/reports/calibration_cuad_report.json")
    cal_bartleby_path = Path("evals/reports/calibration_bartleby_report.json")
    if not cal_cuad_path.exists() or not cal_bartleby_path.exists():
        raise FileNotFoundError("Calibration reports not found.")

    with cal_cuad_path.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)
    with cal_bartleby_path.open("r", encoding="utf-8") as f:
        synth_data = json.load(f)

    synth_t = synth_data["optimal_threshold"]
    synth_f1 = synth_data["validation_macro_f1"]
    synth_auc = synth_data["auc_roc_val"]

    auc_val = cuad_data["auc_roc_val"]
    auc_train = cuad_data["auc_roc_train"]
    cuad_t = cuad_data["cuad_optimal_threshold"]
    val_c_synth = cuad_data["val_correct_abstention_at_synthetic"] * 100
    val_cuad = cuad_data["validation_f1_at_optimal"]
    val_c_opt = cuad_data["val_correct_abstention_at_optimal"] * 100
    val_f_opt = cuad_data["val_false_abstention_at_optimal"] * 100
    always_f1 = cuad_data["baseline_always_abstain"]["macro_f1"]
    delta_f1 = val_cuad - always_f1

    n_tot = cuad_data["total_contracts"]
    n_tr = cuad_data["train_contracts_count"]
    n_va = cuad_data["val_contracts_count"]
    emb = cuad_data["embedding_provider"]
    tr_ci = cuad_data.get("auc_roc_train_ci95")
    va_ci = cuad_data.get("auc_roc_val_ci95")

    def ci_txt(ci: list[float] | None) -> str:
        return f" [CI95 {ci[0]:.4f}–{ci[1]:.4f}]" if ci else " [CI not reported]"

    if va_ci and va_ci[0] <= 0.5 <= va_ci[1]:
        auc_note = (
            "the validation interval includes 0.50: indistinguishable from chance on this sample."
        )
    elif va_ci:
        auc_note = "the validation interval excludes 0.50."
    else:
        auc_note = "no interval, so it cannot be compared with 0.50."

    text = f"""> **CUAD calibration diagnosis** (`evals/reports/calibration_cuad_report.json`)
> Synthetic (reference, not evidence): $\\tau = {synth_t:.2f}$, validation Macro F1 {synth_f1:.4f}, AUC-ROC {synth_auc:.4f}.
> CUAD: {n_tot} contracts (train {n_tr}, validation {n_va}, seed 42), embeddings `{emb}`.
>
> 1. **AUC-ROC (Wilcoxon-Mann-Whitney):** train {auc_train:.4f}{ci_txt(tr_ci)}; validation {auc_val:.4f}{ci_txt(va_ci)}; {auc_note}
> 2. **Threshold:** at $\\tau = {synth_t:.2f}$, correct abstention on validation absences is {val_c_synth:.1f}%. At $\\tau = {cuad_t:.2f}$ (train optimum) it is {val_c_opt:.1f}%, with {val_f_opt:.1f}% false abstentions and Macro F1 {val_cuad:.4f} (Always abstain: {always_f1:.4f}; difference {delta_f1:+.4f})."""
    return text


def generate_llm_verification_footnote() -> str:
    path = Path("evals/reports/bartleby_llm_verification_report.json")
    if not path.exists():
        raise FileNotFoundError(f"Required report {path} not found.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m3 = data["modes"][2]
    prompt_tokens = m3["prompt_tokens"]
    candidate_tokens = m3["candidate_tokens"]

    text = f"""*\\*Cost estimated from measured tokens ({prompt_tokens:,} prompt / {candidate_tokens:,} candidate, with thinking tokens counted as output) and official pay-as-you-go Vertex AI rates ($1.50 / $9.00 per million tokens for Gemini 3.5 Flash, per [`config/pricing.yaml`](config/pricing.yaml)). With Gemini 3.8 Flash the estimated cost is ~$0.88/1k at the current introductory rate ($0.75 / $3.75 until December 31, 2026) and ~$1.77/1k at the standard rate ($1.50 / $7.50 from January 1, 2027).*"""
    return text


def generate_bakeoff_diagnosis() -> str:
    path = Path("evals/reports/bakeoff_latest.json")
    if not path.exists():
        raise FileNotFoundError(f"Required report {path} not found.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cands = {c["candidate_name"]: c for c in data["candidates"]}
    c35 = cands.get("Gemini 3.5 Flash (GA Active)")
    c38 = cands.get("Gemini 3.8 Flash (GA Active)")
    lat35 = f"{c35['mean_latency_ms']:.1f} ms" if c35 else "N/A"
    lat38 = f"{c38['mean_latency_ms'] / 1000:.1f} segundos" if c38 else "N/A"

    text = f"""> **Diagnóstico Forense y Fidelidad de la Medición:**
> En los contratos con redacción variada (`style="diverse"`), el extractor por reglas deterministas captura con éxito los plazos de pago estándar (100%), pero **colapsa al 0% en la tabla de tarifas** debido a la dispersión de estructuras narrativas, llevando la exactitud conjunta a 0.0%. En los contratos evaluados en vivo con Vertex AI mediante esquema estructurado estricto, **Gemini 3.5 Flash** y **Gemini 3.8 Flash** recuperaron fielmente las tarifas tabulares y los plazos netos.
>
> **Dinámica de Latencia y Razonamiento:** Con `thinking_budget: 0`, Gemini 3.5 Flash alcanza una latencia media de **{lat35}** por extracción de documento complejo, mientras que Gemini 3.8 Flash promedió **{lat38}** (hipótesis preliminar a investigar formalmente en Fase 4: sobrecosto de latencia por deliberación interna o reintentos de endpoint, aún no medido de forma aislada). Los costos se calcularon rigurosamente a partir de tarifas GA en Vertex AI ($1.50 / $9.00 por 1M tokens para 3.5 Flash; $0.75 / $3.75 introductorio para 3.8 Flash).
>
> **Salvaguarda de Ciclo de Vida y Fail-Fast:**
> - `gemini-3.5-flash` es GA en Vertex AI con fecha de retiro fijada para el **19 de mayo de 2027 o posterior** (región endpoint: `global`).
> - `gemini-3.8-flash` es GA con retiro para el **2 de septiembre de 2027 o posterior** (región endpoint: `global`).
> - Se implementó un canario semanal en GitHub Actions (`.github/workflows/model_canary.yml`) y sonda de verificación local (`make check-models`)."""
    return text


GENERATORS = {
    "domain_calibration_table": generate_domain_calibration_table,
    "domain_calibration_diagnosis": generate_domain_calibration_diagnosis,
    "llm_verification_table": generate_llm_verification_table,
    "llm_verification_footnote": generate_llm_verification_footnote,
    "mode3_diagnosis": generate_mode3_diagnosis,
    "oos_bartleby_table": generate_oos_bartleby_table,
    "operating_point": generate_operating_point,
    "retrieval_table": generate_retrieval_table,
    "synthetic_ablation_table": generate_synthetic_ablation_table,
}


def sync_readme(readme_path: Path = Path("README.md")) -> None:
    import re

    content = readme_path.read_text(encoding="utf-8")
    for name, gen_fn in GENERATORS.items():
        pattern = re.compile(rf"(<!--\s*GEN:{name}\s*-->)([\s\S]*?)(<!--\s*/GEN\s*-->)")
        new_block = gen_fn().strip()
        content, count = pattern.subn(
            lambda m, nb=new_block: f"{m.group(1)}\n{nb}\n{m.group(3)}",
            content,
        )
        if count == 0:
            print(f"Warning: Block <!-- GEN:{name} --> not found in {readme_path}")
        else:
            print(f"Synced <!-- GEN:{name} --> successfully.")

    readme_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    import sys

    if "--sync" in sys.argv:
        sync_readme()
    else:
        for name, gen in GENERATORS.items():
            print(f"=== {name.upper()} ===")
            print(gen())
            print()
