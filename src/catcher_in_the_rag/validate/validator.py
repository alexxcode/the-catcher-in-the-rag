from __future__ import annotations

from catcher_in_the_rag.schemas import AnomalyType, Contract, Finding

CONTRACT_ANOMALIES = {
    AnomalyType.A01,
    AnomalyType.A04,
    AnomalyType.A05,
    AnomalyType.A06,
}


def validate_citation(finding: Finding, contract: Contract, strict_require: bool = False) -> bool:
    """Validate that a finding's citation exists verbatim in the contract source at the cited offsets.

    Enforces Non-negotiable Principle #2 ('No citation, no accusation').
    Any finding alleging a contractual anomaly (A01, A04, A05, A06) MUST carry a
    verbatim citation in the governing contract. Contract findings without citations
    are strictly discarded.
    """
    if not finding.citation:
        if finding.verification_status == "NOT_VERIFIABLE":
            return True
        return not (strict_require or finding.anomaly_type in CONTRACT_ANOMALIES)

    cit = finding.citation
    if cit.document_id != contract.contract_id:
        return False

    if cit.text not in contract.raw_text:
        return False

    actual_slice = contract.raw_text[cit.start_char : cit.end_char]
    return actual_slice == cit.text


def validate_findings(
    findings: list[Finding],
    contract: Contract,
) -> tuple[list[Finding], list[Finding]]:
    """Strictly validate citations for all findings against the contract source.

    Drops any finding whose citation is not found verbatim in the source document
    (per Non-negotiable Principle #2).
    """
    valid: list[Finding] = []
    dropped: list[Finding] = []

    for f in findings:
        if validate_citation(f, contract):
            valid.append(f)
        else:
            dropped.append(f)

    return valid, dropped
