from __future__ import annotations

from catcher_in_the_rag.schemas import Contract, Finding


def validate_citation(finding: Finding, contract: Contract) -> bool:
    """Validate that a finding's citation exists verbatim in the contract source at the cited offsets."""
    if not finding.citation:
        return True

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
