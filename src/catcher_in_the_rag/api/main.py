from __future__ import annotations

import json
import os
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from catcher_in_the_rag.graph import audit_pipeline
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.schemas import (
    Decision,
    DecisionType,
    Finding,
    ProcurementPackage,
)

load_dotenv()

app = FastAPI(
    title="The Catcher in the RAG — Audit API",
    description="Automated procurement audit API with hybrid contract RAG, verbatim citations, and Bartleby abstention guard.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory Rate Limiter ───────────────────────────────────────────────────
_RATE_LIMIT_BUCKET: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW_SECONDS = 60.0
DEFAULT_MAX_RPM = 30


def _extract_client_ip(request: Request) -> str:
    """Extract real client IP handling Cloud Run reverse proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # First IP in comma-separated list is the original client IP
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def _enforce_rate_limit(client_ip: str) -> None:
    """Enforce rolling rate limit per IP to prevent API exhaustion and DoS."""
    now = time.time()
    max_rpm = int(os.getenv("RATE_LIMIT_RPM", str(DEFAULT_MAX_RPM)))
    bucket = _RATE_LIMIT_BUCKET.setdefault(client_ip, [])
    # Evict timestamps outside rolling window
    valid_bucket = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(valid_bucket) >= max_rpm:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (maximum {max_rpm} requests per minute). Try again shortly.",
        )
    valid_bucket.append(now)
    _RATE_LIMIT_BUCKET[client_ip] = valid_bucket


def _resolve_llm_provider(
    x_api_key: str | None = None,
    authorization: str | None = None,
) -> Any:
    """Determine LLM provider based on PIN authentication to safeguard LLM budget.

    Spend Safeguard:
    - If DEMO_ACCESS_PIN is configured and verified (via X-API-Key or Authorization Bearer),
      returns the active production provider (e.g., vertex_ai).
    - If unauthenticated:
      - If ALLOW_PUBLIC_STUB=true (default): executes safely using zero-cost offline stub.
      - If ALLOW_PUBLIC_STUB=false: rejects with 401 Unauthorized.
    - If DEMO_ACCESS_PIN is not configured (local dev): returns active default provider.
    """
    configured_pin = os.getenv("DEMO_ACCESS_PIN", "").strip()
    if not configured_pin:
        return get_llm_provider()

    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if token and secrets.compare_digest(token, configured_pin):
        return get_llm_provider()

    allow_stub = os.getenv("ALLOW_PUBLIC_STUB", "true").lower() in ("1", "true", "yes")
    if allow_stub:
        return get_llm_provider(provider_type="stub")

    raise HTTPException(
        status_code=401,
        detail="Unauthorized: valid DEMO_ACCESS_PIN required via X-API-Key or Bearer token.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "The Catcher in the RAG"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_health: dict[str, Any] = Field(default_factory=dict)


class AuditRequest(BaseModel):
    package_id: str | None = None
    package: dict[str, Any] | None = None


class AuditResponse(BaseModel):
    package_id: str
    decision: Decision
    validated_findings: list[Finding]
    match_findings: list[Finding]
    clause_findings: list[Finding]
    bartleby_abstained: bool
    bartleby_reason: str | None = None
    requires_human_review: bool
    execution_provider: str = "default"


class ReviewOverrideRequest(BaseModel):
    package_id: str
    override_decision: DecisionType
    notes: str = ""
    auditor_id: str = "human_auditor_1"


class ReviewOverrideResponse(BaseModel):
    package_id: str
    final_decision: DecisionType
    status: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# In-memory store for human overrides during session
_REVIEW_OVERRIDES: dict[str, dict[str, Any]] = {}


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service health check endpoint with active model lifecycle verification."""
    try:
        model_info = get_llm_provider().check_model_health()
        return HealthResponse(model_health=model_info)
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Model health check failed: {err}") from err


@app.get("/api/packages")
def list_packages(data_dir: str = "data/synthetic") -> list[dict[str, Any]]:
    """List available synthetic procurement packages with metadata."""
    path = Path(data_dir)
    if not path.exists():
        return []

    packages = []
    # Sort PKG directories numerically
    for pkg_dir in sorted(path.glob("PKG-*")):
        if not pkg_dir.is_dir():
            continue
        labels_file = pkg_dir / "labels.json"
        labels_data = {}
        if labels_file.exists():
            try:
                with labels_file.open(encoding="utf-8") as f:
                    labels_data = json.load(f)
            except Exception:
                pass

        packages.append(
            {
                "package_id": pkg_dir.name,
                "anomalies": labels_data.get("anomalies", []),
                "expected_decision": labels_data.get("expected_decision", "UNKNOWN"),
            }
        )
    return packages


@app.get("/api/packages/{package_id}")
def get_package(package_id: str, data_dir: str = "data/synthetic") -> dict[str, Any]:
    """Retrieve details and ground-truth labels for a specific package."""
    pkg_dir = Path(data_dir) / package_id
    if not pkg_dir.exists():
        raise HTTPException(status_code=404, detail=f"Package {package_id} not found")

    package_file = pkg_dir / "package.json"
    labels_file = pkg_dir / "labels.json"

    if not package_file.exists():
        raise HTTPException(status_code=404, detail=f"package.json missing for {package_id}")

    with package_file.open(encoding="utf-8") as f:
        package_data = json.load(f)

    labels_data = None
    if labels_file.exists():
        with labels_file.open(encoding="utf-8") as f:
            labels_data = json.load(f)

    return {
        "package": package_data,
        "labels": labels_data,
        "override": _REVIEW_OVERRIDES.get(package_id),
    }


@app.post("/api/audit", response_model=AuditResponse)
def audit_package_endpoint(
    req: AuditRequest,
    request: Request,
    data_dir: str = "data/synthetic",
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AuditResponse:
    """Execute the end-to-end audit agent on a package."""
    client_ip = _extract_client_ip(request)
    _enforce_rate_limit(client_ip)
    provider = _resolve_llm_provider(x_api_key=x_api_key, authorization=authorization)

    package_to_audit: ProcurementPackage | dict[str, Any]

    if req.package is not None:
        package_to_audit = req.package
        package_id = req.package.get("package_id", "CUSTOM-PKG")
    elif req.package_id is not None:
        pkg_file = Path(data_dir) / req.package_id / "package.json"
        if not pkg_file.exists():
            raise HTTPException(status_code=404, detail=f"Package {req.package_id} not found")
        with pkg_file.open(encoding="utf-8") as f:
            data = json.load(f)
        package_to_audit = ProcurementPackage.model_validate(data)
        package_id = req.package_id
    else:
        raise HTTPException(
            status_code=400,
            detail="Either package_id or package payload must be provided",
        )

    final_state = audit_pipeline(package_to_audit, llm_provider=provider)
    provider_name = getattr(provider, "provider_type", type(provider).__name__)

    return AuditResponse(
        package_id=package_id,
        decision=final_state["decision"],
        validated_findings=final_state.get("validated_findings", []),
        match_findings=final_state.get("match_findings", []),
        clause_findings=final_state.get("clause_findings", []),
        bartleby_abstained=final_state.get("bartleby_abstained", False),
        bartleby_reason=final_state.get("bartleby_reason"),
        requires_human_review=final_state.get("requires_human_review", False),
        execution_provider=str(provider_name),
    )


@app.post("/api/review/override", response_model=ReviewOverrideResponse)
def override_decision_endpoint(
    req: ReviewOverrideRequest,
) -> ReviewOverrideResponse:
    """Allow human auditor to record an override or confirmation."""
    _REVIEW_OVERRIDES[req.package_id] = {
        "override_decision": req.override_decision.value,
        "notes": req.notes,
        "auditor_id": req.auditor_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    return ReviewOverrideResponse(
        package_id=req.package_id,
        final_decision=req.override_decision,
        status="Recorded human override successfully",
    )


@app.get("/api/reports/latest")
def get_latest_report(
    reports_dir: str = "evals/reports",
) -> dict[str, Any]:
    """Retrieve the most recent evaluation report JSON."""
    r_dir = Path(reports_dir)
    latest_symlink = r_dir / "latest.json"
    if latest_symlink.exists():
        try:
            with latest_symlink.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to newest timestamped report
    json_files = sorted(r_dir.glob("eval_report_*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="No evaluation reports found in evals/reports")

    with json_files[-1].open(encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/reports")
def list_reports(reports_dir: str = "evals/reports") -> list[str]:
    """List all available evaluation report files."""
    r_dir = Path(reports_dir)
    if not r_dir.exists():
        return []
    return [p.name for p in sorted(r_dir.glob("eval_report_*.json"), reverse=True)]
