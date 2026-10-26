import json
import os
import secrets
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from catcher_in_the_rag.bartleby.guard import BartlebyGuard, load_bartleby_thresholds
from catcher_in_the_rag.graph import audit_pipeline
from catcher_in_the_rag.ingest.cuad_loader import (
    CUADContract,
    CUADLoader,
)
from catcher_in_the_rag.providers.factory import get_llm_provider, resolve_llm_provider
from catcher_in_the_rag.providers.tracing import is_tracing_enabled
from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from catcher_in_the_rag.schemas import (
    DecisionType,
    Finding,
    GoodsReceipt,
    Invoice,
    LineItem,
    PackageLabels,
    ProcurementPackage,
    PurchaseOrder,
    ReceiptItem,
)

load_dotenv()


def _ui_credential() -> str | None:
    """PIN sent to the provider decision only when the auditor unlocked it in the sidebar."""
    if st.session_state.get("is_authenticated"):
        return os.getenv("DEMO_ACCESS_PIN", "").strip() or None
    return None


def execute_audit(package: ProcurementPackage) -> dict[str, Any]:
    """Execute audit via private Cloud Run API (service-to-service) or local in-process pipeline.

    If API_URL is set and the API fails, the audit runs locally with the stub provider and a
    visible warning is shown (no silent fallback).
    """
    api_url = os.getenv("API_URL", "").strip().rstrip("/")
    if api_url:
        api_error = ""
        try:
            import httpx

            headers: dict[str, str] = {}
            pin = _ui_credential()
            if pin:
                headers["X-API-Key"] = pin

            # In Cloud Run service-to-service, fetch OIDC ID token if available
            try:
                import google.auth.transport.requests
                import google.oauth2.id_token

                auth_req = google.auth.transport.requests.Request()
                id_token = google.oauth2.id_token.fetch_id_token(auth_req, api_url)
                if id_token:
                    headers["Authorization"] = f"Bearer {id_token}"
            except Exception:
                pass

            resp = httpx.post(
                f"{api_url}/api/audit",
                json={
                    "package_id": package.package_id,
                    "package": package.model_dump(mode="json"),
                },
                headers=headers,
                timeout=60.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                from catcher_in_the_rag.schemas import Decision, Finding

                return {
                    "package": package,
                    "decision": Decision.model_validate(data["decision"]),
                    "validated_findings": [
                        Finding.model_validate(f) for f in data.get("findings", [])
                    ],
                    "bartleby_abstained": data.get("bartleby_abstained", False),
                    "bartleby_reason": data.get("bartleby_reason"),
                    "execution_mode": f"API ({data.get('execution_provider', 'unknown')})",
                }
            api_error = f"HTTP {resp.status_code}"
        except Exception as exc:
            api_error = type(exc).__name__
        st.warning(f"API no disponible: modo local/stub ({api_error})")
        st.sidebar.warning("Modo activo: local/stub (API no disponible)")
        result = dict(audit_pipeline(package, llm_provider=get_llm_provider(provider_type="stub")))
        result["execution_mode"] = "local/stub (API no disponible)"
        return result

    provider = resolve_llm_provider(_ui_credential())
    result = dict(audit_pipeline(package, llm_provider=provider))
    result["execution_mode"] = f"local/{getattr(provider, 'provider_type', 'unknown')}"
    return result


# ── Streamlit Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="The Catcher in the RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    _max_live_calls = int(os.getenv("MAX_LIVE_CALLS", "0") or 0)
except ValueError:
    _max_live_calls = 0
if _max_live_calls <= 0:
    st.info("Demo en modo stub; sin llamadas a Gemini (MAX_LIVE_CALLS=0).")

# ── Custom CSS for High-End Typography & Visuals ──────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .decision-card-approve {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .decision-card-review {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .decision-card-reject {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .decision-card-abstain {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .citation-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38BDF8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
    }
    .bartleby-banner {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .cuad-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(71, 85, 105, 0.35);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .cuad-badge-present {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .cuad-badge-absent {
        background: rgba(100, 116, 139, 0.15);
        border: 1px solid rgba(100, 116, 139, 0.35);
        color: #94A3B8;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .score-meter {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 6px;
        overflow: hidden;
        height: 8px;
        margin-top: 4px;
    }
    .score-fill {
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_package_list(data_dir: str = "data/synthetic") -> list[dict[str, Any]]:
    """List all available synthetic package IDs and anomalies."""
    path = Path(data_dir)
    if not path.exists():
        return []

    packages = []
    for pkg_dir in sorted(path.glob("PKG-*")):
        if not pkg_dir.is_dir():
            continue
        labels_file = pkg_dir / "labels.json"
        labels_data: dict[str, Any] = {}
        if labels_file.exists():
            try:
                with labels_file.open(encoding="utf-8") as f:
                    labels_data = json.load(f)
            except Exception:
                pass
        packages.append(
            {
                "id": pkg_dir.name,
                "anomalies": labels_data.get("anomalies", []),
                "expected_decision": labels_data.get("expected_decision", "UNKNOWN"),
                "dir": pkg_dir,
            }
        )
    return packages


def load_package_data(
    pkg_dir: Path,
) -> tuple[ProcurementPackage | None, PackageLabels | None]:
    """Load package and ground truth labels from directory."""
    pkg_file = pkg_dir / "package.json"
    labels_file = pkg_dir / "labels.json"

    package = None
    labels = None

    if pkg_file.exists():
        with pkg_file.open(encoding="utf-8") as f:
            package = ProcurementPackage.model_validate(json.load(f))

    if labels_file.exists():
        with labels_file.open(encoding="utf-8") as f:
            labels = PackageLabels.model_validate(json.load(f))

    return package, labels


def load_evaluation_reports(
    reports_dir: str = "evals/reports",
) -> list[tuple[str, Path]]:
    """Load all evaluation JSON reports."""
    path = Path(reports_dir)
    if not path.exists():
        return []
    reports = []
    canonical = path / "canonical_baseline_report.json"
    if canonical.exists():
        n_pkgs = json.loads(canonical.read_text(encoding="utf-8")).get("total_packages", "?")
        reports.append(
            (f"canonical_baseline_report.json ({n_pkgs} paquetes sintéticos)", canonical)
        )
    for report_file in sorted(path.glob("eval_report_*.json"), reverse=True):
        reports.append((report_file.name, report_file))
    return reports


@st.cache_resource(show_spinner=False)
def get_cuad_loader(
    data_path: str = "data/real/cuad/CUADv1.json",
) -> CUADLoader | None:
    """Get cached CUADLoader instance if dataset file exists."""
    path = Path(data_path)
    if not path.exists():
        return None
    try:
        loader = CUADLoader(cuad_json_path=path)
        loader._ensure_loaded()
        return loader
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_cuad_contract_summaries(
    data_path: str = "data/real/cuad/CUADv1.json", limit: int = 50
) -> list[dict[str, Any]]:
    """Load high-level summaries for the first N CUAD contracts."""
    loader = get_cuad_loader(data_path)
    if not loader:
        return []
    contracts = loader.load_contracts(limit=limit)
    summaries = []
    for c in contracts:
        summaries.append(
            {
                "contract_id": c.contract_id,
                "title": c.title,
                "char_length": c.char_length,
                "present_count": len(c.annotations),
                "absent_count": len(c.absent_categories),
                "categories_present": sorted(c.annotations.keys()),
                "categories_absent": sorted(c.absent_categories),
            }
        )
    return summaries


def build_simulated_cuad_package(
    cuad_contract: CUADContract,
    loader: CUADLoader,
    scenario: str = "Clean / Compliant",
) -> ProcurementPackage:
    """Create a realistic procurement package anchored to a real SEC CUAD contract."""
    contract = loader.to_catcher_contract(cuad_contract)
    vendor_id = contract.vendor_id
    vendor_name = contract.vendor_name
    po_id = f"PO-{cuad_contract.contract_id}"
    inv_id = f"INV-{cuad_contract.contract_id}"
    rc_id = f"RC-{cuad_contract.contract_id}"

    today = date.today()
    due_date = today + timedelta(days=contract.payment_terms_days)

    if scenario == "Unauthorized Surcharge (A09 - Melville Guard)":
        inv_items = [
            LineItem(
                item_id="SKU-REAL",
                description="Procured Commercial Services",
                quantity=Decimal("10"),
                unit_price=Decimal("1000.00"),
                total_amount=Decimal("10000.00"),
            ),
            LineItem(
                item_id="FEE-UNAUTH",
                description="Expedited Delivery Surcharge",
                quantity=Decimal("1"),
                unit_price=Decimal("450.00"),
                total_amount=Decimal("450.00"),
            ),
        ]
        total_inv = Decimal("10450.00")
    elif scenario == "Price Variance (A01 - Rate Sheet Violation)":
        inv_items = [
            LineItem(
                item_id="SKU-REAL",
                description="Procured Commercial Services",
                quantity=Decimal("10"),
                unit_price=Decimal("1350.00"),
                total_amount=Decimal("13500.00"),
            )
        ]
        total_inv = Decimal("13500.00")
    else:  # Clean
        inv_items = [
            LineItem(
                item_id="SKU-REAL",
                description="Procured Commercial Services",
                quantity=Decimal("10"),
                unit_price=Decimal("1000.00"),
                total_amount=Decimal("10000.00"),
            )
        ]
        total_inv = Decimal("10000.00")

    po = PurchaseOrder(
        po_id=po_id,
        contract_id=contract.contract_id,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        issue_date=today,
        expected_delivery_date=today,
        line_items=[
            LineItem(
                item_id="SKU-REAL",
                description="Procured Commercial Services",
                quantity=Decimal("10"),
                unit_price=Decimal("1000.00"),
                total_amount=Decimal("10000.00"),
            )
        ],
        total_amount=Decimal("10000.00"),
    )
    receipt = GoodsReceipt(
        receipt_id=rc_id,
        po_id=po_id,
        delivery_date=today,
        received_by="Receiving Clerk",
        items=[
            ReceiptItem(
                item_id="SKU-REAL",
                quantity_received=Decimal("10"),
                quantity_rejected=Decimal("0"),
                condition="ACCEPTABLE",
            )
        ],
    )
    invoice = Invoice(
        invoice_id=inv_id,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        po_id=po_id,
        invoice_date=today,
        payment_due_date=due_date,
        line_items=inv_items,
        total_amount=total_inv,
    )
    return ProcurementPackage(
        package_id=f"PKG-{cuad_contract.contract_id}-SIM",
        contract=contract,
        po=po,
        receipt=receipt,
        invoice=invoice,
    )


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ The Catcher in the RAG")
    st.markdown("*catches the invoice before it falls off the cliff.*")
    st.divider()

    selected_screen = st.radio(
        "Navigation Screen",
        [
            "📋 Review Workspace",
            "📜 CUAD Real SEC Contracts",
            "📊 Evaluation Dashboard",
            "ℹ️ System Architecture",
        ],
        index=0,
    )

    st.divider()
    st.markdown("#### Access & Budget Shield")
    demo_pin_env = os.getenv("DEMO_ACCESS_PIN", "").strip()
    is_production = bool(os.getenv("K_SERVICE") or os.getenv("ENVIRONMENT") == "production")

    if "is_authenticated" not in st.session_state:
        if is_production:
            # Production FAIL-CLOSED: must have a non-empty PIN configured and validated
            st.session_state["is_authenticated"] = False
        else:
            # Local dev convenience: authenticated by default only if no PIN is configured
            st.session_state["is_authenticated"] = not bool(demo_pin_env)

    if demo_pin_env:
        if not st.session_state["is_authenticated"]:
            st.markdown("🔒 **Public Demo Mode** (Zero-Cost / Cached)")
            st.caption(
                "Live Vertex AI calls are locked to protect GCP budget against bot scraping."
            )
            pin_input = st.text_input(
                "Enter Auditor Access PIN", type="password", key="demo_pin_box"
            )
            if st.button("Unlock Live Vertex AI", use_container_width=True):
                if secrets.compare_digest(pin_input.strip(), demo_pin_env):
                    st.session_state["is_authenticated"] = True
                    st.success("Authenticated.")
                    st.rerun()
                else:
                    st.error("Invalid Access PIN.")
        else:
            st.markdown("🔓 **PIN:** unlocked")
            if st.button("Lock Public Demo Mode", use_container_width=True):
                st.session_state["is_authenticated"] = False
                st.rerun()

    if os.getenv("API_URL", "").strip():
        active_mode = "API remota (API_URL)"
    else:
        try:
            _prov = resolve_llm_provider(_ui_credential())
            active_mode = f"local/{getattr(_prov, 'provider_type', 'unknown')}"
        except PermissionError:
            active_mode = "bloqueado (PIN requerido)"
    st.markdown(f"🟢 **Modo activo:** {active_mode}")
    if "audit_results" in st.session_state:
        last_mode = st.session_state["audit_results"].get("execution_mode")
        if last_mode:
            st.caption(f"Última auditoría: {last_mode}")

    st.divider()
    st.markdown("#### System Safeguards")
    _tau = load_bartleby_thresholds()["min_relevance_score"]
    if not active_mode.startswith("local/"):
        _llm_check = "la decide la API"
    elif active_mode == "local/stub":
        _llm_check = "inactiva (stub)"
    else:
        _llm_check = "activa"
    st.markdown(f"🔒 **Bartleby:** τ = {_tau:.2f} (config); verificación LLM 3b: {_llm_check}")
    _last = st.session_state.get("audit_results")
    if _last:
        _fs = _last.get("validated_findings", [])
        _nv = sum(1 for f in _fs if f.verification_status == "NOT_VERIFIABLE")
        st.markdown(
            f"📑 **Citas (última auditoría):** {len(_fs) - _nv} con cita validada, "
            f"{_nv} NOT_VERIFIABLE"
        )
    else:
        st.markdown("📑 **Citas:** sin auditoría en esta sesión")
    st.markdown("🤖 **Decision Logic:** Deterministic Policy Engine")
    if is_tracing_enabled():
        host_preview = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
            "LANGFUSE_HOST", "cloud.langfuse.com"
        )
        st.markdown(
            f"📡 **Langfuse:** claves definidas (`{host_preview[:22]}`); entrega al servidor no medida"
        )
    else:
        st.markdown("📡 **Langfuse:** sin claves (NoOpTracer)")


# ── SCREEN 1: REVIEW WORKSPACE ────────────────────────────────────────────────
if selected_screen == "📋 Review Workspace":
    st.markdown(
        '<div class="hero-title">Procurement Review Workspace</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Three-way matching & governing contract clause verification with verbatim evidence</div>',
        unsafe_allow_html=True,
    )

    packages = load_package_list()

    col_sel, col_action = st.columns([3, 1])

    with col_sel:
        if packages:
            pkg_options = [
                f"{p['id']} — {', '.join(p['anomalies']) if p['anomalies'] else 'A10 (Clean Package)'} (Expected: {p['expected_decision']})"
                for p in packages
            ]
            selected_idx = st.selectbox(
                "Select Package from Evaluation Corpus",
                range(len(packages)),
                format_func=lambda i: pkg_options[i],
            )
            selected_pkg_info = packages[selected_idx]
            package, labels = load_package_data(selected_pkg_info["dir"])
        else:
            st.info(
                "No synthetic packages found in `data/synthetic/`. Please run `make generate-synthetic`."
            )
            package, labels = None, None

    with col_action:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        run_audit_btn = st.button("🔍 Run Audit Agent", use_container_width=True, type="primary")

    if package is not None:
        if "audit_results" not in st.session_state or run_audit_btn:
            with st.spinner("Executing 3-way match, hybrid contract RAG, and Bartleby guard..."):
                st.session_state["audit_results"] = execute_audit(package)
                st.session_state["current_pkg_id"] = package.package_id

        # Make sure results belong to currently selected package
        if st.session_state.get("current_pkg_id") != package.package_id:
            st.session_state["audit_results"] = execute_audit(package)
            st.session_state["current_pkg_id"] = package.package_id

        res = st.session_state["audit_results"]
        decision = res["decision"]
        findings: list[Finding] = res.get("validated_findings", [])
        bartleby_abstained = res.get("bartleby_abstained", False)
        bartleby_reason = res.get("bartleby_reason")

        # ── Decision Banner ───────────────────────────────────────────────────
        if labels is not None:
            match_status = (
                "✅ Aligned with ground truth"
                if decision.decision == labels.expected_decision
                else "⚠️ Review discrepancy"
            )
            anom_str = (
                ", ".join([a.value for a in labels.anomalies])
                if labels.anomalies
                else "A10 (Clean)"
            )
            st.markdown(
                f"<div style='margin-bottom: 0.6rem; font-size: 0.88rem; color: #94A3B8;'>"
                f"<strong>Ground Truth Benchmark:</strong> Expected <code>{labels.expected_decision.value}</code> "
                f"[{anom_str}] &middot; <em>{match_status}</em></div>",
                unsafe_allow_html=True,
            )
        decision_class = {
            DecisionType.APPROVE: "decision-card-approve",
            DecisionType.REVIEW: "decision-card-review",
            DecisionType.REJECT: "decision-card-reject",
            DecisionType.ABSTAIN: "decision-card-abstain",
        }.get(decision.decision, "decision-card-review")

        badge_emoji = {
            DecisionType.APPROVE: "✅",
            DecisionType.REVIEW: "⚠️",
            DecisionType.REJECT: "🛑",
            DecisionType.ABSTAIN: "📜",
        }.get(decision.decision, "ℹ️")

        st.markdown(
            f"""
            <div class="{decision_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 1.4rem; font-weight: 700;">
                        {badge_emoji} Agent Decision: {decision.decision.value}
                    </div>
                    <div style="font-size: 0.95rem; opacity: 0.85;">
                        Package: <strong>{package.package_id}</strong> | Timestamp: {datetime.now().strftime("%H:%M:%S")}
                    </div>
                </div>
                <div style="margin-top: 0.6rem; font-size: 1.05rem;">
                    <strong>Policy Justification:</strong> {decision.justification}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Bartleby Guard Banner (If Abstained) ──────────────────────────────
        if bartleby_abstained:
            st.markdown(
                f"""
                <div class="bartleby-banner">
                    <div style="font-weight: 600; font-size: 1.05rem; color: #38BDF8;">
                        📜 Bartleby Abstention Guard Activated
                    </div>
                    <div style="font-style: italic; color: #CBD5E1; margin: 0.3rem 0;">
                        "I would prefer not to." — Herman Melville, <em>Bartleby, the Scrivener (1853)</em>
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.95rem;">
                        <strong>Abstention Reason:</strong> {bartleby_reason or "No contractual clause found to substantiate the invoice surcharge."}
                        <br/><em>The system strictly abstains from inventing ungrounded interpretations when contractual evidence is absent.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Findings & Citations Section ──────────────────────────────────────
        st.markdown("### 🔍 Verified Findings & Verbatim Citations")
        if not findings:
            st.success(
                "No discrepancies detected. 3-way match and contract terms are fully compliant."
            )
        else:
            for idx, finding in enumerate(findings, start=1):
                with st.container():
                    col_f1, col_f2 = st.columns([3, 1])
                    with col_f1:
                        anom_code = str(
                            finding.anomaly_type.value
                            if hasattr(finding.anomaly_type, "value")
                            else finding.anomaly_type
                        )
                        st.markdown(f"**{idx}. [{anom_code}] {finding.description}**")
                    with col_f2:
                        sev_str = str(
                            finding.severity.value
                            if hasattr(finding.severity, "value")
                            else finding.severity
                        )
                        sev_color = {
                            "ERROR": "#EF4444",
                            "CRITICAL": "#EF4444",
                            "WARNING": "#F59E0B",
                            "INFO": "#38BDF8",
                        }.get(sev_str, "gray")
                        st.markdown(
                            f"<span style='color:{sev_color}; font-weight:600;'>Severity: {sev_str}</span>",
                            unsafe_allow_html=True,
                        )

                    if finding.citation:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">
                                    VERBATIM CLAUSE CITATION &middot; Document: <code>{finding.citation.document_id}</code> &middot; Offsets: <code>[{finding.citation.start_char}:{finding.citation.end_char}]</code>
                                </span>
                                <div style="margin-top: 0.3rem; font-size: 0.95rem; color: #F1F5F9; font-style: italic;">
                                    &ldquo;{finding.citation.text}&rdquo;
                                </div>
                                <div style="margin-top: 0.3rem; font-size: 0.8rem; color: #10B981; font-weight: 600;">
                                    ✓ Verified verbatim in governing contract text
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.divider()

        # ── 4-Way Document Explorer Tabs ──────────────────────────────────────
        st.markdown("### 📑 Package Document Inspector")
        tab_inv, tab_po, tab_receipt, tab_contract = st.tabs(
            ["🧾 Invoice", "📋 Purchase Order", "📦 Goods Receipt", "📜 Governing Contract"]
        )

        with tab_inv:
            inv = package.invoice
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            col_i1.metric("Invoice ID", inv.invoice_id)
            col_i2.metric("Vendor", inv.vendor_name)
            col_i3.metric("Invoice Date", inv.invoice_date.isoformat())
            col_i4.metric("Total Billed", f"${inv.total_amount:,.2f}")

            st.markdown(
                f"**Vendor ID:** `{inv.vendor_id}` | **Associated PO:** `{inv.po_id}` | **Payment Due:** `{inv.payment_due_date.isoformat()}`"
            )

            inv_items = [
                {
                    "Item ID": it.item_id,
                    "Description": it.description,
                    "Quantity": float(it.quantity),
                    "Unit Price": float(it.unit_price),
                    "Total Amount": float(it.total_amount),
                }
                for it in inv.line_items
            ]
            st.dataframe(inv_items, use_container_width=True)

        with tab_po:
            po = package.po
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("PO ID", po.po_id)
            col_p2.metric("Vendor", po.vendor_name)
            col_p3.metric("Issue Date", po.issue_date.isoformat())
            col_p4.metric("PO Total", f"${po.total_amount:,.2f}")

            st.markdown(
                f"**Contract Ref:** `{po.contract_id}` | **Vendor ID:** `{po.vendor_id}` | **Expected Delivery:** `{po.expected_delivery_date.isoformat()}`"
            )
            po_items = [
                {
                    "Item ID": it.item_id,
                    "Description": it.description,
                    "Quantity": float(it.quantity),
                    "Unit Price": float(it.unit_price),
                    "Total Amount": float(it.total_amount),
                }
                for it in po.line_items
            ]
            st.dataframe(po_items, use_container_width=True)

        with tab_receipt:
            rc = package.receipt
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Receipt ID", rc.receipt_id)
            col_r2.metric("PO Ref", rc.po_id)
            col_r3.metric("Delivery Date", rc.delivery_date.isoformat())
            col_r4.metric("Received By", rc.received_by)

            rc_items = [
                {
                    "Item ID": it.item_id,
                    "Quantity Received": float(it.quantity_received),
                    "Quantity Rejected": float(it.quantity_rejected),
                    "Condition": it.condition,
                }
                for it in rc.items
            ]
            st.dataframe(rc_items, use_container_width=True)

        with tab_contract:
            c = package.contract
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            col_c1.metric("Contract ID", c.contract_id)
            col_c2.metric("Vendor", c.vendor_name)
            col_c3.metric("Effective", c.effective_date.isoformat())
            col_c4.metric(
                "Expiration", c.expiration_date.isoformat() if c.expiration_date else "NOT FOUND"
            )

            st.markdown(
                f"**Agreed Payment Terms:** `Net {c.payment_terms_days} days` | **Auto-Renewal:** `{c.auto_renew}`"
            )

            st.markdown("#### Contract Rates")
            rates = [
                {
                    "Item ID": r.item_id,
                    "Description": r.description,
                    "Price Ceiling": float(r.max_unit_price),
                }
                for r in c.rate_sheet
            ]
            st.dataframe(rates, use_container_width=True)

            st.markdown("#### Governing Contract Clauses")
            for clause in c.clauses:
                with st.expander(
                    f"Clause {clause.clause_id}: {clause.title} ({clause.clause_type})"
                ):
                    st.write(clause.text)

        # ── Human Reviewer Action Box ─────────────────────────────────────────
        st.divider()
        st.markdown("### ✍️ Human-in-the-Loop Audit Decision")
        col_ov1, col_ov2 = st.columns([2, 1])

        with col_ov1:
            auditor_notes = st.text_area(
                "Auditor Review Notes",
                placeholder="Enter justification, notes for vendor, or resolution comments...",
                key=f"notes_{package.package_id}",
            )

        with col_ov2:
            st.markdown("<div style='height: 25px'></div>", unsafe_allow_html=True)
            override_action = st.radio(
                "Final Action",
                ["Confirm Agent Decision", "Override: Approve", "Override: Reject"],
                key=f"act_{package.package_id}",
            )
            confirm_btn = st.button(
                "Submit Audit Sign-Off", type="secondary", use_container_width=True
            )

            st.caption("Nota: este botón no guarda nada; solo muestra un mensaje en pantalla.")
            if confirm_btn:
                st.info(
                    f"No guardado: {package.package_id} — {override_action} "
                    "(sin persistencia implementada)"
                )


# ── SCREEN 2: CUAD REAL SEC CONTRACTS ────────────────────────────────────────
elif selected_screen == "📜 CUAD Real SEC Contracts":
    st.markdown(
        '<div class="hero-title">CUAD Real-World Contract Corpus</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">The Atticus Project (CC BY 4.0): commercial agreements filed with the SEC, annotated with clause spans and absence labels</div>',
        unsafe_allow_html=True,
    )

    cuad_loader = get_cuad_loader()
    if cuad_loader is None:
        st.warning(
            "CUAD dataset not found at `data/real/cuad/CUADv1.json`. "
            "Please run `make download-data` to download and unpack the corpus."
        )
    else:
        # Top KPI Metrics Banner
        st.metric("SEC Edgar Contracts (loaded from CUADv1.json)", f"{cuad_loader.count()}")

        st.divider()

        summaries = load_cuad_contract_summaries(limit=50)
        if not summaries:
            st.info("No contracts loaded. Check dataset directory.")
        else:
            col_sel, col_info = st.columns([2, 1])
            with col_sel:
                search_filter = st.text_input(
                    "🔎 Filter Contracts by Title / Keyword",
                    placeholder="e.g. DISTRIBUTOR, ENERGY, SOFTWARE...",
                ).lower()
                filtered_summaries = [
                    s
                    for s in summaries
                    if (
                        not search_filter
                        or search_filter in s["title"].lower()
                        or search_filter in s["contract_id"].lower()
                    )
                ]
                if not filtered_summaries:
                    filtered_summaries = summaries

                options = [
                    f"{s['contract_id']} — {s['title'][:55]} ({s['char_length']:,} chars | {s['present_count']} clauses)"
                    for s in filtered_summaries
                ]
                sel_idx = st.selectbox(
                    "Select SEC Commercial Contract",
                    range(len(filtered_summaries)),
                    format_func=lambda i: options[i],
                )
                selected_meta = filtered_summaries[sel_idx]

            contract_obj: CUADContract | None = None
            contracts = cuad_loader.load_contracts(limit=50)
            for c in contracts:
                if c.contract_id == selected_meta["contract_id"]:
                    contract_obj = c
                    break

            with col_info:
                if contract_obj:
                    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="cuad-card">
                            <div style="font-size: 0.9rem; color: #94A3B8;">Document ID: <code>{contract_obj.contract_id}</code></div>
                            <div style="font-weight: 700; color: #38BDF8; font-size: 1.05rem; margin-top: 2px;">{contract_obj.title[:35]}...</div>
                            <div style="margin-top: 6px; font-size: 0.85rem; color: #E2E8F0;">
                                📏 <strong>Length:</strong> {contract_obj.char_length:,} characters (~{contract_obj.char_length // 4:,} tokens)<br/>
                                ✅ <strong>Present Clauses:</strong> {len(contract_obj.annotations)} / {len(contract_obj.annotations) + len(contract_obj.absent_categories)}<br/>
                                📜 <strong>Absent Categories:</strong> {len(contract_obj.absent_categories)} / {len(contract_obj.annotations) + len(contract_obj.absent_categories)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if contract_obj:
                tab_cats, tab_rag, tab_sim, tab_raw = st.tabs(
                    [
                        "📑 Clause Categories",
                        "⚡ Live Hybrid RAG & Bartleby Playground",
                        "🧾 Simulated Procurement Audit",
                        "📜 Full SEC Contract Text",
                    ]
                )

                with tab_cats:
                    st.markdown("#### Annotated Clause Categories (CUAD Ground Truth)")
                    col_f, col_cat = st.columns([1, 2])
                    with col_f:
                        cat_filter = st.radio(
                            "Filter Category Status",
                            [
                                "All Categories",
                                "Present Clauses Only",
                                "Absent Clauses Only (Bartleby Negatives)",
                            ],
                        )

                    all_categories = sorted(
                        list(contract_obj.annotations.keys()) + list(contract_obj.absent_categories)
                    )
                    if cat_filter == "Present Clauses Only":
                        display_cats = sorted(contract_obj.annotations.keys())
                    elif cat_filter == "Absent Clauses Only (Bartleby Negatives)":
                        display_cats = sorted(contract_obj.absent_categories)
                    else:
                        display_cats = all_categories

                    with col_cat:
                        if display_cats:
                            sel_cat = st.selectbox(
                                "Select Clause Category to Inspect", display_cats
                            )
                        else:
                            sel_cat = None
                            st.info("No categories match the selected filter.")

                    if sel_cat:
                        is_present = sel_cat in contract_obj.annotations
                        if is_present:
                            spans = contract_obj.annotations[sel_cat]
                            st.markdown(
                                f'<span class="cuad-badge-present">✅ PRESENT IN CONTRACT ({len(spans)} verbatim span{"s" if len(spans) > 1 else ""})</span>',
                                unsafe_allow_html=True,
                            )
                            for s_idx, sp in enumerate(spans, 1):
                                st.markdown(
                                    f"""
                                    <div class="citation-box" style="margin-top: 0.8rem;">
                                        <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">
                                            SPAN #{s_idx} &middot; Offsets: <code>[{sp.start_char}:{sp.end_char}]</code> &middot; Length: {len(sp.text)} chars
                                        </div>
                                        <div style="margin-top: 0.4rem; font-size: 0.95rem; color: #F8FAFC; line-height: 1.5;">
                                            &ldquo;{sp.text}&rdquo;
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                '<span class="cuad-badge-absent">📜 ABSENT IN CONTRACT (GROUND TRUTH NEGATIVE FOR BARTLEBY)</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"""
                                <div class="bartleby-banner" style="margin-top: 0.8rem;">
                                    <div style="font-weight: 600; color: #38BDF8;">Ground Truth Absence: Category '{sel_cat}' is not in this contract</div>
                                    <div style="margin-top: 0.4rem; color: #CBD5E1; font-size: 0.95rem;">
                                        In CUAD v1, this category was explicitly verified as absent (<code>answers: []</code>, <code>is_impossible: True</code>).
                                        When an auditor or query searches for terms governed by this category, <strong>The Catcher in the RAG's Bartleby Guard</strong> must strictly decline to assert terms (<code>NO_EVIDENCE</code>) rather than hallucinating plausible terms.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with tab_rag:
                    st.markdown("#### Live Hybrid RAG & Bartleby Abstention Playground")
                    st.markdown(
                        "Execute real-time hybrid retrieval (BM25 lexical + dense vector embeddings) across the entire 50k+ character contract, and inspect Bartleby's deterministic abstention gate."
                    )

                    col_q1, col_q2 = st.columns([3, 1])
                    with col_q1:
                        query_presets = [
                            ("Governing Law and dispute resolution jurisdiction", "Governing Law"),
                            (
                                "Liquidated damages, late delivery penalties, and delay remedies",
                                "Liquidated Damages",
                            ),
                            (
                                "Pricing restrictions, rate adjustments, and fee schedule",
                                "Price Restrictions",
                            ),
                            (
                                "Cap on liability, indemnification, and consequential damages",
                                "Cap On Liability",
                            ),
                            (
                                "Agreement expiration date, term duration, and renewal",
                                "Expiration Date",
                            ),
                            (
                                "Unauthorized delivery surcharge or expedited freight fee (ABSENT)",
                                "Absent Surcharge",
                            ),
                            ("Custom Query...", "Custom"),
                        ]
                        preset_labels = [p[0] for p in query_presets]
                        sel_preset = st.selectbox(
                            "Select Audit Query Preset", range(len(preset_labels)), index=0
                        )

                        if sel_preset == len(query_presets) - 1:
                            query_text = st.text_input(
                                "Enter Custom Audit Query", value="governing law and jurisdiction"
                            )
                        else:
                            query_text = query_presets[sel_preset][0]

                    with col_q2:
                        top_k = st.slider("Top-K Clauses", min_value=1, max_value=5, value=3)
                        run_rag_btn = st.button(
                            "⚡ Run Hybrid Retrieval", type="primary", use_container_width=True
                        )

                    if run_rag_btn or f"rag_cuad_{contract_obj.contract_id}" in st.session_state:
                        st.session_state[f"rag_cuad_{contract_obj.contract_id}"] = (
                            query_text,
                            top_k,
                        )
                        cat_contract = cuad_loader.to_catcher_contract(contract_obj)
                        retriever = ContractRetriever(cat_contract)

                        t0 = time.perf_counter()
                        retrieved = retriever.retrieve(query_text, top_k=top_k)
                        retrieval_ms = (time.perf_counter() - t0) * 1000

                        guard = BartlebyGuard()
                        assessment = guard.evaluate_retrieval(retrieved, query_text)

                        st.markdown(
                            f"<div style='font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem;'>"
                            f"⚡ Retrieved {len(retrieved)} candidate clauses in <strong>{retrieval_ms:.1f} ms</strong> across {len(cat_contract.clauses)} segmented contract sections.</div>",
                            unsafe_allow_html=True,
                        )

                        if assessment.abstained:
                            st.markdown(
                                f"""
                                <div class="bartleby-banner">
                                    <div style="font-weight: 700; font-size: 1.1rem; color: #38BDF8;">
                                        📜 Bartleby Abstention Guard Triggered (NO_EVIDENCE)
                                    </div>
                                    <div style="font-style: italic; color: #CBD5E1; margin: 0.3rem 0;">
                                        "I would prefer not to." — Herman Melville, <em>Bartleby, the Scrivener (1853)</em>
                                    </div>
                                    <div style="color: #F1F5F9; font-size: 0.95rem; margin-top: 0.4rem;">
                                        <strong>Abstention Reason:</strong> {assessment.reason}
                                    </div>
                                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">
                                        ✓ The highest candidate clause scored below the calibrated relevance threshold (0.50). The agent refuses to hallucinate authorization.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="decision-card-approve">
                                    <div style="font-weight: 700; font-size: 1.05rem; color: #10B981;">
                                        ✅ Contract Evidence Found & Verified
                                    </div>
                                    <div style="font-size: 0.95rem; margin-top: 0.2rem; color: #E2E8F0;">
                                        Top retrieved clause scored <strong>{retrieved[0].score:.3f}</strong> (threshold &ge; {guard.min_relevance_score:.2f}). Evidence is sufficient for policy evaluation.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("##### Candidate Scored Clauses")
                        for r_idx, sc in enumerate(retrieved, 1):
                            col_s1, col_s2 = st.columns([3, 1])
                            with col_s1:
                                st.markdown(f"**Rank #{r_idx}: {sc.title}**")
                                st.markdown(
                                    f"<div class='citation-box' style='margin-top: 0.2rem; font-size: 0.9rem;'>"
                                    f"<span style='color: #94A3B8; font-size: 0.8rem;'>Offsets: [{sc.start_char}:{sc.end_char}]</span><br/>"
                                    f"{sc.text[:400]}{'...' if len(sc.text) > 400 else ''}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            with col_s2:
                                st.metric("Hybrid Score", f"{sc.score:.3f}")
                                st.markdown(
                                    f"<div style='font-size: 0.8rem; color: #94A3B8;'>"
                                    f"BM25 Lexical: <code>{sc.bm25_score:.3f}</code><br/>"
                                    f"Dense Vector: <code>{sc.dense_score:.3f}</code>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            st.divider()

                with tab_sim:
                    st.markdown("#### End-to-End Procurement Package Audit Simulation")
                    st.markdown(
                        "Simulate an entire 4-way procurement audit where this real SEC contract acts as the governing legal agreement."
                    )

                    col_sc1, col_sc2 = st.columns([2, 1])
                    with col_sc1:
                        sim_scenario = st.selectbox(
                            "Select Procurement Package Scenario",
                            [
                                "Clean / Compliant (All terms match agreement)",
                                "Unauthorized Surcharge (A09 - Melville Guard Activates)",
                                "Price Variance (A01 - Rate Sheet Violation)",
                            ],
                        )
                    with col_sc2:
                        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                        run_sim_btn = st.button(
                            "🔍 Run Audit with SEC Contract",
                            type="primary",
                            use_container_width=True,
                        )

                    sim_pkg = build_simulated_cuad_package(contract_obj, cuad_loader, sim_scenario)

                    if run_sim_btn or f"sim_res_{contract_obj.contract_id}" in st.session_state:
                        st.session_state[f"sim_res_{contract_obj.contract_id}"] = sim_scenario
                        with st.spinner("Executing 3-way match and contract audit..."):
                            audit_res = execute_audit(sim_pkg)

                        dec = audit_res["decision"]
                        findings_list = audit_res.get("validated_findings", [])
                        b_abstained = audit_res.get("bartleby_abstained", False)

                        dec_class = {
                            DecisionType.APPROVE: "decision-card-approve",
                            DecisionType.REVIEW: "decision-card-review",
                            DecisionType.REJECT: "decision-card-reject",
                            DecisionType.ABSTAIN: "decision-card-abstain",
                        }.get(dec.decision, "decision-card-review")

                        emoji = {
                            DecisionType.APPROVE: "✅",
                            DecisionType.REVIEW: "⚠️",
                            DecisionType.REJECT: "🛑",
                            DecisionType.ABSTAIN: "📜",
                        }.get(dec.decision, "ℹ️")

                        st.markdown(
                            f"""
                            <div class="{dec_class}">
                                <div style="font-size: 1.3rem; font-weight: 700;">
                                    {emoji} Policy Decision: {dec.decision.value}
                                </div>
                                <div style="margin-top: 0.4rem; font-size: 1.0rem;">
                                    <strong>Justification:</strong> {dec.justification}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if b_abstained:
                            st.markdown(
                                f"""
                                <div class="bartleby-banner">
                                    <div style="font-weight: 600; color: #38BDF8;">📜 Bartleby Guard Exercised Abstention</div>
                                    <div style="color: #CBD5E1; font-size: 0.95rem; margin-top: 0.2rem;">
                                        {audit_res.get("bartleby_reason")}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("##### Findings Detected")
                        if not findings_list:
                            st.success("No discrepancies detected. Compliant.")
                        else:
                            for f_idx, f_item in enumerate(findings_list, 1):
                                st.markdown(
                                    f"**{f_idx}. [{f_item.anomaly_type.value if hasattr(f_item.anomaly_type, 'value') else f_item.anomaly_type}] {f_item.description}**"
                                )
                                if f_item.citation:
                                    st.markdown(
                                        f"<div class='citation-box'>"
                                        f"<span style='color: #94A3B8; font-size: 0.8rem;'>Quote from {f_item.citation.document_id}</span><br/>"
                                        f"&ldquo;{f_item.citation.text}&rdquo;"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                with tab_raw:
                    st.markdown("#### Full SEC Contract Raw Text")
                    search_term = st.text_input(
                        "Search within Raw Agreement", placeholder="e.g. indemnity, court"
                    )
                    if search_term:
                        count = contract_obj.raw_text.lower().count(search_term.lower())
                        st.info(f"Found {count} occurrence(s) of '{search_term}'.")
                    st.text_area(
                        "Raw Contract Text (EDGAR Filing)",
                        value=contract_obj.raw_text,
                        height=450,
                    )


# ── SCREEN 3: EVALUATION DASHBOARD ────────────────────────────────────────────
elif selected_screen == "📊 Evaluation Dashboard":
    st.markdown(
        '<div class="hero-title">Reproducible Evaluation Dashboard</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Rigorous benchmark metrics, ablation comparisons, real-world CUAD evaluation, and model bake-off</div>',
        unsafe_allow_html=True,
    )

    eval_mode = st.radio(
        "Select Benchmark Track",
        [
            "📦 Synthetic Baseline (A01–A10, no es evidencia)",
            "📜 Real-World CUAD Contract Benchmark (SEC Edgar)",
            "⚡ Vertex AI Model Bake-Off (Gemini Flash vs Pro vs Rules)",
        ],
        horizontal=True,
    )

    st.divider()

    if eval_mode == "📦 Synthetic Baseline (A01–A10, no es evidencia)":
        reports = load_evaluation_reports()

        if not reports:
            st.warning(
                "No evaluation reports found in `evals/reports/`. Run `make eval-ablations` to produce benchmark reports."
            )
        else:
            report_names = [name for name, _ in reports]
            selected_report_name = st.selectbox("Select Evaluation Report", report_names)
            report_path = dict(reports)[selected_report_name]

            with report_path.open(encoding="utf-8") as f:
                data = json.load(f)

            # ── Top KPI Metric Cards ──────────────────────────────────────────────
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

            total_pkgs = data.get("total_packages", 0)
            acc = data.get("decision_accuracy", 0.0) * 100
            mean_lat = data.get("mean_latency_ms", 0.0)
            clean_fp = data.get("clean_package_false_positive_rate", 0.0) * 100
            cite_val = data.get("citation_validity_rate", 0.0) * 100

            bartleby_info = data.get("bartleby", {})
            correct_abstain = bartleby_info.get("correct_abstention_rate", 1.0) * 100

            kpi1.metric("Decision Accuracy", f"{acc:.1f}%", f"{total_pkgs} pkgs")
            kpi2.metric("Citation Validity", f"{cite_val:.1f}%", "Zero fabricated quotes")
            kpi3.metric("Bartleby Abstention", f"{correct_abstain:.1f}%", "Melville guard")
            kpi4.metric("Clean FP Rate", f"{clean_fp:.1f}%", "Target: 0.0%")
            kpi5.metric("Mean Latency", f"{mean_lat:.1f} ms", "Offline deterministic")

            st.divider()

            # ── Ablation Comparison Section ───────────────────────────────────────
            st.markdown("### 🔬 Ablación sintética: con vs. sin evaluador de cláusulas")
            st.caption(
                "Leído de canonical_baseline_report.json y canonical_norag_report.json "
                "(plantillas sintéticas; no es evidencia empírica)."
            )
            reports_dir = Path("evals/reports")
            base_p = reports_dir / "canonical_baseline_report.json"
            norag_p = reports_dir / "canonical_norag_report.json"
            if base_p.exists() and norag_p.exists():
                base_acc = json.loads(base_p.read_text(encoding="utf-8"))["decision_accuracy"]
                norag_acc = json.loads(norag_p.read_text(encoding="utf-8"))["decision_accuracy"]
                col_ab1, col_ab2 = st.columns(2)
                col_ab1.metric("Exactitud con cláusulas", f"{base_acc * 100:.1f}%")
                col_ab2.metric(
                    "Exactitud sin cláusulas",
                    f"{norag_acc * 100:.1f}%",
                    f"{(norag_acc - base_acc) * 100:+.1f} pts",
                )
            else:
                st.warning("Faltan los reportes canónicos; corre `make eval-ablations`.")

            st.divider()

            # ── Anomaly Taxonomy Per-Class Breakdown ──────────────────────────────
            st.markdown("### 🎯 Anomaly Taxonomy Matrix (A01 – A10)")
            class_metrics = data.get("per_class_metrics", {})
            if class_metrics:
                taxonomy_descriptions = {
                    "A01": "Price > contract rate ceiling",
                    "A02": "Quantity billed > received",
                    "A03": "Duplicate invoice number",
                    "A04": "Payment terms mismatch against contract",
                    "A05": "Late-delivery penalty applicable",
                    "A06": "Contract expired / auto-renewal triggered",
                    "A07": "Unknown vendor or unlinked PO",
                    "A08": "Amount exceeds $50k approval threshold",
                    "A09": "Governing clause absent (must abstain)",
                    "A10": "Clean package (false-positive control)",
                }
                rows = []
                for code in sorted(taxonomy_descriptions.keys()):
                    m = class_metrics.get(code, {})
                    support = m.get("support", 0)
                    prec = m.get("precision", 0.0) * 100
                    rec = m.get("recall", 0.0) * 100
                    f1 = m.get("f1_score", 0.0)
                    rows.append(
                        {
                            "Anomaly ID": code,
                            "Description": taxonomy_descriptions[code],
                            "Support": support,
                            "Precision": f"{prec:.1f}%",
                            "Recall": f"{rec:.1f}%",
                            "F1-Score": f"{f1:.3f}",
                        }
                    )
                st.dataframe(rows, use_container_width=True)

            # ── Failure Cases Inspector ───────────────────────────────────────────
            st.divider()
            st.markdown("### ⚠️ Failure Cases & Edge Case Inspection")
            failures = data.get("failure_cases", [])
            if not failures:
                st.success("Zero failure cases recorded in this evaluation run!")
            else:
                st.info(
                    f"Showing {len(failures)} failure case(s) from this run for diagnostic transparency."
                )
                for f_case in failures[:15]:
                    with st.expander(
                        f"📦 {f_case.get('package_id')} — Expected {f_case.get('expected_decision')}, Got {f_case.get('actual_decision')}"
                    ):
                        st.write(
                            f"**Justification:** {f_case.get('justification') or f_case.get('explanation')}"
                        )

    elif eval_mode == "📜 Real-World CUAD Contract Benchmark (SEC Edgar)":
        cuad_report_path = Path("evals/reports/cuad_benchmark_latest.json")
        if not cuad_report_path.exists():
            st.warning(
                "CUAD benchmark report not found at `evals/reports/cuad_benchmark_latest.json`. Run `make eval-cuad` to produce it."
            )
        else:
            with cuad_report_path.open(encoding="utf-8") as f:
                cuad_data = json.load(f)

            rep = cuad_data.get("report", {})
            st.markdown("### 🏆 Real-World Retrieval & Melville Abstention on SEC Contracts")
            st.markdown(
                "Benchmark executed on authentic commercial contracts filed with the SEC, testing hybrid RAG retrieval against expert-annotated spans and Melville abstention on ground-truth absent clauses."
            )

            emb_prov = rep.get("embedding_provider", "Hybrid BM25 + Dense")
            emb_model = rep.get("embedding_model_id", "text-embedding-005")
            st.caption(
                f"🧬 Active Dense Vector Index: **{emb_prov}** (`{emb_model}`) | Contracts: {rep.get('total_contracts_evaluated', 0)} (Train: {rep.get('train_contracts_count', 0)}, Val: {rep.get('val_contracts_count', 0)})"
            )

            # Metrics row 1: Retrieval
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Recall @ 1", f"{rep.get('recall_at_1', 0.0) * 100:.1f}%", "Top-1 candidate")
            r2.metric("Recall @ 3", f"{rep.get('recall_at_3', 0.0) * 100:.1f}%", "Top-3 candidate")
            r3.metric("Recall @ 5", f"{rep.get('recall_at_5', 0.0) * 100:.1f}%", "Top-5 candidate")
            r4.metric("Mean Reciprocal Rank", f"{rep.get('mean_reciprocal_rank', 0.0):.4f}", "MRR")

            # Metrics row 2: Bartleby Dual-Sided Abstention & Latency
            abs_ci = rep.get("bartleby_correct_abstention_ci95", [0.0, 0.0])
            false_abs_ci = rep.get("bartleby_false_abstention_ci95", [0.0, 0.0])
            b1, b2, b3, b4 = st.columns(4)
            b1.metric(
                "Correct Abstention (Negatives)",
                f"{rep.get('bartleby_correct_abstention_rate', 0.0) * 100:.1f}%",
                f"95% CI: [{abs_ci[0] * 100:.1f}%, {abs_ci[1] * 100:.1f}%]",
            )
            b2.metric(
                "False Abstention (Present)",
                f"{rep.get('bartleby_false_abstention_rate', 0.0) * 100:.1f}%",
                f"95% CI: [{false_abs_ci[0] * 100:.1f}%, {false_abs_ci[1] * 100:.1f}%]",
            )
            b3.metric(
                "Spurious False Assertion",
                f"{rep.get('bartleby_false_assertion_rate', 0.0) * 100:.1f}%",
                "Target: <3.0%",
            )
            b4.metric(
                "Mean Retrieval Latency",
                f"{rep.get('mean_retrieval_latency_ms', 0.0):.1f} ms",
                "SEC Full Texts",
            )

            st.divider()

            # Retrieval depth bar chart
            st.markdown("#### Retrieval Depth vs Dual-Sided Abstention")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("**Recall by Retrieval Depth (k)**")
                recall_chart = {
                    "Recall @ 1": [rep.get("recall_at_1", 0.0) * 100],
                    "Recall @ 3": [rep.get("recall_at_3", 0.0) * 100],
                    "Recall @ 5": [rep.get("recall_at_5", 0.0) * 100],
                }
                st.bar_chart(recall_chart)
            with chart_col2:
                st.markdown(
                    "**Bartleby Abstention Integrity (Ground-Truth Negatives vs Positives)**"
                )
                abstain_chart = {
                    "Correct Abstention on Negatives (%)": [
                        rep.get("bartleby_correct_abstention_rate", 0.0) * 100
                    ],
                    "False Abstention on Present Clauses (%)": [
                        rep.get("bartleby_false_abstention_rate", 0.0) * 100
                    ],
                }
                st.bar_chart(abstain_chart)

            st.divider()

            # Sample cases table
            st.markdown("#### Sample Retrieval Benchmark Cases")
            cases = cuad_data.get("sample_retrieval_cases", [])
            if cases:
                table_rows = []
                for c in cases[:15]:
                    table_rows.append(
                        {
                            "Contract ID": c.get("contract_id"),
                            "Category": c.get("category"),
                            "Query": c.get("query"),
                            "GT Spans": c.get("num_ground_truth_spans"),
                            "Hit @ 1": "✅" if c.get("hit_at_1") else "❌",
                            "Hit @ 3": "✅" if c.get("hit_at_3") else "❌",
                            "Hit @ 5": "✅" if c.get("hit_at_5") else "❌",
                            "MRR": f"{c.get('reciprocal_rank', 0.0):.2f}",
                            "Latency": f"{c.get('latency_ms', 0.0):.1f} ms",
                        }
                    )
                st.dataframe(table_rows, use_container_width=True)

                st.markdown("##### Detailed Case Deep-Dive")
                for c in cases[:5]:
                    with st.expander(
                        f"📜 {c.get('contract_id')} — {c.get('category')} (Hit@5: {c.get('hit_at_5')})"
                    ):
                        st.write(f"**Query:** `{c.get('query')}`")
                        st.write(f"**Ground Truth Spans:** {c.get('num_ground_truth_spans')}")
                        st.write("**Top Retrieved Clauses:**")
                        for idx, p in enumerate(c.get("retrieved_top_5", []), 1):
                            st.markdown(f"*{idx}.* {p}")

    elif eval_mode == "⚡ Vertex AI Model Bake-Off (Gemini Flash vs Pro vs Rules)":
        bakeoff_report_path = Path("evals/reports/bakeoff_latest.json")
        if not bakeoff_report_path.exists():
            st.warning(
                "Bake-off report not found at `evals/reports/bakeoff_latest.json`. Run `make eval-bakeoff` to produce it."
            )
        else:
            with bakeoff_report_path.open(encoding="utf-8") as f:
                bakeoff_data = json.load(f)

            st.markdown("### ⚡ Live Vertex AI Structured Extraction Bake-Off")
            st.markdown(
                "Head-to-head empirical comparison between deterministic rule parsing and GA Gemini models (`gemini-3.5-flash`, `gemini-3.8-flash`) over Google Cloud Vertex AI ADC."
            )

            candidates = bakeoff_data.get("candidates", [])
            if candidates:
                cand_cols = st.columns(len(candidates))
                for idx, cand in enumerate(candidates):
                    with cand_cols[idx]:
                        st.markdown(
                            f"""
                            <div class="cuad-card">
                                <div style="font-weight: 700; color: #38BDF8; font-size: 1.05rem;">{cand.get("candidate_name")}</div>
                                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">Model ID: <code>{cand.get("model_id")}</code></div>
                                <div style="margin-top: 8px; font-size: 0.9rem;">
                                    ⏱️ <strong>Latency:</strong> {cand.get("mean_latency_ms", 0.0):.1f} ms<br/>
                                    💰 <strong>Cost / 1k:</strong> ${cand.get("estimated_cost_per_1k_usd", 0.0):.2f}<br/>
                                    🎯 <strong>Accuracy:</strong> {cand.get("overall_perfect_accuracy", 0.0) * 100:.1f}%<br/>
                                    ⚠️ <strong>Error Rate:</strong> {cand.get("error_rate", 0.0) * 100:.1f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.divider()

                # Detailed comparisons
                bake_table = []
                for cand in candidates:
                    accs = cand.get("field_accuracies", {})
                    bake_table.append(
                        {
                            "Candidate": cand.get("candidate_name"),
                            "Provider": cand.get("provider_type"),
                            "Latency (ms)": f"{cand.get('mean_latency_ms', 0.0):.1f}",
                            "Cost / 1k ($)": f"${cand.get('estimated_cost_per_1k_usd', 0.0):.2f}",
                            "Terms Accuracy": f"{accs.get('payment_terms_days', 0.0) * 100:.0f}%",
                            "Expiration Accuracy": f"{accs.get('expiration_date', 0.0) * 100:.0f}%",
                            "Penalty Accuracy": f"{accs.get('late_delivery_penalty_pct_per_day', 0.0) * 100:.0f}%",
                            "Rate Sheet Accuracy": f"{accs.get('rate_sheet', 0.0) * 100:.0f}%",
                            "Overall Exact Match": f"{cand.get('overall_perfect_accuracy', 0.0) * 100:.1f}%",
                        }
                    )
                st.dataframe(bake_table, use_container_width=True)


# ── SCREEN 3: SYSTEM ARCHITECTURE ─────────────────────────────────────────────
elif selected_screen == "ℹ️ System Architecture":
    st.markdown(
        '<div class="hero-title">System Architecture & Non-Negotiables</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Rigorous design principles for audit agents in procurement workflows</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ```
         PDF / JSON Procurement Package
                    │
         [1 Ingest] → [2 Classify] → [3 Extract (LLM, schema-constrained)]
                                              │
                                 [4 Link PO / Receipt by ID]
                                              │
                                 [5 Match (Deterministic 3-way, tolerances)]
                                              │
                      [6 Retrieve Contract Terms (BM25 + Dense Hybrid RAG)]
                                              │
                      [6b Bartleby: Abstention Gate → NO_EVIDENCE or continue]
                                              │
                      [7 Apply Terms (LLM proposes finding + verbatim quote)]
                                              │
                      [8 Validate (Verbatim quote checked) → Decide (Policy Engine)]
                                              │
                      Approve | Human Review (with evidence) | Reject
        ```
        """
    )

    st.markdown(
        """
        ### Core Principles
        1. **LLM Extracts and Explains; Deterministic Code Decides:** Decisions (`APPROVE`, `REVIEW`, `REJECT`) are issued by a deterministic policy engine, never left to LLM stochasticity.
        2. **Verifiable Citations:** Every finding carries document ID, character span offsets, and a verbatim quote. Findings with non-verbatim quotes are dropped by the validator.
        3. **Abstain Over Guess (`bartleby`):** When the contract is silent (A09), the system declines to hallucinate rules and issues `NO_EVIDENCE`.
        4. **Offline Stub Execution:** The entire benchmark and audit pipeline runs deterministically offline without LLM API calls or cloud credentials.
        """
    )
