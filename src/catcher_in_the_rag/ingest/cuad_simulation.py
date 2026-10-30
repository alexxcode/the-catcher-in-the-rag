"""Simulated procurement packages anchored to a real CUAD contract (UI demo and tests)."""

from datetime import date, timedelta
from decimal import Decimal

from catcher_in_the_rag.ingest.cuad_loader import CUADContract, CUADLoader
from catcher_in_the_rag.schemas import (
    GoodsReceipt,
    Invoice,
    LineItem,
    ProcurementPackage,
    PurchaseOrder,
    ReceiptItem,
)

SCENARIO_MATCH = "Invoice matches contract (expiry may be unverifiable)"
SCENARIO_SURCHARGE_A09 = "Unauthorized Surcharge (A09 - Melville Guard Activates)"
SCENARIO_PRICE_A01 = "Price Variance (A01 - Rate Sheet Violation)"
SIMULATION_SCENARIOS = (SCENARIO_MATCH, SCENARIO_SURCHARGE_A09, SCENARIO_PRICE_A01)

# Appended only to the simulated package of the A01 scenario. The CUAD contract has no rate
# sheet, so without it the A01 finding has no verbatim citation and validate drops it. The
# rate line uses the format the rule-based extractor parses and the matcher cites.
SIMULATED_RATE_ADDENDUM = (
    "\n\nSimulated rate schedule (demo addendum)\n"
    "* SKU-REAL - Procured Commercial Services: Maximum Unit Rate of $1000.00 USD\n"
)


def build_simulated_cuad_package(
    cuad_contract: CUADContract,
    loader: CUADLoader,
    scenario: str = SCENARIO_MATCH,
) -> ProcurementPackage:
    """Create a realistic procurement package anchored to a real SEC CUAD contract."""
    contract = loader.to_catcher_contract(cuad_contract)
    if scenario == SCENARIO_PRICE_A01:
        contract = contract.model_copy(
            update={"raw_text": contract.raw_text + SIMULATED_RATE_ADDENDUM}
        )
    vendor_id = contract.vendor_id
    vendor_name = contract.vendor_name
    po_id = f"PO-{cuad_contract.contract_id}"
    inv_id = f"INV-{cuad_contract.contract_id}"
    rc_id = f"RC-{cuad_contract.contract_id}"

    today = date.today()
    due_date = today + timedelta(days=contract.payment_terms_days)

    if scenario == SCENARIO_SURCHARGE_A09:
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
    elif scenario == SCENARIO_PRICE_A01:
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
    elif scenario == SCENARIO_MATCH:
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
    else:
        msg = f"Unknown scenario: {scenario!r}. Expected one of {SIMULATION_SCENARIOS}."
        raise ValueError(msg)

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
