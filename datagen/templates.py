from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class ItemTemplate:
    item_id: str
    description: str
    base_unit_price: Decimal
    category: str


@dataclass(frozen=True)
class VendorTemplate:
    vendor_id: str
    name: str
    category: str
    items: list[ItemTemplate]


VENDORS: list[VendorTemplate] = [
    VendorTemplate(
        vendor_id="VND-001",
        name="Apex Systems & Hardware LLC",
        category="IT Hardware",
        items=[
            ItemTemplate("SKU-101", "Enterprise Workstation 32GB RAM", Decimal("1450.00"), "IT"),
            ItemTemplate("SKU-102", "Ultra-Wide Monitor 34-inch", Decimal("420.00"), "IT"),
            ItemTemplate("SKU-103", "Ergonomic Mechanical Keyboard", Decimal("110.00"), "IT"),
            ItemTemplate("SKU-104", "USB-C Dual 4K Docking Station", Decimal("195.00"), "IT"),
        ],
    ),
    VendorTemplate(
        vendor_id="VND-002",
        name="BluePeak Cloud Infrastructure Inc",
        category="Cloud Services",
        items=[
            ItemTemplate(
                "SKU-201", "Dedicated Managed Virtual Server (Monthly)", Decimal("850.00"), "Cloud"
            ),
            ItemTemplate("SKU-202", "Cloud Object Storage Block 10TB", Decimal("220.00"), "Cloud"),
            ItemTemplate(
                "SKU-203", "Enterprise API Gateway Tier 2 (Monthly)", Decimal("600.00"), "Cloud"
            ),
        ],
    ),
    VendorTemplate(
        vendor_id="VND-003",
        name="Cascade Logistics & Freight Corp",
        category="Logistics",
        items=[
            ItemTemplate(
                "SKU-301",
                "Overnight Freight Express pallet delivery",
                Decimal("650.00"),
                "Logistics",
            ),
            ItemTemplate(
                "SKU-302", "Standard Regional Truckload route", Decimal("1250.00"), "Logistics"
            ),
            ItemTemplate(
                "SKU-303", "Climate Controlled Warehousing unit/mo", Decimal("380.00"), "Logistics"
            ),
        ],
    ),
    VendorTemplate(
        vendor_id="VND-004",
        name="Delta Facility Operations Group",
        category="Facilities",
        items=[
            ItemTemplate(
                "SKU-401", "Commercial Deep Cleaning Service (Full Floor)", Decimal("750.00"), "Fac"
            ),
            ItemTemplate(
                "SKU-402", "HVAC Quarterly Preventative Maintenance", Decimal("1200.00"), "Fac"
            ),
            ItemTemplate(
                "SKU-403", "Security Patrol Coverage (8-hour shift)", Decimal("320.00"), "Fac"
            ),
        ],
    ),
    VendorTemplate(
        vendor_id="VND-005",
        name="Evergreen Office Furnishings",
        category="Office Supplies",
        items=[
            ItemTemplate(
                "SKU-501", "Ergonomic Mesh Task Chair Executive", Decimal("340.00"), "Furniture"
            ),
            ItemTemplate(
                "SKU-502",
                "Motorized Adjustable Standing Desk 60x30",
                Decimal("580.00"),
                "Furniture",
            ),
            ItemTemplate(
                "SKU-503", "Acoustic Privacy Panel Partition", Decimal("160.00"), "Furniture"
            ),
        ],
    ),
]


def render_contract_text(
    contract_id: str,
    vendor_name: str,
    vendor_id: str,
    effective_date: date,
    expiration_date: date,
    rate_items: list[tuple[str, str, Decimal]],
    payment_terms_days: int = 30,
    late_penalty_pct: Decimal = Decimal("1.0"),
    auto_renew: bool = False,
    include_penalties_clause: bool = True,
) -> tuple[str, list[dict[str, str | int]]]:
    """Render full legal contract text and compute exact character offsets for each clause."""
    renew_phrase = (
        "shall automatically renew for successive terms of one (1) year unless terminated."
        if auto_renew
        else "shall expire unconditionally on the Expiration Date without automatic renewal."
    )

    rates_text_lines = []
    for sku, desc, price in rate_items:
        rates_text_lines.append(f"  * {sku} - {desc}: Maximum Unit Rate of ${price:.2f} USD")
    rates_block = "\n".join(rates_text_lines)

    header = f"""MASTER SERVICES AND PROCUREMENT AGREEMENT
Contract Reference: {contract_id}
Buyer: Acme Industrial Holdings Corp ("Buyer")
Vendor: {vendor_name} (ID: {vendor_id}) ("Vendor")

This Agreement is entered into as of {effective_date.isoformat()} ("Effective Date").
"""

    c1_title = "Section 1. Pricing and Rate Ceilings"
    c1_body = f"""Vendor agrees that all orders issued under this Agreement shall adhere strictly to the negotiated rate sheet below. Vendor shall not bill Buyer at rates exceeding the stated maximum unit prices without an executed written amendment.

{rates_block}
"""

    c2_title = "Section 2. Payment Terms and Invoicing"
    c2_body = f"""All invoices must reference a valid Purchase Order number issued by Buyer. Undisputed invoices shall be payable Net {payment_terms_days} days from the date of confirmed invoice receipt. Any invoice demanding earlier remittance or contrary terms is non-compliant and shall be held for review.
"""

    c3_title = "Section 3. Delivery and Late Delivery Penalties"
    if include_penalties_clause:
        c3_body = f"""Time is of the essence for all deliveries under this Agreement. In the event of unexcused late delivery beyond the scheduled delivery date specified in the Purchase Order, Vendor agrees to a liquidated penalty of {late_penalty_pct:.1f}% per calendar day of delay, calculated upon the total value of the delayed goods. Such penalty shall be deducted from the invoice total prior to payment.
"""
    else:
        c3_body = """Vendor shall use commercially reasonable efforts to meet scheduled delivery timelines. Deliveries are subject to standard delivery terms.
"""

    c4_title = "Section 4. Term and Termination"
    c4_body = f"""This Agreement shall remain in full force until {expiration_date.isoformat()} ("Expiration Date") unless terminated earlier. Upon the Expiration Date, this Agreement {renew_phrase} No new purchase orders or charges dated subsequent to expiration shall be recognized without a formal extension.
"""

    c5_title = "Section 5. Governing Law and Severability"
    c5_body = """This Agreement shall be construed in accordance with the laws of the State of Delaware, excluding its conflict of laws principles.
"""

    clauses_def = [
        ("pricing", c1_title, c1_body),
        ("payment_terms", c2_title, c2_body),
        ("penalties", c3_title, c3_body),
        ("term_and_termination", c4_title, c4_body),
        ("governing_law", c5_title, c5_body),
    ]

    full_text = header
    clause_metadata: list[dict[str, str | int]] = []

    for c_type, title, body in clauses_def:
        full_text += "\n" + title + "\n"
        start_char = len(full_text)
        full_text += body
        end_char = len(full_text)
        clause_metadata.append(
            {
                "clause_id": f"CLAUSE-{c_type.upper()}",
                "clause_type": c_type,
                "title": title,
                "text": body.strip(),
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return full_text, clause_metadata
