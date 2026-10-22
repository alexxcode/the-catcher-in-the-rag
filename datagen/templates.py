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

    c5_title = "Section 5. Warranties and Standard of Performance"
    c5_body = """Vendor warrants that all goods and equipment delivered hereunder shall be new, of merchantable quality, and free from defects in material, design, and workmanship for a minimum period of twelve (12) months following formal acceptance. All services shall be performed diligently by qualified personnel in accordance with highest commercial standards.
"""

    c6_title = "Section 6. Confidentiality and Non-Disclosure"
    c6_body = """Each party agrees to safeguard and hold in strict confidence all proprietary, commercial, and technical information received from the disclosing party. Neither party shall disclose or disseminate confidential information to any third party without obtaining prior written authorization, except where mandated by applicable judicial process.
"""

    c7_title = "Section 7. Intellectual Property and Work Product"
    c7_body = """Buyer shall retain sole, exclusive, and unencumbered ownership of all custom deliverables, specifications, drawings, and work product developed specifically for Buyer pursuant to any Purchase Order issued under this Agreement.
"""

    c8_title = "Section 8. Freight, Transportation and Delivery Protocol"
    c8_body = """Vendor shall arrange and pay for all standard shipping, freight, carriage, and cartage to the delivery facility specified on the Purchase Order. Standard freight costs are fully incorporated into the rate ceilings established in Section 1. Buyer shall under no circumstances reimburse or pay for expedited courier fees, rush delivery surcharges, or emergency transport handling unless explicitly authorized in advance by a written change order executed by Buyer's Vice President of Procurement.
"""

    c9_title = "Section 9. Inspection, Acceptance and Rejection of Goods"
    c9_body = """Buyer reserves the right to inspect all delivered items within ten (10) business days following physical receipt at Buyer's warehouse. Non-conforming, surplus, or damaged goods may be rejected by Buyer and held for removal or return at Vendor's sole expense and risk.
"""

    c10_title = "Section 10. Taxes and Remittance Protocol"
    c10_body = """Prices set forth herein exclude applicable state sales or transaction privilege taxes. Vendor shall clearly itemize all lawful taxes on each individual invoice. Buyer shall not be liable for taxes based on Vendor's net income or corporate franchise.
"""

    c11_title = "Section 11. Indemnification and Defense"
    c11_body = """Vendor shall defend, indemnify, and hold harmless Buyer, its parent corporation, affiliates, directors, and employees against all liabilities, judgments, settlements, losses, and legal costs arising from Vendor's negligence, willful misconduct, or infringement of third-party intellectual property.
"""

    c12_title = "Section 12. Limitation of Direct and Consequential Damages"
    c12_body = """Except with respect to breaches of Section 6 (Confidentiality) or third-party indemnification obligations under Section 11, neither party shall be liable for indirect, incidental, or consequential damages. Total aggregate liability under any order shall not exceed the total contract value paid in the preceding twelve months.
"""

    c13_title = "Section 13. Insurance Requirements"
    c13_body = """Vendor shall procure and maintain throughout the term of this Agreement Commercial General Liability insurance with limits not less than $2,000,000 per occurrence and statutory Worker's Compensation insurance covering all deployed personnel.
"""

    c14_title = "Section 14. Force Majeure and Excusable Delays"
    c14_body = """Neither party shall be deemed in breach of this Agreement for failure or delay in performance resulting from acts of God, armed conflict, civil commotion, severe labor disputes, epidemics, or government embargos beyond reasonable foresight and control, provided written notice is delivered within forty-eight (48) hours.
"""

    c15_title = "Section 15. Subcontracting and Delegation Restrictions"
    c15_body = """Vendor shall not subcontract, delegate, or assign any portion of its obligations under this Agreement to third-party entities without obtaining prior written approval from Buyer's authorized procurement manager.
"""

    c16_title = "Section 16. Books, Records and Audit Rights"
    c16_body = """Vendor shall maintain complete and accurate books and billing records concerning all goods and services invoiced under this Agreement for at least three (3) years following final payment. Buyer or its independent auditors shall have the right to audit such records upon reasonable advance notice.
"""

    c17_title = "Section 17. Compliance with Laws and Anti-Corruption"
    c17_body = """Vendor shall comply in all respects with applicable domestic and international laws, including export controls, fair labor standards, and the U.S. Foreign Corrupt Practices Act (FCPA). Vendor represents that no improper payments or gifts have been offered to Buyer personnel.
"""

    c18_title = "Section 18. Governing Law, Severability and Entire Agreement"
    c18_body = """This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflicts of laws principles. This Agreement, together with executed Purchase Orders, constitutes the entire agreement between the parties and supersedes all prior negotiations or proposals.
"""

    clauses_def = [
        ("pricing", c1_title, c1_body),
        ("payment_terms", c2_title, c2_body),
        ("penalties", c3_title, c3_body),
        ("term_and_termination", c4_title, c4_body),
        ("warranties", c5_title, c5_body),
        ("confidentiality", c6_title, c6_body),
        ("intellectual_property", c7_title, c7_body),
        ("freight_and_carriage", c8_title, c8_body),
        ("inspection_acceptance", c9_title, c9_body),
        ("taxes_invoicing", c10_title, c10_body),
        ("indemnification", c11_title, c11_body),
        ("liability_cap", c12_title, c12_body),
        ("insurance", c13_title, c13_body),
        ("force_majeure", c14_title, c14_body),
        ("subcontracting", c15_title, c15_body),
        ("audit_rights", c16_title, c16_body),
        ("compliance", c17_title, c17_body),
        ("governing_law", c18_title, c18_body),
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
