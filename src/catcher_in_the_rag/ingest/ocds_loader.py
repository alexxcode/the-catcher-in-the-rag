"""Open Contracting Data Standard (OCDS) Ingestion Loader.

Parses public procurement releases conforming to the Open Contracting Data Standard:
- Real buyers, suppliers, categories, and award values
- Contract periods, descriptions, and line items
- Used to seed realistic procurement context and validate schema mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path


@dataclass
class OCDSEntity:
    """An organization (buyer, procuring entity, or supplier)."""

    id: str
    name: str


@dataclass
class OCDSLineItem:
    """A line item procured under an OCDS tender or contract."""

    item_id: str
    description: str
    quantity: Decimal
    unit_amount: Decimal
    currency: str


@dataclass
class OCDSContractRecord:
    """A contract record extracted from an OCDS release."""

    ocid: str
    release_id: str
    buyer: OCDSEntity
    supplier: OCDSEntity | None
    contract_id: str
    title: str
    description: str
    total_amount: Decimal
    currency: str
    start_date: str | None
    end_date: str | None
    items: list[OCDSLineItem] = field(default_factory=list)


class OCDSLoader:
    """Ingests Open Contracting Data Standard (OCDS) releases."""

    def __init__(self, ocds_dir: str | Path = "data/real/ocds") -> None:
        self.ocds_dir = Path(ocds_dir)

    def load_contracts(self) -> list[OCDSContractRecord]:
        """Load contract records from all available OCDS release files."""
        if not self.ocds_dir.exists():
            return []

        records: list[OCDSContractRecord] = []
        for json_path in sorted(self.ocds_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            releases = data.get("releases", [])
            for rel in releases:
                ocid = rel.get("ocid", "")
                rel_id = rel.get("id", "")
                buyer_raw = rel.get("buyer", {})
                buyer = OCDSEntity(
                    id=buyer_raw.get("id", "BUYER-UNKNOWN"),
                    name=buyer_raw.get("name", "Public Buyer Entity"),
                )

                # Extract supplier from awards if available
                supplier: OCDSEntity | None = None
                awards = rel.get("awards", [])
                if awards:
                    sup_list = awards[0].get("suppliers", [])
                    if sup_list:
                        supplier = OCDSEntity(
                            id=sup_list[0].get("id", "VND-UNKNOWN"),
                            name=sup_list[0].get("name", "Awarded Supplier"),
                        )

                contracts = rel.get("contracts", [])
                for c in contracts:
                    val = c.get("value", {})
                    amount = Decimal(str(val.get("amount", "0.0")))
                    currency = val.get("currency", "USD")
                    period = c.get("period", {})

                    items: list[OCDSLineItem] = []
                    for item in c.get("items", []):
                        unit_val = item.get("unit", {}).get("value", {})
                        items.append(
                            OCDSLineItem(
                                item_id=str(item.get("id", "ITEM-1")),
                                description=item.get("description", "Procured Item"),
                                quantity=Decimal(str(item.get("quantity", "1"))),
                                unit_amount=Decimal(str(unit_val.get("amount", "0.0"))),
                                currency=unit_val.get("currency", currency),
                            )
                        )

                    records.append(
                        OCDSContractRecord(
                            ocid=ocid,
                            release_id=rel_id,
                            buyer=buyer,
                            supplier=supplier,
                            contract_id=c.get("id", f"CTR-{ocid}"),
                            title=c.get("title", "Public Procurement Agreement"),
                            description=c.get("description", ""),
                            total_amount=amount,
                            currency=currency,
                            start_date=period.get("startDate"),
                            end_date=period.get("endDate"),
                            items=items,
                        )
                    )

        return records
