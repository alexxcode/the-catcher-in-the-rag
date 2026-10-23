# Data Sources

All third-party datasets are downloaded via scripts; raw corpora are **not committed** to this repository.

| Dataset | Role | License | Download script | URL | Version / Date recorded |
|---|---|---|---|---|---|
| CUAD v1 (510 contracts, 41 clause types) | Contract corpus + retrieval ground truth | CC BY 4.0 | `datagen/download.py --dataset cuad` (`make download-data`) | https://github.com/TheAtticusProject/cuad | v1.0 (aok_v1.0), 2026-10-22 |
| OCDS public procurement data (OCP sample data) | Realistic vendors, categories, amounts for synthetic POs | Apache 2.0 / Open Data | `datagen/download.py --dataset ocds` (`make download-data`) | https://github.com/open-contracting/sample-data | OCDS 1.1 release package, 2026-10-22 |
| FATURA (10k invoice images, 50 layouts) | Extraction robustness set (future image extension) | CC BY 4.0 | `datagen/download.py --dataset fatura` | Optional image extension | Optional |
| Synthetic packages (own generator) | Labeled anomalies, main eval set | Own — CC BY 4.0 or MIT (marked synthetic) | generated via `make generate-synthetic` | Local generator | v0.4.0, 300 packages |

## OCDS Publisher(s)

Recorded publisher details:
- **Publisher Name:** Open Contracting Partnership (OCP)
- **Data Standard:** OCDS 1.1 (JSON schema)
- **Repository URL:** https://github.com/open-contracting/sample-data
- **License:** Apache 2.0 / Open Database License
- **Download Date:** 2026-10-22 (snapshot releases: planning, tender, award, contract, implementation)

**Do not use** datasets under non-commercial terms (e.g. CC BY-NC) or aggregates with mixed/unclear licenses.

## Notes

- Re-verify all licenses at source before publishing results.
- Credit everything in `ATTRIBUTION.md`.
- Generated synthetic data must be marked as synthetic and use only GA (non-preview) models.
