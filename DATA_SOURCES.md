# Data Sources

All third-party datasets are downloaded via scripts; raw corpora are **not committed** to this repository.

| Dataset | Role | License | Download script | URL | Version / Date recorded |
|---|---|---|---|---|---|
| CUAD v1 (510 contracts, 41 clause types) | Contract corpus + retrieval ground truth | CC BY 4.0 | `datagen/download.py --dataset cuad` | https://github.com/TheAtticusProject/cuad | TODO: record commit/date |
| FATURA (10k invoice images, 50 layouts) | Extraction robustness set | CC BY 4.0 | `datagen/download.py --dataset fatura` | TODO: verify URL | TODO |
| ContractNLI | Optional: contradiction/entailment test cases | CC BY 4.0 (re-verify at source) | `datagen/download.py --dataset contractnli` | TODO | TODO |
| LegalBench-RAG (CUAD/ContractNLI subsets) | Retrieval benchmark comparison | Repo: MIT; underlying corpora keep own licenses | `datagen/download.py --dataset legalbench` | TODO | TODO |
| OCDS public procurement data | Realistic vendors, categories, amounts for synthetic POs | Per publisher (record below) | `datagen/download.py --dataset ocds` | TODO: choose publisher(s) | TODO |
| Synthetic packages (own generator) | Labeled anomalies, main eval set | Own — CC BY 4.0 or MIT (mark as synthetic) | generated via `make generate-synthetic` | n/a | n/a |

## OCDS Publisher(s)

TODO (Open Decision #4): Choose 1–2 OCDS publishers and record:
- Publisher name and URL
- Exact license
- Download date and version/snapshot

**Do not use** datasets under non-commercial terms (e.g. CC BY-NC) or aggregates with mixed/unclear licenses.

## Notes

- Re-verify all licenses at source before publishing results.
- Credit everything in `ATTRIBUTION.md`.
- Generated synthetic data must be marked as synthetic and use only GA (non-preview) models.
