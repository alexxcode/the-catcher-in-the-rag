#!/usr/bin/env bash
# Staging smoke: liveness + 1 audit query. Sends no DEMO_ACCESS_PIN, so the API
# runs the zero-cost stub provider (0 Vertex calls). Usage: scripts/smoke.sh <API_URL>
set -euo pipefail
URL="${1:?usage: scripts/smoke.sh <API_URL>}"
TOKEN="$(gcloud auth print-identity-token 2>/dev/null || true)"  # API is private
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")
echo "== GET /api/reports"
curl -fsS "${AUTH[@]}" "$URL/api/reports"; echo
echo "== POST /api/audit (1 query, PKG-0001)"
curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
  -d '{"package_id": "PKG-0001"}' "$URL/api/audit" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["decision"]["decision"], d["execution_provider"])'
echo "smoke OK"
