#!/usr/bin/env bash
# Build + push the image and run `terraform plan`. Does NOT apply: run
# `terraform -chdir=infra/terraform apply tfplan` yourself after reviewing the plan.
# Usage: PROJECT_ID=<gcp-project> REGION=<region> scripts/deploy.sh
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID (GCP project id), e.g. PROJECT_ID=my-project}"
: "${REGION:?set REGION (GCP region), e.g. REGION=us-central1}"
command -v docker >/dev/null || { echo "docker not found: install Docker Desktop with WSL integration (https://docs.docker.com/desktop/wsl/) or docker-ce in WSL" >&2; exit 1; }

TAG="$(git rev-parse --short HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/catcher-images/catcher:${TAG}"
TF=(terraform -chdir=infra/terraform)

echo "== build ${IMAGE}"
docker build -t "${IMAGE}" -f infra/Dockerfile .
echo "== push (Artifact Registry repo 'catcher-images' must exist)"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "${IMAGE}"

echo "== terraform plan (no apply)"
"${TF[@]}" init -input=false
"${TF[@]}" plan -input=false -out=tfplan \
  -var "project_id=${PROJECT_ID}" -var "region=${REGION}" -var "container_image=${IMAGE}"

API_URL="$("${TF[@]}" output -raw api_uri 2>/dev/null || true)"
echo "Service URL: ${API_URL:-<not deployed yet: available after terraform apply>}"
echo "Next (manual): ${TF[*]} apply tfplan"
echo "Smoke: scripts/smoke.sh ${API_URL:-<API_URL>}"
