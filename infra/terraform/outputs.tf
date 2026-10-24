output "documents_bucket" {
  description = "GCS bucket for ingested procurement documents."
  value       = google_storage_bucket.documents.name
}

output "reports_bucket" {
  description = "GCS bucket for evaluation reports."
  value       = google_storage_bucket.reports.name
}

output "artifact_registry_url" {
  description = "Docker registry URL for container images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "service_account_email" {
  description = "Email of least-privilege Cloud Run service account."
  value       = google_service_account.cloud_run_sa.email
}

output "api_uri" {
  description = "Public URL for Catcher Audit API."
  value       = google_cloud_run_v2_service.api.uri
}

output "ui_uri" {
  description = "Public URL for Catcher Streamlit UI."
  value       = google_cloud_run_v2_service.ui.uri
}

output "demo_access_pin" {
  description = "Effective demo access PIN for authorized auditors."
  value       = local.effective_demo_access_pin
  sensitive   = true
}

