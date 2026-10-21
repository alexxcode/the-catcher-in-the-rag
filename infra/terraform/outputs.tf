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

output "db_instance_name" {
  description = "Cloud SQL instance name (for Cloud SQL Auth Proxy connection string)."
  value       = google_sql_database_instance.main.connection_name
}
