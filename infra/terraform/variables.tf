variable "project_id" {
  description = "GCP project ID."
  type        = string
  default     = "the-catcher-in-the-rag"
}

variable "region" {
  description = "GCP region for all resources."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment: dev | staging | prod."
  type        = string
  default     = "dev"
}

variable "db_instance_tier" {
  description = "Cloud SQL instance tier."
  type        = string
  default     = "db-f1-micro" # cheapest; upgrade for prod
}
