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

variable "max_instances" {
  description = "Maximum Cloud Run instances to prevent runaway cost."
  type        = number
  default     = 3
}

variable "container_image" {
  description = "Container image URL in Artifact Registry."
  type        = string
  default     = ""
}

variable "billing_account_id" {
  description = "GCP Billing Account ID for budget alerts (optional)."
  type        = string
  default     = ""
}

variable "monthly_budget_amount" {
  description = "Monthly budget amount in USD for billing alerts."
  type        = number
  default     = 25
}

variable "demo_access_pin" {
  description = "Initial access PIN for authorized demo auditors stored in Secret Manager."
  type        = string
  sensitive   = true
  default     = "audit-2026-secure"
}
