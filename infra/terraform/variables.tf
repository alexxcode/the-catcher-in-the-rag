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

variable "max_live_calls" {
  description = "Per-instance cap on live Vertex calls (MAX_LIVE_CALLS); 0 blocks all."
  type        = number
  default     = 0
}

variable "container_image" {
  description = "Container image URL in Artifact Registry (set by scripts/deploy.sh). Required: no :latest fallback."
  type        = string

  validation {
    condition     = length(trimspace(var.container_image)) > 0
    error_message = "container_image is required: run scripts/deploy.sh, then terraform apply tfplan."
  }
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
  description = "Access PIN for authorized demo auditors stored in Secret Manager. If null, a secure random password is automatically generated. Set via TF_VAR_demo_access_pin."
  type        = string
  sensitive   = true
  default     = null
}

