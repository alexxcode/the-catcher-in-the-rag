# ── Provider ──────────────────────────────────────────────────────────────────
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable APIs ───────────────────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "billingbudgets.googleapis.com",
  ])

  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

# ── Cloud Storage ─────────────────────────────────────────────────────────────
resource "google_storage_bucket" "documents" {
  name                        = "${var.project_id}-documents"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "prod"

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}

resource "google_storage_bucket" "reports" {
  name                        = "${var.project_id}-reports"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.environment != "prod"

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}

# ── Artifact Registry ─────────────────────────────────────────────────────────
resource "google_artifact_registry_repository" "images" {
  repository_id = "catcher-images"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for The Catcher in the RAG services."

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

# ── Least-Privilege Service Account for Cloud Run ─────────────────────────────
resource "google_service_account" "cloud_run_sa" {
  account_id   = "catcher-run-sa"
  display_name = "Cloud Run Service Account for The Catcher in the RAG"
  description  = "Least-privilege identity executing Catcher API and UI workloads."

  depends_on = [google_project_service.apis]
}

# Grant Vertex AI user role (ADC authentication for Gemini & text-embedding-005)
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Secret Manager accessor role
resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Storage viewer role for documents and reports
resource "google_storage_bucket_iam_member" "documents_viewer" {
  bucket = google_storage_bucket.documents.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_storage_bucket_iam_member" "reports_admin" {
  bucket = google_storage_bucket.reports.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ── Secret Manager ────────────────────────────────────────────────────────────
resource "google_secret_manager_secret" "langfuse_public_key" {
  secret_id = "langfuse-public-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "langfuse_secret_key" {
  secret_id = "langfuse-secret-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "demo_access_pin" {
  secret_id = "demo-access-pin"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "random_password" "demo_access_pin" {
  count   = var.demo_access_pin == null ? 1 : 0
  length  = 24
  special = false
}

locals {
  effective_demo_access_pin = var.demo_access_pin != null ? var.demo_access_pin : random_password.demo_access_pin[0].result
}

resource "google_secret_manager_secret_version" "demo_access_pin_version" {
  secret      = google_secret_manager_secret.demo_access_pin.id
  secret_data = local.effective_demo_access_pin
}

# ── Cloud Run Services (API & UI) ─────────────────────────────────────────────
locals {
  default_image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/catcher:latest"
  active_image  = var.container_image != "" ? var.container_image : local.default_image
}

# 1. Audit API Service
resource "google_cloud_run_v2_service" "api" {
  name     = "catcher-api-${var.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email
    timeout         = "60s"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = local.active_image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "SERVICE_MODE"
        value = "api"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_REGION"
        value = var.region
      }
      # With MAX_LIVE_CALLS=0 every Vertex call is blocked, so run fully on stub.
      env {
        name  = "LLM_PROVIDER"
        value = var.max_live_calls == 0 ? "stub" : "vertex_ai"
      }
      env {
        name  = "EMBEDDING_PROVIDER"
        value = var.max_live_calls == 0 ? "stub" : "vertex_ai"
      }
      env {
        name = "DEMO_ACCESS_PIN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.demo_access_pin.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "ALLOW_PUBLIC_STUB"
        value = "true"
      }
      env {
        name  = "MAX_LIVE_CALLS"
        value = tostring(var.max_live_calls)
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 3
        timeout_seconds       = 5
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8000
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.vertex_ai_user,
    google_project_iam_member.secret_accessor,
    google_secret_manager_secret_version.demo_access_pin_version,
  ]
}

# 2. Interactive Streamlit UI Service
resource "google_cloud_run_v2_service" "ui" {
  name     = "catcher-ui-${var.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = local.active_image

      ports {
        container_port = 8501
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "SERVICE_MODE"
        value = "ui"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_REGION"
        value = var.region
      }
      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api.uri
      }
      env {
        name = "DEMO_ACCESS_PIN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.demo_access_pin.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/_stcore/health"
          port = 8501
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 3
        timeout_seconds       = 5
      }

      liveness_probe {
        http_get {
          path = "/_stcore/health"
          port = 8501
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.vertex_ai_user,
    google_project_iam_member.secret_accessor,
    google_secret_manager_secret_version.demo_access_pin_version,
    google_cloud_run_v2_service.api,
  ]
}

# Private API: Invokable exclusively by the UI Service Account (Cloud Run IAM)
resource "google_cloud_run_v2_service_iam_member" "api_private_ui_invoker" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_cloud_run_v2_service_iam_member" "ui_public" {
  location = google_cloud_run_v2_service.ui.location
  name     = google_cloud_run_v2_service.ui.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Budget Alert (Cost Safeguard) ─────────────────────────────────────────────
resource "google_billing_budget" "catcher_budget" {
  count           = var.billing_account_id != "" ? 1 : 0
  billing_account = var.billing_account_id
  display_name    = "Catcher in the RAG — Monthly Cap"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.50
  }
  threshold_rules {
    threshold_percent = 0.80
  }
  threshold_rules {
    threshold_percent = 1.00
    spend_basis       = "CURRENT_SPEND"
  }

  depends_on = [google_project_service.apis]
}
