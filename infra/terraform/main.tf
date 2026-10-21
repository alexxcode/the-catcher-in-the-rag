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
    "sqladmin.googleapis.com",
    "sql-component.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iamcredentials.googleapis.com",
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

# ── Cloud SQL (PostgreSQL + pgvector) ─────────────────────────────────────────
resource "google_sql_database_instance" "main" {
  name             = "catcher-pg-${var.environment}"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = var.db_instance_tier
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"

    database_flags {
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }

    # Dev: public IP with no authorized networks (access via Cloud SQL Auth Proxy only).
    # Prod: TODO replace with VPC private IP.
    ip_configuration {
      ipv4_enabled = true
    }

    backup_configuration {
      enabled    = var.environment == "prod"
      start_time = "02:00"
    }
  }

  deletion_protection = var.environment == "prod"
  depends_on          = [google_project_service.apis]
}

resource "google_sql_database" "catcher" {
  name     = "catcher"
  instance = google_sql_database_instance.main.name
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

resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}
