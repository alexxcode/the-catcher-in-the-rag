terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
  }

  # Remote state in GCS.
  # PREREQUISITE: Create the bucket before running `terraform init`:
  #   gcloud storage buckets create gs://the-catcher-in-the-rag-tf-state \
  #     --project=the-catcher-in-the-rag \
  #     --location=us-central1 \
  #     --uniform-bucket-level-access
  backend "gcs" {
    bucket = "the-catcher-in-the-rag-tf-state"
    prefix = "terraform/state"
  }
}
