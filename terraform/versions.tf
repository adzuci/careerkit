# -----------------------------------------------------------------------------
# versions.tf - Terraform and Provider Requirements
# -----------------------------------------------------------------------------
# This file specifies the required Terraform version and provider configurations
# for provisioning GKE infrastructure on GCP.
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.3.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

# -----------------------------------------------------------------------------
# Google Cloud Provider Configuration
# -----------------------------------------------------------------------------
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# -----------------------------------------------------------------------------
# Kubernetes Provider Configuration
# -----------------------------------------------------------------------------
# The Kubernetes provider is configured to authenticate with the GKE cluster
# using the cluster's endpoint and a short-lived access token.
# This enables Terraform to manage Kubernetes resources (Deployments, Services)
# directly after cluster creation.
# -----------------------------------------------------------------------------
provider "kubernetes" {
  host                   = "https://${module.gke.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.ca_certificate)
}

# Retrieve access token for Kubernetes provider authentication
data "google_client_config" "default" {}
