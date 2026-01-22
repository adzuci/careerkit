# -----------------------------------------------------------------------------
# main.tf - GKE Cluster Infrastructure
# -----------------------------------------------------------------------------
# This file provisions the core GCP infrastructure:
# - VPC Network and Subnet with secondary ranges for pods/services
# - GKE Cluster using the official terraform-google-modules/kubernetes-engine
# - Node pool with autoscaling configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# VPC Network Configuration
# -----------------------------------------------------------------------------
# Create a dedicated VPC for the GKE cluster with a subnet containing
# secondary IP ranges for Kubernetes pods and services.
# -----------------------------------------------------------------------------
module "vpc" {
  source  = "terraform-google-modules/network/google"
  version = "~> 9.0"

  project_id   = var.project_id
  network_name = var.network_name
  routing_mode = "GLOBAL"

  subnets = [
    {
      subnet_name           = var.subnet_name
      subnet_ip             = var.subnet_cidr
      subnet_region         = var.region
      subnet_private_access = true
      description           = "Subnet for GKE cluster"
    }
  ]

  secondary_ranges = {
    (var.subnet_name) = [
      {
        range_name    = "${var.subnet_name}-pods"
        ip_cidr_range = var.pods_cidr_range
      },
      {
        range_name    = "${var.subnet_name}-services"
        ip_cidr_range = var.services_cidr_range
      }
    ]
  }
}

# -----------------------------------------------------------------------------
# GKE Cluster Configuration
# -----------------------------------------------------------------------------
# Uses the official Google Kubernetes Engine Terraform module for a
# production-ready, opinionated cluster configuration.
#
# Module documentation:
# https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google
# -----------------------------------------------------------------------------
module "gke" {
  source  = "terraform-google-modules/kubernetes-engine/google"
  version = "~> 33.0"

  project_id = var.project_id
  name       = var.cluster_name
  region     = var.region
  zones      = var.zones

  # Network configuration
  network           = module.vpc.network_name
  subnetwork        = module.vpc.subnets_names[0]
  ip_range_pods     = "${var.subnet_name}-pods"
  ip_range_services = "${var.subnet_name}-services"

  # Cluster features
  kubernetes_version       = var.kubernetes_version
  regional                 = true
  create_service_account   = true
  remove_default_node_pool = true
  deletion_protection      = false # Set to true for production

  # Security settings
  http_load_balancing        = true
  network_policy             = false
  horizontal_pod_autoscaling = true
  filestore_csi_driver       = false

  # Node pool configuration
  node_pools = [
    {
      name               = var.node_pool_name
      machine_type       = var.machine_type
      min_count          = var.min_node_count
      max_count          = var.max_node_count
      initial_node_count = var.initial_node_count
      disk_size_gb       = var.disk_size_gb
      disk_type          = var.disk_type
      image_type         = "COS_CONTAINERD"
      auto_repair        = true
      auto_upgrade       = true
      preemptible        = false # Set to true for cost savings in non-prod
    }
  ]

  # Node pool OAuth scopes
  node_pools_oauth_scopes = {
    all = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]
  }

  # Node pool labels
  node_pools_labels = {
    all = merge(var.labels, {
      environment = var.environment
    })
  }

  # Node pool tags for firewall rules
  node_pools_tags = {
    all = ["gke-node", var.cluster_name]
  }

  # Cluster labels
  cluster_resource_labels = var.labels

  depends_on = [module.vpc]
}
