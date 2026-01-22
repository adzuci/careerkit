# -----------------------------------------------------------------------------
# variables.tf - Input Variables for GKE Infrastructure
# -----------------------------------------------------------------------------
# This file defines all configurable parameters for the GKE cluster and
# associated resources. Sensible defaults are provided where appropriate.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# GCP Project Configuration
# -----------------------------------------------------------------------------
variable "project_id" {
  description = "The GCP project ID where resources will be created"
  type        = string
}

variable "region" {
  description = "The GCP region for the GKE cluster"
  type        = string
  default     = "us-central1"
}

variable "zones" {
  description = "The GCP zones for the GKE cluster nodes"
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

# -----------------------------------------------------------------------------
# GKE Cluster Configuration
# -----------------------------------------------------------------------------
variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "apollo-demo-cluster"
}

variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "apollo-demo-vpc"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "apollo-demo-subnet"
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "pods_cidr_range" {
  description = "Secondary CIDR range for pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr_range" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "10.2.0.0/20"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the GKE cluster (use 'latest' for the latest available)"
  type        = string
  default     = "latest"
}

# -----------------------------------------------------------------------------
# Node Pool Configuration
# -----------------------------------------------------------------------------
variable "node_pool_name" {
  description = "Name of the default node pool"
  type        = string
  default     = "default-pool"
}

variable "machine_type" {
  description = "Machine type for cluster nodes"
  type        = string
  default     = "e2-medium"
}

variable "min_node_count" {
  description = "Minimum number of nodes per zone (for autoscaling)"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes per zone (for autoscaling)"
  type        = number
  default     = 3
}

variable "initial_node_count" {
  description = "Initial number of nodes per zone"
  type        = number
  default     = 1
}

variable "disk_size_gb" {
  description = "Disk size in GB for each node"
  type        = number
  default     = 50
}

variable "disk_type" {
  description = "Disk type for nodes (pd-standard, pd-ssd, pd-balanced)"
  type        = string
  default     = "pd-standard"
}

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------
variable "app_name" {
  description = "Name of the hello-world application"
  type        = string
  default     = "hello-app"
}

variable "app_namespace" {
  description = "Kubernetes namespace for the application"
  type        = string
  default     = "default"
}

variable "app_image" {
  description = "Container image for the hello-world application"
  type        = string
  default     = "gcr.io/google-samples/hello-app:1.0"
}

variable "app_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 3
}

variable "app_port" {
  description = "Port the application listens on"
  type        = number
  default     = 8080
}

variable "service_port" {
  description = "Port exposed by the LoadBalancer service"
  type        = number
  default     = 80
}

# -----------------------------------------------------------------------------
# Labels and Tags
# -----------------------------------------------------------------------------
variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "demo"
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    managed-by = "terraform"
    project    = "apollo-interview"
  }
}
