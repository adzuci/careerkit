variable "project_id" {
  type        = string
  description = "GCP project ID."
}

variable "region" {
  type        = string
  description = "Primary region."
  default     = "us-central1"
}

variable "environment" {
  type        = string
  description = "Environment label (dev, staging, prod)."
  default     = "dev"
}

variable "zones" {
  type        = list(string)
  description = "Zones for GKE nodes."
  default     = ["us-central1-a", "us-central1-b"]
}

variable "network_name" {
  type        = string
  description = "VPC name."
  default     = "interview-vpc"
}

variable "subnet_name" {
  type        = string
  description = "Primary subnet name."
  default     = "interview-subnet"
}

variable "subnet_cidr" {
  type        = string
  description = "Primary subnet CIDR."
  default     = "10.10.0.0/20"
}

variable "pods_range" {
  type        = string
  description = "Secondary range for GKE pods."
  default     = "10.20.0.0/16"
}

variable "services_range" {
  type        = string
  description = "Secondary range for GKE services."
  default     = "10.30.0.0/20"
}

variable "gke_cluster_name" {
  type        = string
  description = "GKE cluster name."
  default     = "interview-gke"
}

variable "gke_release_channel" {
  type        = string
  description = "GKE release channel."
  default     = "REGULAR"
}

variable "gke_master_ipv4_cidr" {
  type        = string
  description = "Private master CIDR range."
  default     = "172.16.0.0/28"
}

variable "gke_node_machine_type" {
  type        = string
  description = "Node machine type."
  default     = "e2-medium"
}

variable "gke_node_min_count" {
  type        = number
  description = "Min node count."
  default     = 1
}

variable "gke_node_max_count" {
  type        = number
  description = "Max node count."
  default     = 3
}

variable "gke_node_disk_size_gb" {
  type        = number
  description = "Node disk size in GB."
  default     = 50
}

variable "gke_master_authorized_cidr" {
  type        = string
  description = "CIDR allowed to access GKE master endpoint."
  default     = "0.0.0.0/0"
}

variable "gcs_bucket_name" {
  type        = string
  description = "GCS bucket name (must be globally unique)."
}

variable "gcs_location" {
  type        = string
  description = "GCS bucket location."
  default     = "US"
}
