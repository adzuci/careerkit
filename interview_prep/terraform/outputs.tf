output "vpc_name" {
  description = "VPC name."
  value       = google_compute_network.vpc.name
}

output "subnet_name" {
  description = "Subnet name."
  value       = google_compute_subnetwork.primary.name
}

output "gke_cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.primary.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint."
  value       = google_container_cluster.primary.endpoint
}

output "gke_node_service_account" {
  description = "Node service account email."
  value       = google_service_account.gke_nodes.email
}

output "gcs_bucket_name" {
  description = "GCS bucket name."
  value       = google_storage_bucket.app.name
}
