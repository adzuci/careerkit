# -----------------------------------------------------------------------------
# outputs.tf - Terraform Outputs
# -----------------------------------------------------------------------------
# This file defines outputs that provide useful information after deployment:
# - GKE cluster details for kubectl configuration
# - Application service endpoint for testing
# - Network information for reference
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# GKE Cluster Outputs
# -----------------------------------------------------------------------------
output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = module.gke.name
}

output "cluster_endpoint" {
  description = "Endpoint for the GKE cluster API server"
  value       = module.gke.endpoint
  sensitive   = true
}

output "cluster_location" {
  description = "Location (region) of the GKE cluster"
  value       = module.gke.location
}

output "cluster_ca_certificate" {
  description = "Base64 encoded CA certificate for the cluster"
  value       = module.gke.ca_certificate
  sensitive   = true
}

# -----------------------------------------------------------------------------
# kubectl Configuration Command
# -----------------------------------------------------------------------------
output "kubectl_config_command" {
  description = "Command to configure kubectl to access the cluster"
  value       = "gcloud container clusters get-credentials ${module.gke.name} --region ${module.gke.location} --project ${var.project_id}"
}

# -----------------------------------------------------------------------------
# Network Outputs
# -----------------------------------------------------------------------------
output "network_name" {
  description = "Name of the VPC network"
  value       = module.vpc.network_name
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = module.vpc.subnets_names[0]
}

# -----------------------------------------------------------------------------
# Application Service Outputs
# -----------------------------------------------------------------------------
output "hello_app_service_name" {
  description = "Name of the Hello App Kubernetes service"
  value       = kubernetes_service_v1.hello_app.metadata[0].name
}

output "hello_app_external_ip" {
  description = "External IP address of the Hello App LoadBalancer service"
  value       = try(kubernetes_service_v1.hello_app.status[0].load_balancer[0].ingress[0].ip, "Pending - run 'terraform refresh' after a few minutes")
}

output "hello_app_url" {
  description = "URL to access the Hello App"
  value       = "http://${try(kubernetes_service_v1.hello_app.status[0].load_balancer[0].ingress[0].ip, "<pending>")}:${var.service_port}"
}

# -----------------------------------------------------------------------------
# Verification Commands
# -----------------------------------------------------------------------------
output "verification_commands" {
  description = "Commands to verify the deployment"
  value       = <<-EOT
    # 1. Configure kubectl:
    gcloud container clusters get-credentials ${module.gke.name} --region ${module.gke.location} --project ${var.project_id}

    # 2. Check deployment status:
    kubectl get deployments -n ${var.app_namespace}

    # 3. Check pods:
    kubectl get pods -n ${var.app_namespace} -l app=${var.app_name}

    # 4. Check service and external IP:
    kubectl get service ${var.app_name}-service -n ${var.app_namespace}

    # 5. Test the application (replace <EXTERNAL_IP>):
    curl http://<EXTERNAL_IP>:${var.service_port}
  EOT
}

# -----------------------------------------------------------------------------
# Node Pool Information
# -----------------------------------------------------------------------------
output "node_pool_name" {
  description = "Name of the node pool"
  value       = var.node_pool_name
}

output "service_account" {
  description = "Service account used by GKE nodes"
  value       = module.gke.service_account
}
