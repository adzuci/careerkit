# -----------------------------------------------------------------------------
# kubernetes.tf - Kubernetes Resources (Deployment and Service)
# -----------------------------------------------------------------------------
# This file defines the Kubernetes resources for the Hello World application:
# - Deployment: Runs the containerized web application
# - Service: Exposes the application via a LoadBalancer with external IP
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Hello World Deployment
# -----------------------------------------------------------------------------
# Creates a Kubernetes Deployment that runs the hello-app container.
# The deployment is configured with:
# - Multiple replicas for high availability
# - Resource requests/limits for predictable scheduling
# - Liveness and readiness probes for health checking
# -----------------------------------------------------------------------------
resource "kubernetes_deployment_v1" "hello_app" {
  metadata {
    name      = var.app_name
    namespace = var.app_namespace
    labels = {
      app         = var.app_name
      environment = var.environment
      managed-by  = "terraform"
    }
  }

  spec {
    replicas = var.app_replicas

    selector {
      match_labels = {
        app = var.app_name
      }
    }

    template {
      metadata {
        labels = {
          app         = var.app_name
          environment = var.environment
        }
      }

      spec {
        container {
          name  = var.app_name
          image = var.app_image

          port {
            container_port = var.app_port
            protocol       = "TCP"
          }

          # Resource requests and limits
          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "250m"
              memory = "256Mi"
            }
          }

          # Liveness probe - checks if container is running
          liveness_probe {
            http_get {
              path = "/"
              port = var.app_port
            }
            initial_delay_seconds = 10
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          # Readiness probe - checks if container is ready to serve traffic
          readiness_probe {
            http_get {
              path = "/"
              port = var.app_port
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 3
            failure_threshold     = 3
          }
        }

        # Restart policy
        restart_policy = "Always"

        # Spread pods across nodes for high availability
        topology_spread_constraint {
          max_skew           = 1
          topology_key       = "kubernetes.io/hostname"
          when_unsatisfiable = "ScheduleAnyway"
          label_selector {
            match_labels = {
              app = var.app_name
            }
          }
        }
      }
    }

    # Deployment strategy
    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_surge       = "25%"
        max_unavailable = "25%"
      }
    }
  }

  # Wait for GKE cluster to be ready
  depends_on = [module.gke]
}

# -----------------------------------------------------------------------------
# Hello World Service (LoadBalancer)
# -----------------------------------------------------------------------------
# Creates a Kubernetes Service of type LoadBalancer that:
# - Provisions a GCP Network Load Balancer
# - Assigns an external IP address
# - Routes traffic from port 80 to the application pods
# -----------------------------------------------------------------------------
resource "kubernetes_service_v1" "hello_app" {
  metadata {
    name      = "${var.app_name}-service"
    namespace = var.app_namespace
    labels = {
      app         = var.app_name
      environment = var.environment
      managed-by  = "terraform"
    }
    annotations = {
      # Optional: Use this annotation to get a static IP (requires pre-created IP)
      # "cloud.google.com/load-balancer-type" = "External"
    }
  }

  spec {
    type = "LoadBalancer"

    selector = {
      app = var.app_name
    }

    port {
      name        = "http"
      port        = var.service_port
      target_port = var.app_port
      protocol    = "TCP"
    }

    # Session affinity (optional - set to "ClientIP" for sticky sessions)
    session_affinity = "None"
  }

  depends_on = [kubernetes_deployment_v1.hello_app]
}
