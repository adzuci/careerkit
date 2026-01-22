# Apollo Interview Prep: GKE + Terraform Stack

This project demonstrates provisioning a Google Kubernetes Engine (GKE) cluster using Terraform and deploying a "Hello World" web application that is exposed via a LoadBalancer service.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Google Cloud Platform                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        VPC Network                             │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                      GKE Cluster                         │  │  │
│  │  │  ┌─────────────────────────────────────────────────────┐ │  │  │
│  │  │  │                   Node Pool                          │ │  │  │
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │ │  │  │
│  │  │  │  │ Pod     │  │ Pod     │  │ Pod     │             │ │  │  │
│  │  │  │  │hello-app│  │hello-app│  │hello-app│             │ │  │  │
│  │  │  │  └────┬────┘  └────┬────┘  └────┬────┘             │ │  │  │
│  │  │  └───────┼────────────┼────────────┼──────────────────┘ │  │  │
│  │  │          └────────────┼────────────┘                    │  │  │
│  │  │                       │                                  │  │  │
│  │  │              ┌────────┴────────┐                        │  │  │
│  │  │              │ Service (LB)    │                        │  │  │
│  │  │              │ Port 80 → 8080  │                        │  │  │
│  │  │              └────────┬────────┘                        │  │  │
│  │  └───────────────────────┼─────────────────────────────────┘  │  │
│  └──────────────────────────┼────────────────────────────────────┘  │
│                             │                                        │
│                    ┌────────┴────────┐                              │
│                    │ Cloud Load      │                              │
│                    │ Balancer        │                              │
│                    │ External IP     │                              │
│                    └────────┬────────┘                              │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                         ┌────┴────┐
                         │ Internet│
                         │ Users   │
                         └─────────┘
```

## Prerequisites

Before deploying this stack, ensure you have:

1. **Terraform** >= 1.3.0 installed
   ```bash
   terraform version
   ```

2. **Google Cloud SDK (gcloud)** installed and authenticated
   ```bash
   gcloud version
   gcloud auth login
   gcloud auth application-default login
   ```

3. **A GCP Project** with billing enabled and the following APIs enabled:
   - Kubernetes Engine API
   - Compute Engine API
   - Cloud Resource Manager API

   Enable APIs via gcloud:
   ```bash
   gcloud services enable container.googleapis.com \
       compute.googleapis.com \
       cloudresourcemanager.googleapis.com \
       --project=YOUR_PROJECT_ID
   ```

4. **Sufficient IAM permissions** in the GCP project:
   - `roles/container.admin` (Kubernetes Engine Admin)
   - `roles/compute.admin` (Compute Admin)
   - `roles/iam.serviceAccountUser` (Service Account User)

## Quick Start

### Step 1: Clone and Configure

```bash
# Navigate to the terraform directory
cd terraform

# Copy the example tfvars file
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your GCP project ID
# At minimum, set: project_id = "your-gcp-project-id"
```

### Step 2: Initialize Terraform

```bash
terraform init
```

This downloads the required providers and modules:
- `hashicorp/google` - GCP provider
- `hashicorp/kubernetes` - Kubernetes provider
- `terraform-google-modules/network/google` - VPC module
- `terraform-google-modules/kubernetes-engine/google` - GKE module

### Step 3: Review the Plan

```bash
terraform plan
```

Review the planned changes. You should see:
- 1 VPC network with subnet
- 1 GKE cluster with node pool
- 1 Kubernetes Deployment (hello-app)
- 1 Kubernetes Service (LoadBalancer)

### Step 4: Deploy the Stack

```bash
terraform apply
```

Type `yes` when prompted. Deployment typically takes 10-15 minutes (GKE cluster creation is the longest step).

### Step 5: Access the Application

After deployment completes:

1. **Get the external IP:**
   ```bash
   terraform output hello_app_external_ip
   ```

   Or configure kubectl and use:
   ```bash
   # Configure kubectl
   $(terraform output -raw kubectl_config_command)

   # Get service details
   kubectl get service hello-app-service
   ```

2. **Test the application:**
   ```bash
   # Using the external IP from the output
   curl http://<EXTERNAL_IP>:80
   ```

   Expected response:
   ```
   Hello, world!
   Version: 1.0.0
   Hostname: hello-app-xxxxxx-xxxxx
   ```

3. **Open in browser:**
   Navigate to `http://<EXTERNAL_IP>` in your web browser.

## File Structure

```
terraform/
├── README.md              # This documentation
├── versions.tf            # Terraform and provider version constraints
├── variables.tf           # Input variable definitions
├── main.tf                # GKE cluster and VPC infrastructure
├── kubernetes.tf          # Kubernetes Deployment and Service
├── outputs.tf             # Output values (endpoints, IPs, commands)
└── terraform.tfvars.example  # Example variable values
```

## Configuration Options

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_id` | (required) | GCP project ID |
| `region` | `us-central1` | GCP region for the cluster |
| `cluster_name` | `apollo-demo-cluster` | Name of the GKE cluster |
| `machine_type` | `e2-medium` | Node machine type |
| `min_node_count` | `1` | Minimum nodes per zone |
| `max_node_count` | `3` | Maximum nodes per zone |
| `app_replicas` | `3` | Number of application pods |

### Customization Examples

**Use a different region:**
```hcl
region = "us-west1"
zones  = ["us-west1-a", "us-west1-b"]
```

**Increase capacity:**
```hcl
machine_type   = "e2-standard-2"
max_node_count = 5
app_replicas   = 5
```

**Use preemptible nodes for cost savings:**
Edit `main.tf` and set `preemptible = true` in the node pool configuration.

## Verification Commands

After deployment, verify the infrastructure:

```bash
# 1. Check cluster status
gcloud container clusters list --project=YOUR_PROJECT_ID

# 2. Configure kubectl
gcloud container clusters get-credentials apollo-demo-cluster \
    --region us-central1 \
    --project YOUR_PROJECT_ID

# 3. Verify nodes
kubectl get nodes

# 4. Check deployments
kubectl get deployments

# 5. Check pods
kubectl get pods -l app=hello-app

# 6. Check service and external IP
kubectl get service hello-app-service

# 7. Describe the deployment for details
kubectl describe deployment hello-app

# 8. View pod logs
kubectl logs -l app=hello-app --tail=10
```

## Cleanup / Teardown

To destroy all resources and avoid ongoing charges:

```bash
terraform destroy
```

Type `yes` when prompted. This removes:
- The LoadBalancer service and external IP
- The Kubernetes deployment
- The GKE cluster and node pool
- The VPC network and subnet

**Note:** Ensure all resources are destroyed to avoid unexpected charges. Verify in the GCP Console that no orphaned resources remain.

## Troubleshooting

### LoadBalancer IP shows "Pending"

The external IP may take a few minutes to provision. Wait and run:
```bash
terraform refresh
terraform output hello_app_external_ip
```

Or check with kubectl:
```bash
kubectl get service hello-app-service -w
```

### Terraform Apply Fails with API Not Enabled

Enable the required APIs:
```bash
gcloud services enable container.googleapis.com compute.googleapis.com
```

### Insufficient Permissions

Ensure your account has the required IAM roles:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:YOUR_EMAIL" \
    --role="roles/container.admin"
```

### Pods Not Starting

Check pod events:
```bash
kubectl describe pod -l app=hello-app
kubectl get events --sort-by='.lastTimestamp'
```

## Cost Considerations

This demo stack incurs GCP charges for:
- **GKE cluster:** Management fee (~$0.10/hour for standard cluster)
- **Compute nodes:** Based on machine type and count (e2-medium ~$0.03/hour each)
- **Load Balancer:** Network LB forwarding rule (~$0.025/hour)
- **Egress:** Data transfer out of GCP

**Estimated cost:** ~$3-5/day for the default configuration.

**Cost optimization tips:**
- Use `preemptible = true` for nodes (up to 80% savings)
- Reduce `max_node_count` if not needed
- Destroy resources when not in use (`terraform destroy`)

## Interview Discussion Points

When presenting this solution, be prepared to discuss:

1. **Infrastructure as Code Benefits:**
   - Reproducibility and version control
   - Self-documenting infrastructure
   - Easy teardown and recreation

2. **GKE Module Choice:**
   - Using official terraform-google-modules provides production-ready defaults
   - Handles complex configurations (VPC-native clusters, node pool management)

3. **Kubernetes Resource Design:**
   - Deployment with replicas for high availability
   - Rolling update strategy for zero-downtime deployments
   - Health checks (liveness/readiness probes)
   - Resource requests/limits for predictable scheduling

4. **Service Exposure:**
   - LoadBalancer type creates a GCP Network Load Balancer
   - Alternative: Ingress with HTTP(S) Load Balancer for path-based routing

5. **Security Considerations:**
   - VPC-native cluster with dedicated subnet
   - Workload identity (can be enabled in the module)
   - Network policies (can be enabled for pod-to-pod restrictions)

6. **Scaling:**
   - Cluster autoscaler configured via min/max node count
   - HPA can be added for pod-level autoscaling

## Next Steps / Enhancements

For a production-ready deployment, consider adding:

- [ ] **Remote State:** Store Terraform state in GCS bucket
- [ ] **Workload Identity:** More secure service account binding
- [ ] **Ingress Controller:** For path-based routing and TLS
- [ ] **Network Policies:** Restrict pod-to-pod communication
- [ ] **Monitoring:** Cloud Monitoring and Logging integration
- [ ] **CI/CD:** GitOps with Cloud Build or GitHub Actions

## References

- [Terraform GKE Module](https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google)
- [Terraform Kubernetes Provider](https://registry.terraform.io/providers/hashicorp/kubernetes/latest)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Google Hello App Sample](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples/tree/main/quickstarts/hello-app)
