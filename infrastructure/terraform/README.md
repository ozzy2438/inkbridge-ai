# Terraform configuration placeholder
# For a future AWS/GCP deployment

# Infrastructure components:
# - Object storage (S3/GCS) for uploads
# - Container registry (ECR/GCR)
# - Kubernetes cluster or ECS/Cloud Run
# - Redis for job queue
# - PostgreSQL for metadata
# - Monitoring (CloudWatch/Stackdriver)

The implemented self-hosted boundary is documented in
[`docs/protected_storage_lifecycle.md`](../../docs/protected_storage_lifecycle.md). It verifies
repository-external owner-only POSIX permissions and lifecycle evidence, but does not deploy or
certify cloud IAM, KMS, network isolation, immutable audit logs, or backup erasure. Provider-specific
Terraform must be added only after the target account, region, identity trust policy, and retention
authority are known.
