# 0003 - IAM least-privilege for nodes

## Status

Accepted

## Context

GKE nodes require permissions for logging, monitoring, and accessing storage objects. Overly broad roles increase blast radius if a node is compromised.

## Decision

Bind a dedicated node service account to specific logging, monitoring, and storage object viewer roles rather than using the default compute service account or owner-level permissions.

## Consequences

- Node access is scoped to the minimum set of permissions needed for common workloads.
- Additional roles may be required for specialized workloads and should be added explicitly.
