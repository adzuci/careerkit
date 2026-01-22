# 0001 - Use the official GKE module

## Status

Accepted

## Context

This stack includes a GKE cluster and node pool. We can either model these with direct resources (`google_container_cluster` and `google_container_node_pool`) or use the official Terraform module from Google.

## Decision

Use direct resources instead of the official module to keep the configuration concise and easy to reason about during interviews. This minimizes indirection and keeps the discussion focused on GKE concepts rather than module internals.

## Consequences

- The configuration is shorter and easier to explain.
- Some advanced defaults from the official module are not included and would need to be added manually if desired.
