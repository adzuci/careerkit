# 0002 - VPC and NAT design

## Status

Accepted

## Context

The GKE cluster uses private nodes, which require outbound access for image pulls and updates. We need a network design that supports private node egress without exposing nodes publicly.

## Decision

Use a custom VPC with a single primary subnet and secondary ranges for pods and services. Add Cloud NAT through a regional Cloud Router to provide outbound internet access for private nodes.

## Consequences

- Private nodes can reach external endpoints without public IPs.
- NAT adds cost and a dependency on Cloud Router availability.
- The single-subnet design is simple but may need expansion for multi-region growth.
