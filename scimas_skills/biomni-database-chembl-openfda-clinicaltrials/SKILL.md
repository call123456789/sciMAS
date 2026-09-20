---
name: biomni-database-chembl-openfda-clinicaltrials
description: Use to fetch bioactivity, adverse-event, or trial metadata by drug or trial ID.
x-scimas-role: pharma-data-specialist
x-scimas-server: biomni-database
x-scimas-tools:
  - query_chembl
  - query_openfda
  - query_clinicaltrials
---

# ChEMBL / OpenFDA / ClinicalTrials lookups

## When to use
- ChEMBL compound / target queries for known bioactivity values.
- OpenFDA drug adverse-event counts / label sections / recalls.
- ClinicalTrials.gov NCT ID → trial design, endpoints, status.

## Limitations
- Returns research-log strings.
- Heavy use may trigger rate limits on the underlying services.
