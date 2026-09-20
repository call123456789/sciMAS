---
name: pharma-automl-service
description: Use for MADD predictive/generative service status checks and validated training request submission.
x-scimas-role: pharma-ml-engineer
x-scimas-server: pharma-automl
x-scimas-tools:
  - madd_server_state
  - madd_case_state
  - start_madd_ml_training
  - start_madd_generative_training
---
# Pharma AutoML Service

Use this skill to inspect MADD model service state or submit a validated
training request.

Check URL_PRED or URL_GEN availability before claiming a remote service result.
For training, verify the dataset exists, has enough rows, includes all required
feature and target columns, and keeps SMILES strings within the service length
limit. If a local request times out after submission, tell the user to check
the case state rather than treating the timeout as a completed training run.
