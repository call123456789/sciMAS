---
name: biomni-protocols-wetlab-reference
description: Use to look up a wet-lab protocol from protocols.io by keyword or ID and read the step-by-step procedure.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-protocols
x-scimas-tools:
  - search_protocols
  - get_protocol_details
  - list_local_protocols
  - read_local_protocol
---

# Wet-lab protocol reference

## When to use
- User wants a published wet-lab protocol (PCR setup, Western blot, transfection, etc.) and either a text search or a numeric protocol ID.
- User has locally cached protocols and wants them indexed / read.

## Limitations
- ``search_protocols`` / ``get_protocol_details`` hit the protocols.io API; network availability determines success.
- ``list_local_protocols`` / ``read_local_protocol`` look under a local directory configured by BiOMNI; if the directory is empty they return a JSON error rather than a fake list.
