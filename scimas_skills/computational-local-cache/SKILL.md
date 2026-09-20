---
name: computational-local-cache
description: Use for initializing and querying the local SQLite symmetry cache.
x-scimas-role: computational-chemist
x-scimas-server: chemistry-computational
x-scimas-tools:
  - init_local_symmetry_db
  - query_local_symmetry
---
# Computational Local Symmetry Cache

Use this skill only when a step explicitly needs the local symmetry-cache database.

Initialize the database before querying it. If a query misses, hand off to point-group lookup or geometry analysis rather than inventing a cached value.
