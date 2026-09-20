#!/bin/bash
# Runs only on first Postgres data-dir init (empty volume).
# Creates a dedicated DB for the MLflow tracking server.
set -euo pipefail

exists="$(psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1 FROM pg_database WHERE datname='mlflow'")"
if [ "$exists" != "1" ]; then
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE DATABASE mlflow OWNER \"${POSTGRES_USER}\""
fi
