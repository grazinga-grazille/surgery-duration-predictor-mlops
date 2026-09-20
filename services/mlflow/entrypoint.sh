#!/bin/sh
# Build a safe Postgres URI (handles @ : / # in passwords), then start MLflow.
set -eu

BACKEND_URI="$(python - <<'PY'
import os
from urllib.parse import quote_plus

user = quote_plus(os.environ["POSTGRES_USER"])
password = quote_plus(os.environ["POSTGRES_PASSWORD"])
print(f"postgresql://{user}:{password}@postgres:5432/mlflow")
PY
)"

BUCKET="${MLFLOW_ARTIFACT_BUCKET:-mlflow-artifacts}"

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "s3://${BUCKET}/"
