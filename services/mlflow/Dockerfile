# MLflow tracking server (Postgres backend + MinIO artifacts).
FROM ghcr.io/mlflow/mlflow:v2.19.0

RUN pip install --no-cache-dir "psycopg2-binary>=2.9" "boto3>=1.34"

EXPOSE 5000

# Command is provided by docker-compose (mlflow server ...).
