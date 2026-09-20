# MLflow tracking server (Postgres backend + MinIO artifacts).
FROM ghcr.io/mlflow/mlflow:v2.19.0

RUN pip install --no-cache-dir "psycopg2-binary>=2.9" "boto3>=1.34"

COPY services/mlflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
