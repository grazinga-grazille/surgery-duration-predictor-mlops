# Surgery Duration Predictor

OR surgery duration prediction for operating-room scheduling. A Random Forest
(with TF-IDF / SVD procedure-text features) is the production model; Linear
Regression and XGBoost are available for comparison and Optuna tuning.

Built on the
[MLOps for Data Scientists](https://futureproofds.com/blog/mlops-for-data-scientists)
project structure (`uv`, `src/` package, thin scripts, Docker, CI).

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # environment + deps (includes pytest + ruff)
uv run pytest          # unit tests
uv run ruff check .    # lint
```

Optional groups:

```bash
uv sync --group notebook   # Jupyter
uv sync --group research   # surgical_stopwords offline pipeline
```

Copy the env template (never commit a real `.env`):

```bash
cp .env.example .env
```

> **No uv?** `curl -LsSf https://astral.sh/uv/install.sh | sh`
> ([other platforms](https://docs.astral.sh/uv/getting-started/installation/)).

### Data

Place the surgical cases CSV at `data/01_raw/LoadFile.csv` (gitignored — confidential).

`load_data()` keeps only rows with `ScheduledDate <= 2024-04-27` for ML train/test.
Later cases are reserved for monitoring.

---

## Train & evaluate

```bash
uv run scripts/train.py       # fit RF + artifacts → models/*.pkl
uv run scripts/evaluate.py    # test metrics vs booked-duration baseline
```

### Compare models (LR assumptions + XGBoost)

```bash
uv run scripts/compare_models.py
uv run scripts/compare_models.py --skip-rf   # faster
```

### Hyperparameter tuning (Optuna)

64% train / 16% valid / 20% test; minimize validation MAE (minutes).

```bash
uv run scripts/tune_models.py                         # RF + XGBoost
uv run scripts/tune_models.py --model xgboost
uv run scripts/tune_models.py --model random_forest --n-trials 30
```

Best params are written to `models/tuning_*.json`.

---

## Serve

### Streamlit

```bash
uv run streamlit run services/streamlit/app.py
# → http://localhost:8501
```

Business Analysis needs `.streamlit/secrets.toml` (`financial_impact_csv`,
`resource_utilization_csv`). Without it, that tab shows a notice; Prediction /
ML Analysis / About still work.

### FastAPI

Loads the estimator + TF-IDF/SVD from **MLflow MinIO artifacts** when configured
(``.env``: `MLFLOW_LOAD_FROM_ARTIFACTS=true`, `MLFLOW_SERVING_MODEL_FAMILY=xgboost`,
or `MLFLOW_MODEL_URI=runs:/<run_id>/model`). Otherwise uses local `models/*.pkl`.

```bash
# After compare_models has logged model/ + features/ artifacts:
uv run uvicorn main:app --app-dir services/fastapi --reload
# → http://localhost:8000  (GET /model, POST /predict)
```

### Streamlit → API

The Prediction tab POSTs sidebar fields (including procedure description) to
`API_BASE_URL/predict`. FastAPI embeds the text and returns the duration.
Set `API_BASE_URL=http://localhost:8000` locally (Compose uses `http://fastapi:8000`).

### Docker

```bash
docker compose up --build
# Streamlit :8501  |  FastAPI :8000  |  Postgres :5432  |  MinIO :9000/:9001  |  MLflow :5000
# Infra + tracking only:
docker compose up -d postgres minio minio-init mlflow
```

### MLflow

Tracking server (Compose): UI at http://localhost:5000  
Artifacts go to MinIO bucket `mlflow-artifacts` (created by `minio-init`).

Set in `.env` (see `.env.example`):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=...          # same as MINIO_ROOT_USER
AWS_SECRET_ACCESS_KEY=...      # same as MINIO_ROOT_PASSWORD
```

If Postgres was already initialized before MLflow was added, create the DB once:

```bash
docker exec -it surgery_postgres \
  psql -U surgery -d surgery_ml \
  -c 'CREATE DATABASE mlflow OWNER surgery;'
```

Then log runs:

```bash
uv run scripts/compare_models.py          # experiment: surgery-duration-compare
uv run scripts/tune_models.py --n-trials 5   # experiment: surgery-duration-tune
# Champion = lowest test MAE among tuned models in that invocation
```

Use `--no-mlflow` to skip tracking.
---

## What goes where

```
.
├── src/surgery_duration_predictor/   # Installable package
│   ├── data.py                       #   load + ML date cutoff
│   ├── features.py                   #   TF-IDF / SVD + categoricals
│   ├── train.py                      #   RF training
│   ├── predict.py                    #   inference
│   ├── artifacts.py                  #   load models/*.pkl
│   ├── diagnostics.py                #   OLS assumption checks
│   ├── model_comparison.py           #   LR / RF / XGBoost
│   └── tuning.py                     #   Optuna studies
├── scripts/                          # Thin CLIs
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   └── tune_models.py
├── services/
│   ├── streamlit/                    # Streamlit UI (app.py + tabs/)
│   ├── fastapi/                      # FastAPI prediction service
│   ├── mlflow/                       # MLflow tracking server image
│   ├── prefect/ nannyml/ nginx/      # placeholders
├── db/
│   ├── init.sql                      # App schema placeholder
│   └── 02_create_mlflow_db.sh        # Creates mlflow DB on first Postgres init
├── monitoring/                       # Prometheus / Grafana placeholders
├── volumes/                          # Local Postgres + MinIO data (gitignored)
├── surgical_stopwords/               # Offline stopword / TF-IDF research pipeline
├── notebooks/                        # Exploration (imports from src/)
├── tests/
├── config/config.yaml                # Hyperparams, cutoff, cost rates
├── data/                             # Raw data gitignored
├── models/                           # Artifacts gitignored; regenerated by train
├── .env.example                      # Documented env vars (no secrets)
├── docker-compose.yml                # Postgres, MinIO, MLflow, Streamlit, FastAPI
├── .github/workflows/                # CI: ruff + pytest
├── pyproject.toml
└── uv.lock
```

---

## Conventions

- **`src/` layout.** The package must be installed to import it.
- **Notebooks import from `src/`, not the reverse.**
- **Scripts stay thin.** Logic lives in the package; scripts wire I/O.
- **`data/01_raw/` is immutable.** Never edit source data in place.
- **Secrets stay out of git.** Use `.env.example` and ignore `.env` /
  `.streamlit/secrets.toml` / raw CSVs / model pickles.
- **Dependencies are locked** (`uv.lock`) for local, CI, and container parity.
