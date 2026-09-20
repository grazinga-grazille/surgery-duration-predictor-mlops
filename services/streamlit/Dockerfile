# Streamlit service image. Build context = repo root.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY services/streamlit/app.py ./app.py
COPY services/streamlit/tabs ./tabs

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

# --no-sync: deps are baked in at build time; avoid PyPI/DNS at container start
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
