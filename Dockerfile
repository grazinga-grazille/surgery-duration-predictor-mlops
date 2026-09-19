# Containerize the project so it runs the same anywhere.
FROM python:3.12-slim

# Install uv, the same tool you use locally.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY . .

# Install production dependencies from the lockfile (no dev extras).
RUN uv sync --frozen --no-dev

EXPOSE 8501

# Default: launch the Streamlit app.
# Override with `docker run ... uv run uvicorn api.main:app` for the API.
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
