FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY rnd-python/pyproject.toml rnd-python/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev && \
    rm -rf /tmp/uv-cache

FROM base AS development

RUN uv sync --frozen && \
    rm -rf /tmp/uv-cache

COPY . .

RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS production

COPY . .

RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "gunicorn", "src.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log"]