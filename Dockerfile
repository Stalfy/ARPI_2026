FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARPI_BASE_DIR=/app/data/input \
    ARPI_OUTPUT_DIR=/app/data/output \
    ARPI_MAX_DAYS=0

WORKDIR /app

# Use the official uv binary — avoids pip bootstrap and is faster.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock README.md ./

# Install runtime dependencies only (not the local package yet).
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the package.
COPY src/ ./src/

RUN uv sync --frozen --no-dev

VOLUME ["/app/data/input", "/app/data/output"]

# ENTRYPOINT (not CMD) so extra flags (e.g. --compare, --agency RTL) can be appended by the caller.
ENTRYPOINT ["/app/.venv/bin/python", "-m", "arpi.main"]
