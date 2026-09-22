# Multi-stage build: the wheels are compiled once in a build stage and only the installed
# environment is copied forward, which keeps the runtime image free of compilers and caches.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependency manifests first: this layer is cached and only rebuilds when they change,
# so an ordinary source edit does not reinstall TensorFlow.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install ".[deep]"


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Run as a non-root user: a container that does not need root should not have it.
RUN useradd --create-home --uid 1000 forecast
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=forecast:forecast configs/ ./configs/
COPY --chown=forecast:forecast src/ ./src/
COPY --chown=forecast:forecast scripts/ ./scripts/
COPY --chown=forecast:forecast pyproject.toml README.md ./

# Mount points for the things that must not live in the image: input data and run outputs.
RUN mkdir -p data/raw data/processed data/external models reports/figures experiments \
    && chown -R forecast:forecast /app

USER forecast

# Fails fast if the package cannot be imported, so a broken image is caught at start.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import energy_forecast" || exit 1

ENTRYPOINT ["energy-forecast"]
CMD ["--help"]

# Build and run:
#   docker build -t energy-forecast .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#     -v "$PWD/models:/app/models" energy-forecast train --skip-deep
