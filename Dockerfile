# syntax=docker/dockerfile:1.7

# NOTE: CI will override BASE with an immutable digest (e.g., python:3.10-slim@sha256:...)
# Keep a sane default tag for local builds.
ARG BASE=python:3.10-slim

##############################################
# Stage 1: Build project dependencies        #
##############################################
FROM ${BASE} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Python sane defaults for deterministic installs
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Toolchain only for building native deps; never present in the runtime image
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential gcc libffi-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to resolve deps (better cache utilization)
COPY requirements.txt envtool.sh ./

# Ensure envtool.sh is executable
RUN chmod +x envtool.sh

# It must create /app/.venv with all production deps installed
RUN PYTHON_BINARY_OVERRIDE=python3 bash envtool.sh install prod

##############################################
# Stage 2: Minimal runtime image             #
##############################################
FROM ${BASE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Production-safe Python env defaults
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH" \
    TMP_DIR=/tmp \
    LOGS_DIR=False

# Deterministic non-root user
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} app \
 && useradd -u ${APP_UID} -g ${APP_GID} -m -s /usr/sbin/nologin app

WORKDIR /app

# Bring the prepared virtualenv from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

COPY --chown=app:app config /tmp/config

# Copy application code last for better layer caching
# Ensure a proper .dockerignore to avoid leaking secrets
COPY --chown=app:app . /app

# Ensure entrypoint exists and is executable (fail fast if missing)
RUN test -f /app/run_tasks.sh && chmod +x /app/run_tasks.sh

USER app:app

# Exec-form: signals (SIGTERM) propagate correctly for graceful shutdown
ENTRYPOINT ["/app/run_tasks.sh"]
CMD ["update"]
