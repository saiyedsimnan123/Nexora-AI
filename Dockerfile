# =============================================================================
# Nexora AI — Production Container Image
# Multi-stage / minimal footprint Python 3.12 slim container
# =============================================================================

FROM python:3.12-slim-bookworm AS runtime

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_ENV=production

# Security: Create non-root system user and group
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Install pinned production runtime dependencies
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Copy project configuration, migrations, entrypoint, and application source
COPY --chown=appuser:appgroup alembic.ini docker-entrypoint.sh ./
COPY --chown=appuser:appgroup src/ ./src/

# Ensure entrypoint is executable
RUN chmod +x docker-entrypoint.sh

# Switch to unprivileged non-root user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container liveness health check using Python standard library (no curl dependency required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health')"]

# Production ASGI Entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "nexora.server:app", "--host", "0.0.0.0", "--port", "8000"]
