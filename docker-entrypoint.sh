#!/bin/sh
set -e

# =============================================================================
# Nexora AI Production Container Entrypoint
# =============================================================================
# Handles pre-flight tasks (such as opt-in database migration) before handing
# execution over to the target process via exec.
#
# Usage:
#   RUN_MIGRATIONS=true /app/docker-entrypoint.sh uvicorn nexora.server:app ...
# =============================================================================

# Default to false unless explicitly opted in
RUN_MIGRATIONS="${RUN_MIGRATIONS:-false}"

if [ "$RUN_MIGRATIONS" = "true" ] || [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "[nexora-entrypoint] RUN_MIGRATIONS is enabled. Running 'alembic upgrade head'..."
    alembic upgrade head
    echo "[nexora-entrypoint] Alembic database migrations completed successfully."
fi

# Execute the container command directly, replacing PID 1 so signals (SIGTERM/SIGINT)
# are delivered directly to the application process.
exec "$@"
