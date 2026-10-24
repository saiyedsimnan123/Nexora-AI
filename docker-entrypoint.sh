#!/bin/sh
set -e

# =============================================================================
# Nexora AI Production Container Entrypoint
# =============================================================================
# Handles pre-flight tasks (such as opt-in database migration) before handing
# execution over to the target process via exec.
# Supports dynamic $PORT assignment from cloud hosting environments (Railway/Render).
# =============================================================================

# Default to false unless explicitly opted in
RUN_MIGRATIONS="${RUN_MIGRATIONS:-false}"

if [ "$RUN_MIGRATIONS" = "true" ] || [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "[nexora-entrypoint] RUN_MIGRATIONS is enabled. Running 'alembic upgrade head'..."
    alembic upgrade head
    echo "[nexora-entrypoint] Alembic database migrations completed successfully."
fi

# Dynamically adapt port if cloud environment assigns a custom $PORT
if [ -n "$PORT" ] && [ "$PORT" != "8000" ]; then
    new_args=""
    prev=""
    for arg in "$@"; do
        if [ "$prev" = "--port" ]; then
            new_args="$new_args $PORT"
        else
            if [ -n "$new_args" ]; then
                new_args="$new_args $arg"
            else
                new_args="$arg"
            fi
        fi
        prev="$arg"
    done
    eval "set -- $new_args"
fi

# Fallback to offline demo RAG mode if no database URL is provided
if [ -z "$DATABASE_URL" ] && [ -z "$NEXORA_DATABASE_URL" ]; then
    new_args=""
    for arg in "$@"; do
        if [ "$arg" = "nexora.server:app" ]; then
            new_args="$new_args nexora.api.app:app"
        else
            if [ -n "$new_args" ]; then
                new_args="$new_args $arg"
            else
                new_args="$arg"
            fi
        fi
    done
    eval "set -- $new_args"
    export APP_ENV=demo
fi

# Execute the container command directly, replacing PID 1 so signals (SIGTERM/SIGINT)
# are delivered directly to the application process.
exec "$@"
