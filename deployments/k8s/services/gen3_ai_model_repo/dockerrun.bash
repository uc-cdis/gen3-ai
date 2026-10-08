#!/bin/bash

# `exec` so uvicorn replaces bash as PID 1 and receives SIGTERM directly. Without it,
# Kubernetes' shutdown signal stops at bash, in-flight requests are cut, and lifespan
# shutdown (closing the DB pool) never runs.
exec /venv/bin/uvicorn gen3_ai_model_repo.main:app_instance --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 90
