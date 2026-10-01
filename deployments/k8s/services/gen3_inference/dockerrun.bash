#!/bin/bash

# `exec` so uvicorn replaces bash as PID 1 and receives SIGTERM directly. Without it,
# Kubernetes' shutdown signal stops at bash, in-flight requests are cut, and lifespan
# shutdown (closing the DB pool) never runs.
#
# One process per container on purpose - do NOT add `--workers`. FastAPI's guidance for a
# clustered deployment is a single process per container, letting Kubernetes own
# replication: https://fastapi.tiangolo.com/deployment/docker/#replication-number-of-processes
# Scale with replicas instead.
exec /venv/bin/uvicorn gen3_inference.main:app_instance --host 0.0.0.0 --port "${PORT:-8000}" --timeout-graceful-shutdown 90
