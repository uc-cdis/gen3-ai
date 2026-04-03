#!/bin/bash

nginx
uv run --directory /venv gunicorn gen3_ai_model_repo.main:app_instance -k uvicorn.workers.UvicornWorker -c /services/gunicorn.conf.py --user gen3 --group gen3
