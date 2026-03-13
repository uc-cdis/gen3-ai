ARG AZLINUX_BASE_VERSION=3.13-pythonnginx
FROM quay.io/cdis/amazonlinux-base:${AZLINUX_BASE_VERSION} AS base
ARG SERVICE_NAME

# TODO: should this go in base image?
ENV UV_NO_MANAGED_PYTHON=true
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PROJECT_ENVIRONMENT=/venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV OTEL_SERVICE_NAME=${SERVICE_NAME}
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy.monitoring:4318
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf

COPY --chown=gen3:gen3 /libraries /libraries
COPY --chown=gen3:gen3 /services/${SERVICE_NAME}/src /services/${SERVICE_NAME}/src
COPY --chown=gen3:gen3 /services/${SERVICE_NAME}/dockerrun.bash /services
COPY --chown=gen3:gen3 /services/${SERVICE_NAME}/gunicorn.conf.py /services

WORKDIR /services/${SERVICE_NAME}

# Builder stage
FROM base AS builder

USER root
RUN chown -R gen3:gen3 /venv

USER gen3

COPY --chown=gen3:gen3 /services/${SERVICE_NAME}/uv.lock /services/${SERVICE_NAME}
COPY --chown=gen3:gen3 /services/${SERVICE_NAME}/pyproject.toml /services/${SERVICE_NAME}

# locked and frozen ensure lock file is not modified (e.g.
# exact deps are installed as listed)
RUN uv sync -vv --no-dev --all-extras --frozen

# Final stage
FROM base

COPY --from=builder /services/${SERVICE_NAME} /${SERVICE_NAME}
COPY --from=builder /venv /venv
ENV  PATH="/usr/sbin:$PATH"
USER root
RUN mkdir -p /var/log/nginx
RUN chown -R gen3:gen3 /var/log/nginx

# Switch to non-root user 'gen3' for the serving process

USER gen3

CMD ["/bin/bash", "-c", "/services/dockerrun.bash"]
