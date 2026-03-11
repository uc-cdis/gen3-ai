set dotenv-load

setup:
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source .justfile_helpers.bash

  print_header "just setup:" "verifying" "uv" "installation..."
  if command -v uv >/dev/null 2>&1; then
      echo "uv is installed."
      echo "  version: $(uv --version)"
  else
      echo "${YELLOW}** WARNING: uv not found in \$PATH. Installing... **${RESET}"
      curl -LsSf https://astral.sh/uv/install.sh | sh
  fi

  print_header "just setup:" "verifying" "postgres" "installation..."
  if command -v psql >/dev/null 2>&1; then
      echo "PostgreSQL client (psql) is installed."
      echo "  version: $(psql --version)"
  else
      echo "${RED}** ERROR: PostgreSQL client (psql) not found in \$PATH. **${RESET}"
      echo "${RED}** Cannot set up databases. Please install PostgreSQL and rerun. **${RESET}"
      exit 1
  fi

  # TODO check if postgres username and password are available, error if not and tell to set in
  # .env file

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      print_header "just setup:" "setting up postgres db for" "${dir#services/}" "service..."

      if [ ! -f .env ]; then
        echo "${RED}** ERROR: .env file not found in "${dir}". Please create one. **${RESET}"
      fi

      set -a
        source "${dir}/.env"
      set +a

      psql \
        -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" \
        -c "CREATE DATABASE \"${DB_NAME}\" WITH OWNER \"${DB_USER}\";" \
        2>/dev/null || echo "Database already exists."

      # TODO: db migration / initial setup
    fi
  done

install $SERVICE="all":
  #!/usr/bin/env bash

  # this includes some helpers for colored line printing
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    # this forces looping over folders in /services
    # and reduces code duplication by re-calling this recipe
    # with a specific service
    just install_all
  else
    # print_header COMMAND TEXT SERVICE TEXT
    print_header "just install:" "installing" "$SERVICE" "service..."

    uv sync --all-packages --group dev --directory "./services/$SERVICE"
  fi

lock $SERVICE="all":
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just lock_all
  else
    print_header "just lock:" "locking" "$SERVICE" "service..."

    uv lock --directory "./services/$SERVICE" --upgrade
  fi

test $SERVICE="all":
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just test_all
  else
    print_header "just test:" "testing" "$SERVICE" "service..."
    cd "./services/$SERVICE"
    uv run pytest .
    exit_code=$?
    cd -

    overall_exit=$((overall_exit | $exit_code))

    report_error_or_success $overall_exit "just test:" "testing" "$SERVICE" "service!"

    exit $overall_exit
  fi

build $SERVICE="all":
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just build_all
  else
    print_header "just build:" "building" "$SERVICE" "service..."
    docker build -t $SERVICE --build-arg SERVICE_NAME="$SERVICE" .

    report_error_or_success $? "just build:" "building" "$SERVICE" "service!"
  fi

@run $SERVICE:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  print_header "just run:" "running" "$SERVICE" "service..."

  # Opentelemetry in MacOS can cause an error due to forking, to avoid the error:
  # "+[NSMutableString initialize] may have been in progress in another thread when fork() was called."
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=1

  # Start the app with OpenTelemetry and Gunicorn and Uvicorn workers
  uv run --directory "./services/$SERVICE" \
    opentelemetry-instrument \
    gunicorn \
    $SERVICE.main:app_instance \
    -k uvicorn.workers.UvicornWorker \
    -c gunicorn.conf.py \
    --access-logfile - \
    --error-logfile -

@docker_run $SERVICE $EXTERNAL_PORT="8001" $INTERNAL_PORT="4141":
  #!/usr/bin/env bash
  print_header "just docker_run:" "running" "$SERVICE" "service..."
  docker kill $SERVICE
  docker rm $SERVICE
  SERVICE_NAME=$SERVICE docker run --name $SERVICE \
  --env-file "./.env" \
  -v "$PROMETHEUS_MULTIPROC_DIR":"$PROMETHEUS_MULTIPROC_DIR" \
  -p {{EXTERNAL_PORT}}:{{INTERNAL_PORT}} \
  $SERVICE:latest

format $SERVICE="all":
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just format_all
  else
    print_header "just format:" "formatting" "$SERVICE" "service..."
    if [ $SERVICE == "all" ]; then
      for dir in services/*; do
        print_header "just format:" "formatting" "${dir#services/}" "..."
        uv run --directory "./services/${dir#services/}" ruff format
      done
    else
      uv run --directory ./services/$SERVICE ruff format
    fi
  fi

# `just lint $SERVICE`
lint $SERVICE="all":
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just lint_all
  else
    # install to get ruff and other things required for formatting and linting
    just install $SERVICE
    just format $SERVICE

    print_header "just lint:" "ruff check" "$SERVICE" "..."
    uv run --directory "./services/$SERVICE" ruff check ./src --fix

    exit_code=$?
    report_error_if_failed $exit_code "just lint:" "ruff check" "$SERVICE" "service!"
    overall_exit=$((overall_exit | $exit_code))
    echo

    print_header "just lint:" "deptry" "$SERVICE" "..."
    uv run --directory "./services/$SERVICE" deptry ./src

    exit_code=$?
    report_error_if_failed $exit_code "just lint:" "deptry" "$SERVICE" "service!"
    overall_exit=$((overall_exit | $exit_code))
    echo

    print_header "just lint:" "pylint setup and run" "$SERVICE" "..."
    if [ ! -d ~/.gen3/.github ]; then
        git clone git@github.com:uc-cdis/.github.git ~/.gen3/.github
    fi

    cd ./services/$SERVICE && \
    bash ~/.gen3/.github/.github/linters/update_pylint_config.sh && \
    cd -

    uv run --directory "./services/$SERVICE" pylint ./src --rcfile ~/.gen3/.github/.github/linters/.python-lint

    exit_code=$?
    report_error_if_failed $exit_code "just lint:" "pylint setup and run" "$SERVICE" "service!"
    overall_exit=$((overall_exit | $exit_code))

    report_error_or_success $overall_exit "just lint:" "linting" "$SERVICE" "service!"

    exit $overall_exit
  fi

# you can use this instead: `just install`
install_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just install ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

# you can use this instead: `just lint`
lint_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just lint ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just lint:" "linting" "all" "services!"

  exit $overall_exit

# you can use this instead: `just format`
format_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just format ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

# you can use this instead: `just build`
build_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just build ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

# you can use this instead: `just lock`
lock_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just lock ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

# you can use this instead: `just test`
test_all:
  #!/usr/bin/env bash
  source .justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just test ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just test:" "testing" "all" "services!"

  exit $overall_exit
