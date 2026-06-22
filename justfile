set dotenv-load

# show available tasks
default:
    @just --list

_check_dependencies:
    @./scripts/check_dependencies.bash

setup:
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  print_header "just setup:" "verifying" "uv" "installation..."
  if command -v uv >/dev/null 2>&1; then
      echo "uv is installed."
      echo "  version: $(uv --version)"
  else
      echo "${YELLOW}** WARNING: uv not found in \$PATH. Installing... **${RESET}"
      curl -LsSf https://astral.sh/uv/install.sh | sh
  fi

  print_header "just setup:" "verifying" "PostgreSQL client (psql)" "installation..."
  if command -v psql >/dev/null 2>&1; then
      echo "PostgreSQL client (psql) is installed."
      echo "  version: $(psql --version)"
  else
      echo "${RED}** ERROR: PostgreSQL client (psql) not found in \$PATH. **${RESET}"
      echo "${RED}** Cannot set up databases. Please install PostgreSQL and rerun. **${RESET}"
      exit 1
  fi

  print_header "just setup:" "verifying" "dbmate (db migrations tool)" "installation..."
  if command -v dbmate >/dev/null 2>&1; then
      echo "dbmate (db migrations tool) is installed."
      echo "  version: $(dbmate --version)"
  else
      echo "${RED}** ERROR: dbmate (db migrations tool) not found in \$PATH. **${RESET}"
      echo "${RED}** Cannot migrate database. Please install dbmate and rerun. **${RESET}"
      echo "${RED}** See: https://github.com/amacneil/dbmate#installation **${RESET}"
      exit 1
  fi

  print_header "just setup:" "verifying" "pre-commit" "installation..."
  if command -v pre-commit >/dev/null 2>&1; then
      echo "pre-commit is installed."
      echo "  version: $(pre-commit --version)"
  else
      echo "${YELLOW}** WARNING: pre-commit not found in \$PATH. Installing with pip... **${RESET}"
      pip install pre-commit
  fi

  hook_path="$(git rev-parse --git-path hooks/pre-commit)"

  if [[ ! -f "$hook_path" ]]; then
    echo "${YELLOW}** WARNING: pre-commit git hook not found from: `git rev-parse --git-path hooks/pre-commit`. Installing... **${RESET}"
    pre-commit install
  elif ! grep -q 'pre-commit' "$hook_path"; then
    echo "${YELLOW}** WARNING: pre-commit git hook not found from: `git rev-parse --git-path hooks/pre-commit`. Installing... **${RESET}"
    pre-commit install --overwrite
  fi

  echo "pre-commit git hook is installed."

setup_db $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  run_for_service() {
    local service_name="$1"
    local dir="services/$service_name"

    if [ ! -d "$dir" ]; then
      echo "${RED}** ERROR: Service directory '$dir' does not exist. **${RESET}"
      exit 1
    fi

    if [ "$service_name" == "gen3_inference" ]; then
      print_header "just setup_db:" "No PostgreSQL db needed for" "$service_name" "service. Nothing to do."
      return
    fi

    print_header "just setup_db:" "setting up PostgreSQL db for" "$service_name" "service..."

    if [ ! -f "${dir}/.env" ]; then
      echo "${RED}** WARNING: .env file not found in '${dir}'. Will rely on environment variables. **${RESET}"
    else
      echo "Found .env file: ${dir}/.env - Using it to set up database."
      set -a
        source "${dir}/.env"
      set +a
    fi

    set_postgres_defaults

    psql \
      -d postgres \
      -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" \
      -c "CREATE DATABASE \"${PGDATABASE}\" WITH OWNER \"${PGUSER}\";" \
      2>/dev/null || echo "Database already exists."
  }

  if [ "$SERVICE" == "all" ]; then
    for dir in services/*; do
      if [[ -n "${dir#services/}" ]]; then
        run_for_service "${dir#services/}"
      fi
    done
  else
    run_for_service "$SERVICE"
  fi

db_load $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  just _run_dbmate ${SERVICE} load

db_migrate $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  just _run_dbmate ${SERVICE} migrate

db_rollback $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  just _run_dbmate ${SERVICE} down

_run_dbmate $SERVICE $ACTION:
  #!/usr/bin/env bash
  set -euo pipefail

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  run_for_service() {
    local service_name="$1"
    local dir="services/$service_name"

    if [ ! -d "$dir" ]; then
      echo "${RED}** ERROR: Service directory '$dir' does not exist. **${RESET}"
      exit 1
    fi

    if [ "$service_name" == "gen3_inference" ]; then
      print_header "just migrate_db:" "No db needed for" "$service_name" "service. Nothing to do."
      return
    fi

    just setup_db $service_name

    print_header "just migrate_db:" "migrating db for" "$service_name" "service..."

    if [ ! -f "${dir}/.env" ]; then
      echo "${RED}** WARNING: .env file not found in '${dir}'. Will rely on environment variables. **${RESET}"
    else
      echo "Found .env file: ${dir}/.env - Using it to set up database."
      set -a
        source "${dir}/.env"
      set +a
    fi

    set_postgres_defaults

    # run migrations if they exist
    MIGRATIONS_DIR="${dir}/db/migrations"
    if [ -d "$MIGRATIONS_DIR" ]; then
      # defaults for app user if not set
      if [ "$service_name" == "gen3_embeddings" ]; then
        : "${DB_APP_USER:=app_user}"
        : "${DB_APP_USER_PASSWORD:=app_user_password}"

        DB_APP_USER="${DB_APP_USER}" DB_APP_USER_PASSWORD="${DB_APP_USER_PASSWORD}" \
          dbmate -u "${PGDRIVER:=postgresql}://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=disable" \
          -s "${dir}/db/schema.sql" \
          -d "${MIGRATIONS_DIR}" --wait $ACTION

        DB_APP_USER="${DB_APP_USER}" DB_APP_USER_PASSWORD="${DB_APP_USER_PASSWORD}" \
          dbmate -u "${PGDRIVER:=postgresql}://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=disable" \
          -s "${dir}/db/schema.sql" \
          -d "${MIGRATIONS_DIR}" --wait status
      else
        dbmate -u "${PGDRIVER:=postgresql}://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=disable" \
          -s "${dir}/db/schema.sql" \
        -d "${MIGRATIONS_DIR}" --wait $ACTION

        dbmate -u "${PGDRIVER:=postgresql}://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=disable" \
          -s "${dir}/db/schema.sql" \
        -d "${MIGRATIONS_DIR}" --wait status
      fi

      # Check if failed
      if [ $? -ne 0 ]; then
        echo "${RED}** ERROR: Migration failed! **${RESET}"
        exit 1
      fi

      echo "${GREEN}Migrations applied successfully.${RESET}"
    fi
  }

  if [ "$SERVICE" == "all" ]; then
    for dir in services/*; do
      if [[ -n "${dir#services/}" ]]; then
        run_for_service "${dir#services/}"
      fi
    done
  else
    run_for_service "$SERVICE"
  fi

openapi:
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  # regenerate all service specs and then aggregate
  for dir in services/*; do
    if [ -f "${dir}/generate_openapi.py" ]; then
      echo "Generating OpenAPI spec for ${dir#services/}..."
      cd "${dir}" && uv run python generate_openapi.py && cd - >/dev/null
    fi
  done

  python scripts/merge_openapi.py

  # now build a static HTML using redocly
  npx -y @redocly/cli build-docs docs/autogenerated_openapi.json --output docs/api.html

install $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash

  # this includes some helpers for colored line printing
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    # this forces looping over folders in /services
    # and reduces code duplication by re-calling this recipe
    # with a specific service
    just _install_all
  else
    # print_header COMMAND TEXT SERVICE TEXT
    print_header "just install:" "installing" "$SERVICE" "service..."

    echo "Installing common library into service: ${SERVICE}..."
    cd "./services/$SERVICE"
    uv add "common @ file://../../libraries/common"
    cd -

    echo
    echo "uv sync-ing $SERVICE service with --group dev and --all-extras..."
    uv sync --all-packages --group dev --directory "./services/$SERVICE" --all-extras
  fi

lock $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _lock_all
  else
    print_header "just lock:" "locking" "$SERVICE" "service..."

    uv lock --directory "./services/$SERVICE" --upgrade

    just install "$SERVICE"
  fi

test $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _test_all
  else
    print_header "just test:" "testing" "$SERVICE" "service..."
    cd "./services/$SERVICE"
    uv run pytest -n auto . -vv
    exit_code=$?
    cd -

    overall_exit=$((overall_exit | $exit_code))

    report_error_or_success $overall_exit "just test:" "testing" "$SERVICE" "service!"

    exit $overall_exit
  fi

build $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _build_all
  else
    print_header "just build:" "building" "$SERVICE" "service..."
    docker build -t $SERVICE --build-arg SERVICE_NAME="$SERVICE" -f Dockerfile.k8s .

    report_error_or_success $? "just build:" "building" "$SERVICE" "service!"
  fi

@run $SERVICE: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

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
    -c ../../deployments/k8s/services/${SERVICE}/gunicorn.conf.py \
    --access-logfile - \
    --error-logfile -

@docker_run $SERVICE $EXTERNAL_PORT="8001" $INTERNAL_PORT="4141": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  print_header "just docker_run:" "running" "$SERVICE" "service..."

  docker kill "$SERVICE" 2>/dev/null || true
  docker rm "$SERVICE" 2>/dev/null || true
  docker run --name $SERVICE \
  --env-file "services/$SERVICE/.env" \
  -p {{EXTERNAL_PORT}}:{{INTERNAL_PORT}} \
  $SERVICE:latest

format $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _format_all
  else
    print_header "just format:" "formatting" "$SERVICE" "..."
    uv run --directory $SERVICE ruff format
  fi

venv_reset: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash
  for dir in services/*; do
    print_header "just venv_reset:" "removing" "${dir}" ".venv & uv.lock ..."
    rm -r ${dir}/.venv | true
    rm ${dir}/uv.lock | true
  done
  for dir in libraries/*; do
    print_header "just venv_reset:" "removing" "${dir}" ".venv & uv.lock ..."
    rm -r ${dir}/.venv | true
    rm ${dir}/uv.lock | true
  done

snyk $SERVICE="all": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _snyk_all
  else
    print_header "just snyk:" "snyk scanning" "$SERVICE" "service..."
    if [ $SERVICE == "all" ]; then
      for dir in services/*; do
        print_header "just snyk:" "snyk scanning" "${dir#services/}" "..."
        uv --directory "./services/${dir#services/}" export --format cyclonedx1.5 > sbom_${dir#services/}.json
        snyk sbom test --file sbom_${dir#services/}.json --source-dir="./services/${dir#services/}" --experimental
      done
    else
      # export a requirements file without local imports
      # since the local imports are reflected in the overeall requirements and confuse snyk
      uv --directory "./services/${SERVICE}" export --no-emit-local --format requirements.txt > ${SERVICE}_requirements.txt

      # synk, at the moment, requires pip in an env to actually test things. uv envs don't depend on pip
      # so we need to create a new virtual env.
      # keep an eye on: https://github.com/snyk/snyk-python-plugin/issues/259
      # this is a workaround
      pip install virtualenv
      virtualenv .venv_${SERVICE}
      source .venv_${SERVICE}/bin/activate
      pip install -r ${SERVICE}_requirements.txt

      # snyk test
      snyk test --file=${SERVICE}_requirements.txt --package-manager=pip

      exit_code=$?
      report_error_if_failed $exit_code "just snyk:" "scanning" "$SERVICE" "service!"
      overall_exit=$((overall_exit | $exit_code))

      # cleanup
      deactivate
      rm ${SERVICE}_requirements.txt
      rm -r .venv_${SERVICE}

      exit $overall_exit
    fi
  fi

# `just lint $SERVICE`
lint $SERVICE="all" $EXTRA_ARG="": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  if [ $SERVICE == "all" ]; then
    just _lint_all "$EXTRA_ARG"
  else
    # install services to get ruff and other things required for formatting and linting
    if [[ $SERVICE == *services* ]]; then
      just install ${SERVICE#services/}
    fi

    just format $SERVICE

    print_header "just lint:" "ruff check" "$SERVICE" "..."
    start=$(date +%s.%N)
    uv run --directory "$SERVICE" ruff check ./src --fix $EXTRA_ARG
    exit_code=$?
    end=$(date +%s.%N)

    elapsed_ms=$(awk "BEGIN {printf \"%.0f\", ($end-$start)*1000}")
    echo "ruff check finished in $elapsed_ms ms."

    report_error_if_failed $exit_code "just lint:" "ruff check" "$SERVICE" "!"
    overall_exit=$((overall_exit | $exit_code))
    echo

    report_error_or_success $overall_exit "just lint:" "linting" "$SERVICE" "!"

    exit $overall_exit
  fi

# allow an old version with in-line comment: # allow-old-version
update_versions: _check_dependencies
    #!/usr/bin/env bash
    source scripts/.justfile_helpers.bash

    print_header "just update_versions:" "attempting to update" ".github/workflows/*" "uv & just versions..."

    set -euo pipefail
    UV_LATEST=$(curl -s https://api.github.com/repos/astral-sh/uv/releases/latest | jq -r .tag_name)
    JUST_LATEST=$(curl -s https://api.github.com/repos/casey/just/releases/latest | jq -r .tag_name)
    echo "Latest UV:   $UV_LATEST"
    echo "Latest JUST: $JUST_LATEST"

    # sanity check for semver
    semver_regex='^v?[0-9]+\.[0-9]+\.[0-9]+$'
    if ! [[ $UV_LATEST =~ $semver_regex ]]; then
        print_header
        echo "ERROR: UV tag '$UV_LATEST' does not look like a semantic version" >&2
        exit 1
    fi
    if ! [[ $JUST_LATEST =~ $semver_regex ]]; then
        echo "ERROR: JUST tag '$JUST_LATEST' does not look like a semantic version" >&2
        exit 1
    fi

    # update versions in all CI files in .github/workflows/
    for file in .github/workflows/*.yml; do
        echo "Processing $file..."

        # UV_VERSION
        if grep -E "UV_VERSION:.*#[[:space:]]*allow-old-version" "$file" > /dev/null; then
            echo "--> WARNING: Skipping UV_VERSION in $file (# allow-old-version detected)"
        else
            tmp=$(mktemp)
            sed -E "s/(UV_VERSION:[[:space:]]*')[^']*'/\\1${UV_LATEST}'/g" "$file" > "$tmp" && mv "$tmp" "$file"
        fi

        #JUST_VERSION
        if grep -E "JUST_VERSION:.*#[[:space:]]*allow-old-version" "$file" > /dev/null; then
            echo "--> WARNING: Skipping JUST_VERSION in $file (# allow-old-version detected)"
        else
            tmp=$(mktemp)
            sed -E "s/(JUST_VERSION:[[:space:]]*')[^']*'/\\1${JUST_LATEST}'/g" "$file" > "$tmp" && mv "$tmp" "$file"
        fi
    done

    echo "up to date!"


_install_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just install ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

_lint_all $EXTRA_ARG="": _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in libraries/*; do
    if [[ -n "${dir}" ]]; then
      just lint ${dir} "$EXTRA_ARG"
      overall_exit=$((overall_exit | $?))
    fi
  done

  for dir in services/*; do
    if [[ -n "${dir}" ]]; then
      just lint ${dir} "$EXTRA_ARG"
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just lint:" "linting" "all" "services!"

  just _check_uv_modified_files

  exit $overall_exit

_check_uv_modified_files:
    #!/usr/bin/env bash
    source scripts/.justfile_helpers.bash

    modified=False

    echo "Modified files:"

    are_uv_files_modified() {
        local dir=$1
        local files=("uv.lock" "pyproject.toml")
        local found_modification=False

        for file in "${files[@]}"; do
            if [[ -f "${dir}/${file}" ]]; then
                output=$(git diff --name-only "${dir}/${file}")

                if [[ -n "$output" ]]; then
                    echo "$output"
                    found_modification=True
                fi
            fi
        done
        return $( [[ "$found_modification" == "True" ]] && echo 0 || echo 1 )
    }

    for dir in libraries/* services/*; do
        if [[ -d "$dir" ]]; then
            if are_uv_files_modified "$dir"; then
                modified=True
            fi
        fi
    done

    if [[ "$modified" == "True" ]]; then
        echo
        echo "${RED}** WARNING: Installation of services required locally-modified files (see above), so check them in! Otherwise pre-commit linting checks on remote may FAIL! **${RESET}"
        echo
    else
        echo "No modified files. Linting complete."
        echo
    fi


_format_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in libraries/*; do
    if [[ -n "${dir}" ]]; then
      just lint ${dir}
      overall_exit=$((overall_exit | $?))
    fi
  done

  for dir in services/*; do
    if [[ -n "${dir}" ]]; then
      just lint ${dir}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

_build_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just build ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just build:" "building" "all" "services!"

  exit $overall_exit

_lock_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just lock ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  exit $overall_exit

_test_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just test ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just test:" "testing" "all" "services!"

  exit $overall_exit

_snyk_all: _check_dependencies
  #!/usr/bin/env bash
  source scripts/.justfile_helpers.bash

  for dir in services/*; do
    if [[ -n "${dir#services/}" ]]; then
      just snyk ${dir#services/}
      overall_exit=$((overall_exit | $?))
    fi
  done

  report_error_or_success $overall_exit "just snyk:" "snyk scanning" "all" "services!"

  exit $overall_exit
