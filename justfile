# This justfile contains automation for development and CI/CD.
# It support parallel execution of many of the commands at the expense of
# readability. It is mostly bash and there are some helpers in the scripts/
# folder.

set dotenv-load

SERVICES := "gen3_embeddings gen3_inference gen3_ai_model_repo"
LIBRARIES := "common"
PARALLEL := "true"

# Services type-checked by `just typecheck` and `just lint`. gen3_inference is omitted
# because it does not typecheck clean. Run `just typecheck gen3_inference` to see what is
# outstanding.
# TODO / FIXME: add it back here once it passes.
TYPECHECK_SERVICES := "gen3_embeddings gen3_ai_model_repo"

# Pinned to match avto-dev/markdown-lint@v1.5.0, which CI uses.
MARKDOWNLINT_VERSION := "0.26.0"

# List all commands
default:
    #!/usr/bin/env bash
    echo "Description: just recipes for management of Gen3 AI monorepo. See the README.md for concise setup instructions."
    just --list

# Run any command sequentially (Usage: just s {{command}})
s +COMMAND:
    @just PARALLEL=false {{COMMAND}}

# Docker build service(s)
[group('basic')]
build SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just build {}
            just _warn
        else
            for service in {{SERVICES}}; do just build "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        print_header "just build:" "building" "{{SERVICE}}" "service..."
        docker build -t "{{SERVICE}}" --build-arg SERVICE_NAME="{{SERVICE}}" -f Dockerfile.k8s .
    fi

# Install dependencies for service(s)
[group('basic')]
install SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just install {}
            just _warn
        else
            for service in {{SERVICES}}; do just install "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/services/{{SERVICE}}"; fi
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        print_header "just install:" "installing" "$TARGET" "..."
        cd "$TARGET"
        echo "uv sync-ing {{SERVICE}} service..."
        uv sync --all-packages --group dev --all-extras
    fi

# Lock deps, attempting to pull new ones
[group('basic')]
lock SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just lock {}
            just _warn
        else
            for service in {{SERVICES}}; do just lock "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        print_header "just lock:" "locking" "$TARGET" "..."
        uv lock --directory "$TARGET" --upgrade
        just install "{{SERVICE}}"
    fi

# Let you know what tools are missing and try to install
[group('basic')]
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash

    print_header "just setup:" "verifying" "uv" "installation..."
    if command -v uv >/dev/null 2>&1; then
        echo "uv is installed. version: $(uv --version)"
    else
        echo -e "${YELLOW}** WARNING: uv not found in \$PATH. Installing... **${RESET}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    print_header "just setup:" "verifying" "PostgreSQL client (psql)" "installation..."
    if command -v psql >/dev/null 2>&1; then
        echo "psql is installed. version: $(psql --version)"
    else
        echo -e "${RED}** ERROR: psql not found. Please install PostgreSQL. **${RESET}"
        exit 1
    fi

    print_header "just setup:" "verifying" "dbmate" "installation..."
    if command -v dbmate >/dev/null 2>&1; then
        echo "dbmate is installed. version: $(dbmate --version)"
    else
        echo -e "${RED}** ERROR: dbmate not found. See: https://github.com/amacneil/dbmate#installation **${RESET}"
        exit 1
    fi

    print_header "just setup:" "verifying" "pre-commit" "installation..."
    if command -v pre-commit >/dev/null 2>&1; then
        echo "pre-commit is installed. version: $(pre-commit --version)"
    else
        echo -e "${YELLOW}** WARNING: pre-commit not found. Installing... **${RESET}"
        pip install pre-commit
    fi

    hook_path="$(git rev-parse --git-path hooks/pre-commit)"
    if [[ ! -f "$hook_path" ]] || ! grep -q 'pre-commit' "$hook_path"; then
        echo -e "${YELLOW}** WARNING: pre-commit git hook not found or incomplete. Installing... **${RESET}"
        pre-commit install --overwrite
    fi
    echo "pre-commit git hook is installed."

# Detect vulnerabilities in service(s) dependencies
[group('basic')]
snyk SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just snyk {}
            just _warn
        else
            for svc in {{SERVICES}}; do just snyk "$svc"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        print_header "just snyk:" "scanning" "{{SERVICE}}" "service..."

        # export a requirements file without local imports
        # since the local imports are reflected in the overall requirements and confuse snyk
        uv --directory "$TARGET" export --no-emit-local --format requirements.txt > "{{SERVICE}}_requirements.txt"

        # snyk, at the moment, requires pip in an env to actually test things. uv envs don't depend on pip
        # so we need to create a new virtual env.
        # keep an eye on: https://github.com/snyk/snyk-python-plugin/issues/259
        # this is a workaround
        pip install virtualenv >/dev/null 2>&1
        virtualenv ".venv_{{SERVICE}}" >/dev/null 2>&1
        source ".venv_{{SERVICE}}/bin/activate"
        pip install -r "{{SERVICE}}_requirements.txt" >/dev/null 2>&1

        # snyk test
        set +e  # temporarily disable fail-fast so we can capture the exit code
        snyk test --file="{{SERVICE}}_requirements.txt" --package-manager=pip
        exit_code=$?
        set -e  # turn fail-fast back on

        # cleanup
        deactivate
        rm "{{SERVICE}}_requirements.txt"
        rm -rf ".venv_{{SERVICE}}"

        # handle failure reporting
        if [ $exit_code -ne 0 ]; then
            echo -e "${RED}** ERROR: just snyk failed for {{SERVICE}}! **${RESET}"
            exit $exit_code
        fi
    fi

# Run unit tests for service(s)
[group('basic')]
test SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just test {}
            just _warn
        else
            for service in {{SERVICES}}; do just test "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        print_header "just test:" "testing" "$TARGET" "..."
        cd "$TARGET" && uv run pytest --color=yes -n auto . -vv
    fi

# Load empty database with latest version of schema
[group('database')]
db_load SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just _run_dbmate {} load
            just _warn
        else
            for service in {{SERVICES}}; do just _run_dbmate "$service" load; done
        fi
    else
        just _run_dbmate "{{SERVICE}}" load
    fi

# Migrate existing database up using official migrations
[group('database')]
db_migrate SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just _run_dbmate {} migrate
            just _warn
        else
            for service in {{SERVICES}}; do just _run_dbmate "$service" migrate; done
        fi
    else
        just _run_dbmate "{{SERVICE}}" migrate
    fi

# Create a new migration file for schema changes
[group('database')]
db_new_migration SERVICE MIGRATION_NAME: _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    RAW_NAME="{{MIGRATION_NAME}}"
    MIGRATION_NAME_NO_SPACES="${RAW_NAME// /_}"
    just _run_dbmate "{{SERVICE}}" new "$MIGRATION_NAME_NO_SPACES"
    source scripts/new_migration.bash "./services/{{SERVICE}}/db/migrations"

# Rollback last migrationin existing database (relies on downgrade path)
[group('database')]
db_rollback SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just _run_dbmate {} down
            just _warn
        else
            for service in {{SERVICES}}; do just _run_dbmate "$service" down; done
        fi
    else
        just _run_dbmate "{{SERVICE}}" down
    fi

# Create databases based on configuration - does NOT migrate
[group('database')]
db_setup SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just db_setup {}
            just _warn
        else
            for service in {{SERVICES}}; do just db_setup "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        if [ "{{SERVICE}}" = "gen3_inference" ]; then
            print_header "just db_setup:" "No db needed for" "{{SERVICE}}" "service. Skipping."
            exit 0
        fi

        DIR="services/{{SERVICE}}"
        if [ ! -d "$DIR" ]; then
            echo -e "${RED}** ERROR: Directory '$DIR' not found **${RESET}"
            exit 1
        fi

        print_header "just db_setup:" "setting up db for" "{{SERVICE}}" "service..."
        if [ -f "$DIR/.env" ]; then
            set -a
            source "$DIR/.env"
            set +a
        else
            echo -e "${YELLOW}** WARNING: No .env found **${RESET}"
        fi

        service_name="{{SERVICE}}"
        set_postgres_defaults
        psql -d postgres -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -c "CREATE DATABASE \"${PGDATABASE}\" WITH OWNER \"${PGUSER}\";" 2>/dev/null || echo "Database exists."
    fi

# Generate OpenAPI specification and build API docs (`just openapi true` opens the result)
[group('extra helpers')]
openapi OPEN="false":
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
        echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} bash -c 'cd services/{} && [ -f generate_openapi.py ] && uv run python generate_openapi.py || true'
        just _warn
    else
        for service in {{SERVICES}}; do
            (cd "services/$service" && [ -f generate_openapi.py ] && uv run python generate_openapi.py || true)
        done
    fi
    python scripts/merge_openapi.py
    npx -y @redocly/cli build-docs docs/autogenerated_openapi.json --output docs/api.html

    SHOULD_OPEN="$(printf '%s' "{{OPEN}}" | tr '[:upper:]' '[:lower:]')"
    if [ "$SHOULD_OPEN" = "true" ]; then
        print_header "just openapi:" "opening" "docs/api.html" "..."
        # Try to open. Not guaranteed, so warn instead of failing.
        if command -v open >/dev/null 2>&1; then
            open docs/api.html
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open docs/api.html
        else
            echo -e "${YELLOW}** WARNING: no 'open'/'xdg-open' found. Open docs/api.html manually. **${RESET}"
        fi
    fi

# Update versions of tools used in CI
[group('extra helpers')]
update_versions: _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    print_header "just update_versions:" "updating" "CI versions" "..."

    UV_LATEST=$(curl -s https://api.github.com/repos/astral-sh/uv/releases/latest | jq -r .tag_name)
    JUST_LATEST=$(curl -s https://api.github.com/repos/casey/just/releases/latest | jq -r .tag_name)

    if [[ ! $UV_LATEST =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then echo -e "${RED}ERROR: Invalid UV tag${RESET}"; exit 1; fi
    if [[ ! $JUST_LATEST =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then echo -e "${RED}ERROR: Invalid JUST tag${RESET}"; exit 1; fi

    for file in .github/workflows/*.yml; do
        if grep -E "UV_VERSION:.*#[[:space:]]*allow-old-version" "$file" > /dev/null; then
            echo "Skipping UV in $file"
        else
            sed -i.bak -E "s/(UV_VERSION:[[:space:]]*')[^']*'/\\1${UV_LATEST}'/g" "$file"
        fi

        if grep -E "JUST_VERSION:.*#[[:space:]]*allow-old-version" "$file" > /dev/null; then
            echo "Skipping JUST in $file"
        else
            sed -i.bak -E "s/(JUST_VERSION:[[:space:]]*')[^']*'/\\1${JUST_LATEST}'/g" "$file"
        fi
        rm -f "$file.bak"
    done
    echo "Up to date!"

# Delete all .venv and .lock files (irreversible)
[group('extra helpers')]
venv_reset: _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    print_header "just venv_reset:" "Purging" "all .venv and uv.lock files" "..."
    rm -rf services/*/.venv services/*/uv.lock libraries/*/.venv libraries/*/uv.lock 2>/dev/null || true

# Run formatting on service(s)
[group('linting')]
format SERVICE="all": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{LIBRARIES}}" | tr ' ' '\n' | xargs -P 0 -I {} just format libraries/{}
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just format {}
            just _warn
        else
            for lib in {{LIBRARIES}}; do just format "libraries/$lib"; done
            for service in {{SERVICES}}; do just format "$service"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi
        print_header "just format:" "formatting" "$TARGET" "..."
        uv run --directory "$TARGET" ruff format
    fi

# Lint service(s)
[group('linting')]
lint SERVICE="all" EXTRA_ARG="": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{LIBRARIES}}" | tr ' ' '\n' | xargs -P 0 -I {} just lint libraries/{} "{{EXTRA_ARG}}"
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just lint {} "{{EXTRA_ARG}}"
        else
            for lib in {{LIBRARIES}}; do just lint "libraries/$lib" "{{EXTRA_ARG}}"; done
            for service in {{SERVICES}}; do just lint "$service" "{{EXTRA_ARG}}"; done
        fi
        # capture rather than let `set -e` abort: the notices below are most
        # useful precisely when a lint failed, so they must still print
        MARKDOWN_STATUS=0
        just markdown_lint || MARKDOWN_STATUS=$?
        just _check_uv_modified_files

        just PARALLEL="{{PARALLEL}}" _warn
        exit "$MARKDOWN_STATUS"
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        if [[ "$TARGET" == *services* ]]; then just install "{{SERVICE}}"; fi
        just format "{{SERVICE}}"

        print_header "just lint:" "ruff check" "$TARGET" "..."
        uv run --directory "$TARGET" ruff check ./src --fix {{EXTRA_ARG}}

        just sql_lint "{{SERVICE}}"

        # `lint` recurses per-service, so this branch would type-check every service by
        # explicit name. Only run it for the targets we've opted in.
        if [[ " {{TYPECHECK_SERVICES}} {{LIBRARIES}} " == *" $(basename "$TARGET") "* ]]; then
            just typecheck "{{SERVICE}}"
        fi
    fi

# Type check service(s)
[group('linting')]
typecheck SERVICE="all" EXTRA_ARG="": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{LIBRARIES}}" | tr ' ' '\n' | xargs -P 0 -I {} just typecheck libraries/{} "{{EXTRA_ARG}}"
            echo "{{TYPECHECK_SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just typecheck {} "{{EXTRA_ARG}}"
            just _warn
        else
            for lib in {{LIBRARIES}}; do just typecheck "libraries/$lib" "{{EXTRA_ARG}}"; done
            for service in {{TYPECHECK_SERVICES}}; do just typecheck "$service" "{{EXTRA_ARG}}"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        if [[ "$TARGET" == *services* ]]; then just install "{{SERVICE}}"; fi

        print_header "just typecheck:" "ty check" "$TARGET" "..."
        uv run --directory "$TARGET" ty check ./src {{EXTRA_ARG}}
    fi

# Lint .sql files
[group('linting')]
sql_lint SERVICE="all" EXTRA_ARG="": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{SERVICE}}" = "all" ]; then
        if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then
            echo "{{SERVICES}}" | tr ' ' '\n' | xargs -P 0 -I {} just sql_lint {} "{{EXTRA_ARG}}"
            just _warn
        else
            for service in {{SERVICES}}; do just sql_lint "$service" "{{EXTRA_ARG}}"; done
        fi
    else
        source scripts/.justfile_helpers.bash
        TARGET="{{SERVICE}}"
        if [ ! -d "$TARGET" ] && [ -d "services/{{SERVICE}}" ]; then TARGET="services/{{SERVICE}}"; fi

        if [ -d "$TARGET/db/migrations" ]; then
            if [[ "$TARGET" == *services* ]]; then just install "{{SERVICE}}"; fi

            print_header "just sql_lint:" "sqlfluff fix" "$TARGET" "..."
            uv run --directory "$TARGET" sqlfluff fix "./db/migrations/" --dialect postgres || true
            echo

            print_header "just sql_lint:" "sqlfluff lint" "$TARGET" "..."
            echo "!!! This is ALLOWED to fail for now. Read the failures and assess case-by-case. !!!"
            uv run --directory "$TARGET" sqlfluff lint "./db/migrations/" --dialect postgres || true
        fi
    fi

# Lint .md files (repo-wide)
[group('linting')]
markdown_lint EXTRA_ARG="": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash

    # markdownlint is a node tool and node is not a required dependency of this
    # repo, so skip rather than hard-fail. CI still enforces this either way.
    if ! command -v npx >/dev/null 2>&1; then
        echo -e "${YELLOW}** WARNING: node/npx not found, SKIPPING markdown lint. **${RESET}"
        echo -e "${YELLOW}** CI will still check this. Install node to catch it locally. **${RESET}"
        exit 0
    fi

    print_header "just markdown_lint:" "markdownlint --fix" "**/*.md" "..."
    # Note on --ignore:
    #   testfiles/ holds vendored Hugging Face model cards used as test fixtures,
    #   not documentation we author.
    npx --yes "markdownlint-cli@{{MARKDOWNLINT_VERSION}}" \
        --config .github/workflows/validate_markdown_lint.jsonc \
        --ignore-path .gitignore \
        --ignore '**/services/gen3_ai_model_repo/src/gen3_ai_model_repo/routes/testfiles/**' \
        --fix {{EXTRA_ARG}} -- '**/*.md'

# Run a service in docker
[group('run')]
@docker_run SERVICE EXTERNAL_PORT="8001" INTERNAL_PORT="4141": _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    print_header "just docker_run:" "running" "{{SERVICE}}" "service..."
    docker kill "{{SERVICE}}" 2>/dev/null || true
    docker rm "{{SERVICE}}" 2>/dev/null || true
    docker run --name "{{SERVICE}}" --env-file "services/{{SERVICE}}/.env" -p {{EXTERNAL_PORT}}:{{INTERNAL_PORT}} "{{SERVICE}}:latest"

# Run a service using gunicorn
[group('run')]
@run SERVICE: _check_dependencies
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    print_header "just run:" "running" "{{SERVICE}}" "service..."
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=1
    uv run --directory "./services/{{SERVICE}}" opentelemetry-instrument gunicorn {{SERVICE}}.main:app_instance -k uvicorn.workers.UvicornWorker -c ../../deployments/k8s/services/{{SERVICE}}/gunicorn.conf.py --access-logfile - --error-logfile -

_warn:
    @if [[ "{{PARALLEL}}" = "true" && "${GITHUB_ACTIONS:-}" != "true" ]]; then \
        echo -e "\n\033[33mNote: just PARALLEL=\"true\" so above logs may be jumbled. Run again with \`just s <command>\` to run sequentially with colorization.\033[0m"; \
    fi

_check_dependencies:
    @./scripts/check_dependencies.bash

_run_dbmate SERVICE ACTION ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    DIR="services/{{SERVICE}}"

    if [ "{{SERVICE}}" = "gen3_inference" ]; then exit 0; fi
    if [ ! -d "$DIR" ]; then
        echo -e "${RED}** ERROR: '$DIR' does not exist **${RESET}"
        exit 1
    fi

    just db_setup "{{SERVICE}}"
    print_header "just dbmate:" "running migrations for" "{{SERVICE}}" "service..."

    if [ -f "$DIR/.env" ]; then
        set -a
        source "$DIR/.env"
        set +a
    fi

    service_name="{{SERVICE}}"
    set_postgres_defaults

    MIGRATIONS_DIR="${DIR}/db/migrations"
    if [ -d "$MIGRATIONS_DIR" ]; then
        export PGPASSWORD="${PGPASSWORD}"
        CONN_STR="${PGDRIVER:=postgresql}://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=disable"

        if [ "{{SERVICE}}" = "gen3_embeddings" ]; then
            export DB_APP_USER="${DB_APP_USER:=app_user}"
            export DB_APP_USER_PASSWORD="${DB_APP_USER_PASSWORD:=app_user_password}"
        fi

        dbmate -u "$CONN_STR" -s "${DIR}/db/schema.sql" -d "${MIGRATIONS_DIR}" --wait {{ACTION}} {{ARGS}}
        dbmate -u "$CONN_STR" -s "${DIR}/db/schema.sql" -d "${MIGRATIONS_DIR}" --wait status
        echo -e "${GREEN}Migrations applied successfully.${RESET}"
    fi

_check_uv_modified_files:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/.justfile_helpers.bash
    echo
    echo "Modified files:"
    MODIFIED=0
    for file in $(git diff --name-only | grep -E 'uv\.lock|pyproject\.toml' || true); do
        echo "$file"
        MODIFIED=1
    done
    if [ "$MODIFIED" -eq 1 ]; then
        echo -e "\n${RED}** WARNING: Local uv files modified! Check them in! **${RESET}\n"
    fi
