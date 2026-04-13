#!/usr/bin/env bash
set -euo pipefail

missing=()
issues=()

for command in docker; do
  if ! command -v "$command" >/dev/null 2>&1; then
    missing+=("$command")
  fi
done

if ((${#missing[@]} > 0)); then
  printf 'Missing required commands: %s\n' "${missing[*]}" >&2
  printf 'See README.md for installation instructions.\n' >&2
  exit 1
fi

if [[ ! -f .pre-commit-config.yaml ]]; then
  issues+=("Missing .pre-commit-config.yaml")
fi

if [[ ! -f .secrets.baseline ]]; then
  issues+=("Missing .secrets.baseline")
fi

if ! grep -q 'id: detect-secrets' .pre-commit-config.yaml; then
  issues+=("detect-secrets hook is not configured in .pre-commit-config.yaml")
fi

if ((${#issues[@]} > 0)); then
  printf '%s\n' "${issues[@]}" >&2
  exit 1
fi
