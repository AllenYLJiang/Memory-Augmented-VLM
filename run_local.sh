#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python}" -B run_local.py "$@"
