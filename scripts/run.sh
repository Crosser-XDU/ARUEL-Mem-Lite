#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python3}" -m uvicorn fusion_memory.app:create_app --factory \
  --host "${MEMORY_HOST:-127.0.0.1}" --port "${MEMORY_PORT:-18090}" --workers 1 --no-access-log
