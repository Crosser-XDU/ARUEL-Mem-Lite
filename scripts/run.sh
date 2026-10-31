#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/run.py "$@"
