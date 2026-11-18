#!/usr/bin/env bash
set -uo pipefail
cd /root/workplace/ctx-compressor || exit 1
source scripts/reader_environment.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
printf '%s\n' "$$" > logs/reader.pid
date -u +%FT%TZ > logs/reader.started
"$READER_ENV/bin/python" scripts/run_reader.py "$@"
READER_RUN_STATUS=$?
printf '%s\n' "$READER_RUN_STATUS" > logs/reader.exit
date -u +%FT%TZ > logs/reader.finished
exit "$READER_RUN_STATUS"
