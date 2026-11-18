#!/usr/bin/env bash
# Launch through setsid/nohup; this wrapper records PID, status and wall time.
set -uo pipefail
cd /root/workplace/ctx-compressor || exit 1
CTXC_JOB="$1"
shift
[[ "$CTXC_JOB" =~ ^[a-z_]+$ ]] || exit 2
printf '%s\n' "$$" > "logs/$CTXC_JOB.pid"
date -u +%FT%TZ > "logs/$CTXC_JOB.started"
CTXC_START=$SECONDS
bash scripts/run_python.sh "$@"
CTXC_RESULT=$?
printf '%s\n' "$CTXC_RESULT" > "logs/$CTXC_JOB.exit"
printf '%s\n' "$((SECONDS - CTXC_START))" > "logs/$CTXC_JOB.seconds"
date -u +%FT%TZ > "logs/$CTXC_JOB.finished"
exit "$CTXC_RESULT"
