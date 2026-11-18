#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/ctx-compressor
if [[ -f logs/download_musique.pid ]]; then
  read -r musique_pid < logs/download_musique.pid
  while kill -0 "$musique_pid" 2>/dev/null; do
    [[ "$(ps -o stat= -p "$musique_pid")" == Z* ]] && break
    sleep 5
  done
fi
exec /root/anaconda3/bin/python scripts/download_data.py --spec scripts/download_remaining.json
