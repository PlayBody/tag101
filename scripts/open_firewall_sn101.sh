#!/usr/bin/env bash
# Open axon port for SN101 miner (8092 by default).
set -euo pipefail

PORT="${AXON_PORT:-8092}"

if command -v ufw >/dev/null 2>&1; then
  echo "gotosky!!!" | sudo -S ufw allow "${PORT}/tcp" 2>/dev/null || sudo ufw allow "${PORT}/tcp"
  echo "gotosky!!!" | sudo -S ufw status | grep "$PORT" || true
else
  echo "ufw not installed; ensure port $PORT/tcp is open in your provider firewall."
fi
