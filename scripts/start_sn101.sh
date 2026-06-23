#!/usr/bin/env bash
# Start SN101 Tag101 miner under PM2.
#
# Usage:
#   ./start_sn101.sh <coldkey> <hotkey>
#   ./start_sn101.sh sun sun1
#
# Optional env overrides (from miner.pm2.env or shell):
#   MINER_NAME=sn101-miner
#   ENV_FILE=~/tag101/miner.pm2.env
set -euo pipefail

export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"

REPO="${REPO:-$HOME/tag101}"
ENV_FILE="${ENV_FILE:-$REPO/miner.pm2.env}"
NAME="${MINER_NAME:-sn101-miner}"

usage() {
  echo "Usage: $0 <coldkey> <hotkey>" >&2
  echo "Example: $0 sun sun1" >&2
  exit 1
}

COLDKEY="${1:-${WALLET_NAME:-}}"
HOTKEY="${2:-${WALLET_HOTKEY:-}}"
if [[ -z "$COLDKEY" || -z "$HOTKEY" ]]; then
  usage
fi

cd "$REPO"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

# pm2_node parses ENV_FILE via load_env_file(); do not `source` it here because
# unquoted NODE_ARGS (e.g. --netuid 101 ...) would make bash try to run "101".
WALLET_NAME="$COLDKEY"
WALLET_HOTKEY="$HOTKEY"

echo "Wallet: $WALLET_NAME / $WALLET_HOTKEY"
echo "Checking registration on netuid 101..."
.venv/bin/python << PY
import bittensor as bt
sub = bt.Subtensor(network="finney")
w = bt.Wallet(name="${WALLET_NAME}", hotkey="${WALLET_HOTKEY}")
if not sub.is_hotkey_registered(netuid=101, hotkey_ss58=w.hotkey.ss58_address):
    raise SystemExit(
        "Hotkey ${WALLET_NAME}/${WALLET_HOTKEY} is NOT registered on SN101 yet. "
        "Register first, then re-run this script."
    )
print("Registered OK:", w.hotkey.ss58_address)
PY

PYTHON="${REPO}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

"$PYTHON" -m tag101.deploy.pm2_node start \
  --role miner \
  --name "$NAME" \
  --env-file "$ENV_FILE" \
  -- \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY"

pm2 save
echo
echo "Miner started as '$NAME' (PM2 auto-restart + health monitor enabled)"
echo "  pm2 status"
echo "  pm2 logs $NAME"
echo "  pm2 logs ${NAME}-auto-update   # watchdog restarts miner if it stops"
STATE_HOST_DIR="$(grep -E '^STATE_HOST_DIR=' "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)"
echo "Logs dir: ${STATE_HOST_DIR:-$HOME/node-state/tag101-miner}"
