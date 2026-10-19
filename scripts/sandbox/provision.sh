#!/usr/bin/env bash
# Provision a copy-on-write overlay disk for one MACTF peer.
#
# Usage: ./provision.sh [peer] [base-image]
set -euo pipefail

PEER="${1:-swarm}"
BASE="${2:-var/images/base.qcow2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$ROOT/var/images/peer-$PEER.qcow2"

command -v qemu-img >/dev/null 2>&1 || {
    echo "qemu-img not found. Run install_qemu.sh first." >&2
    exit 1
}
[[ -f "$ROOT/$BASE" || -f "$BASE" ]] || {
    echo "Base image not found: $BASE" >&2
    exit 1
}
BASE_PATH="$([[ -f "$BASE" ]] && echo "$BASE" || echo "$ROOT/$BASE")"

mkdir -p "$ROOT/var/images"

if [[ -f "$WORK" && "$WORK" -nt "$BASE_PATH" ]]; then
    echo "Overlay already current: $WORK"
    exit 0
fi
[[ -f "$WORK" ]] && rm -f "$WORK"

echo "Creating overlay $WORK from $BASE_PATH"
qemu-img create -f qcow2 -F qcow2 -b "$BASE_PATH" "$WORK"
qemu-img info "$WORK"
echo "Provisioned $WORK"
