#!/usr/bin/env bash
# Launch an air-gapped QEMU guest for one MACTF peer.
#
# Hardening applied versus the design document:
#   -net none     no network device at all (framework.md P20, P21)
#   -snapshot     guest writes discarded
#   -nographic    no host display channel
#   set -euo pipefail, and every required path is asserted before launch
#
# Usage: ./launch_swarm.sh [peer] [memory-mb] [vcpus]
set -euo pipefail

PEER="${1:-swarm}"
MEMORY="${2:-4096}"
VCPUS="${3:-2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$ROOT/var/images/peer-$PEER.qcow2"
SOCKET="$ROOT/var/run/mactf-$PEER.sock"
SEED="$ROOT/var/cloudinit-$PEER"

command -v qemu-system-x86_64 >/dev/null 2>&1 || {
    echo "qemu-system-x86_64 not found. Run install_qemu.sh first." >&2
    exit 1
}
[[ -f "$WORK" ]] || {
    echo "Overlay missing: $WORK (run provision.sh $PEER)" >&2
    exit 1
}
[[ -f "$SEED/user-data" ]] || {
    echo "Cloud-init seed missing: $SEED/user-data" >&2
    exit 1
}

mkdir -p "$ROOT/var/run"

if [[ -r /dev/kvm && -w /dev/kvm ]]; then
    ACCEL="kvm"
elif qemu-system-x86_64 -accel help 2>&1 | grep -q whpx; then
    ACCEL="whpx"
else
    ACCEL="tcg"
    echo "WARNING: no hardware accelerator available; using TCG (slow)." >&2
fi
CPU="$([[ "$ACCEL" == "whpx" ]] && echo max || echo host)"

echo "Launching peer '$PEER' (accel=$ACCEL, mem=${MEMORY}MB, vcpus=$VCPUS)"
echo "  disk  : $WORK"
echo "  serial: $SOCKET"
echo "  net   : none (air-gapped)"

exec qemu-system-x86_64 \
    -accel "$ACCEL" \
    -machine q35 \
    -cpu "$CPU" \
    -m "$MEMORY" \
    -smp "$VCPUS" \
    -drive "file=$WORK,format=qcow2,if=virtio" \
    -net none \
    -nographic \
    -no-reboot \
    -snapshot \
    -chardev "socket,id=mactf,path=$SOCKET,server=on,wait=off" \
    -device virtio-serial-pci,id=virtio-serial0 \
    -device virtserialport,chardev=mactf,name=mactf.ctl,bus=virtio-serial0.0 \
    -fw_cfg "name=opt/org.cloud-init/seed-dir,string=$SEED"
