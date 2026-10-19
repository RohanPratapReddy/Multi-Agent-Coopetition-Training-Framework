#!/usr/bin/env bash
# Install QEMU and verify KVM access for MACTF on Linux.
#
# KVM is the accelerated backend. Without /dev/kvm QEMU falls back to TCG, which
# is still hardware-virtualized isolation but much slower.
set -euo pipefail

echo "== MACTF sandbox installer (Linux) =="
echo "kernel: $(uname -r)"
echo "user  : $(id -un)"

if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

if command -v qemu-system-x86_64 >/dev/null 2>&1; then
    echo "QEMU already installed: $(qemu-system-x86_64 --version | head -n1)"
else
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update
        $SUDO apt-get install -y qemu-system-x86 qemu-utils
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y qemu-kvm qemu-img
    elif command -v pacman >/dev/null 2>&1; then
        $SUDO pacman -S --noconfirm qemu-full
    else
        echo "No supported package manager found (apt/dnf/pacman)." >&2
        exit 1
    fi
fi

if [[ -e /dev/kvm ]]; then
    if [[ -r /dev/kvm && -w /dev/kvm ]]; then
        echo "KVM available and accessible."
    else
        echo "KVM exists but is not accessible to $(id -un)."
        echo "Add the user to the kvm group, then log out and back in:"
        echo "  sudo usermod -aG kvm $(id -un)"
    fi
else
    echo "WARNING: /dev/kvm is missing. Enable virtualization in firmware"
    echo "         (VT-x/AMD-V) and load the kvm module."
fi

echo
echo "Accelerators reported by QEMU:"
qemu-system-x86_64 -accel help || true

cat <<'EOF'

Next steps:
  1. Download a cloud image, for example:
       wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img
       mv jammy-server-cloudimg-amd64.img var/images/base.qcow2
  2. python training/manage.py sandbox status
  3. python training/manage.py sandbox provision --peer swarm
EOF
