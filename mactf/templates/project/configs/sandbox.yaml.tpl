# Sandbox overlays. QEMU is preferred; the local provider is the fallback.
# Set ALLOW_FALLBACK: false to make hardware isolation mandatory.

SANDBOX:
  PROVIDER: auto
  ISOLATION: hardware
  ALLOW_FALLBACK: true
  REQUIRE_HARDWARE: false
  MEMORY_MB: 4096
  VCPUS: 2
  ACCEL: auto
  NETWORK: none
  SNAPSHOT: true
  BOOT_TIMEOUT: 180
  COMMAND_TIMEOUT: 300
  BASE_IMAGE: var/images/base.qcow2
  WORK_IMAGE: var/images/peer-{peer}.qcow2
  SERIAL_SOCKET: var/run/mactf-{peer}.sock
