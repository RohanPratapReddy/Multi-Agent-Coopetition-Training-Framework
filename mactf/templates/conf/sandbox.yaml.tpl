# conf/sandbox.yaml — owns: SANDBOX. QEMU is preferred; the local provider is the
# fallback. Set ALLOW_FALLBACK: false to make hardware isolation mandatory.

SANDBOX:
  # auto | qemu | local. "auto" probes for QEMU and degrades per ALLOW_FALLBACK.
  PROVIDER: auto
  # hardware | process. "hardware" means a VM or container boundary.
  ISOLATION: hardware
  ALLOW_FALLBACK: true
  REQUIRE_HARDWARE: false

  BASE_IMAGE: var/images/base.qcow2
  # {peer} is replaced with the peer name, so each peer gets its own overlay.
  WORK_IMAGE: var/images/peer-{peer}.qcow2
  SERIAL_SOCKET: var/run/mactf-{peer}.sock

  MEMORY_MB: 4096
  VCPUS: 2
  # auto | kvm | hvf | tcg
  ACCEL: auto
  # none | host. A sandbox that needs the network says so here, not in code.
  NETWORK: none
  SNAPSHOT: true

  BOOT_TIMEOUT: 180
  COMMAND_TIMEOUT: 300

  CLOUD_INIT:
    USER: mactf
    PACKAGES:
      - python3
      - python3-pip
