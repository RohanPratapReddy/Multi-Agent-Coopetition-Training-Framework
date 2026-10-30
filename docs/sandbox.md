# Sandbox

Peers can run behind an isolation provider. Providers implement `SandboxProvider`
(`mactf/sandbox/base.py`): `probe`, `provision`, `exec`, `teardown`, and `capabilities`.

| Provider | Isolation | `exec` |
|---|---|---|
| `local` | none; a subprocess in a working directory | runs on the host |
| `qemu` | a VM from a qcow2 image, with a copy-on-write overlay | through the guest agent |

`mactf sandbox status` probes what is installed, `provision` boots a sandbox, `launch` runs a
command in it, `clean` removes overlays. Capabilities are reported honestly: asking for hardware
isolation on a host without QEMU/KVM is a `CapabilityUnavailable` (HTTP `503`), never a quiet
fallback to `local`. `GET /sandbox/providers` and `client.sandbox.providers()` list the matrix.

## The guest agent

`mactf/sandbox/guest_agent.py` is stdlib-only and standalone (no `mactf` import), because the
framework is not installed in the VM. It is the server; the host's `SocketChannel` is the client.
The wire format is newline-delimited JSON frames (`kind`, `id`, `ts`, `payload`), identical on both
sides. In QEMU the transport is a `virtio-serial` port named `mactf.ctl`
(`/dev/virtio-ports/mactf.ctl` in the guest).

Behaviour that matters:

1. A list command is argv (no shell); a string is explicitly `/bin/sh -c`.
2. Output is truncated per stream at `max_output_bytes` and the response says so.
3. A timeout kills the process group, not just the child.
4. Paths are confined to `allowed_paths` (default `/app`, `/tmp/mactf`) after `realpath`.
5. Nothing raises out of the loop; a bad request becomes an error frame.
6. The frame `id` is echoed so the host can match responses.

## Image requirements

- Linux with Python 3 and `/bin/sh`.
- `virtio-serial` support in the kernel (standard on cloud images).
- The agent installed as a service. `mactf.sandbox.provision.write_cloud_init(..., with_agent=True)`
  writes a cloud-init seed that installs the file and a `mactf-agent.service` unit; provisioning
  does this automatically when the spec has a `serial_socket`.
- `qemu-img` for overlays and a seed-image tool (`genisoimage`, `mkisofs`, or `xorriso`).

## The 127 case

Without a serial channel the host has no way to reach the guest, so `exec` returns exit code
**127** with a message explaining that no serial channel is configured. That is the accurate
outcome: reporting success, or running the command on the host instead, would let a test pass
against something it never ran in. Configure `serial_socket` on the spec to enable `exec`.

`POST /sandbox/exec` requires the `sandbox:exec` scope; see [api.md](api.md).
