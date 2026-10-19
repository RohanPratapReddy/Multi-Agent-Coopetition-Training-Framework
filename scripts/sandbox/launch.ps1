<#
.SYNOPSIS
    Launch an air-gapped QEMU guest for one MACTF peer.

.DESCRIPTION
    Mirrors the docx launch script with fail-closed checks:
      -net none           no network device at all (framework.md P20/P21)
      -snapshot           discard guest writes
      -nographic          no host display channel
      virtio-serial       host<->guest control channel
      -fw_cfg             cloud-init seed without genisoimage

.PARAMETER Peer
    Peer name. Defaults to 'swarm'.

.PARAMETER Accel
    auto | whpx | kvm | tcg. 'auto' probes and picks the strongest available.

.PARAMETER MemoryMB
    Guest memory in MiB.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sandbox\launch.ps1 -Peer alpha -MemoryMB 2048
#>
[CmdletBinding()]
param(
    [string]$Peer = 'swarm',
    [ValidateSet('auto', 'whpx', 'kvm', 'tcg')]
    [string]$Accel = 'auto',
    [int]$MemoryMB = 4096,
    [int]$VCPUs = 2,
    [switch]$NoSnapshot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name not found. Run scripts\sandbox\install_qemu.ps1 first."
    }
}

Assert-Command 'qemu-system-x86_64'

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$work = Join-Path $root "var\images\peer-$Peer.qcow2"
if (-not (Test-Path $work)) {
    throw "Overlay disk missing: $work. Run scripts\sandbox\provision.ps1 -Peer $Peer first."
}

$socketDir = Join-Path $root 'var\run'
New-Item -ItemType Directory -Force -Path $socketDir | Out-Null
$socketPath = Join-Path $socketDir "mactf-$Peer.sock"

if ($Accel -eq 'auto') {
    $available = (& qemu-system-x86_64 -accel help) -join ' '
    if ($available -match 'whpx') { $Accel = 'whpx' }
    elseif ($available -match 'kvm') { $Accel = 'kvm' }
    else { $Accel = 'tcg' }
    Write-Host "Selected accelerator: $Accel" -ForegroundColor Cyan
}

$cpuMode = if ($Accel -eq 'whpx') { 'max' } else { 'host' }

$arguments = @(
    '-accel', $Accel,
    '-machine', 'q35',
    '-cpu', $cpuMode,
    '-m', "$MemoryMB",
    '-smp', "$VCPUs",
    '-drive', "file=$work,format=qcow2,if=virtio",
    '-net', 'none',
    '-nographic',
    '-no-reboot',
    '-chardev', "socket,id=mactf,path=$socketPath,server=on,wait=off",
    '-device', 'virtio-serial-pci,id=virtio-serial0',
    '-device', 'virtserialport,chardev=mactf,name=mactf.ctl,bus=virtio-serial0.0',
    '-fw_cfg', "name=opt/org.cloud-init/seed-dir,string=$root\var\cloudinit-$Peer"
)
if (-not $NoSnapshot) { $arguments += '-snapshot' }

Write-Host "Launching peer '$Peer' (accel=$Accel, mem=${MemoryMB}MB)" -ForegroundColor Cyan
Write-Host "  disk  : $work"
Write-Host "  serial: $socketPath"
Write-Host '  net   : none (air-gapped)'
& qemu-system-x86_64 @arguments
if ($LASTEXITCODE -ne 0) { throw "QEMU exited with code $LASTEXITCODE" }
