<#
.SYNOPSIS
    Provision a copy-on-write overlay disk for one MACTF peer.

.DESCRIPTION
    Creates var\images\peer-<name>.qcow2 backed by the pristine base image, so every
    run starts clean (equivalent to QEMU's -snapshot, but persistent across boots so
    the guest can finish cloud-init).

.PARAMETER Peer
    Peer name. Defaults to 'swarm'.

.PARAMETER Base
    Base qcow2 image. Defaults to var\images\base.qcow2.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sandbox\provision.ps1 -Peer alpha
#>
[CmdletBinding()]
param(
    [string]$Peer = 'swarm',
    [string]$Base = 'var\images\base.qcow2',
    [int]$SizeGB = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not (Get-Command 'qemu-img' -ErrorAction SilentlyContinue)) {
    throw 'qemu-img not found. Run scripts\sandbox\install_qemu.ps1 first.'
}
if (-not (Test-Path $Base)) {
    throw "Base image not found: $Base. Download an Ubuntu cloud image and save it there."
}

$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$imageDir = Join-Path $root 'var\images'
New-Item -ItemType Directory -Force -Path $imageDir | Out-Null
$work = Join-Path $imageDir "peer-$Peer.qcow2"

if (Test-Path $work) {
    $workTime = (Get-Item $work).LastWriteTimeUtc
    $baseTime = (Get-Item $Base).LastWriteTimeUtc
    if ($workTime -ge $baseTime) {
        Write-Host "Overlay already current: $work" -ForegroundColor Green
        return
    }
    Write-Host "Overlay is older than base; rebuilding: $work" -ForegroundColor Yellow
    Remove-Item $work -Force
}

Write-Host "Creating overlay $work from $Base" -ForegroundColor Cyan
& qemu-img create -f qcow2 -F qcow2 -b (Resolve-Path $Base) $work
if ($LASTEXITCODE -ne 0) { throw "qemu-img create failed with code $LASTEXITCODE" }

if ($SizeGB -gt 0) {
    Write-Host "Resizing overlay to ${SizeGB}G" -ForegroundColor Cyan
    & qemu-img resize $work "${SizeGB}G"
    if ($LASTEXITCODE -ne 0) { throw "qemu-img resize failed with code $LASTEXITCODE" }
}

& qemu-img info $work
Write-Host "Provisioned $work" -ForegroundColor Green
