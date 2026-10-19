<#
.SYNOPSIS
    Install QEMU and (optionally) enable Windows Hypervisor Platform for MACTF.

.DESCRIPTION
    QEMU provides the hardware-isolated guest that MACTF uses as a peer sandbox.
    WHPX is the accelerated backend on Windows and requires an elevated shell plus
    a reboot. This script is idempotent: it checks before installing.

.PARAMETER SkipWHPX
    Skip the HypervisorPlatform feature. QEMU will fall back to TCG (software
    emulation), which is slow but still isolated.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sandbox\install_qemu.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipWHPX
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host '== MACTF sandbox installer ==' -ForegroundColor Cyan
Write-Host "Platform: $([System.Environment]::OSVersion.VersionString)"

# 1. QEMU ---------------------------------------------------------------------
if (Test-Command 'qemu-system-x86_64') {
    $version = (& qemu-system-x86_64 --version | Select-Object -First 1)
    Write-Host "QEMU already installed: $version" -ForegroundColor Green
}
else {
    if (-not (Test-Command 'winget')) {
        throw 'winget is unavailable. Install App Installer from the Microsoft Store, or download QEMU from https://qemu.weilnetz.de/'
    }
    Write-Host 'Installing QEMU via winget...' -ForegroundColor Yellow
    & winget install -e --id SoftwareFreedomConservancy.QEMU `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget exited with code $LASTEXITCODE"
    }
    Write-Host 'QEMU installed. You may need to open a new shell for PATH changes.' -ForegroundColor Green
}

# 2. WHPX ---------------------------------------------------------------------
if (-not $SkipWHPX) {
    if (-not (Test-Admin)) {
        Write-Host 'WHPX requires an elevated shell. Run this script as Administrator,' -ForegroundColor Yellow
        Write-Host 'or enable it manually with:' -ForegroundColor Yellow
        Write-Host '  dism /online /enable-feature /featurename:HypervisorPlatform /all' -ForegroundColor Yellow
    }
    else {
        Write-Host 'Enabling Windows Hypervisor Platform (requires a reboot)...' -ForegroundColor Yellow
        & dism /online /enable-feature /featurename:HypervisorPlatform /all /norestart
        if ($LASTEXITCODE -notin @(0, 3010)) {
            throw "dism exited with code $LASTEXITCODE"
        }
        Write-Host 'Enabled. Reboot before using -accel whpx.' -ForegroundColor Green
    }
}

# 3. Verification -------------------------------------------------------------
Write-Host ''
Write-Host 'Accelerators reported by QEMU:' -ForegroundColor Cyan
& qemu-system-x86_64 -accel help
Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Cyan
Write-Host '  1. Download a cloud image, e.g. Ubuntu jammy server cloudimg (amd64).'
Write-Host '  2. Place it at var\images\base.qcow2'
Write-Host '  3. Run: python training\manage.py sandbox status'
Write-Host '  4. Run: python training\manage.py sandbox provision --peer swarm'
