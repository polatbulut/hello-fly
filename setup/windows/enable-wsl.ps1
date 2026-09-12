# Enables the Windows features WSL2 needs. Requires elevation. Does NOT reboot.
$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot 'enable-wsl.log'
function W($m) { $m | Tee-Object -FilePath $log -Append | Out-Null; Write-Host $m }
Set-Content -Path $log -Value "=== enable-wsl $(Get-Date -Format o) ===" -Encoding utf8

W "Elevated: $(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"

W "`n--- feature state BEFORE ---"
foreach ($f in 'Microsoft-Windows-Subsystem-Linux','VirtualMachinePlatform') {
  $s = (Get-WindowsOptionalFeature -Online -FeatureName $f).State
  W ("{0,-38} {1}" -f $f, $s)
}

# Enable both features via DISM. Idempotent; safe to re-run.
foreach ($f in 'Microsoft-Windows-Subsystem-Linux','VirtualMachinePlatform') {
  W "`n--- enabling $f ---"
  $r = Enable-WindowsOptionalFeature -Online -FeatureName $f -All -NoRestart -WarningAction SilentlyContinue
  W "  RestartNeeded=$($r.RestartNeeded)"
}

W "`n--- feature state AFTER ---"
foreach ($f in 'Microsoft-Windows-Subsystem-Linux','VirtualMachinePlatform') {
  $s = (Get-WindowsOptionalFeature -Online -FeatureName $f).State
  W ("{0,-38} {1}" -f $f, $s)
}

# Make sure the boot-time hypervisor is actually armed. If a previous tweak
# (or an anti-cheat workaround) set this to Off, WSL2 silently falls back to
# WSL1 or fails outright, so assert it explicitly.
W "`n--- hypervisorlaunchtype ---"
$bcd = (bcdedit /enum '{current}' | Out-String)
if ($bcd -match 'hypervisorlaunchtype\s+(\w+)') { W "  current: $($Matches[1])" } else { W "  current: (not set; default Auto)" }
bcdedit /set '{current}' hypervisorlaunchtype Auto | Out-Null
W "  set to: Auto"

W "`n--- installing WSL2 kernel / MSI (web download, avoids Store) ---"
[Console]::OutputEncoding = [Text.Encoding]::Unicode
$out = (& wsl.exe --install --no-distribution --web-download 2>&1 | Out-String)
W "  exit=$LASTEXITCODE"
W ($out -replace "`0", '')

W "`n=== DONE. A REBOOT IS REQUIRED before WSL will work. ==="
