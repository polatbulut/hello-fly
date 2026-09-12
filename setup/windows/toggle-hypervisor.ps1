<#
    Toggles the boot-time hypervisor on/off. Run ELEVATED. Requires a reboot.

    WHY THIS EXISTS
    WSL2 needs the Hyper-V hypervisor running at boot. Riot Vanguard (vgk.sys)
    and some other kernel anti-cheats dislike that and may refuse to launch.

    The important thing to know: you do NOT need to uninstall WSL to game.
    Flipping 'hypervisorlaunchtype' is enough, and it is a one-line, fully
    reversible change. WSL and Ubuntu stay installed; WSL2 just won't start
    while it's Off (and `wsl` will error or silently want WSL1).

    USAGE
        .\toggle-hypervisor.ps1 -Mode Off    # gaming:  Vanguard-friendly
        .\toggle-hypervisor.ps1 -Mode Auto   # dev:     WSL2 works
        .\toggle-hypervisor.ps1              # just report current state

    NUCLEAR OPTION (rarely needed) - fully remove the features:
        Disable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
        Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
#>
[CmdletBinding()]
param(
    [ValidateSet('Off','Auto')]
    [string]$Mode
)

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

$bcd = (bcdedit /enum '{current}' | Out-String)
if ($bcd -match 'hypervisorlaunchtype\s+(\w+)') { $cur = $Matches[1] } else { $cur = 'Auto (default, not explicitly set)' }
Write-Host "current hypervisorlaunchtype : $cur"
Write-Host "hypervisor running right now : $((Get-CimInstance Win32_ComputerSystem).HypervisorPresent)"

if (-not $Mode) { Write-Host "`nNo -Mode given; nothing changed."; return }

if (-not $isAdmin) { Write-Error "Needs elevation. Re-run this script as Administrator."; exit 1 }

bcdedit /set '{current}' hypervisorlaunchtype $Mode | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "bcdedit failed (exit $LASTEXITCODE)"; exit 1 }

Write-Host "`nset hypervisorlaunchtype -> $Mode"
Write-Host "REBOOT for this to take effect."
if ($Mode -eq 'Off')  { Write-Host "WSL2 will not start until you set this back to Auto." }
if ($Mode -eq 'Auto') { Write-Host "Vanguard/Valorant may object until you set this back to Off." }
