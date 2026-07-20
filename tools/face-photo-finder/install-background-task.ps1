$ErrorActionPreference = "Stop"

$taskName = "Face Photo Finder Background Catalog"
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $appRoot ".venv\Scripts\pythonw.exe"
$runner = Join-Path $appRoot "run.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "The Face Photo Finder virtual environment was not found at $python"
}

$action = New-ScheduledTaskAction -Execute $python -Argument ('"{0}" --background-scan' -f $runner) -WorkingDirectory $appRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Catalog new and relocated photos and prompt after the same unknown person appears in three new photos."
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Write-Output "Installed '$taskName' to scan once per hour while this user is logged on."
