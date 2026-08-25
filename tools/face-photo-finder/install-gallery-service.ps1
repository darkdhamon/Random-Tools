$ErrorActionPreference = "Stop"
$taskName = "Face Photo Finder Web Gallery"
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $appRoot ".venv\Scripts\pythonw.exe"
$runner = Join-Path $appRoot "run-gallery.py"
$action = New-ScheduledTaskAction -Execute $python -Argument ('"{0}"' -f $runner) -WorkingDirectory $appRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Local-only photo gallery and metadata editor at http://127.0.0.1:8765"
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "Installed and started '$taskName'. Open http://127.0.0.1:8765"
