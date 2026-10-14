param(
    [ValidateSet("Start", "Stop", "Status", "Configure", "Schedule")]
    [string]$Action = "Start",

    [datetime]$ScheduledAt,

    [string]$ScheduledTaskName = "Job Search One-Time Audio Reminder",

    [string]$AudioPath = (Join-Path $PSScriptRoot "Human Required.mp3"),

    [string]$Title = "Human Required",

    [string]$Message = "A CAPTCHA or application blocker needs manual attention. Click Dismiss Alert after you are ready for Codex to continue.",

    [string]$DelayConfigPath = (Join-Path $PSScriptRoot "AudioDeviceDelays.json"),

    [string]$DeviceConfigPath = (Join-Path $PSScriptRoot "AudioDeviceConfig.json"),

    [string]$QueueDir = (Join-Path $PSScriptRoot "Alert Queue"),

    [string]$IncludeDeviceContains = "",

    [int]$DurationSeconds = 0,

    [switch]$StartMuted,

    [ValidateRange(1, 10080)]
    [int]$IdleMuteMinutes = 120,

    [string]$LogPath = (Join-Path $PSScriptRoot "HumanRequiredAudio.log"),

    [string]$PidPath = (Join-Path $PSScriptRoot "HumanRequiredAudio.pid")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ExistingProcess {
    if (-not (Test-Path -LiteralPath $PidPath)) {
        return $null
    }

    $rawPid = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    if (-not $rawPid) {
        return $null
    }

    try {
        return Get-Process -Id ([int]$rawPid) -ErrorAction Stop
    }
    catch {
        return $null
    }
}

function Build-Player {
    $playerProject = Join-Path $PSScriptRoot "HumanRequiredAudioPlayer\HumanRequiredAudioPlayer.csproj"
    $playerExe = Join-Path $PSScriptRoot "HumanRequiredAudioPlayer\bin\Release\net10.0-windows\HumanRequiredAudioPlayer.exe"
    if (-not (Test-Path -LiteralPath $playerProject)) {
        throw "Notification player project not found: $playerProject"
    }

    & dotnet build $playerProject -c Release --nologo | Out-Null
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $playerExe) {
            return $playerExe
        }
        throw "Failed to build HumanRequiredAudioPlayer."
    }

    return $playerExe
}

function ConvertTo-ProcessArgument {
    param(
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $quoted = '"'
    $backslashCount = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount++
            continue
        }

        if ($character -eq '"') {
            $quoted += '\' * (($backslashCount * 2) + 1)
            $quoted += '"'
            $backslashCount = 0
            continue
        }

        if ($backslashCount -gt 0) {
            $quoted += '\' * $backslashCount
            $backslashCount = 0
        }
        $quoted += $character
    }

    if ($backslashCount -gt 0) {
        $quoted += '\' * ($backslashCount * 2)
    }

    return $quoted + '"'
}

function Start-PlayerProcess {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $processInfo.FileName = $FilePath
    $processInfo.WorkingDirectory = $PSScriptRoot
    $processInfo.Arguments = ($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "

    return [System.Diagnostics.Process]::Start($processInfo)
}

if ($Action -eq "Schedule") {
    if (-not $PSBoundParameters.ContainsKey("ScheduledAt")) {
        throw "ScheduledAt is required when Action is Schedule."
    }

    if ($ScheduledAt -le (Get-Date)) {
        throw "ScheduledAt must be in the future."
    }

    if (-not (Test-Path -LiteralPath $AudioPath)) {
        throw "Audio file not found: $AudioPath"
    }

    $resolvedScript = (Resolve-Path -LiteralPath $PSCommandPath).Path
    $resolvedAudio = (Resolve-Path -LiteralPath $AudioPath).Path
    $scheduledArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $resolvedScript,
        "-Action", "Start",
        "-Title", $Title,
        "-Message", $Message,
        "-AudioPath", $resolvedAudio,
        "-DelayConfigPath", $DelayConfigPath,
        "-DeviceConfigPath", $DeviceConfigPath,
        "-QueueDir", $QueueDir,
        "-IdleMuteMinutes", "$IdleMuteMinutes",
        "-LogPath", $LogPath,
        "-PidPath", $PidPath
    )
    if ($IncludeDeviceContains) {
        $scheduledArguments += @("-IncludeDeviceContains", $IncludeDeviceContains)
    }
    if ($DurationSeconds -gt 0) {
        $scheduledArguments += @("-DurationSeconds", "$DurationSeconds")
    }
    if ($StartMuted) {
        $scheduledArguments += "-StartMuted"
    }

    $actionArguments = ($scheduledArguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $scheduledAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArguments -WorkingDirectory $PSScriptRoot
    $scheduledTrigger = New-ScheduledTaskTrigger -Once -At $ScheduledAt
    $scheduledSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    Register-ScheduledTask -TaskName $ScheduledTaskName -Action $scheduledAction -Trigger $scheduledTrigger -Settings $scheduledSettings -Description "One-time Job Search audio reminder. Runs the existing Human Required audio alert application." -Force | Out-Null

    $verifiedTask = Get-ScheduledTask -TaskName $ScheduledTaskName -ErrorAction Stop
    $verifiedInfo = Get-ScheduledTaskInfo -TaskName $ScheduledTaskName -ErrorAction Stop
    if ($verifiedTask.State -eq "Disabled") {
        throw "Scheduled task was created but is disabled: $ScheduledTaskName"
    }

    Write-Output "Scheduled one-time notification audio alert '$ScheduledTaskName' for $($ScheduledAt.ToString('yyyy-MM-dd HH:mm:ss zzz')). Next run: $($verifiedInfo.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss zzz'))."
    exit 0
}

function New-AlertRequestFile {
    param(
        [Parameter(Mandatory)]
        [string]$TargetQueueDir
    )

    if (-not (Test-Path -LiteralPath $AudioPath)) {
        throw "Audio file not found: $AudioPath"
    }

    New-Item -ItemType Directory -Force -Path $TargetQueueDir | Out-Null
    $resolvedDelayConfig = $null
    if (Test-Path -LiteralPath $DelayConfigPath) {
        $resolvedDelayConfig = (Resolve-Path -LiteralPath $DelayConfigPath).Path
    }

    $request = [ordered]@{
        audioPath = (Resolve-Path -LiteralPath $AudioPath).Path
        title = $Title
        message = $Message
        delayConfigPath = $resolvedDelayConfig
        deviceConfigPath = $DeviceConfigPath
        includeDeviceContains = $IncludeDeviceContains
        durationSeconds = $DurationSeconds
        startMuted = [bool]$StartMuted
        idleMuteMinutes = $IdleMuteMinutes
        logPath = $LogPath
    }

    $fileName = "{0:yyyyMMddHHmmssfff}-{1}.json" -f (Get-Date), ([guid]::NewGuid().ToString('N'))
    $path = Join-Path $TargetQueueDir $fileName
    $request | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $path -Encoding UTF8
    return $path
}

if ($Action -eq "Stop") {
    $existing = Get-ExistingProcess
    if ($existing) {
        Stop-Process -Id $existing.Id -Force
    }
    if (Test-Path -LiteralPath $PidPath) {
        Remove-Item -LiteralPath $PidPath -Force
    }
    Write-Output "Stopped notification audio alert."
    exit 0
}

if ($Action -eq "Status") {
    $existing = Get-ExistingProcess
    $queued = 0
    if (Test-Path -LiteralPath $QueueDir) {
        $queued = @(Get-ChildItem -LiteralPath $QueueDir -Filter '*.json' -File).Count
    }

    if ($existing) {
        Write-Output "Notification audio alert is running as PID $($existing.Id). Queued alerts: $queued. Idle audio-mute threshold: $IdleMuteMinutes minutes. Log: $LogPath"
    }
    else {
        Write-Output "Notification audio alert is not running. Queued alerts: $queued. Idle audio-mute threshold: $IdleMuteMinutes minutes. Log: $LogPath"
    }
    exit 0
}

if ($Action -eq "Start") {
    $existing = Get-ExistingProcess
    if ($existing) {
        $queuedPath = New-AlertRequestFile -TargetQueueDir $QueueDir
        Write-Output "Notification audio alert is already running as PID $($existing.Id). Queued alert: $queuedPath"
        exit 0
    }

    if (-not (Test-Path -LiteralPath $AudioPath)) {
        throw "Audio file not found: $AudioPath"
    }
}

$playerExe = Build-Player

if ($Action -eq "Configure") {
    $proc = Start-PlayerProcess -FilePath $playerExe -Arguments @("--configure-audio", "--device-config", $DeviceConfigPath)
    Write-Output "Started notification audio configuration UI as PID $($proc.Id)."
    exit 0
}

$resolvedAudio = (Resolve-Path -LiteralPath $AudioPath).Path
$resolvedDelayConfig = $null
if (Test-Path -LiteralPath $DelayConfigPath) {
    $resolvedDelayConfig = (Resolve-Path -LiteralPath $DelayConfigPath).Path
}

$argsList = @(
    "--audio", $resolvedAudio,
    "--title", $Title,
    "--message", $Message,
    "--device-config", $DeviceConfigPath,
    "--queue-dir", $QueueDir,
    "--idle-mute-minutes", "$IdleMuteMinutes",
    "--log-path", $LogPath
)
if ($resolvedDelayConfig) {
    $argsList += @("--delay-config", $resolvedDelayConfig)
}
if ($IncludeDeviceContains) {
    $argsList += @("--include-device-contains", $IncludeDeviceContains)
}
if ($DurationSeconds -gt 0) {
    $argsList += @("--duration-seconds", "$DurationSeconds")
}
if ($StartMuted) {
    $argsList += "--start-muted"
}

$proc = Start-PlayerProcess -FilePath $playerExe -Arguments $argsList
Set-Content -LiteralPath $PidPath -Value $proc.Id
Write-Output "Started notification audio alert as PID $($proc.Id)."
