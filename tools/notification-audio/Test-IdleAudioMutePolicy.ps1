param(
    [string]$Configuration = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$project = Join-Path $PSScriptRoot "HumanRequiredAudioPlayer\HumanRequiredAudioPlayer.csproj"
$framework = "net10.0-windows"
$player = Join-Path $PSScriptRoot "HumanRequiredAudioPlayer\bin\$Configuration\$framework\HumanRequiredAudioPlayer.exe"

& dotnet build $project -c $Configuration --nologo | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "HumanRequiredAudioPlayer build failed."
}

function Assert-PolicyResult {
    param(
        [int]$IdleMinutes,
        [string]$Expected
    )

    & $player --verify-idle-mute-policy --idle-minutes $IdleMinutes --threshold-minutes 120 --expected-result $Expected
    if ($LASTEXITCODE -ne 0) {
        throw "Idle mute policy failed for $IdleMinutes minutes: expected '$Expected'."
    }
}

Assert-PolicyResult -IdleMinutes 119 -Expected "keep-playing"
Assert-PolicyResult -IdleMinutes 120 -Expected "mute"
Assert-PolicyResult -IdleMinutes 121 -Expected "mute"

Write-Output "Idle audio-mute policy passed: below two hours keeps playing; two hours and above mutes."
