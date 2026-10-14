param(
    [string[]]$SpeakerNameContains = @("Kitchen Speaker", "Bedroom Speaker", "3bd6"),

    [string]$MicrophoneName,

    [switch]$ListMicrophones,

    [int]$RecordSeconds = 10,

    [int]$PlaybackSeconds = 8,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot "Sync Calibration Recordings"),

    [string]$ClickTrackPath = (Join-Path $PSScriptRoot "Sync Calibration Clicks.wav")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-ClickTrackIfMissing {
    if (Test-Path -LiteralPath $ClickTrackPath) {
        return
    }

    ffmpeg -hide_banner -y -f lavfi -i "aevalsrc='if(lt(mod(t,1),0.05),0.95*sin(2*PI*1000*t),0)':s=48000:d=8" -ac 2 -c:a pcm_s16le $ClickTrackPath | Out-Null
}

function Get-WavPeakTimes {
    param([string]$Path)

    $bytes = [IO.File]::ReadAllBytes($Path)
    $offset = 12
    $sampleRate = 0
    $bitsPerSample = 0
    $channels = 0
    $dataOffset = -1
    $dataSize = 0

    while ($offset + 8 -le $bytes.Length) {
        $chunkId = [Text.Encoding]::ASCII.GetString($bytes, $offset, 4)
        $chunkSize = [BitConverter]::ToInt32($bytes, $offset + 4)
        if ($chunkId -eq 'fmt ') {
            $channels = [BitConverter]::ToInt16($bytes, $offset + 10)
            $sampleRate = [BitConverter]::ToInt32($bytes, $offset + 12)
            $bitsPerSample = [BitConverter]::ToInt16($bytes, $offset + 22)
        }
        elseif ($chunkId -eq 'data') {
            $dataOffset = $offset + 8
            $dataSize = $chunkSize
            break
        }
        $offset += 8 + $chunkSize
        if (($chunkSize % 2) -eq 1) { $offset++ }
    }

    if ($dataOffset -lt 0 -or $bitsPerSample -ne 16) {
        throw "Unsupported WAV for peak analysis: $Path"
    }

    $bytesPerSample = 2 * $channels
    $sampleCount = [Math]::Floor($dataSize / $bytesPerSample)
    $window = [Math]::Max(1, [int]($sampleRate * 0.02))
    $windowMaxima = New-Object System.Collections.Generic.List[int]
    $globalMax = 0

    for ($i = 0; $i -lt $sampleCount; $i += $window) {
        $max = 0
        $end = [Math]::Min($sampleCount, $i + $window)
        for ($s = $i; $s -lt $end; $s++) {
            $base = $dataOffset + ($s * $bytesPerSample)
            $sum = 0
            for ($ch = 0; $ch -lt $channels; $ch++) {
                $sample = [BitConverter]::ToInt16($bytes, $base + ($ch * 2))
                $sum += [Math]::Abs($sample)
            }
            $avg = [int]($sum / $channels)
            if ($avg -gt $max) { $max = $avg }
        }

        $windowMaxima.Add($max)
        if ($max -gt $globalMax) { $globalMax = $max }
    }

    $threshold = [Math]::Max(40, [int]($globalMax * 0.55))
    $peaks = New-Object System.Collections.Generic.List[double]
    $lastPeakSample = -$sampleRate

    for ($w = 0; $w -lt $windowMaxima.Count; $w++) {
        $i = $w * $window
        $max = $windowMaxima[$w]

        if ($max -ge $threshold -and ($i - $lastPeakSample) -gt ($sampleRate * 0.45)) {
            $peaks.Add([Math]::Round($i / $sampleRate, 4))
            $lastPeakSample = $i
        }
    }

    return $peaks
}

if ($ListMicrophones) {
    ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1
    exit 0
}

if (-not $MicrophoneName) {
    throw "MicrophoneName is required. Run with -ListMicrophones to find the webcam mic device name."
}

New-ClickTrackIfMissing
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$alertScript = Join-Path $PSScriptRoot "Start-HumanRequiredAudio.ps1"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$results = @()

foreach ($speaker in $SpeakerNameContains) {
    $safe = ($speaker -replace '[^a-zA-Z0-9_-]', '_')
    $recording = Join-Path $OutputDirectory "$timestamp - $safe.wav"
    $ffmpegErr = Join-Path $OutputDirectory "$timestamp - $safe.ffmpeg.log"

    $ffmpegArgs = @(
        '-hide_banner', '-y',
        '-f', 'dshow',
        '-i', "audio=`"$MicrophoneName`"",
        '-t', "$RecordSeconds",
        '-ac', '1',
        '-ar', '48000',
        '-c:a', 'pcm_s16le',
        "`"$recording`""
    )

    $rec = Start-Process -FilePath 'ffmpeg.exe' -ArgumentList $ffmpegArgs -RedirectStandardError $ffmpegErr -WindowStyle Hidden -PassThru
    Start-Sleep -Milliseconds 1000

    & $alertScript -Action Start -Title "Audio Sync Calibration - $speaker" -Message "Playing calibration clicks only on devices matching: $speaker" -AudioPath $ClickTrackPath -IncludeDeviceContains $speaker -DurationSeconds $PlaybackSeconds | Out-Null

    try { Wait-Process -Id $rec.Id -ErrorAction Stop } catch { }
    & $alertScript -Action Stop | Out-Null

    if (-not (Test-Path -LiteralPath $recording)) {
        $results += [pscustomobject]@{
            SpeakerNameContains = $speaker
            Recording = $recording
            PeakTimesSeconds = ''
            FirstPeakSeconds = $null
        }
        continue
    }

    $peaks = @(Get-WavPeakTimes -Path $recording)
    $results += [pscustomobject]@{
        SpeakerNameContains = $speaker
        Recording = $recording
        PeakTimesSeconds = ($peaks -join ', ')
        FirstPeakSeconds = if ($peaks.Count -gt 0) { $peaks[0] } else { $null }
    }
}

$csv = Join-Path $OutputDirectory "$timestamp - sync-results.csv"
$results | Export-Csv -NoTypeInformation -LiteralPath $csv
$results | Format-Table -AutoSize
"Results saved to $csv"
