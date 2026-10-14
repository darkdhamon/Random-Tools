# Notification Audio

A Windows desktop notification utility with a dismissible alert window, looping
audio, queued alerts, per-device audio settings and synchronization calibration.

## Requirements

- Windows with an interactive desktop session and .NET 10 SDK.
- PowerShell to use the launcher and verification scripts.
- NuGet access for NAudio 2.2.1 on the first build.
- FFmpeg only if the optional synchronization script needs to recreate its click track.

## Build and use

Run these commands from this directory:

```powershell
dotnet build HumanRequiredAudioPlayer/HumanRequiredAudioPlayer.csproj -c Release
dotnet build AudioPairCalibrator/AudioPairCalibrator.csproj -c Release
./Start-HumanRequiredAudio.ps1 -Action Configure
./Start-HumanRequiredAudio.ps1 -Action Start -Title 'Review required' -Message 'Items are waiting for your review.'
./Start-HumanRequiredAudio.ps1 -Action Status
./Start-HumanRequiredAudio.ps1 -Action Stop
```

Use the launcher so audio, configuration, log and queue paths resolve to this tool
directory. The included Human Required and CAPTCHA Blues MP3s are available;
pass `-AudioPath` to select another file. `-StartMuted` and `-DurationSeconds`
control initial playback. Configure the desired output devices before using alerts.

The launcher also supports `-Action Schedule -ScheduledAt <datetime>` for a
one-time Windows scheduled task. Creating the tool folder does not schedule or
launch notifications.

## Verification and calibration

```powershell
./Test-IdleAudioMutePolicy.ps1
./Test-NotificationAudioSync.ps1 -ListMicrophones
```

The idle-policy test checks the 119-, 120- and 121-minute boundaries without
playing audio. The synchronization utility is an optional interactive hardware
check that plays clicks and records microphone input; select your own speakers
and microphone instead of relying on the recovered machine-specific defaults.

## Behavior and limitations

- Alerts may temporarily raise enabled output-device volume and restore the
  previous settings when stopped or dismissed.
- Playback automatically mutes after the configured idle period (120 minutes
  by default); an alert can otherwise loop until dismissed.
- Additional alerts are queued while an existing player is running.
- Audio configuration, calibration recordings, queues, process IDs and logs are
  local runtime state and are excluded from Git.
- A successful build and idle-policy test do not verify actual audio devices,
  desktop presentation, or behavior while Windows is locked.
- The recovered player currently builds with two nullable-reference warnings.

This folder contains the recovered source and required audio assets, not imported
repository history. No LinkedIn automation is configured by this tool itself.
