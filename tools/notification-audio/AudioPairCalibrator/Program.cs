using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

var leftFilter = Arg(args, "--left-filter") ?? "Office speaker";
var rightFilter = Arg(args, "--right-filter") ?? "ASUS";
var seconds = int.TryParse(Arg(args, "--seconds"), out var parsedSeconds) ? parsedSeconds : 8;
var leftFreq = double.TryParse(Arg(args, "--left-freq"), out var parsedLeft) ? parsedLeft : 900.0;
var rightFreq = double.TryParse(Arg(args, "--right-freq"), out var parsedRight) ? parsedRight : 1800.0;
var sampleRate = 48000;

using var enumerator = new MMDeviceEnumerator();
var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active).ToList();
var leftDevices = devices.Where(d => d.FriendlyName.Contains(leftFilter, StringComparison.OrdinalIgnoreCase)).ToList();
var rightDevices = devices.Where(d => d.FriendlyName.Contains(rightFilter, StringComparison.OrdinalIgnoreCase)).ToList();

if (leftDevices.Count == 0 || rightDevices.Count == 0)
{
    Console.Error.WriteLine($"Missing target device. Left '{leftFilter}' matches {leftDevices.Count}; right '{rightFilter}' matches {rightDevices.Count}.");
    Console.Error.WriteLine("Active render endpoints:");
    foreach (var d in devices) Console.Error.WriteLine("- " + d.FriendlyName);
    return 2;
}

var disposables = new List<IDisposable>();
var volumes = new List<VolumeSnapshot>();
var outputs = new List<WasapiOut>();

try
{
    foreach (var d in leftDevices)
    {
        AddOutput(d, leftFreq, sampleRate, seconds, outputs, disposables, volumes);
    }
    foreach (var d in rightDevices)
    {
        AddOutput(d, rightFreq, sampleRate, seconds, outputs, disposables, volumes);
    }

    Console.WriteLine("Left frequency " + leftFreq + " Hz -> " + string.Join("; ", leftDevices.Select(d => d.FriendlyName)));
    Console.WriteLine("Right frequency " + rightFreq + " Hz -> " + string.Join("; ", rightDevices.Select(d => d.FriendlyName)));

    var startAt = DateTime.UtcNow.AddMilliseconds(500);
    while (DateTime.UtcNow < startAt) Thread.SpinWait(1000);
    foreach (var output in outputs) output.Play();
    Thread.Sleep(seconds * 1000);
    return 0;
}
finally
{
    foreach (var output in outputs) { try { output.Stop(); } catch { } }
    for (var i = disposables.Count - 1; i >= 0; i--) { try { disposables[i].Dispose(); } catch { } }
    foreach (var v in volumes) { try { v.Device.AudioEndpointVolume.MasterVolumeLevelScalar = v.Volume; v.Device.AudioEndpointVolume.Mute = v.Mute; } catch { } }
}

static void AddOutput(MMDevice device, double freq, int sampleRate, int seconds, List<WasapiOut> outputs, List<IDisposable> disposables, List<VolumeSnapshot> volumes)
{
    try
    {
        var endpointVolume = device.AudioEndpointVolume;
        volumes.Add(new VolumeSnapshot(device, endpointVolume.MasterVolumeLevelScalar, endpointVolume.Mute));
        endpointVolume.Mute = false;
        endpointVolume.MasterVolumeLevelScalar = 1.0f;
    }
    catch { }

    var provider = new PulseToneProvider(sampleRate, freq, seconds);
    var output = new WasapiOut(device, AudioClientShareMode.Shared, false, 100);
    output.Init(provider);
    outputs.Add(output);
    disposables.Add(output);
}

static string? Arg(string[] args, string name)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
    }
    return null;
}

sealed record VolumeSnapshot(MMDevice Device, float Volume, bool Mute);

sealed class PulseToneProvider : ISampleProvider
{
    private readonly int sampleRate;
    private readonly double frequency;
    private readonly int totalSamples;
    private int sample;

    public PulseToneProvider(int sampleRate, double frequency, int seconds)
    {
        this.sampleRate = sampleRate;
        this.frequency = frequency;
        totalSamples = sampleRate * seconds;
        WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 2);
    }

    public WaveFormat WaveFormat { get; }

    public int Read(float[] buffer, int offset, int count)
    {
        var samplesWritten = 0;
        for (var n = 0; n < count / 2 && sample < totalSamples; n++)
        {
            var t = sample / (double)sampleRate;
            var pulseOn = t % 1.0 < 0.08;
            var value = pulseOn ? (float)(0.95 * Math.Sin(2 * Math.PI * frequency * t)) : 0f;
            buffer[offset + samplesWritten++] = value;
            buffer[offset + samplesWritten++] = value;
            sample++;
        }
        Array.Clear(buffer, offset + samplesWritten, count - samplesWritten);
        return count;
    }
}
