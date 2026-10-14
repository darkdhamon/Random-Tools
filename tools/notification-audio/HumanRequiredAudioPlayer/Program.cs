using NAudio.CoreAudioApi;
using NAudio.Wave;
using System.ComponentModel;
using System.Drawing;
using System.Text.Json;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;

internal static class Program
{
    private const int DefaultIdleMuteMinutes = 120;

    [STAThread]
    private static int Main(string[] args)
    {
        if (HasArg(args, "--verify-idle-mute-policy"))
        {
            var idleMinutes = GetOptionalIntArg(args, "--idle-minutes") ?? 0;
            var thresholdMinutes = GetOptionalIntArg(args, "--threshold-minutes") ?? DefaultIdleMuteMinutes;
            var actual = ShouldAutoMuteForIdle(TimeSpan.FromMinutes(idleMinutes), TimeSpan.FromMinutes(thresholdMinutes)) ? "mute" : "keep-playing";
            var expected = GetArgValue(args, "--expected-result");
            return string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase) ? 0 : 2;
        }

        var deviceConfigPath = Path.GetFullPath(GetArgValue(args, "--device-config") ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "AudioDeviceConfig.json"));

        if (HasArg(args, "--configure-audio"))
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(CreateAudioConfigurationForm(deviceConfigPath));
            return 0;
        }

        var queueDir = Path.GetFullPath(GetArgValue(args, "--queue-dir") ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Alert Queue"));
        Directory.CreateDirectory(queueDir);

        var initialRequest = new AlertRequest
        {
            AudioPath = Path.GetFullPath(GetArgValue(args, "--audio") ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Human Required.mp3")),
            Title = GetArgValue(args, "--title") ?? "Human Required",
            Message = GetArgValue(args, "--message") ?? "A CAPTCHA or application blocker needs manual attention. Click Dismiss Alert after you are ready for Codex to continue.",
            DelayConfigPath = GetArgValue(args, "--delay-config"),
            DeviceConfigPath = deviceConfigPath,
            IncludeDeviceContains = GetArgValue(args, "--include-device-contains") ?? string.Empty,
            DurationSeconds = GetOptionalIntArg(args, "--duration-seconds") ?? 0,
            StartMuted = HasArg(args, "--start-muted"),
            IdleMuteMinutes = GetOptionalIntArg(args, "--idle-mute-minutes") ?? DefaultIdleMuteMinutes,
            LogPath = GetArgValue(args, "--log-path") ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "HumanRequiredAudio.log")
        };

        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        var request = initialRequest;
        while (request is not null)
        {
            PlayAlert(request, queueDir);
            request = TakeNextQueuedAlert(queueDir);
        }

        return 0;
    }

    private static void PlayAlert(AlertRequest request, string queueDir)
    {
        if (request.IdleMuteMinutes <= 0)
        {
            request.IdleMuteMinutes = DefaultIdleMuteMinutes;
        }
        if (string.IsNullOrWhiteSpace(request.LogPath))
        {
            request.LogPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "HumanRequiredAudio.log");
        }

        request.AudioPath = Path.GetFullPath(request.AudioPath);
        request.DeviceConfigPath = Path.GetFullPath(request.DeviceConfigPath);
        request.LogPath = Path.GetFullPath(request.LogPath);
        if (!string.IsNullOrWhiteSpace(request.DelayConfigPath))
        {
            request.DelayConfigPath = Path.GetFullPath(request.DelayConfigPath);
        }

        if (!File.Exists(request.AudioPath))
        {
            MessageBox.Show($"Audio file not found: {request.AudioPath}", request.Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        var includeFilters = SplitFilters(request.IncludeDeviceContains);
        var delayRules = LoadDelayRules(request.DelayConfigPath);
        var disposables = new List<IDisposable>();
        var volumeSnapshots = new List<VolumeSnapshot>();
        var playbacks = new List<DevicePlayback>();
        var volumesRestored = false;

        void Log(string message) => AppendLog(request.LogPath, request.Title, message);

        void StopAudioAndRestoreVolume()
        {
            foreach (var playback in playbacks)
            {
                try { playback.Output.Stop(); } catch { }
            }

            if (!volumesRestored)
            {
                RestoreVolumes(volumeSnapshots);
                volumesRestored = true;
            }
        }

        try
        {
            using var enumerator = new MMDeviceEnumerator();
            var allDevices = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.All).ToList();
            var deviceConfig = AudioDeviceConfig.Load(request.DeviceConfigPath);
            deviceConfig.MergeDetectedDevices(allDevices);
            deviceConfig.Save(request.DeviceConfigPath);

            foreach (var device in allDevices.Where(d => d.State == DeviceState.Active))
            {
                if (includeFilters.Count > 0 && !includeFilters.Any(filter => device.FriendlyName.Contains(filter, StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                var setting = deviceConfig.Find(device.ID, device.FriendlyName);
                if (setting is null || !setting.Enabled || AudioDeviceConfig.IsPolicyDisabled(device.FriendlyName))
                {
                    continue;
                }

                try
                {
                    TrySetMaxVolume(device, volumeSnapshots);

                    var reader = new AudioFileReader(request.AudioPath) { Volume = 1.0f };
                    var loop = new LoopStream(reader);
                    var output = new WasapiOut(device, AudioClientShareMode.Shared, false, 200);
                    output.Init(loop);

                    disposables.Add(output);
                    disposables.Add(loop);
                    disposables.Add(reader);

                    var leadMilliseconds = setting.StartLeadMilliseconds ?? GetStartLeadMilliseconds(device.FriendlyName, delayRules);
                    playbacks.Add(new DevicePlayback(device.FriendlyName, leadMilliseconds, output));
                }
                catch
                {
                    // Skip endpoints that are active but unavailable to shared WASAPI playback.
                }
            }

            if (playbacks.Count == 0)
            {
                var filterMessage = includeFilters.Count == 0 ? string.Empty : " Matching filters: " + string.Join(", ", includeFilters) + ".";
                MessageBox.Show("No enabled active speaker endpoints could play the alert." + filterMessage, request.Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            StartWithLeadCompensation(playbacks);
            Log($"Alert started. Idle audio-mute threshold: {request.IdleMuteMinutes} minutes. Start muted: {request.StartMuted}.");
            if (request.StartMuted)
            {
                StopAudioAndRestoreVolume();
            }

            using var form = CreateDismissForm(request.Title, request.Message, playbacks, StopAudioAndRestoreVolume, request.DeviceConfigPath, queueDir, request.StartMuted, TimeSpan.FromMinutes(request.IdleMuteMinutes), Log);
            if (request.DurationSeconds > 0)
            {
                using var closeTimer = new System.Windows.Forms.Timer { Interval = request.DurationSeconds * 1000 };
                closeTimer.Tick += (_, _) => form.Close();
                closeTimer.Start();
                Application.Run(form);
            }
            else
            {
                Application.Run(form);
            }
        }
        finally
        {
            StopAudioAndRestoreVolume();
            for (var i = disposables.Count - 1; i >= 0; i--)
            {
                try { disposables[i].Dispose(); } catch { }
            }
        }
    }

    private static AlertRequest? TakeNextQueuedAlert(string queueDir)
    {
        Directory.CreateDirectory(queueDir);
        var next = Directory.GetFiles(queueDir, "*.json").OrderBy(path => path, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
        if (next is null)
        {
            return null;
        }

        try
        {
            var request = JsonSerializer.Deserialize<AlertRequest>(File.ReadAllText(next), new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            File.Delete(next);
            return request;
        }
        catch
        {
            try { File.Delete(next); } catch { }
            return null;
        }
    }

    private static Form CreateAudioConfigurationForm(string configPath)
    {
        using var enumerator = new MMDeviceEnumerator();
        var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.All).ToList();
        var config = AudioDeviceConfig.Load(configPath);
        config.MergeDetectedDevices(devices);
        config.Save(configPath);

        var rows = new SortableBindingList<AudioDeviceConfigRow>(config.Devices
            .OrderByDescending(d => d.LastState == DeviceState.Active.ToString())
            .ThenBy(d => d.FriendlyName)
            .Select(AudioDeviceConfigRow.FromSetting)
            .ToList());

        var form = new Form
        {
            Text = "Notification Audio Configuration",
            StartPosition = FormStartPosition.CenterScreen,
            Size = new Size(980, 620),
            TopMost = true,
            ShowInTaskbar = true
        };
        EnforceAlwaysOnTop(form);

        var grid = new DataGridView
        {
            Dock = DockStyle.Fill,
            AutoGenerateColumns = false,
            DataSource = rows,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            MultiSelect = false
        };

        grid.Columns.Add(new DataGridViewCheckBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.Enabled), HeaderText = "Enabled", Width = 70 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.FriendlyName), HeaderText = "Device", ReadOnly = true, Width = 280 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.LastState), HeaderText = "State", ReadOnly = true, Width = 80 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.StartLeadMilliseconds), HeaderText = "Lead ms", Width = 80 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.Policy), HeaderText = "Policy", ReadOnly = true, Width = 150 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.LastSeenUtc), HeaderText = "Last seen UTC", ReadOnly = true, Width = 150 });
        grid.Columns.Add(new DataGridViewTextBoxColumn { DataPropertyName = nameof(AudioDeviceConfigRow.DeviceId), HeaderText = "Device ID", ReadOnly = true, Width = 420 });
        foreach (DataGridViewColumn column in grid.Columns)
        {
            column.SortMode = DataGridViewColumnSortMode.Automatic;
        }

        grid.CellFormatting += (_, e) =>
        {
            if (e.RowIndex < 0 || e.RowIndex >= rows.Count) return;
            var row = rows[e.RowIndex];
            if (row.Policy.Length > 0)
            {
                grid.Rows[e.RowIndex].DefaultCellStyle.BackColor = Color.FromArgb(245, 235, 235);
            }
            else if (row.LastState != DeviceState.Active.ToString())
            {
                grid.Rows[e.RowIndex].DefaultCellStyle.BackColor = Color.FromArgb(242, 242, 242);
            }
        };

        var explanation = new Label
        {
            Dock = DockStyle.Top,
            Height = 58,
            Padding = new Padding(12),
            Text = "New playback devices are automatically enabled unless their name contains Steam or Office. Previously seen devices stay listed even when disconnected. Steam and Office devices are disabled by policy.",
            TextAlign = ContentAlignment.MiddleLeft
        };

        var save = NewReadableButton("Save", 130, 44);
        var refresh = NewReadableButton("Refresh Devices", 170, 44);
        var close = NewReadableButton("Close", 130, 44);
        var status = new Label { Text = "", AutoSize = false, TextAlign = ContentAlignment.MiddleLeft, Left = 12, Top = 8, Height = 44 };

        var bottom = new Panel { Dock = DockStyle.Bottom, Height = 62, Padding = new Padding(12), BackColor = Color.WhiteSmoke };
        save.Top = refresh.Top = close.Top = 6;
        bottom.Resize += (_, _) =>
        {
            close.Left = bottom.ClientSize.Width - close.Width - 12;
            refresh.Left = close.Left - refresh.Width - 10;
            save.Left = refresh.Left - save.Width - 10;
            status.Width = Math.Max(80, save.Left - 24);
        };
        bottom.Controls.Add(status);
        bottom.Controls.Add(save);
        bottom.Controls.Add(refresh);
        bottom.Controls.Add(close);

        void SaveRows()
        {
            foreach (var row in rows)
            {
                var setting = config.Devices.FirstOrDefault(d => d.DeviceId == row.DeviceId) ?? new AudioDeviceSetting { DeviceId = row.DeviceId };
                if (!config.Devices.Contains(setting)) config.Devices.Add(setting);
                setting.FriendlyName = row.FriendlyName;
                setting.Enabled = row.Enabled;
                setting.StartLeadMilliseconds = row.StartLeadMilliseconds;
                setting.LastSeenUtc = row.LastSeenUtc;
                setting.LastState = row.LastState;
                setting.Policy = row.Policy;
            }
            config.ApplyPolicyDisabledDevices();
            config.Save(configPath);
            status.Text = "Saved " + DateTime.Now.ToString("HH:mm:ss") + ". Steam and Office devices remain disabled by policy.";
        }

        save.Click += (_, _) => SaveRows();
        close.Click += (_, _) => form.Close();
        refresh.Click += (_, _) =>
        {
            using var refreshedEnumerator = new MMDeviceEnumerator();
            config.MergeDetectedDevices(refreshedEnumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.All).ToList());
            config.Save(configPath);
            rows.Clear();
            foreach (var row in config.Devices.OrderByDescending(d => d.LastState == DeviceState.Active.ToString()).ThenBy(d => d.FriendlyName).Select(AudioDeviceConfigRow.FromSetting))
            {
                rows.Add(row);
            }
            status.Text = "Refreshed " + DateTime.Now.ToString("HH:mm:ss") + ".";
        };

        form.Controls.Add(grid);
        form.Controls.Add(explanation);
        form.Controls.Add(bottom);
        return form;
    }

    private static Form CreateDismissForm(string titleText, string messageText, IReadOnlyCollection<DevicePlayback> playbacks, Action muteAudio, string deviceConfigPath, string queueDir, bool startMuted, TimeSpan idleMuteThreshold, Action<string> log)
    {
        var form = new Form
        {
            Text = titleText,
            StartPosition = FormStartPosition.CenterScreen,
            Size = new Size(760, 560),
            TopMost = true,
            ShowInTaskbar = true,
            FormBorderStyle = FormBorderStyle.FixedDialog,
            MaximizeBox = false,
            MinimizeBox = true
        };
        EnforceAlwaysOnTop(form);

        var title = new Label { Text = titleText, Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 18, FontStyle.Bold), AutoSize = false, TextAlign = ContentAlignment.MiddleCenter, Dock = DockStyle.Top, Height = 58 };
        var message = new Label { Text = messageText, Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 10, FontStyle.Regular), AutoSize = false, TextAlign = ContentAlignment.MiddleCenter, Dock = DockStyle.Top, Height = 86, Padding = new Padding(24, 0, 24, 0) };
        var devices = new Label
        {
            Text = "Playing at max volume on: " + string.Join("; ", playbacks.Select(p => p.FriendlyName + (p.StartLeadMilliseconds > 0 ? $" (+{p.StartLeadMilliseconds}ms lead)" : ""))),
            Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 8, FontStyle.Regular),
            AutoSize = false,
            TextAlign = ContentAlignment.MiddleCenter,
            Dock = DockStyle.Top,
            Height = 62,
            Padding = new Padding(24, 0, 24, 0)
        };
        var audioMuted = startMuted;

        var queueHeading = new Label
        {
            Text = "Alerts (1 playing, 0 queued)",
            Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 10, FontStyle.Bold),
            AutoSize = false,
            TextAlign = ContentAlignment.MiddleLeft,
            Dock = DockStyle.Top,
            Height = 34,
            Padding = new Padding(16, 0, 16, 0)
        };
        var queueList = new ListView
        {
            Dock = DockStyle.Fill,
            View = View.Details,
            FullRowSelect = true,
            GridLines = true,
            HideSelection = false,
            MultiSelect = false,
            AccessibleName = "Alert list",
            AccessibleDescription = "The active alert followed by every alert waiting to play"
        };
        queueList.Columns.Add("Status", 90);
        queueList.Columns.Add("Title", 190);
        queueList.Columns.Add("Message", 410);

        var mute = NewReadableButton("Mute Alert", 150, 46);
        var configure = NewReadableButton("Audio Configuration", 210, 46);
        var dismiss = NewReadableButton("Dismiss Alert", 170, 46);
        var dismissAll = NewReadableButton("Dismiss All", 160, 46);
        dismiss.AccessibleDescription = "Dismiss the currently playing alert and play the next queued alert";
        dismissAll.AccessibleDescription = "Dismiss the current alert and clear every pending alert";
        if (startMuted)
        {
            mute.Text = "Muted";
            mute.Enabled = false;
        }
        mute.Click += (_, _) =>
        {
            muteAudio();
            audioMuted = true;
            mute.Text = "Muted";
            mute.Enabled = false;
            devices.Text = "Audio muted manually. This alert remains visible until dismissed.";
            log("Audio muted manually; alert remains visible.");
        };
        configure.Click += (_, _) => StartConfigurationWindow(deviceConfigPath);
        dismiss.Click += (_, _) => form.Close();
        dismissAll.Click += (_, _) =>
        {
            ClearQueuedAlerts(queueDir);
            form.Close();
        };

        var panel = new Panel { Dock = DockStyle.Bottom, Height = 76 };
        panel.Controls.Add(mute);
        panel.Controls.Add(configure);
        panel.Controls.Add(dismiss);
        panel.Controls.Add(dismissAll);
        panel.Resize += (_, _) =>
        {
            var gap = 10;
            var totalWidth = mute.Width + configure.Width + dismiss.Width + dismissAll.Width + (gap * 3);
            mute.Left = (panel.ClientSize.Width - totalWidth) / 2;
            configure.Left = mute.Right + gap;
            dismiss.Left = configure.Right + gap;
            dismissAll.Left = dismiss.Right + gap;
            mute.Top = configure.Top = dismiss.Top = dismissAll.Top = (panel.ClientSize.Height - mute.Height) / 2;
        };

        var initializedListSelection = false;
        var alertListBoldFont = new Font(queueList.Font, FontStyle.Bold);
        void DisplaySelectedAlert(ListViewItem item)
        {
            form.Text = item.SubItems[1].Text;
            title.Text = item.SubItems[1].Text;
            message.Text = item.SubItems[2].Text;
            foreach (ListViewItem alertItem in queueList.Items)
            {
                alertItem.Text = alertItem.Index == item.Index ? "Active" : alertItem.Index == 0 ? "Playing" : "Pending";
                alertItem.BackColor = alertItem.Index == item.Index ? Color.FromArgb(225, 244, 229) : SystemColors.Window;
                alertItem.Font = alertItem.Index == item.Index ? alertListBoldFont : queueList.Font;
            }
        }

        queueList.SelectedIndexChanged += (_, _) =>
        {
            if (queueList.SelectedItems.Count > 0)
            {
                DisplaySelectedAlert(queueList.SelectedItems[0]);
            }
        };

        void RefreshQueue()
        {
            var queuedAlerts = ReadQueuedAlertSummaries(queueDir);
            queueHeading.Text = $"Alerts (1 playing, {queuedAlerts.Count} queued)";
            var selectedIndex = queueList.SelectedIndices.Count > 0 ? queueList.SelectedIndices[0] : -1;
            queueList.BeginUpdate();
            queueList.Items.Clear();
            var activeItem = new ListViewItem(["Active", titleText, messageText])
            {
                BackColor = Color.FromArgb(225, 244, 229),
                Font = alertListBoldFont
            };
            queueList.Items.Add(activeItem);
            foreach (var queuedAlert in queuedAlerts)
            {
                queueList.Items.Add(new ListViewItem(["Pending", queuedAlert.Title, queuedAlert.Message]));
            }
            if (!initializedListSelection)
            {
                activeItem.Selected = true;
                activeItem.Focused = true;
                initializedListSelection = true;
            }
            else if (selectedIndex >= 0 && selectedIndex < queueList.Items.Count)
            {
                queueList.Items[selectedIndex].Selected = true;
                queueList.Items[selectedIndex].Focused = true;
            }
            queueList.EndUpdate();
            if (queueList.SelectedItems.Count > 0)
            {
                DisplaySelectedAlert(queueList.SelectedItems[0]);
            }
        }

        var queueTimer = new System.Windows.Forms.Timer { Interval = 500 };
        queueTimer.Tick += (_, _) => RefreshQueue();
        var idleTimer = new System.Windows.Forms.Timer { Interval = 5000 };
        idleTimer.Tick += (_, _) =>
        {
            if (audioMuted)
            {
                return;
            }

            var idleTime = GetWindowsUserIdleTime();
            if (!ShouldAutoMuteForIdle(idleTime, idleMuteThreshold))
            {
                return;
            }

            muteAudio();
            audioMuted = true;
            mute.Text = "Muted after 2h idle";
            mute.Enabled = false;
            devices.Text = "Audio automatically muted after two hours of Windows inactivity. This alert remains visible until dismissed.";
            log($"Audio automatically muted after {idleTime.TotalMinutes:F1} idle minutes; alert remains visible.");
        };
        form.Shown += (_, _) =>
        {
            RefreshQueue();
            queueTimer.Start();
            idleTimer.Start();
        };
        form.FormClosed += (_, _) =>
        {
            queueTimer.Dispose();
            idleTimer.Dispose();
            alertListBoldFont.Dispose();
            log("Alert dismissed; advancing to the next queued alert when present.");
        };

        form.Controls.Add(panel);
        form.Controls.Add(queueList);
        form.Controls.Add(queueHeading);
        form.Controls.Add(devices);
        form.Controls.Add(message);
        form.Controls.Add(title);
        form.AcceptButton = dismiss;
        return form;
    }

    internal static bool ShouldAutoMuteForIdle(TimeSpan idleTime, TimeSpan threshold) => idleTime >= threshold;

    private static TimeSpan GetWindowsUserIdleTime()
    {
        var info = new LastInputInfo { Size = (uint)Marshal.SizeOf<LastInputInfo>() };
        if (!GetLastInputInfo(ref info))
        {
            return TimeSpan.Zero;
        }

        var idleMilliseconds = unchecked((uint)Environment.TickCount - info.TickCount);
        return TimeSpan.FromMilliseconds(idleMilliseconds);
    }

    private static void AppendLog(string path, string title, string message)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.AppendAllText(path, $"{DateTimeOffset.Now:O} [{title}] {message}{Environment.NewLine}");
        }
        catch
        {
            // Logging must never prevent an intervention alert from remaining available.
        }
    }

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetLastInputInfo(ref LastInputInfo inputInfo);

    [StructLayout(LayoutKind.Sequential)]
    private struct LastInputInfo
    {
        public uint Size;
        public uint TickCount;
    }

    private static List<QueuedAlertSummary> ReadQueuedAlertSummaries(string queueDir)
    {
        Directory.CreateDirectory(queueDir);
        var summaries = new List<QueuedAlertSummary>();
        foreach (var path in Directory.GetFiles(queueDir, "*.json").OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                var request = JsonSerializer.Deserialize<AlertRequest>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                summaries.Add(request is null
                    ? new QueuedAlertSummary("Unreadable queued alert", Path.GetFileName(path))
                    : new QueuedAlertSummary(request.Title, request.Message));
            }
            catch
            {
                summaries.Add(new QueuedAlertSummary("Unreadable queued alert", Path.GetFileName(path)));
            }
        }
        return summaries;
    }

    private static void ClearQueuedAlerts(string queueDir)
    {
        if (!Directory.Exists(queueDir)) return;
        foreach (var path in Directory.GetFiles(queueDir, "*.json"))
        {
            try { File.Delete(path); } catch { }
        }
    }

    private static void StartConfigurationWindow(string deviceConfigPath)
    {
        try
        {
            var exePath = Environment.ProcessPath;
            if (string.IsNullOrWhiteSpace(exePath))
            {
                return;
            }

            var processInfo = new ProcessStartInfo
            {
                FileName = exePath,
                UseShellExecute = false
            };
            processInfo.ArgumentList.Add("--configure-audio");
            processInfo.ArgumentList.Add("--device-config");
            processInfo.ArgumentList.Add(deviceConfigPath);
            Process.Start(processInfo);
        }
        catch
        {
            // Configuration launch is best-effort and should not interfere with the active alert.
        }
    }
    private static Button NewReadableButton(string text, int width, int height)
    {
        var button = new Button
        {
            Text = text,
            Width = width,
            Height = height,
            BackColor = Color.FromArgb(245, 245, 245),
            ForeColor = Color.Black,
            FlatStyle = FlatStyle.Flat,
            UseVisualStyleBackColor = false,
            TextAlign = ContentAlignment.MiddleCenter,
            Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 10, FontStyle.Bold)
        };

        button.FlatAppearance.BorderColor = Color.Black;
        button.FlatAppearance.BorderSize = 1;
        button.Paint += (_, e) =>
        {
            e.Graphics.Clear(button.BackColor);
            using var borderPen = new Pen(Color.Black);
            e.Graphics.DrawRectangle(borderPen, new Rectangle(0, 0, button.Width - 1, button.Height - 1));
            TextRenderer.DrawText(e.Graphics, button.Text, button.Font, button.ClientRectangle, Color.Black, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis);
        };

        return button;
    }

    private static void EnforceAlwaysOnTop(Form form)
    {
        form.TopMost = true;
        form.Shown += (_, _) =>
        {
            form.TopMost = true;
            form.Activate();
            form.BringToFront();
        };
        form.Activated += (_, _) => form.TopMost = true;

        var timer = new System.Windows.Forms.Timer { Interval = 2000 };
        timer.Tick += (_, _) =>
        {
            if (!form.IsDisposed)
            {
                form.TopMost = true;
            }
        };
        form.FormClosed += (_, _) => timer.Dispose();
        timer.Start();
    }

    private static void StartWithLeadCompensation(List<DevicePlayback> playbacks)
    {
        var groups = playbacks.GroupBy(p => p.StartLeadMilliseconds).OrderByDescending(g => g.Key).ToList();
        var previousLead = groups[0].Key;
        foreach (var group in groups)
        {
            var wait = previousLead - group.Key;
            if (wait > 0) Thread.Sleep(wait);
            foreach (var playback in group) playback.Output.Play();
            previousLead = group.Key;
        }
    }

    private static void TrySetMaxVolume(MMDevice device, List<VolumeSnapshot> snapshots)
    {
        try
        {
            var endpointVolume = device.AudioEndpointVolume;
            snapshots.Add(new VolumeSnapshot(device, endpointVolume.MasterVolumeLevelScalar, endpointVolume.Mute));
            endpointVolume.Mute = false;
            endpointVolume.MasterVolumeLevelScalar = 1.0f;
        }
        catch { }
    }

    private static void RestoreVolumes(List<VolumeSnapshot> snapshots)
    {
        foreach (var snapshot in snapshots)
        {
            try
            {
                snapshot.Device.AudioEndpointVolume.MasterVolumeLevelScalar = snapshot.MasterVolumeLevelScalar;
                snapshot.Device.AudioEndpointVolume.Mute = snapshot.Mute;
            }
            catch { }
        }
    }

    private static List<DelayRule> LoadDelayRules(string? delayConfigPath)
    {
        if (string.IsNullOrWhiteSpace(delayConfigPath) || !File.Exists(delayConfigPath)) return [];
        try
        {
            var json = File.ReadAllText(delayConfigPath);
            var config = JsonSerializer.Deserialize<DelayConfig>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            return config?.Rules ?? [];
        }
        catch { return []; }
    }

    private static int GetStartLeadMilliseconds(string deviceName, List<DelayRule> delayRules)
    {
        foreach (var rule in delayRules)
        {
            if (!string.IsNullOrWhiteSpace(rule.NameContains) && deviceName.Contains(rule.NameContains, StringComparison.OrdinalIgnoreCase))
            {
                return Math.Max(0, rule.StartLeadMilliseconds);
            }
        }
        return 0;
    }

    private static bool HasArg(string[] args, string name) => args.Any(a => string.Equals(a, name, StringComparison.OrdinalIgnoreCase));

    private static string? GetArgValue(string[] args, string name)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        }
        return null;
    }

    private static List<string> GetArgValues(string[] args, string name)
    {
        var values = new List<string>();
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) values.AddRange(SplitFilters(args[i + 1]));
        }
        return values;
    }

    private static List<string> SplitFilters(string? value)
    {
        return string.IsNullOrWhiteSpace(value) ? [] : value.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
    }

    private static int? GetOptionalIntArg(string[] args, string name)
    {
        var value = GetArgValue(args, name);
        return int.TryParse(value, out var parsed) ? parsed : null;
    }
}

internal sealed class AlertRequest
{
    public string AudioPath { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string? DelayConfigPath { get; set; }
    public string DeviceConfigPath { get; set; } = string.Empty;
    public string IncludeDeviceContains { get; set; } = string.Empty;
    public int DurationSeconds { get; set; }
    public bool StartMuted { get; set; }
    public int IdleMuteMinutes { get; set; } = 120;
    public string LogPath { get; set; } = string.Empty;
}

internal sealed class SortableBindingList<T> : BindingList<T>
{
    private bool isSorted;
    private ListSortDirection sortDirection;
    private PropertyDescriptor? sortProperty;

    public SortableBindingList(IList<T> list) : base(list) { }

    protected override bool SupportsSortingCore => true;
    protected override bool IsSortedCore => isSorted;
    protected override PropertyDescriptor? SortPropertyCore => sortProperty;
    protected override ListSortDirection SortDirectionCore => sortDirection;

    protected override void ApplySortCore(PropertyDescriptor prop, ListSortDirection direction)
    {
        if (Items is not List<T> items) return;

        items.Sort((left, right) =>
        {
            var leftValue = prop.GetValue(left);
            var rightValue = prop.GetValue(right);
            var result = CompareValues(leftValue, rightValue);
            return direction == ListSortDirection.Ascending ? result : -result;
        });

        sortProperty = prop;
        sortDirection = direction;
        isSorted = true;
        OnListChanged(new ListChangedEventArgs(ListChangedType.Reset, -1));
    }

    protected override void RemoveSortCore()
    {
        isSorted = false;
        sortProperty = null;
    }

    private static int CompareValues(object? leftValue, object? rightValue)
    {
        if (ReferenceEquals(leftValue, rightValue)) return 0;
        if (leftValue is null) return -1;
        if (rightValue is null) return 1;
        if (leftValue is IComparable comparable) return comparable.CompareTo(rightValue);
        return string.Compare(leftValue.ToString(), rightValue.ToString(), StringComparison.OrdinalIgnoreCase);
    }
}

internal sealed class AudioDeviceConfig
{
    public List<AudioDeviceSetting> Devices { get; set; } = [];

    public static AudioDeviceConfig Load(string path)
    {
        if (!File.Exists(path)) return new AudioDeviceConfig();
        try
        {
            return JsonSerializer.Deserialize<AudioDeviceConfig>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new AudioDeviceConfig();
        }
        catch { return new AudioDeviceConfig(); }
    }

    public void Save(string path)
    {
        ApplyPolicyDisabledDevices();
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        File.WriteAllText(path, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
    }

    public void MergeDetectedDevices(IEnumerable<MMDevice> detectedDevices)
    {
        foreach (var device in detectedDevices)
        {
            var setting = Find(device.ID, device.FriendlyName);
            if (setting is null)
            {
                setting = new AudioDeviceSetting
                {
                    DeviceId = device.ID,
                    FriendlyName = device.FriendlyName,
                    Enabled = !IsPolicyDisabled(device.FriendlyName)
                };
                Devices.Add(setting);
            }

            setting.DeviceId = device.ID;
            setting.FriendlyName = device.FriendlyName;
            setting.LastSeenUtc = DateTime.UtcNow.ToString("O");
            setting.LastState = device.State.ToString();
            setting.Policy = IsPolicyDisabled(device.FriendlyName) ? "Disabled by policy" : string.Empty;
            if (IsPolicyDisabled(device.FriendlyName)) setting.Enabled = false;
        }

        ApplyPolicyDisabledDevices();
    }

    public AudioDeviceSetting? Find(string deviceId, string friendlyName)
    {
        return Devices.FirstOrDefault(d => d.DeviceId == deviceId) ?? Devices.FirstOrDefault(d => d.FriendlyName.Equals(friendlyName, StringComparison.OrdinalIgnoreCase));
    }

    public void ApplyPolicyDisabledDevices()
    {
        foreach (var device in Devices)
        {
            if (IsPolicyDisabled(device.FriendlyName))
            {
                device.Enabled = false;
                device.Policy = "Disabled by policy";
            }
        }
    }

    public static bool IsPolicyDisabled(string friendlyName)
    {
        return friendlyName.Contains("Steam", StringComparison.OrdinalIgnoreCase) || friendlyName.Contains("Office", StringComparison.OrdinalIgnoreCase);
    }
}

internal sealed class AudioDeviceSetting
{
    public string DeviceId { get; set; } = string.Empty;
    public string FriendlyName { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public int? StartLeadMilliseconds { get; set; }
    public string LastSeenUtc { get; set; } = string.Empty;
    public string LastState { get; set; } = string.Empty;
    public string Policy { get; set; } = string.Empty;
}

internal sealed class AudioDeviceConfigRow
{
    public string DeviceId { get; set; } = string.Empty;
    public string FriendlyName { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public int? StartLeadMilliseconds { get; set; }
    public string LastSeenUtc { get; set; } = string.Empty;
    public string LastState { get; set; } = string.Empty;
    public string Policy { get; set; } = string.Empty;

    public static AudioDeviceConfigRow FromSetting(AudioDeviceSetting setting)
    {
        return new AudioDeviceConfigRow
        {
            DeviceId = setting.DeviceId,
            FriendlyName = setting.FriendlyName,
            Enabled = setting.Enabled,
            StartLeadMilliseconds = setting.StartLeadMilliseconds,
            LastSeenUtc = setting.LastSeenUtc,
            LastState = setting.LastState,
            Policy = setting.Policy
        };
    }
}

internal sealed record DevicePlayback(string FriendlyName, int StartLeadMilliseconds, WasapiOut Output);
internal sealed record VolumeSnapshot(MMDevice Device, float MasterVolumeLevelScalar, bool Mute);
internal sealed record QueuedAlertSummary(string Title, string Message);
internal sealed class DelayConfig { public List<DelayRule>? Rules { get; set; } }
internal sealed class DelayRule { public string? NameContains { get; set; } public int StartLeadMilliseconds { get; set; } }

internal sealed class LoopStream(WaveStream sourceStream) : WaveStream
{
    public override WaveFormat WaveFormat => sourceStream.WaveFormat;
    public override long Length => sourceStream.Length;
    public override long Position { get => sourceStream.Position; set => sourceStream.Position = value; }

    public override int Read(byte[] buffer, int offset, int count)
    {
        var totalBytesRead = 0;
        while (totalBytesRead < count)
        {
            var bytesRead = sourceStream.Read(buffer, offset + totalBytesRead, count - totalBytesRead);
            if (bytesRead == 0)
            {
                sourceStream.Position = 0;
                continue;
            }
            totalBytesRead += bytesRead;
        }
        return totalBytesRead;
    }
}

