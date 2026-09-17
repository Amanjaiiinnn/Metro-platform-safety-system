# Metro Platform Safety System — Dual Camera

An edge-AI monitor for the yellow safety line on metro platforms. Up to two cameras watch the platform edge. When someone steps past the line, the system identifies the colour and type of their clothing and plays a pre-recorded warning such as *"Attention! Person in red top, please step back from the yellow line."* The warning can play in English, Hindi and a regional language.

It runs on NXP **i.MX93** (Ethos-U65 NPU) and **i.MX95** (eIQ Neutron NPU) boards, and on an ordinary PC (CPU) for development. Operators use a browser dashboard to place the safety line, tune the cameras and set off-hours.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Command-line options](#command-line-options)
- [Camera sources](#camera-sources)
- [Web dashboard](#web-dashboard)
- [Configuration](#configuration)
- [Models](#models)
- [Voice alerts](#voice-alerts)
- [Clock and schedule](#clock-and-schedule)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)
- [License](#license)

## Features

- **Two independent cameras.** A USB camera can be paired with a phone or IP camera (HTTP MJPEG or RTSP). Camera 2 can be switched on or off without a restart, and camera 1 keeps working if camera 2 fails.
- **Safety-line detection.** YOLO detects people and trains. Each camera has its own line and a marked "alert side" (the track side), which together decide who is too close to the edge.
- **Person description.** A pose model finds the torso so its colour can be measured, and an int8 classifier names the garment. Results are combined over several frames before an alert fires.
- **Multi-language voice alerts.** Pre-recorded WAV clips (not text-to-speech) are joined into one phrase. Each camera has its own cooldown, and all alerts go through one audio queue so they never overlap.
- **Crowd and train handling.** A separate "multiple people" alert covers groups, and alerts pause while a train is at the platform.
- **Camera health checks.** Detection pauses on any camera whose picture is black, blurred or blocked. Disconnected cameras reconnect automatically.
- **Scheduling.** A no-detection window covers the hours when trains don't run. A built-in clock keeps the schedule working on boards with no real-time clock (RTC) or internet.
- **Web dashboard** on port 5050, with live feeds, drag-to-place safety lines, image tuning, thresholds, and audio and schedule settings.
- **NPU acceleration.** The board type is detected automatically. Setting `BACKEND = "CPU"` runs everything on the CPU instead.

## How it works

```
 Camera 1 ─► CaptureThread ─┐
                            ├─► Processing loop (CaptureWidget, CPU core 0)
 Camera 2 ─► CaptureThread ─┘   · camera health check, image tuning, schedule
                                · every frame → dashboard live feed
                                · inference turns alternate cam 1 / cam 2
                                          │
                                          ▼
                                Detector (YOLO) ─► ProximityDetector (zone line)
                                          │
                                          ▼
                                Application (state kept per camera)
                                · train suppression · multi-person check · votes
                                          │
                                          ▼
                                SegmentAnalyzer
                                · pose → torso → colour · garment classifier
                                          │
                                          ▼
                                AlertEngine
                                · per-camera cooldown → console + voice queue

 NPUWorker (CPU core 1) runs every model (YOLO, pose, clothing) on one thread.
```

The numbers below are the defaults from `config.py`.

1. **Capture.** Each camera has its own capture thread feeding a two-frame queue. When the queue is full, the oldest frame is dropped, so a slow pipeline skips stale frames instead of falling behind. Every 2nd frame is marked for inference.
2. **Pre-checks.** Every 3rd frame, the processing loop checks the camera's health. It also applies that camera's image adjustments. Inference is skipped during the no-detection window and while the camera isn't "Live". Every frame still goes to the dashboard.
3. **Turn-taking.** With two cameras running, inference alternates between them. The NPU load is the same as with one camera, and each camera gets about half the inference rate. If the camera whose turn it is sends no frame within 1 s, the other camera takes the turn.
4. **Detection.** YOLO returns Person and Train boxes. The boxes are filtered by the dashboard's confidence thresholds. Duplicates are then removed by non-maximum suppression (NMS): overlap above 0.20 IoU, or a box that sits almost entirely inside another.
5. **Safety-line check.** A person is in the alert zone if the box centre, its bottom-centre or the point 75 % of the way down it is on the alert side of the line or within `threshold_pixels` of the line. Trains only count when they are on the alert side.
6. **Decision.** Each camera's state is tracked separately:
   - A train on the alert side pauses alerts once it has been there for 5 s. Alerts resume 1 s after it leaves.
   - If more than one person is in the zone on 2 inference frames in a row, the system plays the general "multiple people" alert.
   - If exactly one person is in the zone, each inference frame adds one vote for colour and garment.
7. **Description.** The pose model finds the shoulders and hips, and the torso between them is cropped. The torso's colour is measured after lighting is evened out (CLAHE), with the centre of the crop counting most. The garment classifier looks at the top 55 % of the person box, and a garment is only named at 60 % confidence or higher. Frames where no colour can be measured don't count. After 2 frames with a colour, the most common colour and garment are used.
8. **Alert.** `AlertEngine` checks that camera's cooldown, prints the message and queues the voice clips. One audio worker plays alerts from both cameras, one after another.

## Project layout

```
.
├── launcher.py               Board entry point: detects the board, prepares models, runs main.py
├── main.py                   Per-camera state, train suppression, multi-person check, voting
├── config.py                 Settings read at startup, plus shared helpers
├── config.json               Settings the dashboard reads and writes while running
├── config_backup.json        Last saved settings (used by the dashboard's auto-restore)
├── time_sync_state.json      Last clock sync from the dashboard
├── video_widget.py           Camera streams, capture threads, processing loop, overlays
├── detection.py              YOLO pre/post-processing and NMS
├── line_detector.py          Safety-line geometry (ProximityDetector)
├── segment.py                Colour detection, garment classification, vote buffer
├── pose_detector.py          Pose model → torso crop
├── npu_worker.py             The single thread that owns and runs every TFLite model
├── channels.py               One-item handoff between threads (newest item wins)
├── alert_engine.py           Cooldowns, audio queue, WAV joining and playback
├── line_config_server.py     Flask server: dashboard, live feeds, settings and clock API
├── index.html                Dashboard UI
├── mjpeg_relay.py            Host-side relay for cameras the board can't reach
├── static_ip.py              Sets a static Ethernet IP on the board (connmanctl)
├── startup_control.sh        Boot script: picks the install folder and runs launcher.py
├── startup-control.service   systemd unit that runs startup_control.sh
├── models/                   TFLite models and labels.txt
│   └── neutron converted/    Models converted for the i.MX95 NPU (see Models)
└── voices/                   Voice clips, one folder per language
    ├── english/
    ├── hindi/
    └── regional/
```

With `DEBUG = True`, the app also creates `debug_crops/`.

## Requirements

**All platforms**

- Python 3.9 or newer (the board runs 3.12)
- `numpy`, `opencv-python`, `flask`
- A TensorFlow Lite runtime: `tflite_runtime` (usually preinstalled in NXP's board images) or `tensorflow`

**On the board**

- An NXP Linux image that provides the NPU delegate: `/usr/lib/libethosu_delegate.so` (i.MX93) or `/usr/lib/libneutron_delegate.so` (i.MX95)
- `vela` (i.MX93 only). The launcher uses it to compile models for the NPU.
- `aplay` (from alsa-utils) and a speaker for voice alerts
- `connmanctl`, if you use `static_ip.py`

**On a PC**

```bash
pip install numpy opencv-python flask tensorflow
```

You can use `tflite-runtime` instead of `tensorflow` if a build exists for your OS and Python version. Voice playback needs `aplay`, which is Linux-only. On Windows and macOS, alerts appear in the console only.

## Quick start

> Always start the app from the project folder. Paths to models, voices and `config.json` are relative to the folder the app is started from.

### PC (development)

Run `main.py` directly. (`launcher.py` calls `python3`, which often isn't available on Windows.) `--platform` defaults to `PC`, and the backend is always CPU there.

```bash
# One webcam
python main.py --source Camera --camera_device 0

# Two webcams
python main.py --source Camera --camera_device 0 --source_2 Camera --camera_device_2 1

# Webcam plus a phone camera (DroidCam)
python main.py --source Camera --camera_device 0 --source_2 Video --video_path_2 http://<phone-ip>:4747/video

# Two video files (they loop)
python main.py --source Video --video_path path/to/cam1.mp4 --source_2 Video --video_path_2 path/to/cam2.mp4
```

Then open **http://localhost:5050**.

If you don't pass `--source_2`, camera 2 uses the phone-camera URL set in `config.py`. If that camera isn't available, turn **Camera 2** off in the dashboard (Camera Tuning tab). Camera 1 keeps running either way.

OpenCV preview windows only open when a display is detected: a connected monitor on Linux, or `DISPLAY` or `WAYLAND_DISPLAY` set. Otherwise the app runs without windows, and you watch it through the dashboard. To stop, press `q` or `Esc` in a preview window, or `Ctrl+C` in the terminal.

### Board (i.MX93 / i.MX95)

1. Copy the project to `/root/metro` on the board.
2. Set the camera sources in `config.py`: `DETECTION_ON`, `CAMERA_DEVICE`, `DETECTION_ON_2`, `CAMERA_DEVICE_2` and `VIDEO_PATH_2`. `launcher.py` only passes the platform and backend to `main.py`, so all other settings come from `config.py`.
3. On i.MX95 only, put the Neutron models in place (see [Models](#models)).
4. Start the app:

   ```bash
   cd /root/metro
   python3 launcher.py
   ```

   The launcher detects the board, compiles any missing Vela models (i.MX93), then starts `main.py`. The dashboard is at `http://<board-ip>:5050`, and the address is printed at startup.

### Start on boot (systemd)

```bash
cp /root/metro/startup-control.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now startup-control.service
journalctl -u startup-control.service -f    # follow the logs
```

The unit runs `/root/metro/startup_control.sh` and restarts it 3 s after any exit. The script waits 5 s, then runs `launcher.py` from `/root/metro2` if that folder exists, or from `/root/metro` otherwise. The script itself must stay at `/root/metro/startup_control.sh`, even when the code runs from `/root/metro2`.

### Bench network setup

In a typical bench setup, the board is cabled to a host PC that shares its network connection, with the host at `10.42.0.1`.

- Running `python3 static_ip.py` on the board sets its Ethernet address to `10.42.0.2/24`, with gateway `10.42.0.1`. Edit the constants at the top of the file first if your network is different. The dashboard is then at `http://10.42.0.2:5050`.
- If a phone or IP camera is on a network the board can't reach, run `mjpeg_relay.py` on the host (see [Camera sources](#camera-sources)).

## Command-line options

`main.py` takes these options. Their defaults come from `config.py`.

| Option | Values | Default | Purpose |
|---|---|---|---|
| `--source` | `Camera`, `Video` | `Camera` | Type of input for camera 1 |
| `--camera_device` | index or device node | `0` | Camera 1 device for `--source Camera`, e.g. `/dev/video0` |
| `--video_path` | file, index or URL | `videos/Office.mp4` | Camera 1 input for `--source Video` |
| `--source_2` | `Camera`, `Video`, `""` | `Video` | Type of input for camera 2; `""` means same as camera 1 |
| `--camera_device_2` | index or device node | `/dev/video2` | Camera 2 device for `--source_2 Camera` |
| `--video_path_2` | file, index or URL | `http://10.227.155.26:4747/video` | Camera 2 input for `--source_2 Video` |
| `--platform` | `PC`, `i.MX93`, `i.MX95` | `PC` | Target hardware (`launcher.py` sets it) |
| `--backend` | `NPU`, `CPU` | `NPU` | Inference backend (`launcher.py` sets it; always `CPU` on a PC) |

`videos/Office.mp4` is not included in the repository. With `--source Video`, pass `--video_path` yourself.

`mjpeg_relay.py <host:port> [--listen ADDR] [--port N] [--quiet]` forwards a local port to a camera. `--listen` defaults to `0.0.0.0`, and `--port` defaults to the camera's port.

## Camera sources

**`Camera`** opens a local device through V4L2 (Linux), DirectShow (Windows) or AVFoundation (macOS). Give it an index (`0`) or a device node (`/dev/video0`). The app requests 640 × 480 at 30 fps (`FRAME_WIDTH`, `FRAME_HEIGHT`, `FRAMERATE`).

**`Video`** opens anything OpenCV can read from a path or URL:

- a video file, which loops forever
- a camera index
- an `http://` or `https://` MJPEG stream, such as DroidCam at `http://<phone-ip>:4747/video`
- an `rtsp://` IP camera

**Network cameras** are treated as live cameras:

- They connect in the background, so a camera that's offline never holds up the other one.
- The app tries FFmpeg first, then several GStreamer pipelines (many board builds of OpenCV only have GStreamer), then OpenCV's default. It remembers whichever works.
- RTSP connects over TCP with 5-second timeouts. This is set by `RTSP_FFMPEG_OPTIONS`, and an `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable overrides it.

**Reconnects.** If a live source sends no frames for 5 s, the app treats it as disconnected and retries every 3 s. Meanwhile, its tile shows "Connecting..." or "Camera Not Connected".

**Camera 2** is optional. Whether it starts on comes from `camera2_enabled` in `config.json`, with `CAMERA_2_ENABLED` in `config.py` as the fallback. The dashboard toggle takes effect immediately. If camera 2's source is invalid at startup (for example, a missing file), camera 1 runs alone and camera 2's status reads "Camera 2 Source Invalid".

**Phone cameras on another network.** DroidCam serves only one viewer at a time. If the board has no route to the phone, run the relay on a host that is on both networks. For example, the board may be on the host's shared cable while the phone is on office Wi-Fi.

```bash
# On the host
python3 mjpeg_relay.py 10.227.155.26:4747 --listen 10.42.0.1

# On the board, to check the relay is reachable
curl -sI --max-time 5 http://10.42.0.1:4747/video
```

Then point camera 2 at the host in `config.py`:

```python
VIDEO_PATH_2 = "http://10.42.0.1:4747/video"
```

The relay forwards raw TCP, so it works for RTSP cameras too. It has no authentication, so bind it to the board-facing address, as shown above.

## Web dashboard

Open `http://<device-ip>:5050` in a browser on the same network.

**Camera tiles.** Each active camera has a live feed with its own safety line.

- Drag **P1** and **P2** to lay the line along the platform edge.
- Drag **ALERT** to any point on the track side. People on that side of the line trigger alerts.
- Click **Save Line**.

The live feed marks what the app sees:

- **Yellow line:** the safety line currently in use
- **Green boxes:** people
- **Red boxes:** people in the alert zone
- **Blue boxes:** trains
- **Top-left corner:** the display frame rate, and the inference frame rate (`Inf`)

**Top bar.** Shows a status indicator for each camera: Live, Blur / Obstructed, Black Frame, Connecting... or Camera Not Connected. It also shows the schedule state and the board's clock. The strip under the feeds summarises the current settings.

**Settings sidebar.** Press `Ctrl+B` to show or hide it.

| Tab | Settings |
|---|---|
| Camera Tuning | Camera 2 on/off. Brightness, contrast, exposure, saturation and gamma (−100 to 100) for the camera picked in the Cam 1 / Cam 2 selector. Reset button. |
| Alert & Audio | Alert cooldown (10–300 s). Alert volume (0–200 %). Voice languages: English, Hindi, Regional (with none ticked, alerts appear in the console only). |
| Scheduling | No-detection window (From / To). Sync Device Clock. Manual time. Sync Device Time Now. |
| Detection Tuning | Person and Train confidence thresholds (0.05–1.00). |

**How saving works**

- Every change applies immediately and is written to `config.json`.
- **Save Line** and **Save Settings** both save everything (both cameras and all settings) and also copy it to `config_backup.json`.
- If you stop editing for 30 s without saving, the dashboard restores `config_backup.json`, and your unsaved changes are lost.

> **Security:** the dashboard has no login, accepts connections on every network interface, and serves any file in the project folder. Only make port 5050 reachable from a trusted network.

**API endpoints** (for scripts and integrations):

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/video_feed`, `/video_feed2` | MJPEG live feed for camera 1 or camera 2 |
| GET | `/get_lines` | All settings, with lines scaled to the current camera resolution, plus camera health |
| GET | `/camera_status` | Health of each camera, camera 2 on/off, schedule state, device time |
| POST | `/update_lines` | Apply settings. Writes `config.json`, and `config_backup.json` unless `"save_both": false` |
| POST | `/preview_lines` | Apply settings in memory only |
| POST | `/restore_backup` | Copy `config_backup.json` back over `config.json` |
| POST | `/set_device_time` | `{"datetime": "YYYY-MM-DD HH:MM:SS"}` sets the app clock |

## Configuration

Settings live in two files:

- **`config.py`** holds settings read once at startup: camera sources, models, detection tuning, audio processing, clock and debug options. Restart the app after editing it.
- **`config.json`** holds the settings the dashboard reads and writes while the app runs. It has global values plus one block per camera.

> Only edit `config.json` by hand while the app is stopped. The running server keeps its own copy in memory and overwrites the file on the next dashboard change. Also copy your edited file over `config_backup.json`, or the dashboard's 30-second auto-restore may bring back the old values.

### `config.json`

```json
{
  "camera2_enabled": true,
  "person_conf_th": 0.5,
  "train_conf_th": 0.45,
  "alert_cooldown_seconds": 30,
  "alert_volume": 100,
  "tts_languages": ["english", "hindi"],
  "tts_rate": 160,
  "train_off_start": "23:30",
  "train_off_end": "05:00",
  "cameras": {
    "cam1": {
      "p1": [0, 240], "p2": [639, 240], "alert_side_pt": [320, 400],
      "threshold_pixels": 0.0, "frame_width": 640, "frame_height": 480,
      "brightness": 0, "contrast": 0, "exposure": 0, "saturation": 0, "gamma": 0
    },
    "cam2": {
      "p1": [0, 240], "p2": [639, 240], "alert_side_pt": [320, 400],
      "threshold_pixels": 0.0, "frame_width": 640, "frame_height": 480,
      "brightness": 0, "contrast": 0, "exposure": 0, "saturation": 0, "gamma": 0
    }
  }
}
```

Global keys:

| Key | Meaning | Where to change it |
|---|---|---|
| `camera2_enabled` | Camera 2 on or off | Camera Tuning |
| `person_conf_th`, `train_conf_th` | Detection confidence thresholds (minimum 0.05) | Detection Tuning |
| `alert_cooldown_seconds` | Minimum time between alerts from the same camera | Alert & Audio |
| `alert_volume` | Voice volume in percent, 0–200. Above 100, audio starts to distort | Alert & Audio |
| `tts_languages` | Any combination of `"english"`, `"hindi"` and `"regional"` | Alert & Audio |
| `train_off_start`, `train_off_end` | No-detection window as `"HH:MM"`. It can cross midnight, and equal times turn it off | Scheduling |
| `tts_rate` | Not used; kept for compatibility with older files | — |

Per-camera keys, under `cameras.cam1` and `cameras.cam2`:

| Key | Meaning | Where to change it |
|---|---|---|
| `p1`, `p2` | Ends of the safety line, in frame pixels | Drag handles |
| `alert_side_pt` | Any point on the alert (track) side of the line | Drag handle |
| `threshold_pixels` | Buffer distance: a person within this many pixels of the line counts as in the zone, on either side | **`config.json` only** |
| `frame_width`, `frame_height` | Resolution the points were saved at. If the camera's resolution changes, the points are scaled to match | Automatic |
| `brightness`, `contrast`, `exposure`, `saturation`, `gamma` | Image adjustments (−100 to 100), applied before detection | Camera Tuning |

Older single-camera `config.json` files (no `cameras` block) are converted automatically, and their line and image settings become camera 1's.

### `config.py`

These are the settings you're most likely to change:

| Setting | Default | Purpose |
|---|---|---|
| `DETECTION_ON`, `CAMERA_DEVICE`, `VIDEO_PATH` | `"Camera"`, `0`, `"videos/Office.mp4"` | Default source for camera 1 |
| `DETECTION_ON_2`, `CAMERA_DEVICE_2`, `VIDEO_PATH_2` | `"Video"`, `"/dev/video2"`, a DroidCam URL | Default source for camera 2 |
| `CAMERA_2_ENABLED` | `True` | Camera 2 state when `config.json` doesn't set it |
| `BACKEND` | `"NPU"` | Set to `"CPU"` to run the uncompiled models on the board |
| `MODELS` | see [Models](#models) | Model file names |
| `FRAME_WIDTH`, `FRAME_HEIGHT`, `FRAMERATE` | `640`, `480`, `30` | Capture mode requested from local cameras |
| `RUN_MODEL_ON_EVERY_N_FRAME` | `2` | Run inference on every Nth captured frame |
| `INFER_TURN_TIMEOUT_S` | `1.0` | Give the inference turn to the other camera after this long |
| `LINE_DETECTION_ENABLED` | `True` | Set to `False` to alert on any detected person, with no safety line |
| `SEGMENTATION_ENABLED` | `True` | Loads the pose and clothing models. Leave this on: single-person alerts need a colour, and measuring colour needs the pose model |
| `VOTE_WINDOW_FRAMES` | `2` | Number of frames with a measured colour needed before an alert |
| `MIN_GARMENT_CONFIDENCE` | `0.60` | Below this confidence, no garment is named |
| `ALERT_EXCLUDED_COLORS` | orange, brown, beige | Colours treated as "no colour", because they are often skin or station fittings |
| `POSE_FALLBACK_ENABLED` | `False` | If `True`, colour is measured on the upper body instead after 2 pose failures in a row |
| `TRAIN_SUPPRESS_DELAY_S`, `TRAIN_RESUME_DELAY_S` | `5.0`, `1.0` | How long a train must be present before alerts pause, and how soon they resume after it leaves |
| `IOU_TH`, `NMS_CONTAINMENT_TH` | `0.20`, `0.90` | Thresholds for removing duplicate boxes |
| `BOX_WIDTH_SCALE`, `BOX_HEIGHT_SCALE` | `1.2`, `1.0` | Enlarge detected boxes before the line check and cropping |
| `CAMERA_HEALTH_CHECK_STRIDE` | `3` | Check camera health every Nth frame |
| `CAMERA_RECONNECT_INTERVAL_S`, `CAMERA_DISCONNECT_TIMEOUT_S` | `3.0`, `5.0` | How often to retry a camera, and how long without frames counts as disconnected |
| `NPU_INFER_TIMEOUT_S` | `2.0` | Skip the frame if a model doesn't respond within this time |
| `CORE_CPU_ID`, `CORE_NPU_ID` | `0`, `1` | CPU cores that threads are pinned to (Linux only) |
| `ALERT_CLIP_*` | — | Silence trimming, the pause between words, and volume levelling for voice clips |
| `TIME_SOURCE` | `"virtual"` | See [Clock and schedule](#clock-and-schedule) |
| `DEBUG` | `False` | Extra logging and debug images |

## Models

| Role | File (`MODELS` key) | Notes |
|---|---|---|
| Detector | `wificamera.tflite` (`yolo`) | YOLO. `CLASS_NAMES` maps class `0` to Person, and classes `1` and `6` to Train |
| Pose | `yolov8n-pose_fiq.tflite` (`pose`) | Finds shoulder and hip points for the torso crop |
| Clothing | `clothing_classifier_int8.tflite` (`clothing`) | 224 × 224 int8 classifier. Class names are in `models/labels.txt` |

The other `.tflite` files in `models/` are alternatives: `yolo11s_fiq`, `yolo11m_fiq_`, `yolo11l_fiq_`, `newyolo11l_full_integer_quant`, `unfreeze_clothing_classifier_int8` and `model`. They are only loaded if you name them in `MODELS`. If you switch to a detector trained on the standard COCO dataset, remove `1` from `CLASS_NAMES`, because class 1 in COCO is *bicycle*.

The board type decides which version of each model is loaded:

| Platform | Detected by | Backend | Files loaded |
|---|---|---|---|
| i.MX93 | `/usr/lib/libethosu_delegate.so` exists | NPU (Ethos-U65) | `models/<name>_vela.tflite` |
| i.MX95 | `/usr/lib/libneutron_delegate.so` exists | NPU (Neutron) | `models/<name>_neutron.tflite` |
| PC, or `BACKEND = "CPU"` | — | CPU | `models/<name>.tflite` |

- **i.MX93:** if a model's `_vela` file is missing, `launcher.py` compiles it with `vela` into `models/`. After replacing a model, delete its `_vela` file so it gets recompiled.
- **i.MX95:** the launcher doesn't create these files; it only warns when they are missing. The converted models are currently in `models/neutron converted/` under different names. Copy them into place:

  ```bash
  cd /root/metro/models
  cp "neutron converted/wificamera_converted.tflite"               wificamera_neutron.tflite
  cp "neutron converted/yolov8n-pose_fiq_converted.tflite"         yolov8n-pose_fiq_neutron.tflite
  cp "neutron converted/clothing_classifier_int8_converted.tflite" clothing_classifier_int8_neutron.tflite
  ```

- To use a different NPU driver library (delegate), set the `TFLITE_DELEGATE_PATH` environment variable. If the delegate can't load, the app logs `delegate load failed` and tries the CPU. NPU-compiled models usually don't run on the CPU, so treat that log line as an error.

## Voice alerts

Alerts are built from pre-recorded WAV clips in `voices/<language>/`. The language folder must be `english`, `hindi` or `regional`. Use `regional` for whichever local language the site needs.

| Clip | Content | Used for |
|---|---|---|
| `Attention.wav` | "Attention! Person in …" | Single-person alert |
| `<Colour>.wav` | The colour name | Single-person alert |
| `<Garment>.wav` | The garment name | Single-person alert, only when a garment was identified |
| `Away.wav` | "… please step back from the yellow line." | Single-person alert |
| `General.wav` | "Attention! Multiple people near the yellow line, please step back." | Several people in the zone, or a backed-up audio queue |

**File names must match exactly, including capitalisation:**

- **Colours:** the name with each word capitalised and spaces removed: `Black`, `White`, `Gray`, `Cream`, `Pink`, `Red`, `Yellow`, `Khaki`, `Green`, `Cyan`, `SkyBlue`, `Navy`, `Blue`, `Purple`. Orange, brown and beige are excluded by default and never spoken.
- **Garments:** the label from `models/labels.txt` with spaces and punctuation removed: `Blazers`, `Jackets`, `RainJacket`, `Shirts`, `Sweaters`, `Sweatshirts`, `Tops`, `Tshirts`, `Tunics`.

**How playback works**

- **Sequence.** For each ticked language in turn, the app plays Attention → colour → garment → Away as one phrase through `aplay`.
- **Audio processing.** Before playback, silence is trimmed from each clip and a 90 ms pause is added between words. Quiet clips are boosted to a consistent level, then the dashboard volume is applied.
- **Clip format.** All clips in a language should have the same sample rate, bit depth and number of channels. If they don't, the app plays them one at a time, with audible gaps.
- **All or nothing.** If any clip an alert needs is missing, that language stays silent for that alert. The missing file is logged once as `[Alert] Skipping alert -- missing voice clip ...`.
- **One queue.** Alerts from both cameras share one audio queue, so they never overlap. If a camera raises a new alert while one of its earlier alerts is still waiting to play, those waiting alerts are replaced by a single `General.wav` alert.
- **Person still there?** Before starting each language, the app checks that the person is still in that camera's zone. If they've left, playback stops.
- **Console output.** Every alert is also printed to the console:

  ```
  ------------------------------------------------------------
  14:32:07 [Cam 1] Message : Attention! Person in red top, please step back from the yellow line.
  ------------------------------------------------------------
  ```

  When a garment is identified, its name replaces "top", for example "blue Shirts".

## Clock and schedule

The boards have no battery-backed clock and usually no internet, so the app keeps its own clock (`TIME_SOURCE = "virtual"`):

- **Syncing.** The dashboard sends the browser's time to the board:
  - automatically every 5 minutes, while **Sync Device Clock** is on and the dashboard is open
  - immediately, when you click **Sync Device Time Now**

  To enter a time by hand, turn **Sync Device Clock** off first.
- **Restarts.** The last sync is saved in `time_sync_state.json`, so the clock survives an app restart.
- **Reboots.** After a reboot, the clock restarts from the last synced time and will be behind. Click **Sync Device Time Now** after every reboot.
- **Top-bar clock.** Shows the board's time, not the browser's.
- **What uses this clock.** The no-detection window, alert timestamps and the schedule indicator. During the window, the feeds keep streaming but detection and alerts stop.

To use the operating system's clock instead, set `TIME_SOURCE = "system"`. Syncing then runs `date -s`, which only works on Linux and needs root.

## Debugging

Setting `DEBUG = True` in `config.py` enables:

- **Extra logs:** detector output, frame rates for each camera, and the clock and schedule state every 5 s. Also which method opened each network camera, and when trains pause or resume alerts.
- **Images for each alert** in `debug_crops/`:

  | File | Contents |
  |---|---|
  | `*_1_crop.jpg` | The person |
  | `*_2_upper.jpg` | The region the garment classifier sees |
  | `*_3_pose_points.jpg` | Detected pose points |
  | `*_4_torso.jpg` | The region used for colour |
  | `*_5_model_input.jpg` | The exact image given to the classifier |

- **Images for every pose attempt**, in `debug_crops/pose_detected/` and `debug_crops/pose_failed/`. Failed attempts have the reason drawn on the image.

These images are saved without any limit, so only use `DEBUG` for short sessions.

## Troubleshooting

| Problem | What to check |
|---|---|
| Someone crosses the line but there's no alert at all | 1. Detection may be paused: the schedule indicator shows Off-Hours, or the camera isn't Live. 2. A train may be on the alert side, or the camera may be in its cooldown. 3. Otherwise, no colour could be measured. Pose detection needs both shoulders and both hips visible, and orange, brown and beige don't count. With `DEBUG = True`, look in `debug_crops/pose_failed/`, or try `POSE_FALLBACK_ENABLED = True`. |
| An alert appears in the console but there's no sound | 1. Look for `missing voice clip` in the log (see [Known issues](#known-issues)). 2. Check that at least one language is ticked, that the volume isn't 0, and that `aplay voices/english/General.wav` plays. 3. The person may have left the zone before playback started. 4. Voice playback only works on Linux. |
| `[FATAL] Detector: 'yolo' model not loaded on NPUWorker` | The model file for this platform is missing (see [Models](#models)). Under systemd, this error repeats every few seconds. |
| A network camera works on a PC but not on the board | The log line `could not open '<url>' -- tried: ...` lists what was tried. Test from the board with `curl -sI --max-time 5 <url>`. If the camera can't be reached, use `mjpeg_relay.py`. DroidCam accepts only one viewer at a time. |
| Camera 2 status says "Camera 2 Source Invalid" | Fix `VIDEO_PATH_2` or the `--video_path_2` file path, then restart. |
| A camera status says Blur / Obstructed or Black Frame | Clean or uncover the lens, or adjust brightness and exposure. The thresholds are in `check_camera_health()` in `line_config_server.py`: Black means average brightness below 15, and Blur means low sharpness or contrast. |
| One person triggers the "multiple people" alert | The same person was probably detected twice. Lowering `NMS_CONTAINMENT_TH` or `IOU_TH` removes more duplicates, but setting them too low can merge people standing close together. |
| Dashboard changes undo themselves | Changes that aren't saved within 30 s are rolled back. Click Save. |
| Detection turns off at the wrong times | The board's clock is wrong. Click **Sync Device Time Now**, then check the top-bar clock. |
| Low frame rate or lag | Compare the display and `Inf` frame rates on the feed. Check the log for `loaded delegate` to confirm the NPU is in use. Then raise `RUN_MODEL_ON_EVERY_N_FRAME` or turn camera 2 off. |
| Dashboard won't load | Use the address printed at startup (port 5050). On a bench cable setup, `static_ip.py` sets the board to `10.42.0.2`. |

## Known issues

As of 17 September 2026:

- **Voice clips are missing for many alerts.**
  - **Present:** `Attention`, `Away`, `General`; eight colours (`Black`, `Blue`, `Gray`, `Navy`, `Pink`, `Red`, `White`, `Yellow`); and `Coat`, `Saree`, `Shirt`, `Tshirt`.
  - **Missing colours:** `Cream`, `Cyan`, `Green`, `Khaki`, `Purple`, `SkyBlue`.
  - **Missing garments:** all nine. The classifier's labels are plural (`Shirts`, `Tshirts`, …), so `Shirt.wav` and `Tshirt.wav` never match, and `Coat` and `Saree` aren't in `labels.txt`.

  Because playback is all-or-nothing, an alert is only spoken if its colour is one of the eight above **and** no garment was identified. All other alerts appear in the console only. To fix this, record the missing clips, or rename the clips or the labels so they match.
- **i.MX95 model names don't match.** `config.py` expects `models/*_neutron.tflite`, but the converted files are `models/neutron converted/*_converted.tflite`. Copy them as shown in [Models](#models).




