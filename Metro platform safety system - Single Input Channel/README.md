# Metro Safety System

Metro Safety System watches a metro platform through one camera and warns people who step past the yellow line toward the track. It detects people and trains, describes the person by the colour of their top and the kind of garment, and plays recorded voice announcements. A browser dashboard lets you draw the line, tune the camera and thresholds, set a no-detection schedule and set the clock.

It runs in two places:

- **i.MX 93 board**: models run on the Ethos-U65 NPU (Vela-compiled TFLite through `libethosu_delegate.so`), the app starts at boot through systemd, and voice alerts play through `aplay`.
- **Windows PC**: the same pipeline runs on the CPU with TensorFlow's TFLite interpreter, from a webcam or a video file. Alerts are printed to the console; there's no voice playback on Windows.

## Contents

- [What it does](#what-it-does)
- [Quick start on Windows](#quick-start-on-windows)
- [Running on the i.MX 93 board](#running-on-the-imx-93-board)
- [Command-line options](#command-line-options)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Voice clips](#voice-clips)
- [Models](#models)
- [How it runs](#how-it-runs)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

---

## What it does

Every second camera frame goes through the pipeline. The frames in between are still shown, with the latest results drawn on them.

1. **Detect** people and trains with the YOLO model (`models\wificamera.tflite`).
2. **Check the line.** A person is *in the zone* when the centre of their box, their feet (bottom centre) or the point three-quarters down the box is on the alert side of the line, or within the buffer distance of it. A train only counts when its centre is on the alert side; other trains are ignored.
3. **Hold off while a train is in.** Once a train has been on the alert side for 5 seconds, person alerts stop. They start again 1 second after it leaves.
4. **One person in the zone:** the pose model finds the torso between shoulders and hips, and its colour is read in HSV. The clothing classifier labels the top 55% of the person's box, and the label is kept when its confidence is at least 0.60. After 2 frames with a readable colour, the most common colour and garment are announced:
   ```
   Attention! Person in black Shirts, please step back from the yellow line.
   ```
5. **More than one person in the zone** for 2 processed frames: a general alert, `Attention! Multiple people near the yellow line, please step back.`
6. **Cooldown:** after any alert, no new alert fires until the cooldown set on the dashboard has passed.

Detection pauses, while the video keeps streaming, when:

- the camera looks **black** (average brightness below 15) or **blurred or covered** (Laplacian variance below 20, or brightness spread below 8). This is checked every third frame on a 320×240 copy.
- the current time is inside the dashboard's **no-detection window**.
- the camera is **disconnected**, meaning no frames for 5 seconds. The feed then shows *Camera Not Connected*, and the app retries every 3 seconds and reconnects on its own.

---

## Quick start on Windows

### Requirements

- Python 3.12
- `numpy`, `opencv-python`, `flask`, `tensorflow`
- A webcam or a video file

Known-good setup (the models load and run on the CPU): Windows 11, Python 3.12.4, TensorFlow 2.19.0, OpenCV 4.13.0, NumPy 2.1.3, Flask 3.1.2.

```powershell
pip install numpy opencv-python flask tensorflow
```

### Run with the webcam

```powershell
cd "path\to\metro thread"
python main.py
```

- Start it from inside the project folder, because the models, `config.json` and `voices\` are loaded from relative paths.
- It opens webcam 0 at 640×480 and prints the dashboard address. Open **http://localhost:5050** in a browser to see the video.
- Use another camera with `--camera_device 1`.
- **No video window opens on Windows by default**, because display detection only looks for Linux signals. To get one, set `DISPLAY` before starting; then press **q** or **Esc** in the window to quit:
  ```powershell
  $env:DISPLAY = ":0"
  python main.py
  ```
- Without the window, stop the app with **Ctrl+C**.

### Run on a video file

```powershell
python main.py --source Video --video_path "models\METRO LINE 2\WhatsApp Video 2026-07-21 at 1.52.23 PM.mp4"
```

Three sample clips are in `models\METRO LINE 2\`. Video files play at their own frame rate and loop.

If Windows Firewall asks whether Python may accept connections, allow it on private networks so the dashboard can be opened from other devices.

---

## Running on the i.MX 93 board

### Board requirements

- An NXP i.MX 93 Linux image with eIQ: the TFLite runtime (`tflite_runtime`) and the Ethos-U delegate at `/usr/lib/libethosu_delegate.so`
- The `vela` compiler, which `launcher.py` uses to compile the models on first start (skipped when the `_vela.tflite` files already exist)
- Python 3 with `numpy`, OpenCV (`cv2`) and `flask`
- A V4L2 camera (device index 0 by default)
- `aplay` and a speaker, for voice alerts
- `connmanctl`, only for `static_ip.py`

### Deploy

`launcher.py`, the systemd service and the start script all expect **`/root/metro`**. Copy these into it:

```
/root/metro/
├── *.py, index.html, config.json, config_backup.json
├── startup-control.service, startup_control.sh
├── models/wificamera.tflite
├── models/yolov8n-pose_fiq.tflite
├── models/clothing_classifier_int8.tflite
├── models/labels.txt
└── voices/<language>/*.wav        see Voice clips
```

`old meto line\` and the unused models aren't needed on the board.

### Run

```bash
cd /root/metro
python3 launcher.py
```

`launcher.py` recognises the i.MX 93 by its NPU delegate, compiles any missing `models/*_vela.tflite` files with `vela`, then runs:

```bash
python3 /root/metro/main.py --platform "i.MX93" --backend "NPU"
```

If a display is available, a video window opens; press **q** or **Esc** to quit. Otherwise the app runs headless and you use the dashboard.

### Start at boot

```bash
cp /root/metro/startup-control.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now startup-control.service
journalctl -u startup-control.service -f
```

The service waits 10 seconds after boot, runs `launcher.py`, and restarts it 3 seconds after it exits. The last command follows its log.

### Static IP (optional)

`static_ip.py` gives the first Ethernet connection a fixed address through `connmanctl`: **10.42.0.2/24**, gateway 10.42.0.1, DNS 8.8.8.8. Edit the constants at the top of the file first, then run:

```bash
python3 /root/metro/static_ip.py
```

The dashboard is then at `http://10.42.0.2:5050`.

### Setting the clock

The board has no network time and no battery-backed clock, so the no-detection schedule and alert times depend on the time you set from the dashboard: **Scheduling → Sync Device Time Now**.

With `TIME_SOURCE = "virtual"` (the default), the app keeps its own clock in `time_sync_state.json`. It survives app restarts. After a reboot, though, it carries on from the last synced time without counting the time the board was off, so sync again after every reboot.

---

## Command-line options

`main.py`:

| Option | Default | Description |
|---|---|---|
| `--source` | `Camera` | `Camera`: a camera index or device path. `Video`: a file, a camera index, or an `http(s)://` or `rtsp://` stream |
| `--camera_device` | `0` | Camera index, or a device path such as `/dev/video0` (for `--source Camera`) |
| `--video_path` | `videos/Office.mp4` (not included) | File or stream (for `--source Video`) |
| `--platform` | `PC` | `PC` or `i.MX93`; set by `launcher.py` on the board |
| `--backend` | `NPU` | `NPU` or `CPU`; always CPU when the platform is `PC` |

- Cameras open through DirectShow on Windows and V4L2 on the board, at 640×480 and 30 fps.
- `python line_config_server.py` serves the dashboard on its own, without video or detection.

---

## Dashboard

Open `http://<device address>:5050`.

- **Top bar**: camera status (Live, Blur / Obstructed, Black, Not Connected), schedule status, the device clock, and the settings sidebar toggle (`Ctrl+B`).
- **Zone Line**: the live, annotated video. Drag the **P1** and **P2** handles to place the line and the **ALERT** handle onto the track side, then click **Save Line**.
- **Status strip**: the current person and train thresholds, cooldown, volume, voice languages and no-detection window.
- **Settings sidebar** (click **Save Settings** when done):

| Tab | Controls |
|---|---|
| Camera Tuning | Brightness, Contrast, Exposure, Saturation, Gamma (−100 to 100, applied before detection); **Reset Image Adjustments** |
| Alert & Audio | Alert Cooldown (10–300 s), Alert Volume (0–200%; above 100% is louder but distorts), voice languages (English, Hindi, Regional) |
| Scheduling | No-detection window (From/To; an overnight window works), **Sync Device Clock** (on: send this browser's time; off: type a time), **Sync Device Time Now** |
| Detection Tuning | Person and Train confidence thresholds (0.05–1.00) |

- Changes take effect straight away. **Save Line** and **Save Settings** also copy them to `config_backup.json`.
- If you change something and don't save within 30 seconds, the dashboard restores the last saved settings.
- The line buffer (`threshold_pixels`) has no dashboard control; edit it in `config.json`.

---

## Configuration

### `config.json` (written by the dashboard)

| Key | Meaning |
|---|---|
| `p1`, `p2` | Line end points, in pixels |
| `alert_side_pt` | Any point on the track side of the line |
| `threshold_pixels` | Buffer distance around the line that also counts as in the zone |
| `frame_width`, `frame_height` | Resolution the points were saved at; they're rescaled if the camera resolution differs |
| `brightness`, `contrast`, `exposure`, `saturation`, `gamma` | Image adjustments |
| `person_conf_th`, `train_conf_th` | Detection thresholds |
| `alert_cooldown_seconds` | Minimum gap between alerts |
| `alert_volume` | Voice volume, in percent |
| `tts_languages` | Voice languages to play, in order |
| `train_off_start`, `train_off_end` | No-detection window, `HH:MM` |

`tts_rate` is stored but not used.

### `config.py`

| Setting | Default | Meaning |
|---|---|---|
| `MODELS` | wificamera, yolov8n-pose_fiq, clothing_classifier_int8 | Model files in `models/` |
| `RUN_MODEL_ON_EVERY_N_FRAME` | 2 | Run the pipeline on every Nth captured frame |
| `VOTE_WINDOW_FRAMES` | 2 | Frames with a readable colour needed before a single-person alert |
| `MIN_GARMENT_CONFIDENCE` | 0.60 | Minimum clothing classifier confidence |
| `TRAIN_SUPPRESS_DELAY_S` / `TRAIN_RESUME_DELAY_S` | 5.0 / 1.0 | Train hold-off timing, in seconds |
| `ALERT_EXCLUDED_COLORS` | orange, brown, beige | Colours that never trigger an alert |
| `POSE_FALLBACK_ENABLED` | `False` | Read the colour from the upper body after 2 pose failures in a row |
| `IOU_TH`, `NMS_CONTAINMENT_TH` | 0.20, 0.90 | Duplicate-box removal |
| `BOX_WIDTH_SCALE` | 1.2 | Widens detected boxes |
| `TIME_SOURCE` | `virtual` | `virtual` app clock, or `system` to set the OS clock with `date -s` |
| `LINE_DETECTION_ENABLED` | `True` | `False`: every detected person counts, with no line |
| `SEGMENTATION_ENABLED` | `True` | `False`: the pose and clothing models aren't loaded |
| `DEBUG` | `False` | Extra logging; saves person, torso and classifier crops to `debug_crops/` |
| `CORE_CPU_ID`, `CORE_NPU_ID` | 0, 1 | CPU cores for capture/processing and for the models (board only) |
| `NPU_INFER_TIMEOUT_S` | 2.0 | A model call slower than this skips the frame |
| `VOICES_DIR` | `voices` | Voice clip folder |
| `ALERT_CLIP_*` | — | Silence trimming, gaps and loudness normalisation for joined voice clips |

---

## Voice clips

Voice alerts are recorded WAV files, not text-to-speech. `voices\` isn't included, so create it with one folder per language:

```
voices/
├── english/
├── hindi/
└── regional/
```

Every language folder uses the same file names, and names are case-sensitive:

| File | Spoken as |
|---|---|
| `Attention.wav` | "Attention! Person in" |
| `Away.wav` | "please step back from the yellow line." |
| `General.wav` | The multiple-people alert |
| Colour clips | `Black`, `White`, `Gray`, `Cream`, `Pink`, `Red`, `Yellow`, `Khaki`, `Green`, `Cyan`, `SkyBlue`, `Navy`, `Blue`, `Purple` (each `.wav`) |
| Garment clips | `Blazers`, `Jackets`, `RainJacket`, `Shirts`, `Sweaters`, `Sweatshirts`, `Tops`, `Tshirts`, `Tunics` (each `.wav`) |

- A single-person alert plays Attention → colour → garment (when one was recognised) → Away as one stream, with silence trimmed and a 90 ms gap between clips.
- If any clip a language needs is missing, that language is skipped and the missing file is logged once.
- Checked languages play one after another. No further language starts once the person has left the zone.
- Use the same sample rate, bit depth and channel count for every clip in a language (16-bit PCM works best); otherwise the clips are played one at a time, with gaps.
- Orange, brown and beige are never announced, so those clips aren't needed.

---

## Models

| File | Task | Input | Output |
|---|---|---|---|
| `models/wificamera.tflite` | Person (class 0) and Train (class 1) detector | 320×320×3, INT8 | 1×6×2100, INT8 |
| `models/yolov8n-pose_fiq.tflite` | YOLOv8n pose, 17 keypoints | 320×320×3, INT8 | 1×56×2100, INT8 |
| `models/clothing_classifier_int8.tflite` | 9 garment classes | 224×224×3, INT8 | 1×9, INT8 |
| `models/labels.txt` | Garment names: Blazers, Jackets, Rain Jacket, Shirts, Sweaters, Sweatshirts, Tops, Tshirts, Tunics | — | — |

- On the board, `launcher.py` adds `*_vela.tflite` versions next to them for the NPU.
- Also in `models\`, but not used by default: `newyolo11l_full_integer_quant.tflite`, `yolo11l_fiq_.tflite`, `yolo11m_fiq_.tflite`, `yolo11s_fiq.tflite`, `model.tflite`, `unfreeze_clothing_classifier_int8.tflite` and `scrap model\`. To switch models, edit `MODELS` in `config.py`.
- The detector reads YOLOv8-style outputs (box plus class scores per anchor) or standard four-tensor TFLite detection outputs. Class ids are mapped through `CLASS_NAMES`: 0 is Person, and 1 and 6 are Train.

---

## How it runs

```
CaptureThread ──► frame queue (2 frames, drops oldest) ──► processing loop
 (camera or file)                                             │  camera check, image adjustments,
                                                              │  line check, votes, alerts, overlay
                                                              ├──► NPUWorker thread: YOLO, pose, clothing
                                                              ├──► AlertEngine audio thread (aplay)
                                                              └──► dashboard MJPEG stream (Flask, port 5050)
```

- `NPUWorker` is the only thread that runs models. A model call that takes longer than 2 seconds skips that frame.
- On the board, the capture and processing threads share CPU core 0, and `NPUWorker` has core 1.
- Dashboard settings are read on every frame, so changes apply immediately.

---

## Project structure

```
metro thread\
├── launcher.py               Board entry point: detects the board, compiles Vela models, starts main.py
├── main.py                   App: model loading, alert decisions, train hold-off
├── config.py                 Settings, model paths, app clock, schedule helpers
├── video_widget.py           Capture, processing loop, camera checks, overlay
├── npu_worker.py             Single thread that runs all the models
├── channels.py               One-slot mailbox between threads
├── detection.py              YOLO pre- and post-processing, NMS
├── line_detector.py          Line geometry and zone checks
├── segment.py                Colour reading, garment classification, voting
├── pose_detector.py          Torso crop from pose keypoints
├── alert_engine.py           Alerts, cooldown, voice clip playback
├── line_config_server.py     Flask dashboard backend (port 5050)
├── index.html                Dashboard page
├── config.json               Current dashboard settings
├── config_backup.json        Last explicitly saved settings
├── time_sync_state.json      App clock state
├── startup-control.service   systemd unit
├── startup_control.sh        Start script used by the unit
├── static_ip.py              Static Ethernet address through connmanctl
├── models\                   Models, labels, and sample videos in METRO LINE 2\
├── debug_crops\              Debug images when DEBUG is on
└── old meto line\            Earlier version and prototypes (not used)
```

- `old meto line\` holds the previous version of the app, plus `semi-metro line\`, an earlier prototype that used YOLOv8n and SegFormer clothing segmentation (it has its own README).
- `models\METRO LINE 2\` holds the sample videos, `sortimage.py` (entirely commented out) and `v3modelrun.py` (a clothing-classifier test with hard-coded Downloads paths).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `TFLite runtime not installed`, or `Detector: 'yolo' model not loaded` | On Windows, `pip install tensorflow`, and start from the project folder so `models\` is found. On the board, check `tflite_runtime` and the `models/*_vela.tflite` files. |
| No video window on Windows | Expected. Use http://localhost:5050, or set `DISPLAY` (see [Quick start](#run-with-the-webcam)). |
| The dashboard doesn't open from another device | Allow Python through Windows Firewall, and check that port 5050 is reachable. |
| The feed shows **Camera Not Connected** | Check `--camera_device`, and close other apps that use the camera. |
| The camera shows **Blur / Obstructed** on a clean view | Very plain scenes can fail the check. Its limits are in `check_camera_health()` in `line_config_server.py`. |
| Settings jump back after a while | They weren't saved within 30 seconds. Change them again and click **Save**. |
| No voice on the board | Check `voices/<language>/`, look for `missing voice clip` messages in the console, and check that `aplay` can play a WAV file. |
| A single person crossing isn't announced | See [Known issues](#known-issues), item 4. Turn on `DEBUG` to save the crops and see why. |
| Wrong time or schedule after a reboot | **Scheduling → Sync Device Time Now**. |
| `launcher.py` fails on Windows | Run `main.py` directly. |

## Known issues

1. **`launcher.py` doesn't work on Windows.** It calls `python3` and doesn't quote the script path, and this folder's name contains a space.
2. **No video window on Windows** unless `DISPLAY` is set, because display detection only checks Linux signals.
3. **No voice alerts on Windows**, because playback uses `aplay`.
4. **Some people never get a single-person alert.** The alert needs a readable top colour on 2 frames. Tops read as orange, brown or beige are excluded, and frames where the pose model can't find the shoulders don't count (`POSE_FALLBACK_ENABLED = False`). The multiple-people alert doesn't need a colour.
5. **The dashboard has no login**, and its file route serves any file in the project folder (source code, models, `config.json`) to anyone who can reach port 5050.
6. **`voices\` isn't included**, so there's no audio until clips are added.
7. **The default `--video_path` (`videos/Office.mp4`) doesn't exist.**
8. **There's no dashboard control for the line buffer** (`threshold_pixels`).
9. **The comments in `config.py` give example clip names (`Shirt.wav`, `TShirt.wav`) that don't match `labels.txt`.** Use the names in [Voice clips](#voice-clips).
10. **`line_config_server.py` contains an older built-in dashboard page (`_PAGE`) that is never served.** The dashboard is `index.html`.
