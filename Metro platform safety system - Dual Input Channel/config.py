"""
Shared configuration and small utilities for the line-zone detector.
"""

import ctypes
import json
import os
import platform
import threading
import time as _time
from datetime import datetime, time as _dtime, timezone as _timezone
from pathlib import Path

# Runtime defaults. CLI arguments in main.py can override these.
DETECTION_ON     = "Camera"  # "Camera" for Linux/v4l2, "Video" for OpenCV.
VIDEO_PATH       = "videos/Office.mp4"  # Webcam index, video file path, or stream URL. avi runs fine on PC too.
CAMERA_DEVICE    = 0#"/dev/video0"  # v4l2 device node used by the Camera source (opened via cv2/V4L2).

# ---------------------------------------------------------------------
# Camera 2 -- optional second input.
#
# Both cameras are captured independently (one CaptureThread each) and
# both are pushed to the dashboard as their own live feed, but the
# detect -> vote -> alert pipeline ALTERNATES between them: one inference
# frame from cam1, the next from cam2, then cam1 again, and so on. Each
# camera therefore keeps its FULL field of view and gets half the
# inference turns. On a board where inference is the bottleneck -- which
# is the case on the i.MX93 -- that means total NPU load is unchanged
# from a single camera and each camera runs at about half the rate it
# would alone. See CaptureWidget._take_infer_turn() in video_widget.py.
#
# CAMERA_2_ENABLED here is only the startup default -- the dashboard's
# "Camera 2" toggle (config.json's "camera2_enabled") turns it on/off
# live, without a restart. If cam2 never opens or disconnects mid-run,
# cam1 keeps detecting on its own and cam2's tile shows the usual
# "Camera Not Connected" placeholder with its own health pill.
# ---------------------------------------------------------------------
CAMERA_2_ENABLED = True

# The two cameras do NOT have to be the same KIND of input. DETECTION_ON
# above picks camera 1's kind; DETECTION_ON_2 picks camera 2's:
#
#   "Camera" -- a local device opened through v4l2/DirectShow. Uses
#               CAMERA_DEVICE_2 below, and the capture resolution is
#               forced to FRAME_WIDTH x FRAME_HEIGHT.
#   "Video"  -- anything OpenCV/FFmpeg opens by path or URL: a file, an
#               rtsp:// IP camera, or an http:// MJPEG phone camera such
#               as DroidCam. Uses VIDEO_PATH_2 below, at whatever
#               resolution the source itself provides.
#   ""       -- follow camera 1 (both cameras the same kind).
#
# So a USB camera on cam1 + a phone/IP camera on cam2 is just
# DETECTION_ON = "Camera" with DETECTION_ON_2 = "Video", which is the
# mixed setup this defaults to.
DETECTION_ON_2   = "Video"
CAMERA_DEVICE_2  = "/dev/video2"  # v4l2 device node for cam2 (DETECTION_ON_2 = "Camera").

# DroidCam serves MJPEG over HTTP -- the path is "/video" (some builds
# also accept "/mjpegfeed?640x480"), NOT rtsp://. Port 4747 is its
# default. Note a DroidCam device only accepts ONE client at a time, so a
# phone reachable at two addresses (e.g. a LAN address and a Tailscale
# 100.x one) is still a single camera, not two.
VIDEO_PATH_2     = "http://10.227.155.26:4747/video"

# Canonical camera ids, in pipeline/round-robin order. "cam1" is always
# present; "cam2" only participates while it's enabled. These same ids are
# the keys of config.json's "cameras" block and the suffix on every
# per-camera API (read_camera_config, set_camera_health, update_frame...).
CAMERA_IDS    = ("cam1", "cam2")
CAMERA_LABELS = {"cam1": "Cam 1", "cam2": "Cam 2"}

# ---------------------------------------------------------------------
# RTSP / network sources.
#
# Either camera can be an IP camera instead of a local device: use
# --source Video (DETECTION_ON = "Video") and give VIDEO_PATH /
# VIDEO_PATH_2 an rtsp:// URL. video_widget already classifies rtsp://,
# http:// and https:// sources as LIVE, so they get the same reconnect
# handling as a USB camera rather than being treated as a finite file.
#
# OpenCV's FFmpeg backend has no usable connect timeout by default:
# measured here, opening an rtsp:// URL whose camera was unreachable
# blocked inside cv2.VideoCapture() for ~109 seconds before giving up.
# These options bound that. `timeout` is the current FFmpeg spelling and
# `stimeout` the older one -- both are passed so this works whichever
# build the board ships; FFmpeg ignores the one it doesn't know. Values
# are in MICROseconds. TCP transport is preferred over the UDP default:
# on a lossy link UDP produces torn/green frames that the blur/black
# camera-health check can't distinguish from a genuinely bad lens.
#
# Set to "" to leave OpenCV's defaults alone. An OPENCV_FFMPEG_CAPTURE_OPTIONS
# already present in the environment always wins, so this can be
# overridden per deployment without editing code.
RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|timeout;5000000|stimeout;5000000"

if RTSP_FFMPEG_OPTIONS and not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS

PLATFORM_CHOICES = ("PC", "i.MX93", "i.MX95")
BACKEND_CHOICES  = ("NPU", "CPU")
BACKEND          = "NPU"  # shared default backend, used by launcher.py and main.py

# ---------------------------------------------------------------------
# Model files. One source of truth per logical model (MODELS), instead
# of hand-writing three near-duplicate constants per variant
# (YOLO_MODEL / YOLO_VELA_MODEL / YOLO_NEUTRON_MODEL, ...) that only ever
# differed by a filename suffix. get_model_path()/get_model_filename()
# derive whichever variant is actually needed:
#   "original" -- plain CPU .tflite (PC, or i.MX + CPU backend)
#   "vela"     -- vela-compiled, for i.MX93's Ethos-U65 NPU
#   "neutron"  -- neutron-converted, for i.MX95's NPU
# ---------------------------------------------------------------------
MODEL_DIR = "models"

MODELS = {
    "yolo": "wificamera.tflite",
    "pose": "yolov8n-pose_fiq.tflite",
    "clothing": "clothing_classifier_int8.tflite",
}

_MODEL_VARIANT_SUFFIX = {
    "original": "",
    "vela": "_vela",
    "neutron": "_neutron",
}

LABELS_PATH = f"{MODEL_DIR}/labels.txt"


def get_model_filename(model: str, variant: str = "original") -> str:
    """Bare filename (no directory) for `model` in the given `variant`."""
    base = MODELS[model]
    suffix = _MODEL_VARIANT_SUFFIX[variant]
    return base.replace(".tflite", f"{suffix}.tflite")


def get_model_path(model: str, variant: str = "original") -> str:
    """MODEL_DIR-relative path for `model` in the given `variant`."""
    return f"{MODEL_DIR}/{get_model_filename(model, variant)}"


# Bare filenames, kept as named constants: launcher.py checks/builds
# paths against its own on-device MODEL_DIR (an absolute
# "/root/metro/models/" path -- see launcher.py -- different from the
# relative MODEL_DIR used here), so it needs just the filename, without
# this module's MODEL_DIR baked in.
YOLO_MODEL     = get_model_filename("yolo")
POSE_MODEL     = get_model_filename("pose")
CLOTHING_MODEL = get_model_filename("clothing")

YOLO_VELA_MODEL     = get_model_filename("yolo", "vela")
POSE_VELA_MODEL     = get_model_filename("pose", "vela")
CLOTHING_VELA_MODEL = get_model_filename("clothing", "vela")

YOLO_NEUTRON_MODEL     = get_model_filename("yolo", "neutron")
POSE_NEUTRON_MODEL     = get_model_filename("pose", "neutron")
CLOTHING_NEUTRON_MODEL = get_model_filename("clothing", "neutron")

# Full paths (MODEL_DIR-relative) -- what everything except launcher.py
# actually loads models from.
YOLO_MODEL_PATH     = get_model_path("yolo")
POSE_MODEL_PATH     = get_model_path("pose")
CLOTHING_MODEL_PATH = get_model_path("clothing")

YOLO_VELA_MODEL_PATH     = get_model_path("yolo", "vela")
POSE_VELA_MODEL_PATH     = get_model_path("pose", "vela")
CLOTHING_VELA_MODEL_PATH = get_model_path("clothing", "vela")

YOLO_NEUTRON_MODEL_PATH     = get_model_path("yolo", "neutron")
POSE_NEUTRON_MODEL_PATH     = get_model_path("pose", "neutron")
CLOTHING_NEUTRON_MODEL_PATH = get_model_path("clothing", "neutron")

DEBUG = False

FRAMERATE       = 30

FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

# ---------------------------------------------------------------------
# Time source: the i.MX93 board has no internet (no NTP) and no
# battery-backed RTC. In theory `date -s` (run from the dashboard's
# "Sync device time now" / line_config_server.py's /set_device_time)
# should fix the OS clock and everything else -- is_schedule_off(), the
# [Clock] print, alert timestamps, the dashboard's polled device time --
# just follows datetime.now() from there. In practice, on this board
# that OS-level set hasn't reliably stuck (something -- a stale
# fake-hwclock timer, systemd-timesyncd, an overlay-fs quirk -- appears
# to silently revert it), so TIME_SOURCE lets the whole app bypass the
# OS clock entirely instead of fighting whatever's resetting it:
#
#   "system"  -- trust the OS system clock. /set_device_time calls
#                `date -s` as before. Use this once the underlying board
#                issue is found/fixed and `date -s` actually holds.
#   "virtual" -- ignore the OS clock. /set_device_time instead anchors
#                an in-app offset to time.monotonic() (which nothing --
#                not `date -s`, not a rogue timesyncd -- can disturb;
#                see the whole time.time()->time.monotonic() fix
#                elsewhere in this codebase for why that distinction
#                matters) and persists it to TIME_SYNC_STATE_PATH so an
#                app restart (not a reboot -- there's no RTC to recover
#                real elapsed downtime from either way) keeps ticking
#                from the right place instead of resetting to zero.
#
# Every place in this codebase that needs "the current time" calls
# get_current_time() below instead of datetime.now() directly, so this
# one flag controls all of them consistently -- schedule window checks,
# the [Clock] print, alert timestamps, and the dashboard's polled clock.
TIME_SOURCE          = "virtual"  # "system" or "virtual"
TIME_SYNC_STATE_PATH = "time_sync_state.json"

# SEGMENT
IOU_TH                  = 0.20

# Extra NMS safety net, on top of the standard IoU check above. Plain IoU
# (intersection / union) misses a real failure mode this pipeline was
# hitting: two boxes for the SAME physical person where one is much
# smaller than the other (e.g. a spurious torso-only box sitting fully
# inside the real full-body box). In that case the union is dominated by
# the big box, so IoU comes out low even though the small box contributes
# nothing new -- both boxes survive NMS, and main.py then sees ">1 person"
# and fires the generic multi-person alert instead of the correct
# single-person one.
#
# NMS_CONTAINMENT_TH is checked as intersection / min(area_i, area_j) --
# i.e. "what fraction of the SMALLER box's own area is covered by the
# other box". A near-total containment ratio (this high) essentially only
# happens when one box is a degenerate duplicate sitting inside the other;
# two genuinely different people standing close together, even shoulder to
# shoulder, essentially never approach this because each still has its own
# uncovered area (the other person's differently-positioned limbs/torso).
# Kept deliberately high/conservative for exactly that reason -- this
# should suppress same-person duplicates, not merge two real people into
# one box.
NMS_CONTAINMENT_TH      = 0.90
BOX_WIDTH_SCALE         = 1.2
BOX_HEIGHT_SCALE        = 1.0

# NOTE: no ALERT_COOLDOWN_SECONDS here -- same reasoning as
# PERSON_CONF_TH above. video_widget.py pushes config.json's
# alert_cooldown_seconds into AlertEngine.cooldown_seconds every frame,
# so main.py just passes AlertEngine's own literal startup default.

SEGMENTATION_ENABLED    = True
LINE_DETECTION_ENABLED  = True

# ---------------------------------------------------------------------
# Multi-language spoken alerts (pre-recorded WAV clips, NOT TTS).
#
# At install time, drop recorded clips FLAT under each language folder --
# no colors/ or garments/ subfolders:
#
#   voices/
#     english/
#       Attention.wav       -- "Attention! Person in"
#       Away.wav            -- "please step back from the yellow line."
#       General.wav         -- standalone multi-person alert, e.g.
#                               "Attention! Multiple people near the
#                               yellow line, please step back." Played
#                               INSTEAD of Attention/Color/Garment/Away
#                               whenever more than one person is in the
#                               zone at once (no single person to
#                               describe), or whenever a new alert would
#                               otherwise queue up behind one already
#                               waiting to play -- see AlertEngine.
#       <Color>.wav         -- one per color name _hsv_to_name() in
#                               segment.py can return, PascalCase
#                               (e.g. Black.wav, SkyBlue.wav)
#       <Garment>.wav       -- one per label in labels.txt, spaces
#                               stripped (e.g. Shirt.wav, TShirt.wav)
#     hindi/      -- same flat file layout, Hindi recordings
#     regional/   -- same flat file layout, whatever regional language
#                    this site needs -- code has no opinion on which one.
#
# Filenames must match EXACTLY (case-sensitive) in every language
# folder -- e.g. TShirt.wav, not TShirt.wav in one folder and
# Tshirt.wav in another. A mismatch is treated as "missing", on
# purpose, so a naming slip surfaces immediately instead of being
# silently patched over.
#
# AlertEngine plays Attention -> [color] -> [garment] -> Away, in that
# order, once per language the dashboard has checked, back to back. The
# garment clip is only expected when a garment WAS confidently
# classified -- there's no generic fallback clip for "garment unknown".
#
# ALL-OR-NOTHING per language: if even ONE clip a given alert needs is
# missing on disk, NONE of that language's clips play -- the missing
# path is logged once (not spammed every alert), see
# AlertEngine._play_language_alert().
# ---------------------------------------------------------------------
VOICES_DIR = "voices" #"/root/metro/voices"

# ---------------------------------------------------------------------
# Concatenated alert playback (AlertEngine._concat_wav_clips) plays
# Attention -> [Color] -> [Garment] -> Away as ONE continuous aplay
# stream to avoid the process-spawn/ALSA-reopen gap between separate
# aplay calls. That alone isn't enough, though: each recorded clip
# almost always has its own bit of dead air baked into the start/end of
# the recording (mic lag, room noise floor before/after the word), and
# raw-concatenating preserves that as an audible gap even inside one
# continuous stream. These two settings control trimming that dead air
# off each clip before joining them, plus a small deliberate pause
# re-inserted between clips so back-to-back words don't run together.
#
#   ALERT_CLIP_SILENCE_TRIM_THRESHOLD -- fraction (0.0-1.0) of a clip's
#       full-scale amplitude below which a sample counts as "silence".
#       Raise it if trimming is leaving audible dead air; lower it if
#       trimming is clipping into the start/end of the spoken word.
#   ALERT_CLIP_TRIM_PAD_MS -- milliseconds of the original silence kept
#       on each side of the trimmed audio, so words aren't clipped
#       abruptly right at the waveform's edge.
#   ALERT_CLIP_GAP_MS -- milliseconds of clean silence inserted BETWEEN
#       clips after trimming, for a natural word-to-word cadence. Set to
#       0 for clips to run directly into each other with no gap at all.
# ---------------------------------------------------------------------
ALERT_CLIP_SILENCE_TRIM_THRESHOLD = 0.03
ALERT_CLIP_TRIM_PAD_MS             = 40
ALERT_CLIP_GAP_MS                  = 90
# Recorded voice clips are rarely leveled consistently with each other --
# some were recorded quieter than others, often well under the format's
# full range on purpose to leave the person recording them some
# headroom. Without normalizing, the volume slider was scaling FROM
# whatever quiet level a given clip happened to be recorded at, so even
# the slider's old maximum (150%) of an already-quiet clip still sounded
# quiet -- there just wasn't much signal there to multiply.
# ALERT_CLIP_NORMALIZE_ENABLED scales each trimmed clip's peak up to
# ALERT_CLIP_NORMALIZE_TARGET (a fraction of full-scale) BEFORE the
# user's volume percentage is applied, so "100%" means "as loud as this
# clip can go without clipping" consistently across every clip, not
# whatever level it happened to be recorded at. Never scales a clip DOWN
# -- a clip already at/above the target is left alone.
ALERT_CLIP_NORMALIZE_ENABLED = True
ALERT_CLIP_NORMALIZE_TARGET  = 0.92

# Spoken-alert volume, as a percentage (0-200). 100 = each (now
# normalized) clip played at its full, non-clipping level. Below 100
# attenuates; above 100 pushes louder still by intentionally clipping the
# peaks -- useful in a noisy environment, but audibly distorts, more so
# the higher it goes. Adjustable live from the dashboard's Alert & Audio
# tab ("alert_volume" in config.json); this is just the fallback used
# before the dashboard has ever saved a value.
#ALERT_VOLUME_DEFAULT = 100

# Language codes the dashboard checkboxes can send. Anything outside this
# set is dropped by line_config_server.py's sanitizer before it ever
# reaches AlertEngine.
TTS_LANGUAGE_CHOICES = ("english", "hindi", "regional")

# Segment / clothing classifier settings
IMG_H                        = 224
IMG_W                        = 224

# How long a Train must be visible on the alert side before person alerts
# are suppressed. Persons crossing before this window expires still trigger alerts.
TRAIN_SUPPRESS_DELAY_S       = 5.0

# How long after the Train disappears from the alert side before person
# alerts are re-enabled. Prevents false alerts firing on people still
# boarding/alighting while the train is leaving.
TRAIN_RESUME_DELAY_S         = 1.0

# Voting buffer — number of *inference frames* to collect before deciding the
# final colour + garment for an alert. Both the HSV colour and the clothing
# classifier run on each inference frame; after this many frames the plurality
# winner for each is used to build the alert description.
# Set to 1 to disable voting and alert on every single inference frame (old behaviour).
VOTE_WINDOW_FRAMES           = 2
MIN_GARMENT_CONFIDENCE       = 0.60   # ignore predictions below this confidence

# ---------------------------------------------------------------------
# Core pinning (i.MX93 = 2x Cortex-A55). One dedicated NPU-inference
# thread on its own core; capture + all cv2/CPU work share the other.
# No effect on platforms without sched_setaffinity (e.g. Windows) --
# _pin_thread() below silently no-ops there.
# ---------------------------------------------------------------------
CORE_CPU_ID = 0   # capture thread + processing thread (cv2, HSV, line-cross, drawing)
CORE_NPU_ID = 1   # NPUWorker only -- yolo/pose/clothing invoke()

# How long the processing thread waits for one NPU result before giving up
# and treating that call as "skip this frame". Keep comfortably above your
# worst observed single invoke() time.
NPU_INFER_TIMEOUT_S = 2.0

# Every Nth captured frame goes through the full YOLO->pose->clothing
# pipeline; the frames in between are still displayed/pushed to the live
# preview (never dropped), just redrawn with the most recent detection
# result instead of running inference again. Set to 1 to run the pipeline
# on every frame (old behaviour).
RUN_MODEL_ON_EVERY_N_FRAME = 2

# With two cameras running, inference turns alternate cam1 -> cam2 ->
# cam1... (see CAMERA_2_ENABLED above). This is how long the loop waits
# for the camera whose turn it is to actually deliver a pipeline frame
# before the OTHER camera takes the turn instead. Without it, a cam that
# stalls, disconnects, or is simply slower than the other would freeze
# inference for both -- every frame from the healthy camera would be
# skipped forever waiting for a turn that never comes back.
INFER_TURN_TIMEOUT_S = 1.0

# Depth of the capture -> processing FIFO. This is a DROP-OLDEST buffer:
# if the processing side falls behind, the oldest unread (frame,
# run_pipeline) tuple is discarded and the newest one takes its slot,
# rather than CaptureThread blocking on a full queue. A live safety
# tripwire wants the freshest frame with bounded latency, not a growing
# backlog of stale ones. Kept small on purpose -- 2 is enough to absorb
# normal timing jitter between capture and processing without letting
# latency build up.
FRAME_QUEUE_MAXSIZE = 2

# How often (in captured frames) the camera-health check (blur/black/
# obstruction detection) actually runs. The check itself is a full-frame
# resize + grayscale conversion + Laplacian variance -- not free on a
# single shared CPU core. A few hundred ms of extra latency before
# noticing a blocked lens is an acceptable trade for not paying that cost
# on every single frame. Set to 1 to check every frame (old behaviour).
CAMERA_HEALTH_CHECK_STRIDE = 3

CAMERA_RECONNECT_INTERVAL_S  = 3.0
CAMERA_DISCONNECT_TIMEOUT_S  = 5.0
POSE_KEYPOINT_CONF_TH = 0.25   # minimum keypoint confidence to accept

# Pose torso crop refinement -- the raw shoulder/hip keypoint bbox lands
# exactly ON the joints, not on the fabric edge, so it's trimmed inward
# before being handed to HSV. Fractions are of the RAW bbox's own
# width/height (not the full person crop).
#   TOP    -- pulls down from the shoulder line, past the collar/neckline
#             (often skin, not shirt fabric)
#   BOTTOM -- pulls up from the hip line, above belt/waistband texture
#   X      -- pulls in from both sides, in case the person isn't
#             perfectly frontal to the camera and the shoulder-to-hip
#             bbox is wider than the actual torso silhouette
TORSO_MARGIN_TOP_FRAC    = 0.10
TORSO_MARGIN_BOTTOM_FRAC = 0.10
TORSO_MARGIN_X_FRAC      = 0.08

# Sanity floor on the RAW keypoint bbox, as a fraction of the person crop's
# own width/height. If the 4 keypoints collapsed into a bbox smaller than
# this (misdetected/occluded keypoints), the box is almost certainly noise
# -- reject it (segment.py then has no colour to vote for this frame)
# rather than average colour over garbage geometry.
TORSO_MIN_WIDTH_FRAC  = 0.15
TORSO_MIN_HEIGHT_FRAC = 0.15

# When pose-based torso extraction fails (low keypoint confidence,
# degenerate bbox, hip-above-shoulder geometry, or a bbox smaller than
# TORSO_MIN_WIDTH_FRAC/TORSO_MIN_HEIGHT_FRAC -- see
# PoseDetector.get_torso_crop()) for this many CONSECUTIVE vote frames in
# a row, SegmentAnalyzer.collect_vote() can fall back to running colour
# detection on the upper-body crop (GARMENT_CROP_UPPER_FRAC -- the same
# crop already computed for the garment classifier) instead of skipping
# that frame's vote entirely. Gated by POSE_FALLBACK_ENABLED below --
# currently OFF, so a missing torso always skips the frame regardless of
# this streak count. Kept here, ready to flip on later.
POSE_FAIL_FALLBACK_STREAK = 2

# Master switch for the fallback described above.
# False (current default): no shoulder points detected -> that frame is
#   always skipped for colour/alert purposes, no exceptions.
# True: after POSE_FAIL_FALLBACK_STREAK consecutive pose failures, colour
#   detection runs on the upper-body crop instead, and a frame with that
#   fallback colour DOES count toward the vote/alert.
POSE_FALLBACK_ENABLED = False

# Colors that should never trigger an alert even when successfully
# detected -- e.g. skin-tone-adjacent colors (orange/brown/beige) that
# tend to be false positives from exposed skin, wood/metal fixtures, or
# station surfaces bleeding into the torso crop, rather than an actual
# garment color worth alerting on. Matched case-insensitively against the
# exact strings _hsv_to_name() can return: "black", "white", "gray",
# "brown", "cream", "beige", "pink", "red", "orange", "yellow", "khaki",
# "green", "cyan", "sky blue", "navy", "blue", "purple". Treated exactly
# like the no-color case for alerting (AlertEngine.trigger() gets
# has_color=False) -- but silently: no debug print, no debug-crop save,
# since the color WAS genuinely detected here, this is a deliberate
# filter rather than a failure worth diagnosing.
ALERT_EXCLUDED_COLORS = [
    "orange",
    "brown",
    "beige",
]

# Crop ratios for clothing classification (configurable)
# Clothing model receives upper body crop (e.g. top 55% of person)
GARMENT_CROP_UPPER_FRAC      = 0.55


# How fast the color vote fades from the torso crop's center toward its
# edges (Gaussian sigma, as a fraction of the crop's width/height). This is
# NOT tuned to any specific scene/object -- it encodes a general assumption
# that for a reasonably-framed person box, the actual garment sits near the
# middle of the torso region, while anything that leaked in from outside
# the person (wall fixture, bag, railing, another person's limb, etc., on
# either side) sits nearer the edges. Smaller = more aggressive suppression
# of edge pixels; larger = closer to the old flat/unweighted average.
TORSO_CENTRALITY_SIGMA_FRAC   = 0.35

# Unified per-detection debug dump: crop + upper region + torso region + the
# exact image handed to the garment classifier, all for one person, saved
# together so they can be compared side by side. Gated by DEBUG, saved on
# every call, no throttling -- meant for short, hands-on debugging
# sessions, not to be left running unattended.
DEBUG_CROPS_DIR                = "debug_crops"

CLASS_NAMES = {
    0: "Person",
    1: "Train",
    6: "Train",
}

CLASS_COLORS_BGR = {
    "Person": (51, 242, 26),
    "Train":  (255, 191, 26),
}

_DEFAULT_COLOR_BGR = (255, 255, 255)

_LABEL_SIZE_CACHE = {}

def resolve_model_paths(platform_name: str, backend: str):
    """Pick which model files to load for a given platform/backend.

    - PC has no NPU: always the plain CPU tflite models, regardless of backend.
    - i.MX93 + NPU : vela-compiled models (Ethos-U65).
    - i.MX95 + NPU : neutron-converted models (eIQ Neutron NPU).
    - i.MX + CPU   : plain tflite models.
    """
    if platform_name == "PC" or backend != "NPU":
        return YOLO_MODEL_PATH, CLOTHING_MODEL_PATH, POSE_MODEL_PATH
    if platform_name == "i.MX95":
        return YOLO_NEUTRON_MODEL_PATH, CLOTHING_NEUTRON_MODEL_PATH, POSE_NEUTRON_MODEL_PATH
    return YOLO_VELA_MODEL_PATH, CLOTHING_VELA_MODEL_PATH, POSE_VELA_MODEL_PATH


# Vendor NPU delegate library for each board. Same paths launcher.py itself
# probes at startup to auto-detect which board it's running on.
NPU_DELEGATE_PATHS = {
    "i.MX93": "/usr/lib/libethosu_delegate.so",
    "i.MX95": "/usr/lib/libneutron_delegate.so",
}

def resolve_npu_delegate_path(platform_name: str, backend: str):
    """Delegate .so path to load for this platform/backend, or None.

    None means "run on plain CPU, no delegate" — returned both when backend
    isn't "NPU" and when the platform has no known delegate mapped (e.g.
    "PC"). Callers should treat both cases identically.
    """
    if backend != "NPU":
        return None
    return NPU_DELEGATE_PATHS.get(platform_name)


# ---------------------------------------------------------------------
# Virtual clock (only used when TIME_SOURCE == "virtual" -- see that
# flag's comment above for why this exists).
#
# State is just two numbers: the wall-clock epoch the operator told us
# was correct at sync time, and the time.monotonic() reading at that
# same instant. "Now" is then always epoch_at_sync + however much
# monotonic time has elapsed since -- monotonic never jumps, gets reset
# by date -s, or gets fought by some other process, so this can't drift
# out from under us the way the OS clock apparently does on this board.
# ---------------------------------------------------------------------
_virtual_clock_lock = threading.Lock()
_virtual_clock_state = {"epoch_at_sync": None, "monotonic_at_sync": None}

def _load_virtual_clock_state():
    """Restore the last-known sync on startup, if any. This survives an
    app restart (monotonic is a system-wide clock that keeps counting
    across process restarts) but NOT a reboot (monotonic resets to ~0 at
    boot) -- detected below by the saved monotonic reading being larger
    than what this fresh boot has reached so far, in which case we keep
    the last-known wall time but re-anchor it to *this* boot's monotonic
    clock (best available guess with no RTC; the alternative is starting
    from nothing at all, which is worse).
    """
    global _virtual_clock_state
    try:
        if not os.path.exists(TIME_SYNC_STATE_PATH):
            return
        with open(TIME_SYNC_STATE_PATH) as f:
            saved = json.load(f)
        now_mono = _time.monotonic()
        if saved.get("monotonic_at_sync", 0) > now_mono:
            _virtual_clock_state = {
                "epoch_at_sync": saved.get("epoch_at_sync"),
                "monotonic_at_sync": now_mono,
            }
        else:
            _virtual_clock_state = saved
    except Exception as exc:
        print(f"[Config] Could not load virtual clock state: {exc}")


def set_virtual_clock(epoch_seconds: float) -> None:
    """Anchor the virtual clock to a real wall-clock value right now.
    Called from line_config_server.py's /set_device_time when
    TIME_SOURCE == "virtual", instead of shelling out to `date -s`.
    """
    global _virtual_clock_state
    with _virtual_clock_lock:
        _virtual_clock_state = {
            "epoch_at_sync": float(epoch_seconds),
            "monotonic_at_sync": _time.monotonic(),
        }
        state = dict(_virtual_clock_state)
    try:
        tmp = TIME_SYNC_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, TIME_SYNC_STATE_PATH)
    except Exception as exc:
        print(f"[Config] Could not persist virtual clock state: {exc}")


def get_current_time() -> datetime:
    """The single source of truth for 'what time is it' across this
    whole app -- respects TIME_SOURCE. Everywhere that used to call
    datetime.now() directly (schedule checks, the [Clock] print, alert
    timestamps, the dashboard's polled device time) calls this instead,
    so the one TIME_SOURCE flag controls all of them consistently.
    """
    if TIME_SOURCE != "virtual":
        return datetime.now()
    with _virtual_clock_lock:
        state = dict(_virtual_clock_state)
    if state["epoch_at_sync"] is None:
        # Never synced this boot yet -- fall back to the (possibly wrong)
        # OS clock, since that's still the least-wrong thing available
        # until the operator hits "Sync device time now" once.
        return datetime.now()
    elapsed = _time.monotonic() - state["monotonic_at_sync"]
    # utcfromtimestamp(), NOT fromtimestamp(): epoch_at_sync was encoded by
    # set_device_time()/set_virtual_clock() by treating the synced wall-clock
    # digits as UTC (see line_config_server.py's /set_device_time), on
    # purpose, so that this whole round-trip never depends on the board's
    # OS timezone setting. fromtimestamp() would instead reinterpret the
    # epoch through the board's *local* system timezone -- on this board
    # that's a no-op (board TZ = UTC) so it happened to look right here,
    # but it's silently wrong on any board configured to a non-UTC
    # timezone, and it's exactly what caused the topbar clock's epoch-based
    # display to double-apply a timezone offset. utcfromtimestamp() decodes
    # with the same fixed UTC convention every encode call now uses, so the
    # two sides always cancel out correctly regardless of board TZ.
    # fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None), NOT
    # utcfromtimestamp(): behaviorally identical (decode via UTC, return a
    # naive datetime) but utcfromtimestamp() is deprecated as of Python
    # 3.12, which is what this board runs.
    return datetime.fromtimestamp(state["epoch_at_sync"] + elapsed, tz=_timezone.utc).replace(tzinfo=None)


_load_virtual_clock_state()


_schedule_window_cache = {"key": None, "start_t": None, "end_t": None}


def _parse_schedule_window(start_str: str, end_str: str):
    """Parse (and cache) the "HH:MM" start/end strings into _dtime objects.

    is_schedule_off() below is called on every processed frame from
    video_widget.py (30-60x/sec), but start_str/end_str only actually
    change when the operator edits the Scheduling tab -- re-splitting,
    re-int()-ing, and re-constructing _dtime objects from the exact same
    two strings every single frame is pure waste at that call rate.
    Single-entry cache, same drop-and-rebuild-once pattern as
    video_widget.py's gamma LUT cache: as soon as either string changes,
    the old entry is dropped and reparsed once, then reused every frame
    until it changes again.

    Raises ValueError/TypeError exactly like the old inline parsing did --
    callers still need their own try/except around this.
    """
    key = (start_str, end_str)
    if _schedule_window_cache["key"] == key:
        return _schedule_window_cache["start_t"], _schedule_window_cache["end_t"]

    sh, sm = (int(p) for p in str(start_str).split(":")[:2])
    eh, em = (int(p) for p in str(end_str).split(":")[:2])
    start_t = _dtime(sh, sm)
    end_t = _dtime(eh, em)

    _schedule_window_cache["key"] = key
    _schedule_window_cache["start_t"] = start_t
    _schedule_window_cache["end_t"] = end_t
    return start_t, end_t


def is_schedule_off(start_str: str, end_str: str, now: _dtime = None) -> bool:
    """True if the given local time falls inside the [start_str, end_str)
    no-detection window (both "HH:MM" strings, same fields the dashboard's
    Scheduling tab writes to config.json as train_off_start/train_off_end).

    Handles the overnight case where start > end (e.g. "23:30" -> "05:00")
    by wrapping past midnight. start == end disables the window entirely
    (never off) -- same convention the dashboard's own JS schedule badge
    already uses in index.html's updateScheduleBadge(), kept identical
    here so the on-screen badge and the actual enforced window agree.

    `now` defaults to get_current_time() -- the board's system clock or
    the virtual clock, depending on TIME_SOURCE (see that flag's comment
    above) -- rather than datetime.now() directly, so schedule
    enforcement follows whichever clock the rest of the app is using.
    """
    if not start_str or not end_str:
        return False
    try:
        start_t, end_t = _parse_schedule_window(start_str, end_str)
    except (ValueError, TypeError):
        return False

    if start_t == end_t:
        return False

    now_t = now if now is not None else get_current_time().time()

    if start_t < end_t:
        return start_t <= now_t < end_t
    return now_t >= start_t or now_t < end_t


def monitor_connected() -> bool:
    """Best-effort detection of a physically connected display.

    Checks the kernel DRM connector status first (works even before any
    compositor has started), then falls back to WAYLAND_DISPLAY/DISPLAY
    env vars. If detection itself fails for any reason, assume no
    monitor — same safe "run headless" default as before.
    """
    try:
        drm_dir = Path("/sys/class/drm")
        if drm_dir.is_dir():
            for status_file in drm_dir.glob("*/status"):
                try:
                    if status_file.read_text().strip() == "connected":
                        return True
                except OSError:
                    continue
    except OSError:
        pass

    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def _model_label(model_path: str) -> str:
    stem = Path(model_path).stem
    part = stem.split("_")[0]
    for i, char in enumerate(part):
        if char.isdigit():
            return part[:i].upper() + part[i:]
    return part.upper()


def _pin_thread(core: int):
    """Pin the calling thread to a CPU core on Linux when supported."""
    if platform.system() != "Linux" or not hasattr(os, "sched_setaffinity"):
        return
    try:
        # os.gettid() (Python 3.9+) is the correct, arch-independent way to
        # get this. The previous implementation called the raw gettid
        # syscall via ctypes with a hardcoded number (178), which is only
        # correct on aarch64 — on x86_64 (SYS_gettid=186) it silently pinned
        # the wrong thread ID. Keep the ctypes path only as a fallback for
        # Python <3.9, and look the syscall number up per-architecture
        # instead of assuming aarch64.
        if hasattr(os, "gettid"):
            tid = os.gettid()
        else:
            syscall_numbers = {"x86_64": 186, "aarch64": 178}
            arch = platform.machine()
            if arch not in syscall_numbers:
                raise RuntimeError(f"No known gettid syscall number for architecture '{arch}'")
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            tid = libc.syscall(syscall_numbers[arch])

        os.sched_setaffinity(tid, {core})
        print(f"[CPU] Thread {tid} -> core {core}")
    except Exception as exc:
        print(f"[CPU] Pin to core {core} skipped: {exc}")