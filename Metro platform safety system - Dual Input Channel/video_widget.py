import os
import platform
import queue
import threading
import time
import cv2
import numpy as np

from config import (
    CLASS_COLORS_BGR,
    _DEFAULT_COLOR_BGR,
    _LABEL_SIZE_CACHE,
    DEBUG,
    CORE_CPU_ID,
    RUN_MODEL_ON_EVERY_N_FRAME,
    FRAME_QUEUE_MAXSIZE,
    CAMERA_HEALTH_CHECK_STRIDE,
    CAMERA_RECONNECT_INTERVAL_S,
    CAMERA_DISCONNECT_TIMEOUT_S,
    CAMERA_LABELS,
    INFER_TURN_TIMEOUT_S,
    _model_label,
    _pin_thread,
    monitor_connected,
    is_schedule_off,
    get_current_time,
)
from line_detector import ProximityDetector
from line_config_server import (
    set_frame_resolution,
    update_frame,
    check_camera_health,
    set_camera_health,
    camera2_enabled,
)


# Gamma LUT cache -- gamma only changes when the operator moves the UI
# slider (rare), but apply_image_adjustments() used to rebuild the whole
# 256-entry np.power LUT on every single frame regardless. Single-entry
# cache: as soon as gamma changes, the old entry is dropped and rebuilt
# once, then reused every frame until it changes again.
_gamma_lut_cache = {}


def _get_gamma_lut(gamma: float) -> np.ndarray:
    key = round(gamma)
    lut = _gamma_lut_cache.get(key)
    if lut is None:
        inv_gamma = 2.0 ** (-key / 50.0)  # 0.33 … 3.0 for ±100
        indices = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip(np.power(indices, inv_gamma) * 255.0, 0, 255).astype(np.uint8)
        _gamma_lut_cache.clear()  # bounded to one entry -- gamma rarely changes
        _gamma_lut_cache[key] = lut
    return lut


def apply_image_adjustments(frame: np.ndarray, config: dict) -> np.ndarray:
    if frame is None or not isinstance(config, dict) or not config:
        return frame

    brightness = float(config.get("brightness", 0))
    contrast   = float(config.get("contrast",   0))
    saturation = float(config.get("saturation", 0))
    exposure   = float(config.get("exposure",   0))
    gamma      = float(config.get("gamma",      0))

    if brightness == 0 and contrast == 0 and saturation == 0 and exposure == 0 and gamma == 0:
        return frame  # nothing to do — skip entirely, zero overhead

    # NOTE: every OpenCV/numpy op below returns a NEW array; we never
    # write back into the original `frame` buffer, so no copy needed.
    img = frame

    # --- brightness / contrast / exposure ---
    # contrast -100..100 → c_factor 0.0..2.0 (alpha component)
    # exposure -100..100 → e_factor 0.0..2.0 (gain component)
    c_factor = max(0.0, 1.0 + contrast  / 100.0)
    e_factor = max(0.0, 1.0 + exposure  / 100.0)
    alpha    = c_factor * e_factor
    beta     = brightness   # additive shift, clips at 0/255 automatically

    if alpha != 1.0 or beta != 0:
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- saturation (vectorised, no full-frame float cast) ---
    if saturation != 0:
        s_factor = max(0.0, 1.0 + saturation / 100.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          # uint8 H,S,V
        s   = hsv[:, :, 1].astype(np.float32) * s_factor   # only S channel
        hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- gamma (fully vectorised numpy LUT, no Python loop) ---
    # gamma > 0 → inv_gamma < 1 → pixel^(<1) → brighter (lift shadows)
    # gamma < 0 → inv_gamma > 1 → pixel^(>1) → darker   (deepen blacks)
    # gamma == 0 → skip (neutral)
    if gamma != 0:
        lut = _get_gamma_lut(gamma)
        img = cv2.LUT(img, lut)

    return img


def draw_overlay(
    frame: np.ndarray,
    detections_with_alerts: list,
    line_detection_enabled: bool,
    proximity_detector,
    inference_fps: float = None,
    display_fps: float = None,
) -> None:
    """Paint boxes, the alert-zone line, and the FPS readout onto `frame` in
    place — exactly what the on-screen window shows (when one is available).

    display_fps is the actual rate frames are being shown/pushed at (every
    captured frame, since none are dropped for display -- see
    CaptureThread). inference_fps is the rate the YOLO->pose->clothing
    pipeline is actually running at, which is lower whenever
    PIPELINE_FRAME_STRIDE > 1. Showing both makes the throttle's effect
    visible on screen instead of just in the logs.
    """
    if display_fps is not None:
        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (51, 242, 26),
            2,
        )
    if inference_fps is not None:
        cv2.putText(
            frame,
            f"Inf: {inference_fps:.1f} fps",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (51, 242, 26),
            2,
        )

    if line_detection_enabled and proximity_detector is not None:
        proximity_detector.draw_line_opencv(frame)

    for item in detections_with_alerts:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        (x1, y1, x2, y2, label, conf), is_alert = item
        color = (0, 0, 255) if is_alert else CLASS_COLORS_BGR.get(label, _DEFAULT_COLOR_BGR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        ty = max(y1 - 6, 16)

        if label not in _LABEL_SIZE_CACHE:
            _LABEL_SIZE_CACHE[label] = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )[0]
        tw, th = _LABEL_SIZE_CACHE[label]
        tw += 40

        cv2.rectangle(
            frame,
            (x1, ty - th - 4),
            (x1 + tw + 4, ty + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
        )


_NETWORK_URL_PREFIXES = ("http://", "https://", "rtsp://")


def _backend_available(name: str) -> bool:
    """Whether this OpenCV build actually ships the named capture backend
    (e.g. "CAP_FFMPEG"). Builds differ a lot here: the desktop wheels
    carry FFmpeg, while the board's OpenCV is commonly built against
    GStreamer instead -- which is exactly why a network URL that opens
    fine on a PC can fail on the board with no useful error."""
    flag = getattr(cv2, name, None)
    if flag is None:
        return False
    try:
        return flag in cv2.videoio_registry.getBackends()
    except Exception:
        # Registry not queryable on this build -- let the open attempt
        # itself be the test rather than ruling the backend out here.
        return True


def _gstreamer_pipelines(url: str) -> list:
    """Candidate GStreamer pipelines for a network camera, best first.

    GStreamer, unlike FFmpeg, will NOT take a bare media URL and work out
    what to do with it. Handed "http://phone:4747/video" as a plain
    source string it fails with the unhelpful "Internal data stream
    error / unable to start pipeline" pair. It needs the whole chain
    spelled out: fetch -> demux -> decode -> convert -> hand to OpenCV.

    More than one variant is offered because phone-camera apps disagree
    about framing: most wrap their JPEGs in a multipart stream
    (multipartdemux), a few just stream them, and decodebin can sometimes
    work it out when neither guess fits. Trying them in turn costs one
    failed open each and turns "it doesn't work" into "this exact
    pipeline works", which is otherwise very hard to discover remotely.

    appsink is configured drop=true max-buffers=1 sync=false so the
    pipeline always yields the FRESHEST frame instead of queueing a
    backlog -- the same drop-oldest policy CaptureThread applies further
    down, and the right trade for a live tripwire.

    location is quoted so a URL containing '?' or '&' (e.g.
    "/mjpegfeed?640x480") survives gst_parse_launch's own tokenizer.
    """
    sink = "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"

    if url.startswith("rtsp://"):
        return [
            ("GStreamer/rtsp-tcp",
             f'rtspsrc location="{url}" latency=0 protocols=tcp timeout=5000000 ! '
             f"decodebin ! {sink}"),
            ("GStreamer/rtsp-udp",
             f'rtspsrc location="{url}" latency=0 ! decodebin ! {sink}'),
        ]

    # HTTP MJPEG (DroidCam, IP Webcam, most phone-camera apps).
    # timeout/retries stop a single failed fetch from hanging the pipeline
    # open forever; without them souphttpsrc can sit waiting indefinitely.
    src = (f'souphttpsrc location="{url}" is-live=true do-timestamp=true '
           "timeout=5 retries=2 keep-alive=true")
    return [
        ("GStreamer/mjpeg", f"{src} ! multipartdemux ! jpegdec ! {sink}"),
        ("GStreamer/jpeg",  f"{src} ! jpegdec ! {sink}"),
        ("GStreamer/decodebin", f"{src} ! decodebin ! {sink}"),
    ]


def _make_placeholder_frame(width: int, height: int, message: str) -> np.ndarray:
    """A solid frame with a centered status message, pushed to the live
    preview (/video_feed) whenever no capture source is currently open --
    startup with no camera plugged in yet, or a mid-session disconnect.
    Without this, /video_feed just stalls (nothing new ever pushed to
    line_config_server's MJPEG queue) and the dashboard's <img> looks
    like a generic network failure. Pushing an explicit frame here makes
    "camera not connected" visually obvious on the live feed itself, on
    top of the status pill (which reads this over /camera_status).
    """
    frame = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(width, height) / 480.0 * 0.8)
    thickness = 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
    x = max(0, (width - tw) // 2)
    y = max(th, (height + th) // 2)
    cv2.putText(frame, message, (x, y), font, scale, (60, 60, 220), thickness, cv2.LINE_AA)
    return frame
class CaptureThread(threading.Thread):
    """Owns cv2.VideoCapture.read(). Runs continuously, independent of how
    fast the processing side keeps up.

    The capture -> processing handoff is drop-oldest: if the processing
    loop has fallen behind and the queue (FRAME_QUEUE_MAXSIZE) is full,
    the oldest unread (frame, run_pipeline) tuple is discarded to make
    room for the newest one, instead of this thread blocking on a full
    queue. A blocking put() here would pause camera reads and let
    end-to-end latency grow unbounded whenever the pipeline stalls (e.g.
    an NPU timeout) -- exactly the wrong tradeoff for a live proximity
    alert, where a late alert is worse than an occasionally-dropped
    display frame. The freshest frame is always worth more than an old
    one still sitting in the queue.

    Each item put on the queue is (frame, run_pipeline): run_pipeline is
    True only every PIPELINE_FRAME_STRIDE-th captured frame -- the
    processing loop uses that flag to decide whether to run
    detect->pose->clothing on this frame, or just redraw the previous
    result on it. The stride is counted here, at the raw capture rate, not
    on whatever the processing loop happens to pull -- keeps it one clean,
    predictable knob (config.PIPELINE_FRAME_STRIDE) instead of stacking
    with anything else.

    Pinned to the same core as the processing loop (they time-slice;
    capture is I/O-bound and yields naturally).
    """

    def __init__(self, cap, frame_queue: "queue.Queue", is_live_source: bool,
                 frame_delay_s: float, core_id: int = CORE_CPU_ID,
                 pipeline_stride: int = RUN_MODEL_ON_EVERY_N_FRAME,
                 disconnect_timeout_s: float = CAMERA_DISCONNECT_TIMEOUT_S):
        super().__init__(daemon=True, name="CaptureThread")
        self.cap = cap
        self.frame_queue = frame_queue
        self.is_live_source = is_live_source
        self.frame_delay_s = frame_delay_s
        self.core_id = core_id
        self.pipeline_stride = max(1, int(pipeline_stride))
        self.disconnect_timeout_s = float(disconnect_timeout_s)
        self._stop_event = threading.Event()
        self._disconnected_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def is_disconnected(self) -> bool:
        return self._disconnected_event.is_set()
    def run(self):
        _pin_thread(self.core_id)

        cap_fps_counter = 0
        cap_fps_time = time.monotonic()
        capture_index = 0
        last_ok_time = time.monotonic()

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            ok, frame = self.cap.read()

            if not ok:
                if self.is_live_source:
                    if time.monotonic() - last_ok_time >= self.disconnect_timeout_s:
                        print(
                            f"[Video] No frames for {self.disconnect_timeout_s:.0f}s -- "
                            "treating source as disconnected."
                        )
                        self._disconnected_event.set()
                        break
                    time.sleep(0.01)
                    continue
                self.frame_queue.put(None)  # EOF sentinel for file sources
                break

            last_ok_time = time.monotonic()
            cap_fps_counter += 1
            now = time.monotonic()
            if now - cap_fps_time >= 1.0:
                cap_fps_counter = 0
                cap_fps_time = now

            run_pipeline = (capture_index % self.pipeline_stride == 0)
            capture_index += 1

            # Drop-oldest handoff: normal operation never fills the queue,
            # since only every Nth frame is expensive. If the processing
            # loop has genuinely stalled (e.g. an NPU timeout) and the
            # queue is full, discard the oldest still-queued frame instead
            # of blocking capture -- keeps latency bounded and always
            # hands the processing loop the freshest frame available.
            try:
                self.frame_queue.put_nowait((frame, run_pipeline))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # drop the stalest queued frame
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait((frame, run_pipeline))
                except queue.Full:
                    pass  # lost a race with the consumer -- fine, next frame will land

            if not self.is_live_source:
                # Pace file playback to roughly its native fps -- a live
                # camera paces itself via its own frame-ready timing, but
                # a file would otherwise decode as fast as the CPU allows.
                remaining = self.frame_delay_s - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)


class CameraStream:
    """One camera, end to end: its cv2.VideoCapture, its CaptureThread,
    its own ProximityDetector (each camera has its own zone line, so the
    detectors can never be shared), its own health status, its own
    live-preview feed, and its own reconnect clock.

    Everything here used to live directly on CaptureWidget, which was
    correct while there was exactly one camera. Pulling it into its own
    object is what lets cam2 fail, disconnect, or be switched off entirely
    without cam1 noticing: each stream opens, reconnects, loops a finished
    file and reports health completely independently of the other.

    What deliberately does NOT live here is anything the pipeline decides:
    detection, voting and alerting stay on CaptureWidget/Application so
    there is still exactly ONE thread calling into NPUWorker. (NPUWorker's
    request/result channels are single-slot and assume a single caller --
    two streams inferring on their own threads would steal each other's
    results.)
    """

    def __init__(self, app, cam_id: str, source: str, video_path, camera_device):
        self.app = app
        self.cam_id = cam_id
        self.label = CAMERA_LABELS.get(cam_id, cam_id)

        # Only true for an http:// / rtsp:// URL -- selects the
        # multi-backend open ladder in _blocking_open().
        self._is_network_source = False
        # Backend that actually worked last time, so reconnects go
        # straight to it instead of re-walking the whole ladder.
        self._proven_attempt = None

        if source == "Camera":
            source_text = str(camera_device)
            self._is_live_source = True
            # A numeric camera device ("0", "1", ...) must be passed to
            # cv2.VideoCapture as an int to open a device INDEX. Passed as
            # a str, OpenCV's bindings instead treat it as a FILENAME to
            # open -- which doesn't exist -- and isOpened() comes back
            # False with no other clue why. A device path like
            # "/dev/video0" isn't digit-only, so it's left as a string.
            video_source = int(source_text) if source_text.isdigit() else source_text
            requested_w, requested_h, requested_fps = (
                app.frame_width,
                app.frame_height,
                app.framerate,
            )
            if platform.system() == "Windows":
                open_backend = cv2.CAP_DSHOW
            elif platform.system() == "Darwin":
                open_backend = cv2.CAP_AVFOUNDATION
            else:
                open_backend = cv2.CAP_V4L2
        else:
            if not video_path:
                flag = "--video_path" if cam_id == "cam1" else "--video_path_2"
                raise ValueError(
                    f"{self.label}: a Video source requires {flag} "
                    "/path/to/file, an http:// or rtsp:// URL, or a camera index."
                )
            source_text = str(video_path)
            self._is_network_source = source_text.startswith(_NETWORK_URL_PREFIXES)
            self._is_live_source = source_text.isdigit() or self._is_network_source
            if not self._is_live_source and not os.path.exists(source_text):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            video_source = int(source_text) if source_text.isdigit() else source_text
            requested_w = requested_h = requested_fps = None
            open_backend = None

        self.source_text = source_text
        self.video_source = video_source
        self.open_backend = open_backend
        self.requested_w = requested_w
        self.requested_h = requested_h
        self.requested_fps = requested_fps

        self.cap = None
        # Placeholder geometry until _finalize_open() first succeeds and
        # replaces these with the real values.
        self._vid_w = app.frame_width
        self._vid_h = app.frame_height
        self._vid_fps = app.framerate or 25.0
        self._frame_delay_s = (1.0 / self._vid_fps) if self._vid_fps else 0.04

        # Built by _finalize_open() once a real resolution is known --
        # ProximityDetector's line coordinates are frame-space, so it
        # can't be constructed against the placeholder geometry above.
        self.proximity_detector = None
        self.frame_queue = None
        self.capture_thread = None

        # Reconnect pacing. The old single-camera code could afford to
        # BLOCK in a sleep loop until its one camera showed up
        # (_wait_for_capture); with two cameras that would stall the other
        # camera's detection for as long as this one stays unplugged, so
        # retries are rate-limited by a deadline instead of a sleep.
        self._next_open_attempt = 0.0
        self._placeholder = None
        self._placeholder_msg = None
        self._logged_missing = False
        # In-flight background open (see _start_open_thread). The
        # generation counter invalidates a capture that arrives after a
        # release(), so a slow RTSP connect can't resurrect a stream the
        # operator has since switched off.
        self._open_thread = None
        self._open_result = None
        self._open_generation = 0

        self.window_name = None
        self.window_created = False

        # Per-camera display/pipeline bookkeeping.
        self.frame_counter = 0
        self.last_detections_with_alerts = []
        self.cam_status = "live"
        self.cam_msg = "Live"
        self.inference_fps = 0.0
        self._inf_fps_counter = 0
        self._inf_fps_time = time.monotonic()
        self.display_fps = 0.0
        self._disp_fps_counter = 0
        self._disp_fps_time = time.monotonic()

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def is_open(self) -> bool:
        return self.cap is not None

    def _blocking_open(self):
        """Open self.video_source and return the capture, or None.

        This is the part that can BLOCK -- for a webcam or a file it
        returns in milliseconds, but for an unreachable network source it
        can sit here for a minute or more, which is why _start_open_thread
        runs it off the processing loop. Never raises: a camera that isn't
        plugged in yet must be a retry, not a crash.
        """
        for attempt in self._open_attempts():
            name, source, backend = attempt
            try:
                cap = (
                    cv2.VideoCapture(source, backend)
                    if backend is not None
                    else cv2.VideoCapture(source)
                )
            except Exception as exc:
                print(f"[Video] {self.label}: {name} open failed for '{self.source_text}': {exc}")
                continue

            if not cap.isOpened():
                cap.release()
                if self._is_network_source and DEBUG:
                    print(f"[Video] {self.label}: {name} could not open '{self.source_text}'")
                continue

            if self.requested_w and self.requested_h:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_h)
            if self.requested_fps:
                cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

            if self._is_network_source and self._proven_attempt != attempt:
                # Worth one line: which backend a network camera actually
                # opened with is the single most useful fact when the same
                # URL works on a PC and not on the board.
                print(f"[Video] {self.label}: opened via {name}")
            self._proven_attempt = attempt
            return cap

        return None

    def _open_attempts(self):
        """The (name, source, backend) combinations to try, in order.

        A local device or file has exactly one sensible way to open, so
        this yields a single entry and behaves as before.

        A NETWORK camera does not. OpenCV auto-selects a backend, and
        which one it picks depends entirely on how that particular
        OpenCV was built: the desktop wheels pick FFmpeg, which takes a
        media URL directly; the board's build picks GStreamer, which does
        not -- it needs a full pipeline description and otherwise dies
        with "Internal data stream error / unable to start pipeline".
        Auto-selection therefore silently does the wrong thing on exactly
        one of the two machines. Rather than hard-code either answer,
        every viable option is tried in order of preference and the one
        that works is remembered for subsequent reconnects.
        """
        if not self._is_network_source:
            yield ("default", self.video_source, self.open_backend)
            return

        if self._proven_attempt is not None:
            yield self._proven_attempt        # reconnect: go straight to what worked

        url = self.source_text
        if _backend_available("CAP_FFMPEG"):
            # Preferred where it exists: handles http MJPEG and rtsp from
            # the URL alone, and honours config.RTSP_FFMPEG_OPTIONS'
            # connect timeouts. Absent from many embedded OpenCV builds
            # (the i.MX93 image is GStreamer-only), hence the fallbacks.
            yield ("FFmpeg", url, cv2.CAP_FFMPEG)
        if _backend_available("CAP_GSTREAMER"):
            for name, pipeline in _gstreamer_pipelines(url):
                yield (name, pipeline, cv2.CAP_GSTREAMER)
        # Last resort: let OpenCV choose. Rarely helps if the above
        # failed, but costs nothing and covers an exotic build.
        yield ("auto", url, None)

    def _finalize_open(self, cap) -> bool:
        """Everything that has to happen once a capture is actually open,
        back on the processing thread: read its real geometry and
        (re)build what depends on it -- ProximityDetector (its line
        coordinates are frame-space), the capture->processing queue, and
        CaptureThread.
        """
        self.cap = cap
        self._vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.app.frame_width
        self._vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.app.frame_height
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS) or self.app.framerate or 25.0
        self._frame_delay_s = (1.0 / self._vid_fps) if self._vid_fps else 0.04

        print(
            f"[Video] {self.label}: source connected : "
            f"{self._vid_w}x{self._vid_h} @ {self._vid_fps:.2f} fps"
        )
        set_frame_resolution(self._vid_w, self._vid_h, self.cam_id)

        # Detector / SegmentAnalyzer (and its internal PoseDetector) are
        # built once by Application, already wired to app.npu_worker --
        # one SegmentAnalyzer per camera, see Application.contexts.
        self.proximity_detector = ProximityDetector(
            frame_width=self._vid_w,
            frame_height=self._vid_h,
            cam_id=self.cam_id,
        )

        self._start_capture_thread()

        self._logged_missing = False
        self._placeholder = None
        self.cam_status, self.cam_msg = "live", "Live"
        set_camera_health("live", "Live", self.cam_id)
        return True

    def _start_capture_thread(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self.capture_thread = CaptureThread(
            self.cap, self.frame_queue, self._is_live_source, self._frame_delay_s,
            core_id=CORE_CPU_ID,
        )
        self.capture_thread.start()

    def _start_open_thread(self):
        """Run the blocking half of opening this source on a throwaway
        thread, so the processing loop never waits on it.

        This matters most for network sources. cv2.VideoCapture() on an
        RTSP URL whose camera is powered off or unreachable does not fail
        fast: measured on this codebase, a plain unreachable rtsp:// URL
        blocked inside VideoCapture() for ~109 SECONDS, and ~30s even with
        FFmpeg's own timeouts configured (see RTSP_FFMPEG_OPTIONS in
        config.py). Called inline, that would freeze the shared processing
        loop -- and with it the OTHER camera's detection -- for the whole
        duration. A camera that isn't there must cost the working camera
        nothing.

        The generation counter makes a result that arrives after a
        release() harmless: the stale thread releases its own capture
        instead of handing back one nobody is expecting any more.
        """
        gen = self._open_generation
        self._open_result = None

        def worker():
            cap = self._blocking_open()
            if gen != self._open_generation:
                if cap is not None:
                    cap.release()  # superseded while we were blocked
                return
            self._open_result = cap

        self._open_thread = threading.Thread(
            target=worker, daemon=True, name=f"Open-{self.cam_id}"
        )
        self._open_thread.start()

    def ensure_open(self) -> bool:
        """Non-blocking counterpart of the old _wait_for_capture(): gets
        the source opening if it isn't open, at most once every
        CAMERA_RECONNECT_INTERVAL_S, and keeps the dashboard informed in
        the meantime.

        Returns True if this stream currently has a capture to read from.
        Returning False is a normal, expected state -- the caller simply
        moves on to the other camera this iteration.
        """
        if self.cap is not None:
            return True

        # An attempt started on an earlier pass may still be blocked in
        # cv2.VideoCapture(). Never start a second one on top of it --
        # against a permanently dead RTSP URL that would pile up a new
        # thread every reconnect interval.
        if self._open_thread is not None:
            if self._open_thread.is_alive():
                self._report_not_open("Connecting...")
                return False
            self._open_thread.join()          # already finished -- instant
            self._open_thread = None
            cap, self._open_result = self._open_result, None
            if cap is not None:
                return self._finalize_open(cap)
            self._report_not_open("Camera Not Connected")
            if not self._logged_missing:
                self._logged_missing = True
                tried = ", ".join(name for name, _, _ in self._open_attempts())
                # Printed unconditionally (not just under DEBUG) for a
                # NETWORK camera: when a URL won't open on the board, the
                # list of backends actually attempted is the single most
                # useful line in the log -- it distinguishes "OpenCV has
                # no FFmpeg" from "every pipeline was tried and the host
                # is simply unreachable". Local devices stay quiet as
                # before; an unplugged USB camera is self-explanatory.
                if self._is_network_source:
                    print(
                        f"[Video] {self.label}: could not open '{self.source_text}' -- "
                        f"tried: {tried}. Retrying every {CAMERA_RECONNECT_INTERVAL_S:.0f}s. "
                        "If every backend fails, check the host is reachable FROM THIS "
                        "DEVICE (curl -sI --max-time 5 <url>)."
                    )
                elif DEBUG:
                    print(
                        f"[Video] {self.label}: source '{self.source_text}' not available -- "
                        f"retrying every {CAMERA_RECONNECT_INTERVAL_S:.0f}s"
                    )
            return False

        now = time.monotonic()
        if now < self._next_open_attempt:
            self._report_not_open("Camera Not Connected")
            return False

        self._next_open_attempt = now + CAMERA_RECONNECT_INTERVAL_S
        self._start_open_thread()
        self._report_not_open("Connecting...")
        return False

    def _report_not_open(self, message: str):
        """Publish one "this camera has no picture" state: the health pill
        and the placeholder frame on its live feed always say the same
        thing. "Connecting..." vs "Camera Not Connected" is a real
        distinction for an RTSP source, where an open legitimately takes
        seconds before it either succeeds or gives up."""
        self.cam_status, self.cam_msg = "no_camera", message
        set_camera_health("no_camera", message, self.cam_id)
        self.show_placeholder(message)

    def request_immediate_reopen(self):
        """Skip the remaining reconnect wait and try to open on the next
        pass. Used when something changed that makes an immediate retry
        worthwhile -- the operator switching camera 2 back on, or a file
        source restarting -- rather than making them wait out an interval
        that was scheduled for a camera that wasn't coming back."""
        self._next_open_attempt = 0.0

    def show_placeholder(self, message: str):
        """Feed this camera's live preview a solid frame with a status
        message, so the dashboard tile shows "Camera Not Connected"
        instead of a stalled <img> that just looks like a network error.
        Rebuilt only when the message or the geometry changes."""
        if (
            self._placeholder is None
            or self._placeholder_msg != message
            or self._placeholder.shape[1::-1] != (self._vid_w, self._vid_h)
        ):
            self._placeholder = _make_placeholder_frame(self._vid_w, self._vid_h, message)
            self._placeholder_msg = message
        update_frame(self._placeholder, self.cam_id)

    def release(self, health=("no_camera", "Camera Not Connected")):
        """Tear down everything _finalize_open() built, so the next
        reconnect attempt starts from a clean slate.

        An in-flight background open is invalidated rather than waited
        on: joining it could block this thread for the full ~109s an
        unreachable RTSP connect takes, which is exactly what moving the
        open off this thread was meant to avoid. The bumped generation
        makes that thread release its own capture when it finally
        returns. Its reference is deliberately KEPT, so ensure_open()
        won't stack a second opener on top of one still running.
        """
        self._open_generation += 1
        self._open_result = None

        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.capture_thread = None
        self.frame_queue = None
        self.last_detections_with_alerts = []
        if health:
            self.cam_status, self.cam_msg = health
            set_camera_health(health[0], health[1], self.cam_id)

    def destroy_window(self):
        if self.window_created and self.window_name:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        self.window_created = False

    # ------------------------------------------------------------------
    # Frame handoff
    # ------------------------------------------------------------------
    def poll(self):
        """Take the freshest frame this camera has ready, without blocking.

        Non-blocking is the whole point: the old loop could sit in
        frame_queue.get(timeout=2.0) because there was nothing else to do
        meanwhile. Now a blocking wait on one camera is dead time for the
        other, so the processing loop polls each stream in turn and only
        sleeps when NEITHER has anything.

        Returns (frame, run_pipeline), or None when nothing is ready, or
        the string "eof" for a finite source that has ended.
        """
        if self.frame_queue is None:
            return None
        try:
            item = self.frame_queue.get_nowait()
        except queue.Empty:
            return None
        if item is None:  # EOF sentinel from CaptureThread (file sources)
            return "eof"
        return item

    def check_disconnected(self) -> bool:
        """True if the capture thread has given up on a live source. The
        stream is released here so ensure_open() starts reconnecting on
        the normal interval."""
        if self.capture_thread is None or not self.capture_thread.is_disconnected():
            return False
        print(f"[Video] {self.label}: source disconnected -- will attempt to reconnect.")
        self.release()
        return True

    def _loop_file_source(self) -> bool:
        """Restart a finite file source from frame 0 WITHOUT tearing down
        and reopening cv2.VideoCapture.

        The old behavior on EOF was to release() the capture and reopen
        the same path. Reopening the exact same file immediately after
        release() is flaky with OpenCV/FFmpeg on this board -- it can fail
        intermittently -- and when it does, the reconnect path takes over:
        camera health goes to "no_camera" and the dashboard is fed the
        "Camera Not Connected" placeholder every
        CAMERA_RECONNECT_INTERVAL_S, silently (DEBUG is normally False),
        forever. That's the exact "video plays once, then the dashboard
        gets stuck on Camera Not Connected" symptom.

        Seeking the SAME still-open capture object back to frame 0
        sidesteps that whole reopen path for local files. Returns True if
        the seek succeeded and a fresh CaptureThread was started against
        the same self.cap (Thread objects can't be restarted, so a new
        one is built); False if seeking isn't supported for this source,
        in which case the caller falls back to release + reopen.
        """
        if self.cap is None or self._is_live_source:
            return False
        try:
            seeked = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as exc:
            print(f"[Video] {self.label}: seek-to-start failed ({exc}) -- falling back to reopen.")
            return False
        if not seeked:
            return False

        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)

        self._start_capture_thread()
        if DEBUG:
            print(f"[Video] {self.label}: looping file source '{self.source_text}' back to start.")
        return True

    def handle_eof(self, show_window: bool):
        """A finite source ended: loop it in place if we can, otherwise
        release it so ensure_open() reopens it from scratch."""
        print(f"[Video] {self.label}: end of file reached.")
        if show_window:
            cv2.waitKey(1000)
        self.last_detections_with_alerts = []
        if not self._loop_file_source():
            self.release()
            # Reopen immediately rather than after the reconnect interval
            # -- the file is still there, this is just a restart.
            self.request_immediate_reopen()

    # ------------------------------------------------------------------
    # FPS counters (per camera -- each runs at its own rate)
    # ------------------------------------------------------------------
    def tick_display_fps(self, headless: bool):
        self._disp_fps_counter += 1
        now = time.monotonic()
        if now - self._disp_fps_time >= 1.0:
            self.display_fps = self._disp_fps_counter / (now - self._disp_fps_time)
            self._disp_fps_counter = 0
            self._disp_fps_time = now
            if headless and DEBUG:
                print(f"[Video] {self.label} display    : {self.display_fps:.1f} fps", flush=True)

    def tick_inference_fps(self, headless: bool):
        now = time.monotonic()
        if now - self._inf_fps_time >= 1.0:
            self.inference_fps = self._inf_fps_counter / (now - self._inf_fps_time)
            self._inf_fps_counter = 0
            self._inf_fps_time = now
            if headless and DEBUG:
                print(f"[Video] {self.label} inference  : {self.inference_fps:.1f} fps", flush=True)

    def count_inference(self):
        self._inf_fps_counter += 1


class CaptureWidget:
    """Processing loop for ALL cameras: pulls the freshest available frame
    from each stream in turn and runs
    detect -> line-cross -> vote -> alert -> overlay -> preview on it.
    Pinned to CORE_CPU_ID, same as the capture threads.

    ONE loop, not one per camera, and that is deliberate on two counts:

    1. NPUWorker's request/result channels are single-slot and assume a
       single caller (see its infer() docstring). Two processing threads
       would hand each other's results back and forth and time out.
    2. The whole point of the two-camera design is that the pipeline's
       throughput is SHARED, not doubled. Every captured frame from both
       cameras is still displayed and streamed to the dashboard, but
       inference alternates between them -- one pipeline frame from cam1,
       the next from cam2 -- so each camera keeps its full field of view
       and takes half the inference turns. Where inference is the
       bottleneck (the board), that is the same total NPU load as one
       camera, each running at about half its solo rate.
       _take_infer_turn() is where that alternation is enforced.

    All NPU inference (YOLO/pose/clothing) is delegated to app.npu_worker,
    which runs on its own dedicated, pinned core (see npu_worker.py). This
    class and CaptureThread never touch a TFLite interpreter.
    """

    def __init__(self, app):
        self.app = app
        self.detector = app.detector
        self.headless = not monitor_connected()

        self.streams = {
            "cam1": CameraStream(
                app, "cam1", app.source, app.video_path, app.camera_device
            ),
        }

        # Camera 2 is strictly optional, so a bad cam2 source must never
        # take cam1 down with it. CameraStream.__init__ rejects a missing
        # video file or an empty path outright (correct for the primary
        # camera -- starting blind is worse than failing loudly), so cam2's
        # construction is caught here and downgraded to "cam2 unavailable":
        # cam1 runs normally, the dashboard reports why, and the operator
        # can fix the path and restart without the whole system being down
        # in the meantime.
        cam2 = None
        try:
            cam2 = CameraStream(
                app, "cam2",
                # Camera 2's own input kind, not camera 1's -- a USB
                # camera and a phone/IP camera are a normal pairing.
                getattr(app, "source_2", None) or app.source,
                getattr(app, "video_path_2", None),
                getattr(app, "camera_device_2", None),
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[Video] Camera 2 unavailable -- continuing with camera 1 only: {exc}")
            set_camera_health("no_camera", "Camera 2 Source Invalid", "cam2")
        self.streams["cam2"] = cam2

        # Which camera is allowed to run inference on its next pipeline
        # frame, and since when. See _take_infer_turn().
        self._infer_turn = "cam1"
        self._infer_turn_since = time.monotonic()

        self._cam2_active = False
        self._last_clock_print = 0.0
        self._quit = False

    # ------------------------------------------------------------------
    # Which cameras are in play
    # ------------------------------------------------------------------
    def _all_streams(self) -> list:
        """Every stream that actually exists, enabled or not.

        cam2 is None when its source was rejected at startup, so every
        loop over the streams has to go through here -- iterating
        self.streams.values() directly walks straight into that None.
        """
        return [s for s in self.streams.values() if s is not None]

    def _active_streams(self) -> list:
        """cam1 always; cam2 only while the dashboard toggle is on.

        Read fresh every iteration rather than cached at startup, so
        flipping "Camera 2" in the dashboard takes effect on the next
        frame instead of needing a restart.
        """
        streams = [self.streams["cam1"]]
        if camera2_enabled() and self.streams["cam2"] is not None:
            streams.append(self.streams["cam2"])
        return streams

    def _sync_cam2_enabled(self):
        """React to the dashboard's Camera 2 toggle being flipped.

        Turning it OFF has to do more than stop reading frames: the
        capture is released (so the device is free for anything else),
        the tile is left showing an explicit "Camera 2 Disabled" frame
        rather than a frozen last image, and cam2's detection state is
        cleared so a stale vote buffer or a stale "person still in zone"
        can't influence anything after it's switched back on.
        """
        stream = self.streams["cam2"]
        if stream is None:
            return  # cam2's source was rejected at startup -- nothing to toggle
        enabled = camera2_enabled()
        if enabled == self._cam2_active:
            return
        self._cam2_active = enabled
        if enabled:
            stream.request_immediate_reopen()  # don't wait out the reconnect interval
            print("[Video] Camera 2 enabled.")
        else:
            stream.release(health=("no_camera", "Camera 2 Disabled"))
            stream.destroy_window()
            stream.show_placeholder("Camera 2 Disabled")
            self._suspend_detection(stream)
            if self._infer_turn == "cam2":
                self._infer_turn = "cam1"
                self._infer_turn_since = time.monotonic()
            print("[Video] Camera 2 disabled.")

    # ------------------------------------------------------------------
    # Inference turn-taking
    # ------------------------------------------------------------------
    def _take_infer_turn(self, stream: CameraStream) -> bool:
        """Decide whether THIS camera gets to run the pipeline on this
        frame, and hand the turn to the next camera if it does.

        With one camera active this is always True and costs nothing --
        the turn just keeps landing back on the same stream. With two, it
        is what makes inference alternate cam1 -> cam2 -> cam1 instead of
        each camera independently running the pipeline every
        RUN_MODEL_ON_EVERY_N_FRAME frames (which would double NPU load and
        halve the frame rate of both).

        The INFER_TURN_TIMEOUT_S escape hatch matters as much as the
        alternation: if the camera holding the turn stalls, disconnects,
        or is simply slower than the other, the waiting camera takes the
        turn instead of skipping inference forever on a turn that is never
        coming back.
        """
        now = time.monotonic()
        if self._infer_turn != stream.cam_id:
            if now - self._infer_turn_since < INFER_TURN_TIMEOUT_S:
                return False
            if DEBUG:
                print(
                    f"[Video] {stream.label}: taking inference turn from "
                    f"'{self._infer_turn}' after {INFER_TURN_TIMEOUT_S:.1f}s of silence."
                )
        self._advance_infer_turn(stream)
        return True

    def _advance_infer_turn(self, current: CameraStream):
        open_ids = [s.cam_id for s in self._active_streams() if s.is_open()]
        nxt = current.cam_id
        if len(open_ids) > 1:
            try:
                idx = open_ids.index(current.cam_id)
            except ValueError:
                idx = -1
            nxt = open_ids[(idx + 1) % len(open_ids)]
        self._infer_turn = nxt
        self._infer_turn_since = time.monotonic()

    # ------------------------------------------------------------------
    # Detection state helpers
    # ------------------------------------------------------------------
    def _suspend_detection(self, stream: CameraStream):
        """Clear everything this camera's pipeline was mid-way through.

        Called whenever detection stops running for a camera -- bad camera
        frame, train off-hours, disconnect, cam2 switched off. Without
        this the vote buffer keeps half a description around, and
        AlertEngine's audio worker keeps acting on a person_in_zone flag
        that no frame has confirmed since. Both are strictly per camera:
        clearing cam2's state must never touch cam1's.
        """
        ctx = self.app.context(stream.cam_id) if hasattr(self.app, "context") else None
        if ctx is not None:
            ctx.segment_analyzer.reset_votes()
        if hasattr(self.app, "alert_engine"):
            self.app.alert_engine.set_person_in_zone(stream.cam_id, False)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        _pin_thread(CORE_CPU_ID)

        label = _model_label(self.app.model_path)
        for stream in self._all_streams():
            stream.window_name = f"{label} - {stream.label} ({self.app.source}) (press q to quit)"

        if DEBUG:
            print(
                "[Video] Headless mode - no display, terminal alerts only. Ctrl+C to quit."
                if self.headless
                else "[Video] Windowed mode - press q to quit."
            )

        while not self._quit:
            self._sync_cam2_enabled()
            streams = self._active_streams()

            progressed = False
            for stream in streams:
                if not stream.ensure_open():
                    continue

                item = stream.poll()
                if item is None:
                    # Nothing ready this pass. A live source that has gone
                    # quiet for CAMERA_DISCONNECT_TIMEOUT_S is torn down
                    # here and reconnected on the usual interval; a
                    # momentary stall just waits for the next pass.
                    stream.check_disconnected()
                    continue

                if item == "eof":
                    stream.handle_eof(self._window_open(stream))
                    self._suspend_detection(stream)
                    continue

                progressed = True
                self._process_frame(stream, item)

            if self._pump_window_keys():
                break

            if not progressed:
                # Neither camera had a frame ready -- yield the core
                # instead of spinning on it. Short enough to stay well
                # inside one frame interval at any realistic frame rate.
                time.sleep(0.002)

        for stream in self._all_streams():
            stream.release(health=None)
            stream.destroy_window()
        if not self.headless:
            cv2.destroyAllWindows()
        print("[Video] Finished.")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------
    def _process_frame(self, stream: CameraStream, item):
        raw_frame, run_pipeline = item
        stream.frame_counter += 1

        # Check camera health on raw unadjusted camera input. Throttled
        # to every CAMERA_HEALTH_CHECK_STRIDE frames -- it's a full
        # resize + grayscale + Laplacian-variance pass, not free on a
        # single shared CPU core, and a few hundred ms of extra latency
        # before noticing a blocked/blurred lens is an acceptable
        # trade. Frames in between reuse the last known status.
        if stream.frame_counter % CAMERA_HEALTH_CHECK_STRIDE == 0:
            stream.cam_status, stream.cam_msg = check_camera_health(raw_frame)
            set_camera_health(stream.cam_status, stream.cam_msg, stream.cam_id)

        # Halt model inference when this camera is black, blurry, or obstructed.
        # The OTHER camera is unaffected -- a covered lens on cam2 says
        # nothing about cam1's view.
        if stream.cam_status != "live":
            run_pipeline = False
            stream.last_detections_with_alerts = []
            self._suspend_detection(stream)

        stream.proximity_detector.load()
        cfg = getattr(stream.proximity_detector, "config", {})
        if cfg:
            # Detection thresholds, cooldown, volume and languages are
            # GLOBAL settings -- read from whichever camera's flattened
            # config view is at hand, identical in both.
            p_th = cfg.get("person_conf_th", 0.5)
            t_th = cfg.get("train_conf_th", 0.45)
            self.detector.update_thresholds(p_th, t_th)
            # Image adjustments, however, are this camera's own.
            frame = apply_image_adjustments(raw_frame, cfg)
            # Dynamically update alert cooldown from UI slider
            cooldown = cfg.get("alert_cooldown_seconds", 30)
            if hasattr(self.app, "alert_engine"):
                self.app.alert_engine.cooldown_seconds = float(cooldown)

                # Dynamically update which language(s) the spoken alert
                # plays in from the dashboard's checkboxes. A fresh list
                # object each time (never mutated in place) so the audio
                # worker thread reading it mid-playback stays safe
                # without needing a lock -- see AlertEngine's comment.
                tts_languages = cfg.get("tts_languages", [])
                if isinstance(tts_languages, list):
                    self.app.alert_engine.active_languages = list(tts_languages)

                # Dynamically update spoken-alert volume from the
                # dashboard's Alert & Audio volume slider. Stored in
                # config.json as a 0-150 percentage (matching the
                # slider's own units); AlertEngine.volume itself is a
                # fraction (1.0 = clip's original level) -- see
                # AlertEngine._apply_volume() for the actual scaling
                # and its own clamp.
                volume_pct = cfg.get("alert_volume", 100)
                self.app.alert_engine.volume = float(volume_pct) / 100.0

            # Train off-hours (metro not running): suspend the entire
            # detect->pose->clothing pipeline for this frame, same
            # treatment as a bad camera frame above. The raw feed
            # still gets pushed to the live preview via update_frame()
            # further down -- only inference + alerting stops. Applies
            # to both cameras, since the schedule is one global window.
            if is_schedule_off(cfg.get("train_off_start", ""), cfg.get("train_off_end", "")):
                run_pipeline = False
                stream.last_detections_with_alerts = []
                self._suspend_detection(stream)
        else:
            frame = raw_frame

        now_ts = time.monotonic()
        if now_ts - self._last_clock_print >= 5.0:
            self._last_clock_print = now_ts
            off = is_schedule_off(cfg.get("train_off_start", ""), cfg.get("train_off_end", ""))
            if DEBUG:
                print(
                    f"[Clock] {get_current_time().strftime('%H:%M:%S')} | "
                    f"Train window {cfg.get('train_off_start', '?')} -> {cfg.get('train_off_end', '?')} | "
                    f"{'OFF (no detection)' if off else 'ON (detecting)'}",
                    flush=True,
                )

        # run_pipeline is the frame-stride gate (every Nth captured frame
        # of THIS camera); _take_infer_turn is the cam1/cam2 alternation
        # gate on top of it. A frame that loses the turn is still
        # displayed and streamed, just redrawn with this camera's most
        # recent detection result.
        if run_pipeline and self._take_infer_turn(stream):
            ctx = self.app.context(stream.cam_id)
            if DEBUG:
                ctx.segment_analyzer.last_timing = {"pose_ms": 0.0, "cloth_ms": 0.0}

            yolo_start = time.monotonic()
            last_detections = self.detector.detect(frame)
            yolo_ms = (time.monotonic() - yolo_start) * 1000.0
            if yolo_ms > 0:
                stream.count_inference()

            if self.app.line_detection_enabled:
                last_detections = stream.proximity_detector.filter_train_detections(last_detections)
                alert_flags = stream.proximity_detector.update(last_detections, reload=False)
                stream.last_detections_with_alerts = list(zip(last_detections, alert_flags))
                self.app.handle_alert_state(
                    frame, stream.last_detections_with_alerts, stream.cam_id
                )
            else:
                stream.last_detections_with_alerts = [(det, False) for det in last_detections]
                self.app.handle_person_detection_events(
                    frame, last_detections, stream.cam_id
                )
        # else: this frame is skipped from the pipeline (stride, or the
        # other camera's inference turn). last_detections_with_alerts is
        # simply left as-is from this camera's last processed frame, and
        # still gets drawn + displayed below -- the frame itself is never
        # dropped, only the expensive detect->pose->clothing work is.

        stream.tick_display_fps(self.headless)
        stream.tick_inference_fps(self.headless)

        draw_overlay(
            frame,
            stream.last_detections_with_alerts,
            self.app.line_detection_enabled,
            stream.proximity_detector,
            stream.inference_fps,
            stream.display_fps,
        )
        update_frame(frame, stream.cam_id)  # pushes to this camera's live-preview stream

        self._show_window(stream, frame)

    # ------------------------------------------------------------------
    # Optional on-screen windows (dev/desktop only -- headless on the board)
    # ------------------------------------------------------------------
    def _window_open(self, stream: CameraStream) -> bool:
        return stream.window_created

    def _show_window(self, stream: CameraStream, frame):
        if self.headless:
            return
        if not stream.window_created:
            try:
                # WINDOW_AUTOSIZE locks the window to the exact frame
                # dimensions -- no user-resize, always matches console WxH.
                cv2.namedWindow(stream.window_name, cv2.WINDOW_AUTOSIZE)
                stream.window_created = True
            except cv2.error as exc:
                print(f"[Video] Could not open a display window, falling back to headless: {exc}")
                self.headless = True
                return
        cv2.imshow(stream.window_name, frame)

    def _pump_window_keys(self) -> bool:
        """One waitKey for all windows -- returns True if the operator
        asked to quit. Called once per loop pass rather than once per
        camera, so two open windows don't each add their own 1ms wait."""
        if self.headless or not any(s.window_created for s in self._all_streams()):
            return False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._quit = True
            return True
        return False
