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


class CaptureWidget:
    """Processing loop: pulls the freshest available frame, runs
    detect -> line-cross -> vote -> alert -> overlay -> preview. Pinned to
    CORE_CPU_ID, same as CaptureThread.

    All NPU inference (YOLO/pose/clothing) is delegated to app.npu_worker,
    which runs on its own dedicated, pinned core (CORE_NPU_ID) -- see
    npu_worker.py. This class and CaptureThread never touch a TFLite
    interpreter.
    """

    def __init__(self, app):
        self.app = app

        if app.source == "Camera":
            source_text = str(app.camera_device)
            self._is_live_source = True
            # A numeric camera_device ("0", "1", ...) must be passed to
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
            if not app.video_path:
                raise ValueError(
                    "--source Video requires --video_path /path/to/file, URL, or camera index."
                )
            source_text = str(app.video_path)
            self._is_live_source = source_text.isdigit() or source_text.startswith(
                ("http://", "https://", "rtsp://")
            )
            if not self._is_live_source and not os.path.exists(source_text):
                raise FileNotFoundError(f"Video file not found: {app.video_path}")
            video_source = int(source_text) if source_text.isdigit() else app.video_path
            requested_w = requested_h = requested_fps = None
            open_backend = None

        self.source_text = source_text
        self.video_source = video_source
        self.open_backend = open_backend
        self.requested_w = requested_w
        self.requested_h = requested_h
        self.requested_fps = requested_fps
        self.cap = None
        self._window_created = False
        # real values the moment _open_capture() first succeeds. Used to
        # before a source has ever connected.
        self._vid_w = app.frame_width
        self._vid_h = app.frame_height
        self._vid_fps = app.framerate or 25.0
        self._frame_delay_s = (1.0 / self._vid_fps) if self._vid_fps else 0.04
        self.headless = not monitor_connected()

        self.detector = app.detector

        # Built by _open_capture() once a real resolution is known --
        # ProximityDetector's line coordinates are frame-space, so it
        # can't be constructed against the placeholder geometry above.
        self.proximity_detector = None
        self.frame_queue = None
        self.capture_thread = None
        self.inference_fps = 0.0
        self._inf_fps_counter = 0
        self._inf_fps_time = time.monotonic()
        self._last_clock_print = 0.0
        self.display_fps = 0.0
        self._disp_fps_counter = 0
        self._disp_fps_time = time.monotonic()
    def _open_capture(self) -> bool:
        """Try to open self.video_source. Never raises -- returns False on
        any failure so the caller can retry instead of crashing the whole
        process over a camera that isn't plugged in yet.
        On success, (re)builds everything that depends on the real
        resolution: ProximityDetector (its line coordinates are
        frame-space), the capture->processing queue, and CaptureThread.
        """
        try:
            cap = (
                cv2.VideoCapture(self.video_source, self.open_backend)
                if self.open_backend is not None
                else cv2.VideoCapture(self.video_source)
            )
        except Exception as exc:
            print(f"[Video] Error opening source '{self.source_text}': {exc}")
            return False

        if not cap.isOpened():
            cap.release()
            return False
        if self.requested_w and self.requested_h:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_h)
        if self.requested_fps:
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        self.cap = cap
        self._vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.app.frame_width
        self._vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.app.frame_height
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS) or self.app.framerate or 25.0
        self._frame_delay_s = (1.0 / self._vid_fps) if self._vid_fps else 0.04


        print(f"[Video] Source connected : {self._vid_w}x{self._vid_h} @ {self._vid_fps:.2f} fps")
        set_frame_resolution(self._vid_w, self._vid_h)

        # Detector / SegmentAnalyzer (and its internal PoseDetector) are
        # built once by Application, already wired to app.npu_worker.

        self.proximity_detector = ProximityDetector(
            frame_width=self._vid_w,
            frame_height=self._vid_h,
        )

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self.capture_thread = CaptureThread(
            self.cap, self.frame_queue, self._is_live_source, self._frame_delay_s,
            core_id=CORE_CPU_ID,
        )

        set_camera_health("live", "Live")
        return True
    def _release_capture(self):
        """Tear down everything _open_capture() built, so the next
        reconnect attempt starts from a clean slate."""
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.capture_thread = None
        self.frame_queue = None
    def _loop_file_source(self) -> bool:
        """Restart a finite file source from frame 0 WITHOUT tearing down
        and reopening cv2.VideoCapture.

        The old behavior on EOF was to release() the capture and reopen
        the same path via _open_capture() (see _wait_for_capture()).
        Reopening the exact same file immediately after release() is
        flaky with OpenCV/FFmpeg on this board -- it can fail
        intermittently -- and when it does, _wait_for_capture() drops into
        its retry loop: camera health goes to "no_camera" and the
        dashboard is fed the "Camera Not Connected" placeholder frame
        every CAMERA_RECONNECT_INTERVAL_S, silently (DEBUG is normally
        False), forever. That's the exact "video plays once, then the
        dashboard gets stuck on Camera Not Connected" symptom.

        Seeking the SAME still-open capture object back to frame 0
        sidesteps that whole reopen path for local files. Returns True if
        the seek succeeded and a fresh CaptureThread was started against
        the same self.cap (Thread objects can't be restarted, so a new
        one is built); False if seeking isn't supported for this source,
        in which case the caller should fall back to the old
        release/reopen path.
        """
        if self.cap is None or self._is_live_source:
            return False
        try:
            seeked = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as exc:
            print(f"[Video] Seek-to-start failed ({exc}) -- falling back to reopen.")
            return False
        if not seeked:
            return False

        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=2.0)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self.capture_thread = CaptureThread(
            self.cap, self.frame_queue, self._is_live_source, self._frame_delay_s,
            core_id=CORE_CPU_ID,
        )
        self.capture_thread.start()
        if DEBUG:
            print(f"[Video] Looping file source '{self.source_text}' back to start.")
        return True

    def _handle_file_eof(self, show_window: bool) -> bool:
        """Common EOF handling for _run_session(): prints the usual
        message, then tries to loop a file source in place via
        _loop_file_source() instead of ending the session.

        Returns True if playback was restarted in place -- the caller
        should `continue` its receive loop and reset any per-session
        detection state. Returns False if this is a live source, or if
        looping wasn't possible -- the caller should return "eof" as
        before, which falls back to the old release/reopen path in run().
        """
        print("[Video] End of file reached.")
        if show_window:
            cv2.waitKey(1000)
        if not self._loop_file_source():
            return False
        if hasattr(self.app, "segment_analyzer"):
            self.app.segment_analyzer.reset_votes()
        if hasattr(self.app, "alert_engine"):
            self.app.alert_engine.person_in_zone = False
        return True

    def _wait_for_capture(self):
        """Blocks until _open_capture() succeeds, retrying on an interval
        and keeping the dashboard informed the whole time. This is the
        core of "camera missing shouldn't need a restart": the Flask
        dashboard (started independently in main.py) stays reachable and
        shows a live "Camera Not Connected" status + placeholder preview
        frame for as long as this loop runs, and the pipeline picks the
        source up on its own the moment it becomes available.
        """
        placeholder = None
        while not self._open_capture():
            set_camera_health("no_camera", "Camera Not Connected")
            if placeholder is None or placeholder.shape[1::-1] != (self._vid_w, self._vid_h):
                placeholder = _make_placeholder_frame(
                    self._vid_w, self._vid_h, "Camera Not Connected"
                )
            update_frame(placeholder)
            if DEBUG:
                print(
                    f"[Video] Source '{self.source_text}' not available -- "
                    f"retrying in {CAMERA_RECONNECT_INTERVAL_S:.0f}s"
                )
            time.sleep(CAMERA_RECONNECT_INTERVAL_S)

    def run(self):
        _pin_thread(CORE_CPU_ID)

        label = _model_label(self.app.model_path)
        win_name = f"{label} - {self.app.source} (press q to quit)"

        while True:
            self._wait_for_capture()
            if not self.headless and not self._window_created:
                try:
                # WINDOW_AUTOSIZE locks the window to the exact frame
                # dimensions — no user-resize, always matches console W×H.
                    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                    self._window_created = True
                except cv2.error as exc:
                    print(f"[Video] Could not open a display window, falling back to headless: {exc}")
                    self.headless = True

            show_window = self._window_created

            if DEBUG:
                print(
                    "[Video] Headless mode - no display, terminal alerts only. Ctrl+C to quit."
                    if self.headless
                    else "[Video] Windowed mode - press q to quit."
                )

            self.capture_thread.start()

            reason = self._run_session(win_name, show_window)
            self._release_capture()
            if reason == "quit":
                break
            if reason == "eof":
                # Normal case: _run_session() already looped a file source
                # in place via _handle_file_eof()/_loop_file_source(), so
                # reaching "eof" here means that in-place seek failed (or
                # this is some other finite, non-live source that doesn't
                # support seeking). Fall back to the old release/reopen
                # path -- worse (flaky reopen, dashboard can get stuck on
                # "Camera Not Connected"), but still recovers eventually
                # for sources the seek-based fix doesn't cover.
                if not self._is_live_source:
                    continue
                break
            set_camera_health("no_camera", "Camera Not Connected")
            if hasattr(self.app, "segment_analyzer"):
                self.app.segment_analyzer.reset_votes()
            if hasattr(self.app, "alert_engine"):
                self.app.alert_engine.person_in_zone = False
        if self._window_created:
            cv2.destroyAllWindows()
        print("[Video] Finished.")
    def _run_session(self, win_name: str, show_window: bool) -> str:
        """Runs detect -> line-cross -> vote -> alert -> overlay -> preview
        against the CURRENTLY open self.cap/self.capture_thread, until one
        of three things ends the session. Returns which one:
          "quit"         -- operator pressed q / Esc in the display window.
          "eof"          -- a file source reached its end.
          "disconnected" -- a live source stopped delivering frames (see
                             CaptureThread.is_disconnected()); run() will
                             release this session and reconnect.
        """
        frame_counter = 0
        last_detections_with_alerts = []
        cam_status, cam_msg = "live", "Live"  # reused between throttled health checks

        while True:
            try:
                item = self.frame_queue.get(timeout=2.0)
            except queue.Empty:
                item = "TIMEOUT"

            if item == "TIMEOUT":
                if self.capture_thread.is_disconnected():
                    print("[Video] Source disconnected -- will attempt to reconnect.")
                    return "disconnected"
                if self._is_live_source:
                    # Live source stalled momentarily -- keep waiting for
                    # the capture thread to recover instead of exiting.
                    continue
                if self._handle_file_eof(show_window):
                    last_detections_with_alerts = []
                    continue
                return "eof"
            if item is None:
                if self._handle_file_eof(show_window):
                    last_detections_with_alerts = []
                    continue
                return "eof"

            raw_frame, run_pipeline = item
            frame_counter += 1

            # Check camera health on raw unadjusted camera input. Throttled
            # to every CAMERA_HEALTH_CHECK_STRIDE frames -- it's a full
            # resize + grayscale + Laplacian-variance pass, not free on a
            # single shared CPU core, and a few hundred ms of extra latency
            # before noticing a blocked/blurred lens is an acceptable
            # trade. Frames in between reuse the last known status.
            if frame_counter % CAMERA_HEALTH_CHECK_STRIDE == 0:
                cam_status, cam_msg = check_camera_health(raw_frame)
                set_camera_health(cam_status, cam_msg)

            # Halt model inference when camera is black, blurry, or obstructed
            if cam_status != "live":
                run_pipeline = False
                last_detections_with_alerts = []
                if hasattr(self.app, "segment_analyzer"):
                    self.app.segment_analyzer.reset_votes()
                # No detection is running, so there's no way to know if
                # anyone's still in the zone -- clear it rather than let
                # AlertEngine's audio worker keep acting on whatever the
                # last good frame happened to say.
                if hasattr(self.app, "alert_engine"):
                    self.app.alert_engine.person_in_zone = False

            self.proximity_detector.load()
            cfg = getattr(self.proximity_detector, "config", {})
            if cfg:
                p_th = cfg.get("person_conf_th", 0.5)
                t_th = cfg.get("train_conf_th", 0.45)
                self.detector.update_thresholds(p_th, t_th)
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
                # treatment as a bad camera frame below. The raw feed
                # still gets pushed to the live preview via update_frame()
                # further down -- only inference + alerting stops.
                if is_schedule_off(cfg.get("train_off_start", ""), cfg.get("train_off_end", "")):
                    run_pipeline = False
                    last_detections_with_alerts = []
                    if hasattr(self.app, "segment_analyzer"):
                        self.app.segment_analyzer.reset_votes()
                    # Same reasoning as the camera-health case above --
                    # detection is suspended, so the last known
                    # person_in_zone value is stale and shouldn't keep
                    # driving the audio worker.
                    if hasattr(self.app, "alert_engine"):
                        self.app.alert_engine.person_in_zone = False
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

            if run_pipeline:
                if DEBUG:
                    self.app.segment_analyzer.last_timing = {"pose_ms": 0.0, "cloth_ms": 0.0}

                yolo_start = time.monotonic()
                last_detections = self.detector.detect(frame)
                yolo_ms = (time.monotonic() - yolo_start) * 1000.0
                if yolo_ms > 0:
                    self._inf_fps_counter += 1

                if self.app.line_detection_enabled:
                    last_detections = self.proximity_detector.filter_train_detections(last_detections)
                    alert_flags = self.proximity_detector.update(last_detections, reload=False)
                    last_detections_with_alerts = list(zip(last_detections, alert_flags))
                    self.app.handle_alert_state(frame, last_detections_with_alerts)
                else:
                    last_detections_with_alerts = [(det, False) for det in last_detections]
                    self.app.handle_person_detection_events(frame, last_detections)
            # else: this frame is skipped from the pipeline (stride).
            # last_detections_with_alerts is simply left as-is from the
            # last processed frame, and still gets drawn + displayed below
            # -- the frame itself is never dropped, only the expensive
            # detect->pose->clothing work is.

            self._disp_fps_counter += 1
            now = time.monotonic()
            if now - self._disp_fps_time >= 1.0:
                self.display_fps = self._disp_fps_counter / (now - self._disp_fps_time)
                self._disp_fps_counter = 0
                self._disp_fps_time = now
                if self.headless and DEBUG:
                    print(f"[Video] Display    : {self.display_fps:.1f} fps", flush=True)

            if now - self._inf_fps_time >= 1.0:
                self.inference_fps = self._inf_fps_counter / (now - self._inf_fps_time)
                self._inf_fps_counter = 0
                self._inf_fps_time = now
                if self.headless and DEBUG:
                    print(f"[Video] Inference  : {self.inference_fps:.1f} fps", flush=True)

            self._draw_overlay(frame, last_detections_with_alerts)
            update_frame(frame)  # pushes frame to the Flask live-preview stream

            if show_window:
                cv2.imshow(win_name, frame)
 
            # frame_latency_ms = (time.monotonic() - loop_start) * 1000.0
            # if DEBUG:
            #     print(f"[Video] Frame latency: {frame_latency_ms:.1f} ms (frame #{frame_counter})")
 
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"


    def _draw_overlay(self, frame: np.ndarray, detections_with_alerts: list) -> None:
        draw_overlay(
            frame,
            detections_with_alerts,
            self.app.line_detection_enabled,
            self.proximity_detector,
            self.inference_fps,
            self.display_fps,
        )