import json
import math
import os

import cv2

# Hoisted to module scope, not inside load() -- load() runs on every
# processed frame (video_widget.py calls it unconditionally, every frame,
# to pick up threshold/image-adjustment changes even on non-pipeline
# frames), so a "from line_config_server import _read_config" statement
# sitting inside that function was being re-executed 30-60x/sec for no
# reason -- sys.modules caching makes the re-import cheap, but it's still
# needless overhead on the same CPU core that also owns capture. Resolved
# once here instead; ImportError still falls back to the config.json
# polling path below exactly like before, for any caller that doesn't
# have line_config_server available (e.g. standalone/test use).
try:
    from line_config_server import _read_config as _server_read_config
except ImportError:
    _server_read_config = None


CONFIG_PATH = "config.json"
MATCH_RADIUS = 80.0


class ProximityDetector:
    def __init__(self, config_path=CONFIG_PATH, frame_width=640, frame_height=480):
        self.config_path = config_path
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.p1 = (0, self.frame_height // 2)
        self.p2 = (self.frame_width - 1, self.frame_height // 2)
        self._alert_sign = 1
        self.threshold = 0.0
        self.mtime = 0.0
        self.config = {}
        self.tracks = []
        self._next_track_id = 1
        self.last_crossing_events = []
        self.load()

    def set_resolution(self, width, height):
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.p1 = self._clamp(self.p1)
        self.p2 = self._clamp(self.p2)

    def _clamp(self, pt):
        x = max(0, min(int(pt[0]), self.frame_width - 1))
        y = max(0, min(int(pt[1]), self.frame_height - 1))
        return (x, y)

    def load(self):
        cfg = _server_read_config() if _server_read_config is not None else None

        if cfg is None:
            try:
                mtime = os.path.getmtime(self.config_path)
                if mtime == self.mtime:
                    return
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                self.mtime = mtime
            except (json.JSONDecodeError, OSError):
                return

        if not cfg:
            return

        self.config = cfg

        raw_p1 = cfg.get("p1", [0, self.frame_height // 2])
        raw_p2 = cfg.get("p2", [self.frame_width - 1, self.frame_height // 2])
        raw_alert = cfg.get("alert_side_pt", [self.frame_width // 2, self.frame_height - 1])
        raw_threshold = float(cfg.get("threshold_pixels", 0.0))

        # The stored coordinates are in whatever frame_width/frame_height
        # was active when they were saved (config.get("frame_width"/
        # "frame_height")). If the live capture resolution has since
        # changed, scale proportionally before clamping -- same rescale
        # line_config_server.py's /get_lines route already does for the
        # browser UI. Without this, the UI and the actual detection line
        # silently disagree whenever resolution changes (e.g. the camera
        # renegotiates a different mode than what's in config.json).
        cfg_w = cfg.get("frame_width") or self.frame_width
        cfg_h = cfg.get("frame_height") or self.frame_height
        if cfg_w and cfg_h and (cfg_w, cfg_h) != (self.frame_width, self.frame_height):
            sx = self.frame_width / cfg_w
            sy = self.frame_height / cfg_h
            raw_p1 = [raw_p1[0] * sx, raw_p1[1] * sy]
            raw_p2 = [raw_p2[0] * sx, raw_p2[1] * sy]
            raw_alert = [raw_alert[0] * sx, raw_alert[1] * sy]
            raw_threshold *= (sx + sy) / 2.0

        self.p1 = self._clamp(raw_p1)
        self.p2 = self._clamp(raw_p2)
        self.threshold = raw_threshold

        ref = self._clamp(raw_alert)
        ref_dist = self._signed_distance(ref[0], ref[1])
        if ref_dist > 0:
            self._alert_sign = 1
        elif ref_dist < 0:
            self._alert_sign = -1
        else:
            self._alert_sign = 1

    def filter_train_detections(self, detections: list) -> list:
        """Keep only Train detections whose centroid is on the alert side
        of the line; drop the rest (non-Person detections never reach
        _box_is_alert's multi-point check, so this centroid-only test is
        deliberately simpler than the Person alert logic below)."""
        if not detections:
            return []
        filtered = []
        for det in detections:
            x1, y1, x2, y2, label, conf = det
            if label == "Train":
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if not self._centroid_is_alert(cx, cy):
                    continue
            filtered.append(det)
        return filtered

    def update(self, detections, reload: bool = True):
        """Return alert-side flags and store crossing events in last_crossing_events.

        reload=False lets a caller that already called load() earlier in
        the same frame (video_widget.py does, every frame, to pick up
        threshold/image-adjustment changes even on non-pipeline frames)
        skip doing the config dict-copy + clamp + signed-distance-recompute
        work a second time here.
        """
        if reload:
            self.load()
        self.last_crossing_events = []

        detections = self.filter_train_detections(detections)

        if not detections:
            self.tracks = []
            return []

        alert_flags = []
        new_tracks = []
        used_track_ids = set()

        for det in detections:
            x1, y1, x2, y2, label, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Checked against the line using centroid + feet/lower-body points for robust alert detection
            is_alert = label == "Person" and self._box_is_alert(x1, y1, x2, y2)

            matched_track = None
            best_d = float("inf")
            for track in self.tracks:
                if track["id"] in used_track_ids or track["label"] != label:
                    continue
                d = math.hypot(cx - track["cx"], cy - track["cy"])
                if d < best_d and d < MATCH_RADIUS:
                    best_d = d
                    matched_track = track

            if matched_track is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                was_alert = False
            else:
                track_id = matched_track["id"]
                used_track_ids.add(track_id)
                was_alert = matched_track["is_alert"]

            if is_alert and not was_alert:
                self.last_crossing_events.append({
                    "track_id": track_id,
                    "detection": det,
                })

            new_tracks.append({
                "id": track_id,
                "cx": cx,
                "cy": cy,
                "label": label,
                "is_alert": is_alert,
            })
            alert_flags.append(is_alert)

        self.tracks = new_tracks
        return alert_flags

    def _signed_distance(self, px, py):
        x1, y1 = self.p1
        x2, y2 = self.p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0
        return ((px - x1) * dy - (py - y1) * dx) / length

    def _box_is_alert(self, x1, y1, x2, y2):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if self._centroid_is_alert(cx, cy):
            return True
        if self._centroid_is_alert(cx, float(y2)):
            return True
        if self._centroid_is_alert(cx, y1 + (y2 - y1) * 0.75):
            return True
        return False

    def _centroid_is_alert(self, cx, cy):
        sd = self._signed_distance(cx, cy)
        if abs(sd) <= self.threshold:
            return True
        return (sd * self._alert_sign) > 0

    def draw_line_opencv(self, frame):
        cv2.line(frame, self.p1, self.p2, (0, 255, 255), 2)