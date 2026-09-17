import argparse
import os
import signal
import time

from config import (
    BOX_HEIGHT_SCALE,
    BOX_WIDTH_SCALE,
    LABELS_PATH,
    DETECTION_ON,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAMERATE,
    IOU_TH,
    LINE_DETECTION_ENABLED,
    SEGMENTATION_ENABLED,
    PLATFORM_CHOICES,
    BACKEND_CHOICES,
    BACKEND,
    VIDEO_PATH,
    CAMERA_DEVICE,
    VOTE_WINDOW_FRAMES,
    CORE_NPU_ID,
    NPU_INFER_TIMEOUT_S,
    TRAIN_SUPPRESS_DELAY_S,
    TRAIN_RESUME_DELAY_S,
    resolve_model_paths,
    DEBUG,
    _model_label,
)
from alert_engine import AlertEngine
from detection import Detector
from segment import SegmentAnalyzer
from npu_worker import NPUWorker
from video_widget import CaptureWidget
from line_config_server import start_server

MULTI_PERSON_CONFIRM_FRAMES = 2


class Application:
    def __init__(self, args):
        yolo_model_path, clothing_model_path, pose_model_path = resolve_model_paths(args.platform, args.backend)
        self.model_path = yolo_model_path
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.framerate = FRAMERATE
        self.iou_threshold = IOU_TH
        self.source = args.source
        self.video_path = args.video_path
        self.camera_device = args.camera_device
        self.platform = args.platform
        self.backend = args.backend
        self.debug_detections = DEBUG
        self.line_detection_enabled = LINE_DETECTION_ENABLED
        self.box_width_scale = BOX_WIDTH_SCALE
        self.box_height_scale = BOX_HEIGHT_SCALE

        self._multi_person_streak = 0

        # Train suppression state variables
        self._train_suppress_at: float | None = None
        self._train_resume_at: float | None = None
        self._train_was_present: bool = False

        self.npu_worker = NPUWorker(
            model_paths={
                "yolo": yolo_model_path,
                "pose": pose_model_path if SEGMENTATION_ENABLED else "",
                "clothing": clothing_model_path if SEGMENTATION_ENABLED else "",
            },
            platform=args.platform,
            backend=args.backend,
            core_id=CORE_NPU_ID,
            infer_timeout_s=NPU_INFER_TIMEOUT_S,
        )
        self.npu_worker.start()
        if not self.npu_worker.wait_ready(timeout=60.0):
            raise RuntimeError("NPUWorker failed to load models within 60s.")

        self.alert_engine = AlertEngine(cooldown_seconds=30.0)

        self.detector = Detector(
            npu_worker=self.npu_worker,
            iou_threshold=self.iou_threshold,
            debug=self.debug_detections,
            box_width_scale=self.box_width_scale,
            box_height_scale=self.box_height_scale,
        )
        self.segment_analyzer = SegmentAnalyzer(
            npu_worker=self.npu_worker,
            labels_path=LABELS_PATH,
            enabled=SEGMENTATION_ENABLED,
        )

        self.capture_widget = CaptureWidget(self)

    def run(self):
        self.capture_widget.run()
        self.npu_worker.stop()

    def _confirm_multi_person(self) -> bool:
        self._multi_person_streak += 1
        return self._multi_person_streak >= MULTI_PERSON_CONFIRM_FRAMES

    def _update_train_suppression(self, train_present: bool) -> bool:
        """Update train suppression timing logic using config settings:
        - TRAIN_SUPPRESS_DELAY_S (5.0s): Delay after train appears before full suppression
        - TRAIN_RESUME_DELAY_S: Delay after train leaves before alerts resume (see config.py)

        NOTE on vote-buffer resets -- READ BEFORE TOUCHING THIS AGAIN:
        This is the ONLY place that clears segment_analyzer's vote
        buffer for train-related transitions:
          * LEADING edge (train just appeared) -- whatever was mid-vote
            for a previous person is stale, so it's dropped.
          * TRAILING edge (train just left) -- any partial votes
            collected during the grace window right before suppression
            kicked in are dropped too, so they can't silently blend with
            fresh votes collected once alerts resume.

        Callers (handle_alert_state / handle_person_detection_events)
        must NOT also call segment_analyzer.reset_votes() just because
        train_present is True for the current frame. That was a real
        bug that used to live in both of those methods: it fired every
        single frame a train was anywhere in view (not just on the
        appear/leave edges), so a fresh vote collected earlier in that
        same frame got wiped before it could ever reach
        VOTE_WINDOW_FRAMES. Net effect: no single-person alert could
        ever fire while a train was on screen -- including during the
        5s grace window where alerts are explicitly supposed to still
        work. Don't reintroduce it.
        """
        now = time.monotonic()

        # Leading edge: train just appeared
        if train_present and not self._train_was_present:
            if self._train_suppress_at is None:
                self._train_suppress_at = now + TRAIN_SUPPRESS_DELAY_S
                if DEBUG:
                    print(f"[Train] Train detected — suppression activates in {TRAIN_SUPPRESS_DELAY_S}s")
            self._train_resume_at = None
            self.segment_analyzer.reset_votes()

        # Trailing edge: train just left
        if not train_present and self._train_was_present:
            self._train_resume_at = now + TRAIN_RESUME_DELAY_S
            self._train_suppress_at = None
            # Drop any partial vote buffer collected right before/while
            # the train was present (grace-window votes) so it doesn't
            # blend with whatever shows up once alerts resume.
            self.segment_analyzer.reset_votes()
            if DEBUG:
                print(f"[Train] Train left — alerts resume in {TRAIN_RESUME_DELAY_S}s")

        self._train_was_present = train_present

        # Evaluate suppression status
        if train_present:
            if self._train_suppress_at is not None and now >= self._train_suppress_at:
                return True
        elif self._train_resume_at is not None:
            if now < self._train_resume_at:
                return True
            else:
                self._train_resume_at = None
                self._train_suppress_at = None
                if DEBUG:
                    print("[Train] Train resume delay ended — alerts resumed.")

        return False

    def handle_alert_state(self, frame, detections_with_alerts):
        """Collect one vote per frame for any person inside the zone."""
        train_present = any(
            str(det[4]).lower() == "train" for det, _ in detections_with_alerts
        )

        # NOTE: do NOT call self.segment_analyzer.reset_votes() here just
        # because train_present is True -- _update_train_suppression()
        # already owns vote-buffer resets on the appear/leave edges. See
        # its docstring for why a per-frame reset here silently defeats
        # the whole grace-period window.
        if self._update_train_suppression(train_present):
            self._multi_person_streak = 0
            self.alert_engine.person_in_zone = False
            return

        alertable_persons = [
            det for det, is_alert in detections_with_alerts
            if is_alert and str(det[4]).lower() == "person"
        ]

        self.alert_engine.person_in_zone = bool(alertable_persons)

        if not alertable_persons:
            self._multi_person_streak = 0
            self.segment_analyzer.reset_votes()
            return

        if len(alertable_persons) > 1:
            self.segment_analyzer.reset_votes()
            if self._confirm_multi_person():
                self.alert_engine.trigger_general()
            return

        self._multi_person_streak = 0

        x1, y1, x2, y2 = alertable_persons[0][:4]
        self.segment_analyzer.collect_vote(frame, (x1, y1, x2, y2))

        if self.segment_analyzer.votes_collected < VOTE_WINDOW_FRAMES:
            return

        if not self.alert_engine.can_trigger():
            self.segment_analyzer.reset_votes()
            return

        description, has_color, color, garment = self.segment_analyzer.get_voted_description()
        self.alert_engine.trigger(description, has_color=has_color, color=color, garment=garment)

    def handle_person_detection_events(self, frame, detections):
        """Collect one vote per frame for any detected person."""
        train_present = any(str(det[4]).lower() == "train" for det in detections)

        # See the matching note in handle_alert_state(): no per-frame
        # reset_votes() here -- _update_train_suppression() owns it.
        if self._update_train_suppression(train_present):
            self._multi_person_streak = 0
            self.alert_engine.person_in_zone = False
            return

        person_dets = [det for det in detections if str(det[4]).lower() == "person"]

        self.alert_engine.person_in_zone = bool(person_dets)

        if not person_dets:
            self._multi_person_streak = 0
            self.segment_analyzer.reset_votes()
            return

        if len(person_dets) > 1:
            self.segment_analyzer.reset_votes()
            if self._confirm_multi_person():
                self.alert_engine.trigger_general()
            return

        self._multi_person_streak = 0

        x1, y1, x2, y2 = person_dets[0][:4]
        self.segment_analyzer.collect_vote(frame, (x1, y1, x2, y2))

        if self.segment_analyzer.votes_collected < VOTE_WINDOW_FRAMES:
            return

        if not self.alert_engine.can_trigger():
            self.segment_analyzer.reset_votes()
            return
        description, has_color, color, garment = self.segment_analyzer.get_voted_description()
        self.alert_engine.trigger(description, has_color=has_color, color=color, garment=garment)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO TFLite line-zone detector for i.MX93/Linux camera and computer OpenCV sources."
    )
    parser.add_argument(
        "--source",
        default=DETECTION_ON,
        choices=["Camera", "Video"],
        help="Camera opens a v4l2 device node directly via OpenCV. Video uses OpenCV webcam/file/stream.",
    )
    parser.add_argument(
        "--video_path",
        default=VIDEO_PATH,
        help="OpenCV source for --source Video: camera index, file path, or stream URL.",
    )
    parser.add_argument(
        "--camera_device",
        default=CAMERA_DEVICE,
        help="v4l2 device node for --source Camera (e.g. /dev/video0).",
    )
    parser.add_argument(
        "--platform",
        default="PC",
        choices=list(PLATFORM_CHOICES),
        help="Target board, normally supplied by launcher.py after it detects the hardware.",
    )
    parser.add_argument(
        "--backend",
        default=BACKEND,
        choices=list(BACKEND_CHOICES),
        help="Inference backend in use, normally supplied by launcher.py. Ignored on PC (always CPU).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args()
    if args.platform == "PC":
        args.backend = "CPU"

    yolo_model_path, clothing_model_path, pose_model_path = resolve_model_paths(args.platform, args.backend)
    label = _model_label(yolo_model_path)

    print("=" * 60)
    print(f"  {label} Detection")
    print(f"  Platform   : {args.platform}   Backend : {args.backend}")
    source_detail = ""
    if args.source == "Video":
        source_detail = f" -> {args.video_path}"
    elif args.source == "Camera":
        source_detail = f" -> {args.camera_device}"
    print(f"  Source     : {args.source}{source_detail}")
    print(f"  Resolution : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Confidence : Person/Train -- set via dashboard (config.json)   IoU : {IOU_TH}")
    print("  Alert gap  : set via dashboard (config.json)")
    print(f"  YOLO       : {yolo_model_path}")
    print(f"  Clothing   : {clothing_model_path}")
    print(f"  Pose       : {pose_model_path}")
    print("=" * 60)

    try:
        app = Application(args)
        start_server()
        app.run()
    except FileNotFoundError as exc:
        print(f"\n[Video] {exc}")
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user, shutting down...")
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        import traceback

        traceback.print_exc()

    print("[Pipeline] Exited cleanly.")
    os._exit(0)