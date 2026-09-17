from collections import Counter

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from config import (
    IMG_H,
    IMG_W,
    MIN_GARMENT_CONFIDENCE,
    GARMENT_CROP_UPPER_FRAC,
    TORSO_CENTRALITY_SIGMA_FRAC,
    ALERT_EXCLUDED_COLORS,
    POSE_FAIL_FALLBACK_STREAK,
    POSE_FALLBACK_ENABLED,
    DEBUG,
    DEBUG_CROPS_DIR,
)
from pose_detector import PoseDetector


class SegmentAnalyzer:
    """Describe a person crop for alert text.

    Garment type  — int8 TFLite classifier, inference via the shared
                     NPUWorker thread (see npu_worker.py); only
                     preprocessing/postprocessing happen here.
    Color         — cv2 HSV, no NPU involved at all -- this was already
                     pure CPU work before the threading refactor and stays
                     exactly as it was.
    """

    MODEL_NAME = "clothing"

    def __init__(
        self,
        npu_worker=None,
        labels_path: str = "",
        enabled: bool = True,
        pt_pose_model_path: str = "",
    ):
        self.enabled      = bool(enabled)
        self.npu_worker   = npu_worker
        self.labels_path  = labels_path

        self._loaded      = False
        self._available   = False
        self._model_info  = None
        self.labels: list = []

        # Stashed during the current describe_person() call so _save_debug_crops
        # can access them without changing method return signatures.
        self._last_torso_crop = None
        self._last_garment_model_input_bgr = None

        # Whether the MOST RECENT describe_person() call actually detected
        # a color (as opposed to skipping it -- frame-quality gate, pose
        # rejection, etc.). Public (no underscore) since callers like
        # AlertEngine need to check it: describe_person()'s return value
        # alone can't distinguish "color detected" from "no color, only a
        # garment label" once both collapse into plain strings.
        self.last_color_detected = False

        # Buffered debug lines from garment classification (scale/zp/range,
        # top/conf/all) -- held here instead of printed immediately, so
        # they can be dropped together with the debug-crop save below when
        # this detection ends up with no color. A frame with no color
        # won't alert either, so printing garment-classifier internals for
        # it is just console noise.
        self._debug_lines: list = []

        # Precomputed once so the per-frame membership check in
        # describe_person is a cheap lowercase set lookup.
        self._excluded_colors_lower = {c.lower() for c in ALERT_EXCLUDED_COLORS}

        # Pose detector — separate module, TFLite path shares this same
        # NPUWorker so pose invoke() also runs on the dedicated NPU core.
        self._pose_detector = PoseDetector(
            npu_worker=npu_worker,
            pt_model_path=pt_pose_model_path,
        )

        # ------------------------------------------------------------------
        # Vote buffer — accumulates per-inference-frame colour + garment
        # results. Flushed (voted on + reset) by get_voted_description() after
        # VOTE_WINDOW_FRAMES frames are collected, or reset immediately by
        # reset_votes() when no person is detected in a frame.
        # ------------------------------------------------------------------
        self._color_votes:   list = []   # str | None per frame
        self._garment_votes: list = []   # str | None per frame
        self._last_valid_debug_data = None  # (debug_lines, crop, upper) from last valid frame
        self.last_timing = {"pose_ms": 0.0, "cloth_ms": 0.0}

        # Consecutive collect_vote() calls (for whatever's currently
        # occupying the vote buffer -- reset together with it) where pose
        # failed to produce a torso crop. See POSE_FAIL_FALLBACK_STREAK /
        # POSE_FALLBACK_ENABLED in config.py for how this is used below.
        self._consecutive_pose_failures = 0

        # Reused across every _normalize_illumination() call instead of
        # constructing a fresh cv2.CLAHE object per vote frame -- the
        # clip/tile params never change, so there's nothing gained by
        # rebuilding it every time someone's in the alert zone (exactly
        # the path where per-frame latency matters most).
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Single-entry cache for _centrality_weights()'s Gaussian mask,
        # keyed by (img_h, img_w) -- same drop-and-rebuild-once pattern as
        # video_widget.py's gamma LUT cache. Consecutive vote frames for
        # the same person almost always hand in a torso crop of identical
        # or near-identical size, so recomputing the full np.mgrid + exp
        # mask from scratch every single vote is wasted work most of the
        # time.
        self._centrality_cache = {"key": None, "weights": None}


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # -- Vote buffer properties -----------------------------------------

    @property
    def votes_collected(self) -> int:
        """Number of valid (colour-detected) frames currently in the vote buffer."""
        return len(self._color_votes)

    def reset_votes(self):
        """Clear the vote buffer without firing an alert.

        Call this when no person is detected in the zone for a frame, so
        stale votes from a previous person don't bleed into the next.
        """
        self._color_votes   = []
        self._garment_votes = []
        self._last_valid_debug_data = None
        self.last_timing = {"pose_ms": 0.0, "cloth_ms": 0.0}
        self._consecutive_pose_failures = 0

    def collect_vote(self, frame_bgr, bbox):
        """Run one inference frame through colour + garment detection.

        The frame is only counted (added to the vote buffer) when a valid
        colour is detected. Frames with no colour are silently skipped so
        they don't dilute the window or delay the alert.

        Does NOT fire any alert — call get_voted_description() once
        votes_collected >= VOTE_WINDOW_FRAMES to tally.

        Parameters
        ----------
        frame_bgr : np.ndarray
            The full camera frame.
        bbox : tuple
            (x1, y1, x2, y2) person bounding box.
        """
        crop = self._crop(frame_bgr, bbox)
        self._debug_lines = []

        if crop.size == 0:
            return   # empty crop — skip, don't count

        self._last_garment_model_input_bgr = None
        upper = self._upper_region(crop)

        # ── Garment classification ────────────────────────────────────────
        t_cloth = time.time()
        garment_raw = None
        if self.enabled:
            self._load_model_once()
            if self._available:
                garment_raw = self._classify_garment(upper)
        cloth_ms = (time.time() - t_cloth) * 1000.0

        # ── Colour detection ──────────────────────────────────────────────
        t_pose = time.time()
        color_raw = None
        pose_torso = self._pose_detector.get_torso_crop(crop)
        if pose_torso is not None and pose_torso.size > 0:
            self._consecutive_pose_failures = 0
            color_raw = self._hsv_color_from_torso(pose_torso)
        else:
            # No shoulder keypoints -> no torso crop. By default
            # (POSE_FALLBACK_ENABLED=False) this frame is skipped entirely
            # for colour purposes -- color_raw stays None, so the
            # "no colour = skip, don't count" check below drops it.
            # If POSE_FALLBACK_ENABLED is flipped on, a genuine RUN of
            # POSE_FAIL_FALLBACK_STREAK consecutive failures instead falls
            # back to colour detection on the upper-body crop, so the vote
            # window isn't starved just because pose can't lock on for
            # this particular person/pose.
            self._consecutive_pose_failures += 1
            if POSE_FALLBACK_ENABLED and self._consecutive_pose_failures >= POSE_FAIL_FALLBACK_STREAK:
                fallback_torso = upper
                if fallback_torso is not None and fallback_torso.size > 0:
                    color_raw = self._hsv_color_from_torso(fallback_torso)
                    if DEBUG:
                        self._debug_lines.append(
                            f"[Segment] Pose detection failed ({self._consecutive_pose_failures} consecutive frames) -> used upper-body fallback crop"
                        )
            elif DEBUG:
                self._debug_lines.append(
                    f"[Segment] No shoulder points detected ({self._consecutive_pose_failures} "
                    "consecutive frames) -> frame skipped, no colour vote"
                )
        pose_ms = (time.time() - t_pose) * 1000.0

        self.last_timing = {"pose_ms": pose_ms, "cloth_ms": cloth_ms}

        # Apply excluded-color filter
        if color_raw and color_raw.lower() in self._excluded_colors_lower:
            color_raw = None

        # Only count this frame if a valid colour was detected.
        # No colour = no information worth voting on — skip silently.
        if color_raw is None:
            self._debug_lines = []
            return

        # Store debug info for this valid frame (printed only when alert triggers)
        pose_vis = getattr(self._pose_detector, "last_pose_vis", None)
        self._last_valid_debug_data = (list(self._debug_lines), crop, upper, pose_vis)
        self._debug_lines = []

        self._color_votes.append(color_raw)
        self._garment_votes.append(garment_raw)


    def get_voted_description(self) -> tuple:
        """Tally the vote buffer and return the plurality winners.

        Returns
        -------
        (description: str, has_color: bool, color_winner: str|None, garment_winner: str|None)
            description    — alert-ready string, e.g. ``"red Shirt"``.
            has_color      — True if a valid colour was detected across the window.
            color_winner   — raw colour name (e.g. "red"), or None.
            garment_winner — raw garment label from labels.txt (e.g. "Shirt"), or None.
                             AlertEngine slugifies this itself to find the
                             matching voices/<lang>/garments/<slug>.wav clip.

        Side effect: resets the vote buffer.
        """
        color_counter   = Counter(self._color_votes)
        garment_counter = Counter(v for v in self._garment_votes if v)

        color_winner   = color_counter.most_common(1)[0][0]   if color_counter   else None
        garment_winner = garment_counter.most_common(1)[0][0] if garment_counter else None

        has_color = bool(color_winner)

        garment_display = garment_winner.title() if garment_winner else None

        if has_color and garment_display:
            description = f"{color_winner} {garment_display}"
        elif garment_display:
            description = garment_display
        elif has_color:
            description = f"{color_winner} top"
        else:
            description = "person"

        self.last_color_detected = has_color

        if DEBUG and has_color and self._last_valid_debug_data is not None:
            debug_lines, crop, upper, pose_vis = self._last_valid_debug_data
            for line in debug_lines:
                print(line)
            self._save_debug_crops(crop, upper, color_winner, garment_winner, pose_vis=pose_vis)

        self.reset_votes()

        return description, has_color, color_winner, garment_winner

    def describe_person(self, frame_bgr, bbox) -> str:
        """Single-frame convenience path (preserves existing API).

        Collects one vote and immediately flushes — identical to the old
        single-frame behaviour. Use collect_vote() + get_voted_description()
        for the multi-frame voting flow.
        """
        self.collect_vote(frame_bgr, bbox)
        description, has_color, _color, _garment = self.get_voted_description()
        return description

    # ------------------------------------------------------------------
    # Model loading — reads labels.txt + grabs shape/quantization info
    # from the shared NPUWorker (already loaded on its own thread).
    # ------------------------------------------------------------------
    def _load_model_once(self):
        if self._loaded:
            return
        self._loaded = True

        if not self.labels_path:
            self._available = False
            print("[Segment] No labels_path — color-only fallback.")
            return

        try:
            with open(self.labels_path, "r") as f:
                self.labels = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            self._available = False
            print(f"[Segment] Failed to read labels — color-only fallback: {exc}")
            return

        info = self.npu_worker.get_model_info(self.MODEL_NAME) if self.npu_worker else None
        if info is None:
            self._available = False
            print("[Segment] 'clothing' model not loaded on NPUWorker — color-only fallback.")
            return

        self._model_info = info
        self._available = True
        if DEBUG:
            print(f"[Segment] Labels ({len(self.labels)}): {self.labels}")

    # ------------------------------------------------------------------
    # Garment classification (int8 TFLite, invoke via NPUWorker)
    # ------------------------------------------------------------------
    def _preprocess(self, frame_bgr):
        # Resize first so the BGR->RGB swap only touches 224x224 pixels,
        # not the full upper-body crop.
        img = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        img = img[:, :, ::-1]

        # Stash exactly what's about to go into the model (still uint8,
        # pre-quantization -- quantizing doesn't change the visual content,
        # just how it's numerically encoded) so describe_person() can save
        # it via _save_debug_crops().
        self._last_garment_model_input_bgr = img[:, :, ::-1].copy()

        arr = img.astype(np.float32)              # raw 0-255, no scaling, no ImageNet norm
        arr = arr[np.newaxis, ...]

        inp_scale = self._model_info["input_scale"]
        inp_zp    = self._model_info["input_zero"]

        pre_quant = arr / inp_scale + inp_zp
        if DEBUG:
            sat = np.count_nonzero((pre_quant <= -128) | (pre_quant >= 127)) / pre_quant.size
            self._debug_lines.append(
                f"[Segment] scale={inp_scale} zp={inp_zp} "
                f"range=({pre_quant.min():.1f},{pre_quant.max():.1f}) saturated={sat*100:.1f}%"
            )

        # Reuses pre_quant instead of recomputing arr/inp_scale+inp_zp a
        # second time — same expression was computed twice in the original.
        arr_int8 = np.clip(np.round(pre_quant), -128, 127).astype(np.int8)
        return arr_int8

    def _classify_garment(self, crop_bgr):
        try:
            x = self._preprocess(crop_bgr)

            outputs_raw = self.npu_worker.infer(self.MODEL_NAME, x)
            if outputs_raw is None:
                return None  # NPU busy/timed out -- sacrifice this vote frame

            out_detail = self._model_info["output_details"][0]
            out_scale, out_zp = out_detail.get("quantization", (0.0, 0))
            raw = outputs_raw[0]                                          # INT8 (1,9)
            probs = (raw.astype(np.float32) - out_zp) * out_scale         # already real probabilities
            probs = probs[0]                                              # (9,)
            # no softmax here — the model already produced it

            top_idx  = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            if DEBUG:
                self._debug_lines.append(
                    f"[Segment] top={self.labels[top_idx]} conf={top_conf:.3f} all={np.round(probs,2)}"
                )

            if top_conf < MIN_GARMENT_CONFIDENCE:
                return None
            if top_idx < len(self.labels):
                return self.labels[top_idx]
            return None
        except Exception as exc:
            print(f"[Segment] Inference error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Color detection — cv2 HSV. Unchanged: this never touched the NPU.
    # ------------------------------------------------------------------
    def _normalize_illumination(self, hsv: np.ndarray) -> np.ndarray:
        """Equalize the V channel with CLAHE so a dim crop and a bright
        crop of the SAME actual garment color end up with similar V
        before the fixed thresholds in _hsv_color_from_torso are applied.

        Only V is touched. H is untouched. S is left alone too --
        under a plain brightness/gain change, HSV saturation
        S = (max-min)/max is scale-invariant (the k cancels out), so
        the instability was coming from V pushing past the fixed
        black/white cutoffs, not from S itself drifting.

        CLAHE (as opposed to a single global histogram stretch) also
        keeps this well-behaved when only part of the torso crop is
        lit unevenly (e.g. one shoulder catching more light).

        Reuses self._clahe (built once in __init__) instead of
        constructing a new cv2.CLAHE object on every call -- this runs
        once per vote frame while a person is in the alert zone, i.e.
        exactly the path where shaving per-frame CPU cost matters most.
        """
        v_u8 = np.clip(hsv[:, :, 2], 0, 255).astype(np.uint8)
        v_eq = self._clahe.apply(v_u8).astype(np.float32)
        hsv_eq = hsv.copy()
        hsv_eq[:, :, 2] = v_eq
        return hsv_eq

    def _hsv_color_from_torso(self, torso_bgr) -> str:
        """Run weighted HSV colour analysis on an already-cropped torso.

        Called with the pose-defined shoulder-to-hip crop from
        PoseDetector.get_torso_crop().
        """
        if torso_bgr is None or torso_bgr.size == 0:
            return "unknown"

        self._last_torso_crop = torso_bgr

        hsv_raw = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        raw_mean_v = float(np.mean(hsv_raw[:, :, 2]))
        if raw_mean_v < 18:
            return "black"

        hsv = self._normalize_illumination(hsv_raw)
        img_h, img_w = hsv.shape[:2]
        h = hsv[:, :, 0].reshape(-1)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)

        weight = self._centrality_weights(img_h, img_w).reshape(-1)
        total_w = float(np.sum(weight)) or 1.0

        black_ratio = float(np.sum(weight[v < 55])) / total_w
        white_ratio = float(np.sum(weight[(s < 35) & (v > 185)])) / total_w
        gray_ratio  = float(np.sum(weight[(s < 35) & (v >= 55) & (v <= 185)])) / total_w

        if black_ratio >= 0.38:
            return "black"
        if white_ratio >= 0.38:
            return "white"
        if gray_ratio  >= 0.45:
            return "gray"

        color_mask = (s >= 35) & (v >= 45)
        color_w = float(np.sum(weight[color_mask]))
        if color_w < max(20.0, total_w * 0.08):
            if black_ratio >= white_ratio and black_ratio >= gray_ratio:
                return "black"
            if white_ratio >= gray_ratio:
                return "white"
            return "gray"

        h_sel = h[color_mask]
        s_sel = s[color_mask]
        v_sel = v[color_mask]
        w_sel = weight[color_mask]

        vote_weights = np.clip(s_sel, 1, 255) * np.clip(v_sel, 1, 255) * w_sel

        n_bins = 30
        bin_width = 180.0 / n_bins
        bin_idx = (h_sel / bin_width).astype(np.int32) % n_bins
        bin_votes = np.bincount(bin_idx, weights=vote_weights, minlength=n_bins)

        wrap_votes = bin_votes.copy()
        wrap_votes[0] += bin_votes[-1]
        wrap_votes[-1] += bin_votes[0]

        peak_bin = int(np.argmax(wrap_votes))
        peak_bins = {(peak_bin - 1) % n_bins, peak_bin, (peak_bin + 1) % n_bins}
        peak_mask = np.isin(bin_idx, list(peak_bins))

        h_peak = h_sel[peak_mask]
        s_peak = s_sel[peak_mask]
        v_peak = v_sel[peak_mask]
        w_peak = vote_weights[peak_mask]

        angles  = h_peak * (np.pi / 90.0)
        sin_sum = float(np.sum(np.sin(angles) * w_peak))
        cos_sum = float(np.sum(np.cos(angles) * w_peak))
        hue     = (np.arctan2(sin_sum, cos_sum) * 90.0 / np.pi) % 180.0

        sat = self._weighted_percentile(s_peak, w_peak, 65)
        val = self._weighted_percentile(v_peak, w_peak, 65)
        return self._hsv_to_name(hue, sat, val)

    def _centrality_weights(self, img_h: int, img_w: int) -> np.ndarray:
        """Gaussian weight mask centered on the torso crop, fading toward
        the edges.

        General reasoning, not tuned to any one scene: for a reasonably
        framed person box, the actual garment is expected to sit near the
        middle of the torso region, while anything that leaked into the box
        from outside the person -- a wall-mounted object, a bag, a railing,
        another person's arm, whatever it is, on whatever side it's on --
        sits nearer the edges. Down-weighting by distance from center
        reduces that leakage's influence on the color vote without
        assuming anything about its color, shape, or size.

        Cached by (img_h, img_w) in self._centrality_cache -- consecutive
        vote frames for the same person almost always hand in a torso crop
        of identical or near-identical size, so rebuilding this np.mgrid +
        exp mask from scratch on every single vote is wasted work most of
        the time. Same single-entry, drop-and-rebuild-once pattern as
        video_widget.py's gamma LUT cache.
        """
        key = (img_h, img_w)
        if self._centrality_cache["key"] == key:
            return self._centrality_cache["weights"]

        yy, xx = np.mgrid[0:img_h, 0:img_w].astype(np.float32)
        cy, cx = (img_h - 1) / 2.0, (img_w - 1) / 2.0
        sigma_x = max(img_w * TORSO_CENTRALITY_SIGMA_FRAC, 1.0)
        sigma_y = max(img_h * TORSO_CENTRALITY_SIGMA_FRAC, 1.0)
        weights = np.exp(
            -(((xx - cx) ** 2) / (2 * sigma_x ** 2) + ((yy - cy) ** 2) / (2 * sigma_y ** 2))
        )
        self._centrality_cache = {"key": key, "weights": weights}
        return weights

    @staticmethod
    def _weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
        """Weighted equivalent of np.percentile(values, pct).

        Same intent as the original unweighted percentile call (a "typical"
        value that's robust to a handful of extreme pixels, not a plain
        mean) -- just respecting each pixel's centrality weight instead of
        counting every pixel equally.
        """
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v_sorted = values[order]
        w_sorted = weights[order]
        cum = np.cumsum(w_sorted)
        if cum[-1] <= 0:
            return float(np.mean(values))
        cutoff = (pct / 100.0) * cum[-1]
        idx = min(int(np.searchsorted(cum, cutoff)), len(v_sorted) - 1)
        return float(v_sorted[idx])

    @staticmethod
    def _hsv_to_name(h: float, s: float, v: float) -> str:
        """Map HSV values to a color name string.
        OpenCV ranges: H 0-179, S 0-255, V 0-255.
        """
        if v < 50:
            return "black"
        if s < 28:
            return "white" if v > 185 else "gray"

        if 5 <= h <= 27:
            if v < 115 and s > 45:
                return "brown"
            if s < 70 and v > 175:
                return "cream" if h >= 18 else "beige"

        if h < 5 or h >= 172:   return "pink"  if (v > 180 and s < 150) else "red"
        if h < 18:               return "orange"
        if h < 35:               return "yellow"
        if h < 48 and s < 100 and v > 145:
                                 return "khaki"
        if h < 82:               return "green"
        if h < 96:               return "cyan"  if s > 90 else "sky blue"
        if h < 130:              return "navy"  if v < 120 else "blue"
        if h < 150:              return "purple"
        if h < 170:              return "pink"
        return "red"

    # ------------------------------------------------------------------
    # Crop / region helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _crop(frame_bgr, bbox):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return frame_bgr[0:0, 0:0]
        return frame_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _upper_region(crop_bgr, frac: float = GARMENT_CROP_UPPER_FRAC):
        h = crop_bgr.shape[0]
        return crop_bgr[: max(1, int(h * frac)), :]

    # ------------------------------------------------------------------
    # Unified per-detection debug dump — only when DEBUG=True, no throttle.
    # ------------------------------------------------------------------
    def _save_debug_crops(self, crop_bgr, upper_bgr, color: str, garment_label, pose_vis=None):
        if not DEBUG:
            return
        try:
            out_dir = Path(DEBUG_CROPS_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if crop_bgr is not None and crop_bgr.size:
                cv2.imwrite(str(out_dir / f"{stamp}_1_crop.jpg"), crop_bgr)
            if upper_bgr is not None and upper_bgr.size:
                cv2.imwrite(str(out_dir / f"{stamp}_2_upper.jpg"), upper_bgr)
            if pose_vis is not None and getattr(pose_vis, "size", 0):
                cv2.imwrite(str(out_dir / f"{stamp}_3_pose_points.jpg"), pose_vis)
            if self._last_torso_crop is not None and self._last_torso_crop.size:
                cv2.imwrite(str(out_dir / f"{stamp}_4_torso.jpg"), self._last_torso_crop)
            if self._last_garment_model_input_bgr is not None and self._last_garment_model_input_bgr.size:
                cv2.imwrite(
                    str(out_dir / f"{stamp}_5_model_input.jpg"),
                    self._last_garment_model_input_bgr,
                )

            print(
                f"[Segment] Debug crops saved: {out_dir}/{stamp}_*  "
                f"(color={color}, garment={garment_label})"
            )
        except Exception as exc:
            print(f"[Segment] Debug crop save failed: {exc}")