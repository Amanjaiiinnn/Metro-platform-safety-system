"""
pose_detector.py
----------------
Preprocessing + postprocessing for the pose model. TFLite inference itself
goes through the shared NPUWorker thread (see npu_worker.py) so only ONE
thread ever touches NPU interpreters.

The .pt (ultralytics) path is kept for PC/dev only and is NOT run through
NPUWorker -- it stays synchronous on whichever thread calls
get_torso_crop(), same as before this refactor.

Always used by segment.py for the HSV colour crop (shoulder-to-hip torso
region). If pose detection itself fails for a given frame (no keypoints
confident enough, or the resulting bbox is too small -- see
TORSO_MIN_WIDTH_FRAC/TORSO_MIN_HEIGHT_FRAC), get_torso_crop() returns
None and that frame simply contributes no colour vote.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

from config import (
    DEBUG,
    DEBUG_CROPS_DIR,
    POSE_KEYPOINT_CONF_TH,
    TORSO_MARGIN_TOP_FRAC,
    TORSO_MARGIN_BOTTOM_FRAC,
    TORSO_MARGIN_X_FRAC,
    TORSO_MIN_WIDTH_FRAC,
    TORSO_MIN_HEIGHT_FRAC,
)


class PoseDetector:
    MODEL_NAME = "pose"

    # COCO keypoint indices for torso corners
    _KP_L_SHOULDER = 5
    _KP_R_SHOULDER = 6
    _KP_L_HIP      = 11
    _KP_R_HIP      = 12
    _REQUIRED_KP   = {_KP_L_SHOULDER, _KP_R_SHOULDER, _KP_L_HIP, _KP_R_HIP}

    def __init__(self, npu_worker=None, pt_model_path: str = ""):
        """
        npu_worker    : shared NPUWorker instance (TFLite path). Pass None
                         to force the .pt path.
        pt_model_path : ultralytics .pt weights, PC/dev only. Used only
                         when npu_worker has no 'pose' model loaded.
        """
        self.npu_worker = npu_worker
        self._is_tflite = False
        self._model = None  # ultralytics YOLO, .pt path only

        self.last_pose_vis = None

        info = npu_worker.get_model_info(self.MODEL_NAME) if npu_worker else None
        if info is not None:
            self._is_tflite = True
            self.input_h = info["input_h"]
            self.input_w = info["input_w"]
            self.input_dtype = info["input_dtype"]
            self.input_scale = info["input_scale"]
            self.input_zero = info["input_zero"]
        elif pt_model_path:
            self._load_pt(pt_model_path)
        else:
            print(
                "[PoseDetector] No pose model loaded (NPUWorker has none, "
                "no .pt given) -- colour crop disabled."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_torso_crop(self, person_crop_bgr):
        self.last_pose_vis = None
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        if self._is_tflite:
            return self._run_tflite(person_crop_bgr)
        if self._model is not None:
            return self._run_pt(person_crop_bgr)
        return None

    def _save_pose_debug(self, person_crop_bgr, is_success: bool, points: dict = None, torso_box: tuple = None, reason: str = ""):
        """Visualizes detected pose points & torso box on person crop, saving to debug folders."""
        if not DEBUG or person_crop_bgr is None or person_crop_bgr.size == 0:
            return None

        try:
            vis = person_crop_bgr.copy()
            h, w = vis.shape[:2]

            # 1. Draw connecting lines between torso points if available
            if points:
                pairs = [
                    (self._KP_L_SHOULDER, self._KP_R_SHOULDER, (0, 255, 0)),
                    (self._KP_L_SHOULDER, self._KP_L_HIP, (255, 255, 0)),
                    (self._KP_R_SHOULDER, self._KP_R_HIP, (255, 255, 0)),
                    (self._KP_L_HIP, self._KP_R_HIP, (255, 0, 0)),
                ]
                for kp1, kp2, col in pairs:
                    if kp1 in points and kp2 in points:
                        cv2.line(vis, points[kp1], points[kp2], col, 2)

                kp_labels = {
                    self._KP_L_SHOULDER: "L_Sh",
                    self._KP_R_SHOULDER: "R_Sh",
                    self._KP_L_HIP: "L_Hip",
                    self._KP_R_HIP: "R_Hip",
                }
                for idx, pt in points.items():
                    px, py = pt
                    cv2.circle(vis, (px, py), 4, (0, 255, 0) if is_success else (0, 0, 255), -1)
                    lbl = kp_labels.get(idx, str(idx))
                    cv2.putText(vis, lbl, (px + 4, max(12, py - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # 2. Draw torso bounding box if available
            if torso_box is not None:
                x1, y1, x2, y2 = torso_box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis, "Torso Crop", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 3. Header status banner
            banner_h = max(22, int(h * 0.08))
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

            status_text = "POSE: OK" if is_success else f"POSE FAIL: {reason}"
            status_color = (51, 242, 26) if is_success else (0, 0, 255)
            cv2.putText(vis, status_text, (6, max(15, banner_h - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1)

            # 4. Save to subfolder (pose_detected / pose_failed)
            subfolder = "pose_detected" if is_success else "pose_failed"
            out_dir = Path(DEBUG_CROPS_DIR) / subfolder
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prefix = "ok" if is_success else "fail"
            out_path = out_dir / f"{stamp}_{prefix}.jpg"
            cv2.imwrite(str(out_path), vis)

            self.last_pose_vis = vis
            return vis
        except Exception as exc:
            if DEBUG:
                print(f"[PoseDetector] Debug vis save failed: {exc}")
            return None

    def _load_pt(self, model_path):
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            self._model = UltralyticsYOLO(model_path)
            self._is_tflite = False
            print(f"[PoseDetector] PyTorch pose loaded: {model_path}")
        except Exception as exc:
            print(f"[PoseDetector] PyTorch load failed ({exc}) -- colour crop disabled")
            self._model = None

    # ------------------------------------------------------------------
    # TFLite path -- preprocess here, invoke on NPUWorker, postprocess here
    # ------------------------------------------------------------------
    def _run_tflite(self, person_crop_bgr):
        h, w = person_crop_bgr.shape[:2]

        # Resize first so the BGR->RGB swap only touches model-input pixels,
        # not the full person crop.
        img = cv2.resize(person_crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]

        if np.issubdtype(self.input_dtype, np.floating):
            inp = (img.astype(np.float32) / 255.0).astype(self.input_dtype)
        else:
            inp = img.astype(np.float32) / 255.0
            if self.input_scale > 0:
                inp = inp / self.input_scale + self.input_zero
            inp = np.clip(
                inp, np.iinfo(self.input_dtype).min, np.iinfo(self.input_dtype).max
            ).astype(self.input_dtype)

        inp = np.expand_dims(inp, axis=0)

        outputs_raw = self.npu_worker.infer(self.MODEL_NAME, inp)
        if outputs_raw is None:
            self._save_pose_debug(person_crop_bgr, False, reason="NPU timeout / no output")
            return None  # NPU busy/timed out -- sacrifice this vote frame

        info = self.npu_worker.get_model_info(self.MODEL_NAME)
        out_detail = info["output_details"][0]
        out_raw = outputs_raw[0]
        out_scale, out_zp = out_detail.get("quantization", (0.0, 0))
        if out_scale and out_scale > 0 and not np.issubdtype(out_raw.dtype, np.floating):
            out = (out_raw.astype(np.float32) - out_zp) * out_scale
        else:
            out = out_raw.astype(np.float32)

        out = np.squeeze(out)
        if out.ndim != 2:
            self._save_pose_debug(person_crop_bgr, False, reason="Invalid output dimensions")
            return None
        if out.shape[0] < out.shape[1] and out.shape[0] in range(5, 100):
            out = out.T
        if out.shape[1] < 56:
            self._save_pose_debug(person_crop_bgr, False, reason="Output shape too small (<56)")
            return None

        scores = out[:, 4]
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < POSE_KEYPOINT_CONF_TH:
            self._save_pose_debug(person_crop_bgr, False, reason=f"Low person conf ({scores[best_idx]:.2f} < {POSE_KEYPOINT_CONF_TH})")
            return None
        best_row = out[best_idx]

        return self._extract_crop_tflite(person_crop_bgr, best_row, w, h)

    # ------------------------------------------------------------------
    # PyTorch path (.pt via ultralytics) -- PC/dev only, unaffected by NPUWorker
    # ------------------------------------------------------------------
    def _run_pt(self, person_crop_bgr):
        try:
            results = self._model.predict(
                person_crop_bgr, conf=POSE_KEYPOINT_CONF_TH, verbose=False
            )
            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                self._save_pose_debug(person_crop_bgr, False, reason="No keypoints found")
                return None

            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

            person_kpts = kpts_xy[0]
            person_conf = kpts_conf[0] if kpts_conf is not None else np.ones(17)

            h, w = person_crop_bgr.shape[:2]

            # ------------------------------------------------------------------
            # SHOULDER GUARD: both shoulder keypoints must be physically present
            # (non-zero coordinates).  If either shoulder is missing the torso
            # bbox will be wrong, so skip this frame entirely.
            # ------------------------------------------------------------------
            for sh_idx in (self._KP_L_SHOULDER, self._KP_R_SHOULDER):
                x, y = person_kpts[sh_idx]
                if x == 0 and y == 0:
                    side = "L" if sh_idx == self._KP_L_SHOULDER else "R"
                    if DEBUG:
                        print(
                            f"[PoseDetector] Frame skipped – shoulder keypoint "
                            f"{side} missing (0,0)"
                        )
                    self._save_pose_debug(person_crop_bgr, False, reason=f"Shoulder {side} missing (0,0)")
                    return None  # skip frame – no colour vote

            points = {}
            for idx in self._REQUIRED_KP:
                x, y = person_kpts[idx]
                conf = float(person_conf[idx])
                if conf < POSE_KEYPOINT_CONF_TH or (x == 0 and y == 0):
                    self._save_pose_debug(person_crop_bgr, False, points=points, reason=f"Keypoint {idx} missing or low-conf ({conf:.2f})")
                    return None
                points[idx] = (int(x), int(y))

            return self._bbox_crop(person_crop_bgr, points, w, h)

        except Exception as exc:
            print(f"[PoseDetector] PyTorch inference error: {exc}")
            self._save_pose_debug(person_crop_bgr, False, reason=f"PyTorch error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _extract_crop_tflite(self, person_crop_bgr, best_row, w, h):
        # ------------------------------------------------------------------
        # SHOULDER GUARD: both shoulder keypoints must be physically present
        # (non-zero coordinates).  If either shoulder is missing the torso
        # bbox will be wrong, so skip this frame entirely.
        # ------------------------------------------------------------------
        for sh_idx in (self._KP_L_SHOULDER, self._KP_R_SHOULDER):
            base = 5 + sh_idx * 3
            kx = float(best_row[base])
            ky = float(best_row[base + 1])
            if kx == 0 and ky == 0:
                side = "L" if sh_idx == self._KP_L_SHOULDER else "R"
                if DEBUG:
                    print(
                        f"[PoseDetector] Frame skipped – shoulder keypoint "
                        f"{side} missing (0,0)"
                    )
                self._save_pose_debug(person_crop_bgr, False, reason=f"Shoulder {side} missing (0,0)")
                return None  # skip frame – no colour vote

        points = {}
        for idx in self._REQUIRED_KP:
            base = 5 + idx * 3
            kx = float(best_row[base])
            ky = float(best_row[base + 1])
            kconf = float(best_row[base + 2])

            if kconf < 0:
                kconf = 1.0 / (1.0 + np.exp(-kconf))

            if max(abs(kx), abs(ky)) <= 2.0:
                px = int(kx * w)
                py = int(ky * h)
            else:
                px = int(kx * (w / float(self.input_w)))
                py = int(ky * (h / float(self.input_h)))

            if kconf < POSE_KEYPOINT_CONF_TH or (kx == 0 and ky == 0):
                self._save_pose_debug(person_crop_bgr, False, points=points, reason=f"Keypoint {idx} missing or low-conf ({kconf:.2f})")
                return None

            points[idx] = (px, py)

        return self._bbox_crop(person_crop_bgr, points, w, h)

    def _bbox_crop(self, person_crop_bgr, points, w, h):
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        x1_raw, x2_raw = min(xs), max(xs)
        y1_raw, y2_raw = min(ys), max(ys)

        box_w = x2_raw - x1_raw
        box_h = y2_raw - y1_raw
        if box_w <= 0 or box_h <= 0:
            self._save_pose_debug(person_crop_bgr, False, points=points, reason="Degenerate keypoint box (w/h <= 0)")
            return None

        shoulder_y = (points[self._KP_L_SHOULDER][1] + points[self._KP_R_SHOULDER][1]) / 2.0
        hip_y = (points[self._KP_L_HIP][1] + points[self._KP_R_HIP][1]) / 2.0
        if hip_y <= shoulder_y:
            self._save_pose_debug(person_crop_bgr, False, points=points, reason="Inverted geometry (hips above shoulders)")
            return None

        if box_w < w * TORSO_MIN_WIDTH_FRAC or box_h < h * TORSO_MIN_HEIGHT_FRAC:
            self._save_pose_debug(person_crop_bgr, False, points=points, reason=f"Torso box too small ({box_w}x{box_h})")
            return None

        mx = int(box_w * TORSO_MARGIN_X_FRAC)
        my_top = int(box_h * TORSO_MARGIN_TOP_FRAC)
        my_bot = int(box_h * TORSO_MARGIN_BOTTOM_FRAC)

        x1 = max(0, x1_raw + mx)
        x2 = min(w - 1, x2_raw - mx)
        y1 = max(0, y1_raw + my_top)
        y2 = min(h - 1, y2_raw - my_bot)

        if x2 <= x1 or y2 <= y1:
            self._save_pose_debug(person_crop_bgr, False, points=points, reason="Margin trimmed box invalid")
            return None

        # Successful torso extraction
        self._save_pose_debug(person_crop_bgr, True, points=points, torso_box=(x1, y1, x2, y2), reason="OK")
        return person_crop_bgr[y1:y2, x1:x2].copy()




#########################################################################################################

# """
# pose_detector.py
# ----------------
# Preprocessing + postprocessing for the pose model. TFLite inference itself
# goes through the shared NPUWorker thread (see npu_worker.py) so only ONE
# thread ever touches NPU interpreters.

# The .pt (ultralytics) path is kept for PC/dev only and is NOT run through
# NPUWorker -- it stays synchronous on whichever thread calls
# get_torso_crop(), same as before this refactor.

# POSE_ENABLED = True  -> used by segment.py for HSV colour crop
# POSE_ENABLED = False -> segment.py falls back to static fraction-based crop
# """

# import math

# import numpy as np
# import cv2

# from config import (
#     POSE_ENABLED,
#     POSE_KEYPOINT_CONF_TH,
#     TORSO_MARGIN_TOP_FRAC,
#     TORSO_MARGIN_BOTTOM_FRAC,
#     TORSO_MARGIN_X_FRAC,
#     TORSO_MIN_WIDTH_FRAC,
#     TORSO_MIN_HEIGHT_FRAC,
# )


# class PoseDetector:
#     MODEL_NAME = "pose"

#     # COCO keypoint indices for torso corners
#     _KP_L_SHOULDER = 5
#     _KP_R_SHOULDER = 6
#     _KP_L_HIP      = 11
#     _KP_R_HIP      = 12
#     _REQUIRED_KP   = {_KP_L_SHOULDER, _KP_R_SHOULDER, _KP_L_HIP, _KP_R_HIP}

#     def __init__(self, npu_worker=None, pt_model_path: str = ""):
#         """
#         npu_worker    : shared NPUWorker instance (TFLite path). Pass None
#                          to force the .pt path.
#         pt_model_path : ultralytics .pt weights, PC/dev only. Used only
#                          when npu_worker has no 'pose' model loaded.
#         """
#         self.npu_worker = npu_worker
#         self._is_tflite = False
#         self._model = None  # ultralytics YOLO, .pt path only

#         info = npu_worker.get_model_info(self.MODEL_NAME) if npu_worker else None
#         if info is not None:
#             self._is_tflite = True
#             self.input_h = info["input_h"]
#             self.input_w = info["input_w"]
#             self.input_dtype = info["input_dtype"]
#             self.input_scale = info["input_scale"]
#             self.input_zero = info["input_zero"]
#         elif pt_model_path:
#             self._load_pt(pt_model_path)
#         else:
#             print(
#                 "[PoseDetector] No pose model loaded (NPUWorker has none, "
#                 "no .pt given) -- colour crop disabled."
#             )

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------
#     def get_torso_crop(self, person_crop_bgr):
#         if not POSE_ENABLED:
#             return None
#         if person_crop_bgr is None or person_crop_bgr.size == 0:
#             return None
#         if self._is_tflite:
#             return self._run_tflite(person_crop_bgr)
#         if self._model is not None:
#             return self._run_pt(person_crop_bgr)
#         return None

#     def _load_pt(self, model_path):
#         try:
#             from ultralytics import YOLO as UltralyticsYOLO
#             self._model = UltralyticsYOLO(model_path)
#             self._is_tflite = False
#             print(f"[PoseDetector] PyTorch pose loaded: {model_path}")
#         except Exception as exc:
#             print(f"[PoseDetector] PyTorch load failed ({exc}) -- colour crop disabled")
#             self._model = None

#     # ------------------------------------------------------------------
#     # TFLite path -- preprocess here, invoke on NPUWorker, postprocess here
#     # ------------------------------------------------------------------
#     def _run_tflite(self, person_crop_bgr):
#         h, w = person_crop_bgr.shape[:2]

#         # Resize BEFORE the BGR->RGB swap, not after -- same reasoning as
#         # detection.py/segment.py: interpolation doesn't care about channel
#         # order, and person crops are typically larger than the pose
#         # model's input, so this does the swap on far fewer pixels.
#         img = cv2.resize(person_crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
#         img = img[:, :, ::-1]  # BGR -> RGB view, no extra cv2 call

#         if np.issubdtype(self.input_dtype, np.floating):
#             inp = (img.astype(np.float32) / 255.0).astype(self.input_dtype)
#         else:
#             inp = img.astype(np.float32) / 255.0
#             if self.input_scale > 0:
#                 inp = inp / self.input_scale + self.input_zero
#             inp = np.clip(
#                 inp, np.iinfo(self.input_dtype).min, np.iinfo(self.input_dtype).max
#             ).astype(self.input_dtype)

#         inp = np.expand_dims(inp, axis=0)

#         outputs_raw = self.npu_worker.infer(self.MODEL_NAME, inp)
#         if outputs_raw is None:
#             return None  # NPU busy/timed out -- sacrifice this vote frame

#         info = self.npu_worker.get_model_info(self.MODEL_NAME)
#         out_detail = info["output_details"][0]
#         out_raw = outputs_raw[0]
#         out_scale, out_zp = out_detail.get("quantization", (0.0, 0))
#         if out_scale and out_scale > 0 and not np.issubdtype(out_raw.dtype, np.floating):
#             out = (out_raw.astype(np.float32) - out_zp) * out_scale
#         else:
#             out = out_raw.astype(np.float32)

#         out = np.squeeze(out)
#         if out.ndim != 2:
#             return None
#         if out.shape[0] < out.shape[1] and out.shape[0] in range(5, 100):
#             out = out.T
#         if out.shape[1] < 56:
#             return None

#         scores = out[:, 4]
#         best_idx = int(np.argmax(scores))
#         if scores[best_idx] < POSE_KEYPOINT_CONF_TH:
#             return None
#         best_row = out[best_idx]

#         return self._extract_crop_tflite(person_crop_bgr, best_row, w, h)

#     # ------------------------------------------------------------------
#     # PyTorch path (.pt via ultralytics) -- PC/dev only, unaffected by NPUWorker
#     # ------------------------------------------------------------------
#     def _run_pt(self, person_crop_bgr):
#         try:
#             results = self._model.predict(
#                 person_crop_bgr, conf=POSE_KEYPOINT_CONF_TH, verbose=False
#             )
#             result = results[0]
#             if result.keypoints is None or len(result.keypoints) == 0:
#                 return None

#             kpts_xy = result.keypoints.xy.cpu().numpy()
#             kpts_conf = (
#                 result.keypoints.conf.cpu().numpy()
#                 if result.keypoints.conf is not None
#                 else None
#             )

#             person_kpts = kpts_xy[0]
#             person_conf = kpts_conf[0] if kpts_conf is not None else np.ones(17)

#             h, w = person_crop_bgr.shape[:2]
#             points = {}
#             for idx in self._REQUIRED_KP:
#                 x, y = person_kpts[idx]
#                 conf = float(person_conf[idx])
#                 if conf < POSE_KEYPOINT_CONF_TH or (x == 0 and y == 0):
#                     return None
#                 points[idx] = (int(x), int(y))

#             return self._bbox_crop(person_crop_bgr, points, w, h)

#         except Exception as exc:
#             print(f"[PoseDetector] PyTorch inference error: {exc}")
#             return None

#     # ------------------------------------------------------------------
#     # Shared helpers
#     # ------------------------------------------------------------------
#     def _extract_crop_tflite(self, person_crop_bgr, best_row, w, h):
#         points = {}
#         for idx in self._REQUIRED_KP:
#             base = 5 + idx * 3
#             kx = float(best_row[base])
#             ky = float(best_row[base + 1])
#             kconf = float(best_row[base + 2])

#             if kconf < 0:
#                 # Scalar sigmoid on a single Python float -- math.exp avoids
#                 # numpy's per-call dispatch overhead that np.exp carries
#                 # even for one value. Only up to 4 of these per crop, but
#                 # it's free.
#                 kconf = 1.0 / (1.0 + math.exp(-kconf))

#             if kconf < POSE_KEYPOINT_CONF_TH or (kx == 0 and ky == 0):
#                 return None

#             if max(abs(kx), abs(ky)) <= 2.0:
#                 px = int(kx * w)
#                 py = int(ky * h)
#             else:
#                 px = int(kx * (w / float(self.input_w)))
#                 py = int(ky * (h / float(self.input_h)))

#             points[idx] = (px, py)

#         return self._bbox_crop(person_crop_bgr, points, w, h)

#     @classmethod
#     def _bbox_crop(cls, person_crop_bgr, points, w, h):
#         xs = [p[0] for p in points.values()]
#         ys = [p[1] for p in points.values()]
#         x1_raw, x2_raw = min(xs), max(xs)
#         y1_raw, y2_raw = min(ys), max(ys)

#         box_w = x2_raw - x1_raw
#         box_h = y2_raw - y1_raw
#         if box_w <= 0 or box_h <= 0:
#             return None

#         shoulder_y = (points[cls._KP_L_SHOULDER][1] + points[cls._KP_R_SHOULDER][1]) / 2.0
#         hip_y = (points[cls._KP_L_HIP][1] + points[cls._KP_R_HIP][1]) / 2.0
#         if hip_y <= shoulder_y:
#             return None

#         if box_w < w * TORSO_MIN_WIDTH_FRAC or box_h < h * TORSO_MIN_HEIGHT_FRAC:
#             return None

#         mx = int(box_w * TORSO_MARGIN_X_FRAC)
#         my_top = int(box_h * TORSO_MARGIN_TOP_FRAC)
#         my_bot = int(box_h * TORSO_MARGIN_BOTTOM_FRAC)

#         x1 = max(0, x1_raw + mx)
#         x2 = min(w - 1, x2_raw - mx)
#         y1 = max(0, y1_raw + my_top)
#         y2 = min(h - 1, y2_raw - my_bot)

#         if x2 <= x1 or y2 <= y1:
#             return None
#         return person_crop_bgr[y1:y2, x1:x2].copy()
