import cv2
import numpy as np

from config import (
    CLASS_NAMES,
    BOX_WIDTH_SCALE,
    BOX_HEIGHT_SCALE,
    NMS_CONTAINMENT_TH,
)


class Detector:
    """YOLO -- preprocessing + postprocessing only.

    Actual inference (set_tensor/invoke/get_tensor) happens on the shared
    NPUWorker thread; this class never touches a TFLite interpreter
    directly. See npu_worker.py.
    """

    MODEL_NAME = "yolo"

    def __init__(
        self,
        npu_worker,
        iou_threshold: float = 0.45,
        debug: bool = False,
        box_width_scale: float = BOX_WIDTH_SCALE,
        box_height_scale: float = BOX_HEIGHT_SCALE,
        # Startup fallback only -- main.py doesn't pass these anymore.
        # The real values are fully dashboard-driven: video_widget.py
        # reloads config.json every frame and calls update_thresholds()
        # with whatever the Detection Tuning tab currently has, before
        # this class's threshold is ever actually used to filter a
        # detection.
        person_conf_th: float = 0.5,
        train_conf_th: float = 0.45,
    ):
        self.npu_worker = npu_worker
        self.person_conf_th = float(person_conf_th)
        self.train_conf_th = float(train_conf_th)
        self.class_thresholds = {
            "Person": self.person_conf_th,
            "Train": self.train_conf_th,
        }
        self.iou_threshold = float(iou_threshold)
        self.containment_threshold = float(NMS_CONTAINMENT_TH)
        self.box_width_scale = float(box_width_scale)
        self.box_height_scale = float(box_height_scale)
        self.debug = bool(debug)
        self._debug_printed = False
        info = npu_worker.get_model_info(self.MODEL_NAME)
        if info is None:
            raise RuntimeError(
                "Detector: 'yolo' model not loaded on NPUWorker -- check model_paths."
            )
        self.input_h = info["input_h"]
        self.input_w = info["input_w"]
        self.input_dtype = info["input_dtype"]
        self.input_scale = info["input_scale"]
        self.input_zero = info["input_zero"]
        self.output_details = info["output_details"]

        if self.debug:
            print(f"[Detector] Input: {self.input_w}x{self.input_h} {self.input_dtype}")
            for detail in self.output_details:
                print(
                    "[Detector] Output:",
                    detail["name"],
                    detail["shape"],
                    detail["dtype"],
                    detail.get("quantization"),
                )

    def update_thresholds(self, person_conf_th: float, train_conf_th: float):
        self.person_conf_th = float(person_conf_th)
        self.train_conf_th = float(train_conf_th)
        self.class_thresholds["Person"] = self.person_conf_th
        self.class_thresholds["Train"] = self.train_conf_th

    def detect(self, frame: np.ndarray) -> list:
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        outputs_raw = self.npu_worker.infer(self.MODEL_NAME, input_tensor)
        if outputs_raw is None:
            return []  # NPU busy/timed out -- sacrifice this frame

        outputs = [
            self._dequantize_output(o, d) for o, d in zip(outputs_raw, self.output_details)
        ]
        boxes, scores, class_ids = self._parse_outputs(outputs, orig_w, orig_h)
        if self.debug and not self._debug_printed:
            print(f"[Detector] Parsed candidates above threshold: {len(boxes)}")
            if scores:
                print(f"[Detector] Top scores: {sorted(scores, reverse=True)[:10]}")
            self._debug_printed = True

        if not boxes:
            return []

        min_th = min(self.person_conf_th, self.train_conf_th)
        keep = self._nms(boxes, scores, min_th, self.iou_threshold, self.containment_threshold)
        if not keep:
            return []

        detections = []
        for idx in keep:
            x, y, w, h = boxes[idx]
            class_id = int(class_ids[idx])
            label = self._label_for_class(class_id)
            if label is None:
                continue
            detections.append(
                [
                    int(max(0, x)),
                    int(max(0, y)),
                    int(min(orig_w - 1, x + w)),
                    int(min(orig_h - 1, y + h)),
                    label,
                    float(scores[idx]),
                ]
            )
        return detections

    @staticmethod
    def _nms(boxes: list, scores: list, score_threshold: float, iou_threshold: float,
              containment_threshold: float = None) -> list:
        """Greedy NMS in plain NumPy -- drop-in replacement for
        cv2.dnn.NMSBoxes so this module no longer touches the cv2.dnn
        submodule at all. Same algorithm: sort by score descending, keep
        the top box, discard every remaining box whose IoU with it exceeds
        iou_threshold, repeat with what's left. Returns kept indices, same
        order cv2.dnn.NMSBoxes returns them in (highest score first).

        Boxes are already filtered to their own per-class threshold in
        _parse_outputs, so score_threshold here is mostly a no-op safety
        net -- kept for behavioral parity with the old call signature.

        A box is ALSO discarded if it's almost entirely contained inside
        the box being kept (intersection / smaller-box-area >=
        containment_threshold), even when its IoU is below iou_threshold.
        Plain IoU alone misses this case: when one box is much smaller
        than the other, their union is dominated by the big box, so IoU
        stays low even though the small box is really just a degenerate
        duplicate sitting inside the real one -- e.g. a spurious
        torso-only box inside the true full-body box for the same person.
        Left at the default of None, this check is skipped entirely, so
        existing callers that don't pass it see identical behavior to
        before this was added.
        """
        if not boxes:
            return []

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)

        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]

        order = np.where(scores_arr >= score_threshold)[0]
        order = order[np.argsort(-scores_arr[order], kind="stable")]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            rest = order[1:]
            if rest.size == 0:
                break

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            suppressed = iou > iou_threshold
            if containment_threshold is not None:
                min_area = np.minimum(areas[i], areas[rest])
                containment = np.where(min_area > 0, inter / min_area, 0.0)
                suppressed = suppressed | (containment >= containment_threshold)

            order = rest[~suppressed]

        return keep

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Resize BEFORE swapping channels, not after. Interpolation is
        # per-pixel and doesn't care about channel order, so the result is
        # identical either way -- but resize is done once on the full
        # camera frame (e.g. 640x480) regardless of order, while the
        # BGR->RGB swap's cost scales with pixel count. Doing the swap on
        # the already-downscaled 320x320 image instead of the original
        # touches ~6x fewer pixels for that step, every single frame.
        image = cv2.resize(frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        image = image[:, :, ::-1]  # BGR -> RGB, cheap numpy view instead of a cv2 call

        if np.issubdtype(self.input_dtype, np.floating):
            image = (image.astype(np.float32) / 255.0).astype(self.input_dtype)
        else:
            image = image.astype(np.float32) / 255.0
            if self.input_scale and self.input_scale > 0:
                image = image / self.input_scale + self.input_zero
            image = np.clip(image, np.iinfo(self.input_dtype).min, np.iinfo(self.input_dtype).max)
            image = image.astype(self.input_dtype)

        return np.expand_dims(image, axis=0)

    @staticmethod
    def _dequantize_output(output: np.ndarray, detail: dict) -> np.ndarray:
        scale, zero = detail.get("quantization", (0.0, 0))
        if scale and scale > 0 and not np.issubdtype(output.dtype, np.floating):
            return (output.astype(np.float32) - zero) * scale
        return output.astype(np.float32)

    def _parse_outputs(self, outputs, orig_w, orig_h):
        if len(outputs) >= 4:
            return self._parse_tflite_detection_outputs(outputs, orig_w, orig_h)

        output = np.squeeze(outputs[0])
        if output.ndim != 2:
            raise RuntimeError(f"Unsupported model output shape: {outputs[0].shape}")

        # YOLO exports commonly use [channels, candidates]. Convert to candidates rows.
        if output.shape[0] < output.shape[1] and output.shape[0] in range(5, 200):
            output = output.T

        num_cols = output.shape[1]
        if num_cols < 6:
            return [], [], []

        # Vectorized replacement for the old per-row Python loop (up to
        # ~2100 candidates/frame from a 320x320 model). Same math, same
        # per-class thresholds, just done as array ops instead of one
        # Python-level iteration per candidate -- this was the single
        # biggest CPU cost in the detect() path on PC/CPU.
        cx, cy, bw, bh = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        if num_cols == 7:
            objectness = output[:, 4]
            class_scores = output[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        else:
            class_scores = output[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        # Per-row threshold lookup, vectorized: build an array of
        # thresholds (one per candidate) matching each candidate's
        # predicted class, in a handful of masked assignments instead of
        # a dict.get() per candidate.
        thresholds = np.full(scores.shape, np.inf, dtype=np.float32)
        known_mask = np.zeros(scores.shape, dtype=bool)
        for cid, label in CLASS_NAMES.items():
            rows = class_ids == cid
            if rows.any():
                thresholds[rows] = self.class_thresholds[label]
                known_mask[rows] = True

        keep = known_mask & (scores >= thresholds)
        if not np.any(keep):
            return [], [], []

        cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        x1, y1, x2, y2 = self._scale_boxes(cx, cy, bw, bh, orig_w, orig_h)

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return [], [], []

        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        return boxes, scores.tolist(), class_ids.astype(int).tolist()

    def _parse_tflite_detection_outputs(self, outputs, orig_w, orig_h):
        boxes_raw = np.squeeze(outputs[0])
        class_ids_raw = np.squeeze(outputs[1]).astype(np.int32)
        scores_raw = np.squeeze(outputs[2])
        count = int(np.squeeze(outputs[3])) if np.size(outputs[3]) == 1 else len(scores_raw)

        boxes, scores, class_ids = [], [], []
        for i in range(min(count, len(scores_raw))):
            score = float(scores_raw[i])
            class_id = int(class_ids_raw[i])
            if class_id not in CLASS_NAMES:
                continue
            label = CLASS_NAMES[class_id]
            th = self.class_thresholds[label]
            if score < th:
                continue

            ymin, xmin, ymax, xmax = boxes_raw[i]
            x1 = xmin * orig_w
            y1 = ymin * orig_h
            x2 = xmax * orig_w
            y2 = ymax * orig_h

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1) * self.box_width_scale
            bh = (y2 - y1) * self.box_height_scale

            x1 = int(cx - bw / 2.0)
            y1 = int(cy - bh / 2.0)
            x2 = int(cx + bw / 2.0)
            y2 = int(cy + bh / 2.0)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        return boxes, scores, class_ids

    def _scale_boxes(self, cx, cy, bw, bh, orig_w, orig_h):
        """Array version of _scale_box for a batch of candidates at once.

        The normalized-vs-pixel-scale decision is made once from the
        overall max across the whole batch instead of per-row -- valid
        because every candidate comes from the same model output, so
        they're all in the same coordinate convention already.
        """
        cx = cx.astype(np.float32, copy=True)
        cy = cy.astype(np.float32, copy=True)
        bw = bw.astype(np.float32, copy=True)
        bh = bh.astype(np.float32, copy=True)

        peak = max(
            float(np.abs(cx).max(initial=0.0)),
            float(np.abs(cy).max(initial=0.0)),
            float(np.abs(bw).max(initial=0.0)),
            float(np.abs(bh).max(initial=0.0)),
        )
        if peak <= 2.0:
            cx *= orig_w
            bw *= orig_w
            cy *= orig_h
            bh *= orig_h
        else:
            cx *= orig_w / self.input_w
            bw *= orig_w / self.input_w
            cy *= orig_h / self.input_h
            bh *= orig_h / self.input_h

        bw *= self.box_width_scale
        bh *= self.box_height_scale

        x1 = (cx - bw / 2.0).astype(np.int32)
        y1 = (cy - bh / 2.0).astype(np.int32)
        x2 = (cx + bw / 2.0).astype(np.int32)
        y2 = (cy + bh / 2.0).astype(np.int32)
        return x1, y1, x2, y2

    def _scale_box(self, cx, cy, bw, bh, orig_w, orig_h):
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
            cx *= orig_w
            bw *= orig_w
            cy *= orig_h
            bh *= orig_h
        else:
            cx *= orig_w / self.input_w
            bw *= orig_w / self.input_w
            cy *= orig_h / self.input_h
            bh *= orig_h / self.input_h

        bw *= self.box_width_scale
        bh *= self.box_height_scale

        x1 = int(cx - bw / 2.0)
        y1 = int(cy - bh / 2.0)
        x2 = int(cx + bw / 2.0)
        y2 = int(cy + bh / 2.0)
        return x1, y1, x2, y2

    @staticmethod
    def _label_for_class(class_id: int) -> str:
        if class_id in CLASS_NAMES:
            return CLASS_NAMES[class_id]
        return None