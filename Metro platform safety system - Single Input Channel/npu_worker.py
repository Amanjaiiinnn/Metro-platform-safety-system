"""
npu_worker.py
-------------
One dedicated thread that owns EVERY TFLite interpreter that talks to the
NPU (YOLO detector, pose model, clothing classifier) and is the ONLY
thread that ever calls set_tensor/invoke/get_tensor on any of them.

Why one thread instead of one-thread-per-model:
The i.MX93's Ethos-U65 is a single physical NPU core -- it runs one graph
at a time no matter how many Python threads call invoke() concurrently.
Three threads hammering it would just add thread-switching overhead on
top of the NPU's own internal serialization, for zero extra throughput.
What actually helps is keeping this ONE thread pinned to its own CPU core
so it's never preempted by, and never competes with, the capture/cv2 work
running on the other core.

Model loading happens INSIDE run(), on this thread, not in __init__.
TFLite interpreters (especially with a delegate loaded) are safest used
from the same thread that created and allocated them, so nothing ever
calls set_tensor/invoke/get_tensor from a different thread than the one
that built the interpreter.
"""

import os
import threading
import time
from pathlib import Path

import numpy as np

from config import resolve_npu_delegate_path, _pin_thread
from channels import SingleSlotChannel


class _ModelHandle:
    """Everything needed to run one already-loaded TFLite model."""

    __slots__ = (
        "interpreter", "input_index", "output_details",
        "input_h", "input_w", "input_dtype", "input_scale", "input_zero",
    )

    def __init__(self, interpreter, input_details, output_details):
        self.interpreter = interpreter
        self.input_index = input_details["index"]
        self.output_details = output_details
        shape = input_details["shape"]  # [1, H, W, 3]
        self.input_h = int(shape[1])
        self.input_w = int(shape[2])
        self.input_dtype = input_details["dtype"]
        scale, zero = input_details.get("quantization", (0.0, 0))
        self.input_scale = float(scale) if scale else 0.0
        self.input_zero = int(zero) if zero else 0


class NPUWorker(threading.Thread):
    """Runs on one dedicated, pinned core. Owns yolo/pose/clothing
    interpreters. Talks to the rest of the app through two SingleSlotChannels
    (request + result) -- both drop-oldest: if the processing thread submits
    a newer job before this thread picked up the previous one, the previous
    one is silently discarded rather than queued.
    """

    def __init__(self, model_paths: dict, platform: str, backend: str,
                 core_id: int = 1, infer_timeout_s: float = 2.0):
        """
        model_paths : {"yolo": path, "pose": path, "clothing": path}
                      Any value can be "" / None to skip loading that model
                      (e.g. pose/clothing disabled, or resolved to a .pt
                      path -- this worker only ever loads .tflite models;
                      a .pt pose model stays on PoseDetector's own path,
                      outside this worker, for PC/dev use).
        """
        super().__init__(daemon=True, name="NPUWorker")
        self.model_paths = dict(model_paths)
        self.platform = str(platform)
        self.backend = str(backend)
        self.core_id = core_id
        self.infer_timeout_s = float(infer_timeout_s)

        self._handles: dict = {}
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._seq_lock = threading.Lock()
        self._seq = 0

        self.request_channel = SingleSlotChannel()
        self.result_channel = SingleSlotChannel()

    # ------------------------------------------------------------------
    # Public API -- called from OTHER threads
    # ------------------------------------------------------------------
    def wait_ready(self, timeout=30.0) -> bool:
        return self._ready_event.wait(timeout)

    def get_model_info(self, name: str):
        """Input/output shape info for a loaded model, or None if that
        model wasn't loaded. Handles are created once at startup and never
        mutated afterward, so this is safe to call from any thread without
        locking -- but only after wait_ready() has returned True."""
        handle = self._handles.get(name)
        if handle is None:
            return None
        return {
            "input_h": handle.input_h,
            "input_w": handle.input_w,
            "input_dtype": handle.input_dtype,
            "input_scale": handle.input_scale,
            "input_zero": handle.input_zero,
            "output_details": handle.output_details,
        }

    def infer(self, model_name: str, input_tensor: np.ndarray):
        """Blocking call: submit input_tensor to `model_name`, wait for the
        matching result. Returns a list of raw output tensors, or None if
        the worker didn't answer within infer_timeout_s (NPU stalled, or
        that model isn't loaded) -- callers must treat None as "skip this
        frame", never raise on it.
        """
        if model_name not in self._handles:
            return None

        with self._seq_lock:
            self._seq += 1
            seq = self._seq

        self.request_channel.put({"model": model_name, "input": input_tensor, "seq": seq})

        deadline = time.monotonic() + self.infer_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            result = self.result_channel.get(timeout=remaining)
            if result is None:
                return None
            if result["seq"] == seq:
                return result["outputs"]
            # A stale result from a superseded job -- only possible if a
            # caller submits again before consuming its own previous
            # result, which nothing in this codebase does. Ignore and keep
            # waiting for ours.

    def stop(self):
        self._stop_event.set()
        self.request_channel.put(None)  # wake the loop so it can exit promptly

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self):
        _pin_thread(self.core_id)

        self._load_all_models()
        self._ready_event.set()

        while not self._stop_event.is_set():
            job = self.request_channel.get(timeout=0.5)
            if job is None:
                continue
            self._handle_job(job)

    def _handle_job(self, job):
        handle = self._handles.get(job["model"])
        if handle is None:
            self.result_channel.put({"model": job["model"], "seq": job["seq"], "outputs": None})
            return
        try:
            handle.interpreter.set_tensor(handle.input_index, job["input"])
            handle.interpreter.invoke()
            outputs = [handle.interpreter.get_tensor(d["index"]) for d in handle.output_details]
        except Exception as exc:
            print(f"[NPUWorker] Inference error on '{job['model']}': {exc}")
            outputs = None
        self.result_channel.put({"model": job["model"], "seq": job["seq"], "outputs": outputs})

    # ------------------------------------------------------------------
    # Model loading -- runs on THIS thread only, once, at startup
    # ------------------------------------------------------------------
    def _load_all_models(self):
        for name, path in self.model_paths.items():
            if not path:
                continue
            handle = self._load_one(name, path)
            if handle is not None:
                self._handles[name] = handle

    def _load_one(self, name: str, model_path: str):
        suffix = Path(model_path).suffix.lower()
        if suffix != ".tflite":
            print(
                f"[NPUWorker] '{name}': {model_path} is not .tflite -- skipping "
                "(a .pt model stays on its own class, outside this worker)."
            )
            return None

        try:
            Interpreter, load_delegate = self._load_tflite_runtime()
        except RuntimeError as exc:
            print(f"[NPUWorker] TFLite runtime unavailable: {exc}")
            return None

        delegates = []
        delegate_path = os.environ.get("TFLITE_DELEGATE_PATH")
        delegate_source = "TFLITE_DELEGATE_PATH override"
        if not delegate_path:
            delegate_path = resolve_npu_delegate_path(self.platform, self.backend)
            delegate_source = "auto (platform/backend)"

        if delegate_path:
            try:
                delegates.append(load_delegate(delegate_path))
                print(f"[NPUWorker] '{name}': loaded delegate ({delegate_source}): {delegate_path}")
            except Exception as exc:
                print(f"[NPUWorker] '{name}': delegate load failed ({exc}) -- falling back to CPU.")
                delegates = []

        try:
            # num_threads=1: this thread has its own dedicated core and is
            # the ONLY caller of invoke() for any model -- there's no
            # multithreading left for TFLite itself to usefully do.
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=delegates or None,
                num_threads=1,
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()
            return _ModelHandle(interpreter, input_details, output_details)
        except Exception as exc:
            print(f"[NPUWorker] '{name}': failed to load ({exc})")
            return None

    @staticmethod
    def _load_tflite_runtime():
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
            return Interpreter, load_delegate
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter, load_delegate
                return Interpreter, load_delegate
            except ImportError as exc:
                raise RuntimeError(
                    "TFLite runtime not installed. Install tflite-runtime or tensorflow."
                ) from exc