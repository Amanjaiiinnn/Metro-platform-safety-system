import os
import re
import queue
import threading
import time
import wave
import subprocess

import numpy as np

from config import (
    VOICES_DIR,
    TTS_LANGUAGE_CHOICES,
    get_current_time,
    ALERT_CLIP_SILENCE_TRIM_THRESHOLD,
    ALERT_CLIP_TRIM_PAD_MS,
    ALERT_CLIP_GAP_MS,
    ALERT_CLIP_NORMALIZE_ENABLED,
    ALERT_CLIP_NORMALIZE_TARGET,
)

class AlertEngine:
    """Terminal alert manager with a global cooldown.

    The cooldown is global on purpose: if multiple people cross the line in the
    same frame, only the first accepted crossing prints an alert.

    Spoken alerts are pre-recorded WAV clips, one folder per language under
    VOICES_DIR (see the big comment in config.py for the exact layout) --
    NOT text-to-speech. `active_languages` controls which language folders
    get played, and can hold any combination of TTS_LANGUAGE_CHOICES.
    video_widget.py updates it live from the dashboard's checkboxes
    (config.json's "tts_languages"), the same way it already does for
    cooldown_seconds. An empty list means: still print the alert to the
    console, just no audio.

    `volume` is a fraction (1.0 = clip's own normalized, full-but-not-
    clipping level, 0.0 = silent, up to 2.0 = +100% -- intentionally
    clipping the peaks for extra loudness) applied to every clip's PCM
    samples right before playback. video_widget.py updates it live from
    the dashboard's Alert & Audio volume slider (config.json's
    "alert_volume", stored there as a 0-150 percentage), same pattern as
    cooldown_seconds and active_languages.
    """

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        voices_dir: str = VOICES_DIR,
        languages=None,
        volume: float = 1.0,
    ):
        self.cooldown_seconds = float(cooldown_seconds)
        self.voices_dir = voices_dir
        # A fresh list, not a shared/aliased one -- this instance must be
        # free to have video_widget.py reassign it every frame (from
        # config.json's "tts_languages") without that mutating some
        # shared default list elsewhere.
        self.active_languages = list(languages) if languages else ["english"]
        self.volume = float(volume)
        self._last_alert_time = 0.0
        self._tts_queue = None
        self._warned_missing = set()

        # Whether a person is currently inside the alert zone THIS frame.
        # Updated live, every processed frame, from main.py's
        # handle_alert_state()/handle_person_detection_events() -- same
        # "plain attribute, no lock" pattern video_widget.py already uses
        # for active_languages/volume/cooldown_seconds (safe because
        # CPython attribute assignment is atomic). The audio worker below
        # reads this BETWEEN languages so it can stop playing further
        # languages the moment the person steps back, instead of
        # unconditionally finishing every checked language regardless of
        # whether anyone is still there to hear it.
        self.person_in_zone = False

        self._start_audio_worker()

    def can_trigger(self) -> bool:
        return (time.monotonic() - self._last_alert_time) >= self.cooldown_seconds

    def trigger(
        self,
        description: str,
        has_color: bool = True,
        ignore_cooldown: bool = False,
        color: str = None,
        garment: str = None,
    ) -> bool:
        # If Segmenter.describe_person() couldn't detect a color for this
        # person this frame (frame-quality gate rejected it, or pose
        # rejected the torso crop as noise), don't alert at all -- an
        # alert with a garment label but no color, or with a color-less
        # "person" description, is exactly the wrong-detection case this
        # is meant to prevent. Returning here also does NOT touch
        # _last_alert_time, so a suppressed frame doesn't eat into the
        # cooldown window for the next (hopefully color-detected) frame.
        if not has_color:
            return False

        if not ignore_cooldown and not self.can_trigger():
            return False

        self._last_alert_time = time.monotonic()
        timestamp = get_current_time().strftime("%H:%M:%S")
        message = f"Attention! Person in {description}, please step back from the yellow line."

        # Console text is always English -- it's a developer/operator log
        # line, not the spoken alert, so it isn't affected by which
        # language checkboxes are on.
        print("-" * 60, flush=True)
        print(f"{timestamp} Message : {message}", flush=True)
        print("-" * 60, flush=True)

        self._queue_alert_job(general=False, color=color, garment=garment)
        return True

    def trigger_general(self, ignore_cooldown: bool = False) -> bool:
        """Fire the generic 'multiple people' alert instead of a single
        person's color/garment description.

        Called by main.py when more than one alertable person is in the
        zone in the SAME frame -- there's no single "the" person left to
        describe, so this skips the per-person vote/description pipeline
        entirely and goes straight to a dedicated multi-person phrase
        (voices/<language>/General.wav -- see _play_general_language_alert).

        Shares the same global cooldown as trigger() (both write/read the
        same _last_alert_time) -- a general alert doesn't get to jump the
        queue ahead of the normal cooldown just because more people are
        involved.
        """
        if not ignore_cooldown and not self.can_trigger():
            return False

        self._last_alert_time = time.monotonic()
        timestamp = get_current_time().strftime("%H:%M:%S")
        message = "Attention! Multiple people near the yellow line, please step back."

        print("-" * 60, flush=True)
        print(f"{timestamp} Message : {message}", flush=True)
        print("-" * 60, flush=True)

        self._queue_alert_job(general=True)
        return True

    def _queue_alert_job(self, general: bool, color: str = None, garment: str = None):
        """Common enqueue path for both trigger() and trigger_general().

        Collapses to a general job -- discarding whatever's still WAITING
        in the queue -- whenever a backlog already exists (>=1 job not yet
        started): reading out a color/garment description for a person
        who may no longer even be the one still standing there is more
        confusing than a plain "multiple people" warning, and a backlog
        this deep usually means more than one person triggered it anyway.

        The job currently IN FLIGHT (already popped off the queue by the
        worker and mid-playback) is never touched here -- it always
        finishes its own clip sequence; only jobs still waiting their turn
        get discarded.
        """
        if not (self._tts_queue is not None and self.active_languages):
            return
        if not general and not color:
            return  # nothing to describe -- mirrors the old has_color guard

        if self._tts_queue.qsize() >= 1:
            general = True
            color = None
            garment = None
            self._drain_pending_jobs()

        self._tts_queue.put({"general": general, "color": color, "garment": garment})

    def _drain_pending_jobs(self):
        """Discard every job still WAITING in the queue (not the one
        currently in flight, if any -- the worker already removed that one
        from the queue before this can run). Used right before collapsing
        to a general alert so a stale, superseded specific description
        never gets played after the general one.
        """
        if self._tts_queue is None:
            return
        while True:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # WAV-clip playback -- one dedicated daemon thread, same reasoning as
    # the old TTS worker: a single persistent worker draining a queue, so
    # a slow/late alert never blocks the capture/inference loop.
    # ------------------------------------------------------------------
    def _start_audio_worker(self):
        self._tts_queue = queue.Queue()

        def worker():
            while True:
                job = self._tts_queue.get()
                is_general = bool(job.get("general"))
                color = job.get("color")
                garment = job.get("garment")
                # Snapshot: active_languages can be reassigned (a new list
                # object) by video_widget.py mid-playback -- finish this
                # alert with whatever was current when it was queued.
                for language in list(self.active_languages):
                    if language not in TTS_LANGUAGE_CHOICES:
                        continue
                    # Re-checked before EVERY language, not just once up
                    # front: the person is known to have been in the zone
                    # when this alert was queued (trigger() only queues on
                    # a valid vote), so the first language always plays.
                    # If they've stepped back by the time a later
                    # language's turn comes up, there's no one left to
                    # warn -- stop instead of finishing the queued
                    # language list regardless. Each language's own clip
                    # sequence (Attention->Color->Garment->Away) is still
                    # never cut mid-phrase; this only decides whether to
                    # START the next one.
                    if not self.person_in_zone:
                        break
                    if is_general:
                        self._play_general_language_alert(language)
                    else:
                        self._play_language_alert(language, color, garment)

        threading.Thread(target=worker, daemon=True).start()

    def _play_general_language_alert(self, language: str):
        """Standalone multi-person alert -- ONE dedicated pre-recorded
        clip per language, e.g.:
            voices/<language>/General.wav
                -- "Attention! Multiple people near the yellow line,
                   please step back."

        Unlike _play_language_alert, this is never spliced together from
        Attention/Color/Garment/Away pieces -- with more than one person
        in the zone there's no single description to build, just this one
        fixed phrase. Missing-file handling follows the same pattern as
        the specific-alert path: warn once, then stay silent for that
        language rather than guessing at a substitute.
        """
        lang_dir = os.path.join(self.voices_dir, language)
        path = os.path.join(lang_dir, "General.wav")
        if not os.path.isfile(path):
            self._warn_missing_once(path)
            return
        self._play_clip_with_volume(path)

    def _play_language_alert(self, language: str, color: str, garment: str):
        """Attention -> [Color] -> [Garment] -> Away, all from ONE flat
        voices/<language>/ folder -- no colors/ or garments/ subfolders.
        e.g. color "black" -> Black.wav, garment "Shirt" -> Shirt.wav.
        A garment clip is only expected when a garment WAS classified --
        there's no generic "top"/unknown-garment fallback clip.

        Filenames must match EXACTLY (case-sensitive) across every
        language folder -- e.g. TShirt.wav, not Tshirt.wav in one folder
        and TShirt.wav in another. This is intentional: a silent
        case-insensitive fallback would hide a genuine naming mistake
        instead of surfacing it as "missing".

        ALL-OR-NOTHING per language: if even ONE clip this alert needs is
        missing on disk, NONE of them are played for this language -- a
        half-played "Attention... [silence]... please step back" is worse
        than staying silent for that language. Missing paths are printed
        once (not spammed every alert). Other checked languages are
        evaluated independently and unaffected by this one's gap.
        """
        lang_dir = os.path.join(self.voices_dir, language)

        candidate_names = ["Attention.wav"]
        if color:
            candidate_names.append(f"{self._pascal_case(color)}.wav")
        if garment:
            candidate_names.append(f"{re.sub(r'[^a-zA-Z0-9]', '', str(garment))}.wav")
        candidate_names.append("Away.wav")

        resolved_paths = []
        missing_paths = []
        for name in candidate_names:
            path = os.path.join(lang_dir, name)
            (resolved_paths if os.path.isfile(path) else missing_paths).append(path)

        if missing_paths:
            for path in missing_paths:
                self._warn_missing_once(path)
            return  # all-or-nothing -- don't play a partial clip sequence

        self._play_concatenated(resolved_paths)

    def _play_concatenated(self, clip_paths):
        """Play all of clip_paths as ONE continuous stream instead of one
        aplay process per clip.

        Every separate `aplay` invocation forks a new process AND
        re-opens/re-closes the ALSA device from scratch -- on this board
        that round-trip is the actual source of the audible gap between
        "Attention" / color / garment / "Away", not anything about the
        clips themselves. Concatenating the raw PCM samples in memory and
        handing them to a single `aplay` call (device opened once, closed
        once) collapses that to one continuous phrase.

        Falls back to the old one-process-per-clip behavior (with the
        gaps) if the clips can't be safely concatenated -- e.g. mismatched
        sample rate/width/channels across files, or a read error -- rather
        than silently dropping the alert.
        """
        merged = self._concat_wav_clips(clip_paths)
        if merged is None:
            for clip_path in clip_paths:
                self._play_clip_with_volume(clip_path)
            return

        raw_bytes, framerate, sampwidth, channels = merged
        raw_bytes = self._apply_volume(raw_bytes, sampwidth, self.volume)
        alsa_format = {1: "U8", 2: "S16_LE", 3: "S24_3LE", 4: "S32_LE"}.get(sampwidth, "S16_LE")
        try:
            subprocess.run(
                ["aplay", "-q", "-t", "raw", "-r", str(framerate),
                 "-f", alsa_format, "-c", str(channels), "-"],
                input=raw_bytes,
                check=False,
            )
        except Exception as exc:
            print(f"[Alert] Audio playback error (concatenated clip): {exc}")

    def _play_clip_with_volume(self, clip_path: str):
        """Fallback single-clip playback (mismatched-format case only).

        Reads the clip's own raw PCM, scales it by self.volume the same
        way the concatenated path does, and pipes it to aplay in raw mode
        instead of handing aplay the file path directly -- a direct
        `aplay clip.wav` call has no volume control of its own, so this is
        what makes the volume slider work on this fallback path too.
        """
        try:
            with wave.open(clip_path, "rb") as wf:
                framerate, sampwidth, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
                raw_bytes = wf.readframes(wf.getnframes())
            if ALERT_CLIP_NORMALIZE_ENABLED:
                raw_bytes = self._normalize_peak(raw_bytes, sampwidth, channels, ALERT_CLIP_NORMALIZE_TARGET)
            raw_bytes = self._apply_volume(raw_bytes, sampwidth, self.volume)
            alsa_format = {1: "U8", 2: "S16_LE", 3: "S24_3LE", 4: "S32_LE"}.get(sampwidth, "S16_LE")
            subprocess.run(
                ["aplay", "-q", "-t", "raw", "-r", str(framerate),
                 "-f", alsa_format, "-c", str(channels), "-"],
                input=raw_bytes,
                check=False,
            )
        except Exception as exc:
            print(f"[Alert] Audio playback error ({clip_path}): {exc}")

    @staticmethod
    def _apply_volume(raw: bytes, sampwidth: int, volume: float) -> bytes:
        """Scale raw PCM sample amplitudes by `volume`.

        volume == 1.0 leaves the clip at its own recorded level (the
        common case -- skipped entirely, no float round-trip for the
        default). Below 1.0 attenuates; above 1.0 amplifies, hard-clipped
        to the format's valid sample range so an aggressive setting
        distorts at the peaks instead of wrapping into noise. Clamped to
        0.0-2.0 here (not just at the UI/config layer) so a bad or
        stale value written straight into config.json some other way
        can't blow out the speaker.

        Applied to the already-merged, already-gapped stream (see
        _play_concatenated) rather than per-clip before merging -- the
        silence gap between words is already sample value 0, so scaling
        it again is a no-op, and doing this once on the final stream
        means every clip is scaled by the exact same factor.
        """
        volume = max(0.0, min(2.0, float(volume)))
        if volume == 1.0 or len(raw) == 0:
            return raw

        dtype = AlertEngine._DTYPE_FOR_SAMPWIDTH.get(sampwidth)
        if dtype is None:
            return raw  # unsupported width (e.g. 24-bit) -- leave unscaled

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if dtype == np.uint8:
            # Unsigned 8-bit PCM silence sits at 128, not 0 -- scale
            # around that midpoint, not around zero.
            scaled = (samples - 128.0) * volume + 128.0
            lo, hi = 0, 255
        else:
            scaled = samples * volume
            info = np.iinfo(dtype)
            lo, hi = info.min, info.max

        scaled = np.clip(scaled, lo, hi)
        return scaled.astype(dtype).tobytes()

    @staticmethod
    def _normalize_peak(raw: bytes, sampwidth: int, channels: int, target: float) -> bytes:
        """Scale one already-trimmed clip so its loudest sample reaches
        `target` (a fraction of the format's full-scale amplitude, e.g.
        0.92) -- never scaled DOWN, only up, and only if it isn't already
        at/above target.

        This is what makes the volume slider's 100% actually mean
        something consistent: without it, "100%" played each clip at
        whatever level it happened to be recorded at, and clips recorded
        quietly (common -- leaves the person recording them some
        headroom) stayed quiet even at the slider's maximum, because
        there just wasn't much amplitude there to multiply. Normalizing
        first means every clip's 100% is "as loud as THIS clip can go
        without clipping," and the slider's job past that is purely
        extra, intentionally-clipping boost.
        """
        dtype = AlertEngine._DTYPE_FOR_SAMPWIDTH.get(sampwidth)
        if dtype is None or len(raw) == 0:
            return raw  # unsupported width (e.g. 24-bit) or empty clip

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if dtype == np.uint8:
            centered = samples - 128.0
            full_scale = 127.0
        else:
            centered = samples
            full_scale = float(np.iinfo(dtype).max)

        peak = float(np.abs(centered).max()) if centered.size else 0.0
        if peak <= 0:
            return raw  # silent clip -- nothing to normalize against

        gain = (full_scale * target) / peak
        if gain <= 1.0:
            return raw  # already at/above target -- never turn a clip down here

        scaled = centered * gain
        if dtype == np.uint8:
            scaled = np.clip(scaled, -full_scale, full_scale) + 128.0
            lo, hi = 0, 255
        else:
            info = np.iinfo(dtype)
            lo, hi = info.min, info.max
            scaled = np.clip(scaled, lo, hi)

        return scaled.astype(dtype).tobytes()

    @staticmethod
    def _concat_wav_clips(clip_paths):
        """Read every clip in clip_paths, trim each clip's own leading/
        trailing dead air, and join them with a small fixed gap in between.

        Returns (raw_pcm_bytes, framerate, sampwidth, channels), or None
        if any clip fails to open/read, or the clips don't all share the
        same format (rate/width/channels) -- concatenating mismatched PCM
        without resampling would just produce garbled/pitched-wrong
        audio, so that case is refused rather than attempted.

        Playing every clip back-to-back as ONE aplay stream (see
        _play_concatenated) already removes the process-spawn/ALSA-reopen
        gap between separate aplay calls. That alone wasn't enough,
        though -- each recorded clip almost always has its own bit of
        silence baked into the start/end of the recording itself (mic
        lag, room noise floor), and raw-concatenating preserved that as
        an audible gap even inside one continuous stream, which is
        exactly why 4 separate recordings still sounded like 4 separate
        recordings. _trim_silence() strips that dead air off each clip
        first; _make_gap() then re-inserts one small, consistent pause
        between clips for a natural word-to-word cadence instead of
        whatever uneven silence each recording happened to have.
        """
        frames = []
        params = None
        for path in clip_paths:
            try:
                with wave.open(path, "rb") as wf:
                    current = (wf.getframerate(), wf.getsampwidth(), wf.getnchannels())
                    if params is None:
                        params = current
                    elif current != params:
                        print(
                            f"[Alert] Voice clip format mismatch ({path}: "
                            f"{current} vs expected {params}) -- falling back "
                            "to per-clip playback for this alert."
                        )
                        return None
                    raw = wf.readframes(wf.getnframes())
                    raw = AlertEngine._trim_silence(
                        raw, wf.getsampwidth(), wf.getnchannels(), wf.getframerate()
                    )
                    if ALERT_CLIP_NORMALIZE_ENABLED:
                        raw = AlertEngine._normalize_peak(
                            raw, wf.getsampwidth(), wf.getnchannels(), ALERT_CLIP_NORMALIZE_TARGET
                        )
                    frames.append(raw)
            except Exception as exc:
                print(f"[Alert] Could not read voice clip {path}: {exc} -- "
                      "falling back to per-clip playback for this alert.")
                return None

        if params is None:
            return None
        framerate, sampwidth, channels = params

        gap = AlertEngine._make_gap(framerate, sampwidth, channels, ALERT_CLIP_GAP_MS)
        merged = gap.join(frames) if gap else b"".join(frames)
        return merged, framerate, sampwidth, channels

    # Numpy dtype for each WAV sample width AlertEngine expects to see.
    # 24-bit (sampwidth == 3) has no native numpy integer dtype and isn't
    # a format these voice clips use in practice, so it's deliberately
    # left out -- _trim_silence() no-ops for it below.
    _DTYPE_FOR_SAMPWIDTH = {1: np.uint8, 2: np.int16, 4: np.int32}

    @staticmethod
    def _trim_silence(raw: bytes, sampwidth: int, channels: int, framerate: int) -> bytes:
        """Strip near-silent samples off the start and end of one clip's
        raw PCM, keeping a small pad on each side so the word isn't
        clipped right at its own edge.

        "Silence" is anything under ALERT_CLIP_SILENCE_TRIM_THRESHOLD
        (config.py) of the format's full-scale amplitude -- tune that if
        trimming leaves audible dead air (raise it) or clips into the
        spoken word itself (lower it).
        """
        dtype = AlertEngine._DTYPE_FOR_SAMPWIDTH.get(sampwidth)
        if dtype is None or len(raw) == 0:
            return raw  # unsupported width (e.g. 24-bit) or empty clip -- leave as-is

        samples = np.frombuffer(raw, dtype=dtype)
        if channels > 1:
            usable = (len(samples) // channels) * channels
            samples = samples[:usable].reshape(-1, channels)

        if dtype == np.uint8:
            # Unsigned 8-bit PCM silence sits at 128, not 0 -- center it
            # before taking magnitude.
            mag = np.abs(samples.astype(np.int16) - 128)
            full_scale = 128.0
        else:
            mag = np.abs(samples.astype(np.int64))
            full_scale = float(np.iinfo(dtype).max)

        if channels > 1:
            mag = mag.max(axis=1)  # loudest channel per frame

        above = np.where(mag > full_scale * ALERT_CLIP_SILENCE_TRIM_THRESHOLD)[0]
        if above.size == 0:
            return raw  # whole clip is at/under threshold -- nothing to trim safely

        pad_frames = int(framerate * ALERT_CLIP_TRIM_PAD_MS / 1000.0)
        start = max(0, int(above[0]) - pad_frames)
        end = min(mag.shape[0], int(above[-1]) + 1 + pad_frames)

        trimmed = samples[start:end]
        return trimmed.reshape(-1).astype(dtype).tobytes()

    @staticmethod
    def _make_gap(framerate: int, sampwidth: int, channels: int, gap_ms: float) -> bytes:
        """Raw PCM silence of the given duration, in the same format as
        the clips being joined -- inserted between clips (not before the
        first or after the last) so words have a small, consistent pause
        instead of running together or keeping each clip's own uneven
        silence.
        """
        if gap_ms <= 0:
            return b""
        n_bytes = int(framerate * gap_ms / 1000.0) * sampwidth * channels
        if n_bytes <= 0:
            return b""
        # Unsigned 8-bit PCM silence is the byte value 128; every other
        # width used here is signed, where silence is the zero byte.
        return bytes([128]) * n_bytes if sampwidth == 1 else b"\x00" * n_bytes

    def _warn_missing_once(self, path: str):
        if path in self._warned_missing:
            return
        self._warned_missing.add(path)
        print(f"[Alert] Skipping alert -- missing voice clip (will keep skipping silently): {path}")

    @staticmethod
    def _pascal_case(name: str) -> str:
        """'sky blue' -> 'SkyBlue', 'black' -> 'Black' -- matches the
        <Color>.wav filename convention (Black.wav, SkyBlue.wav, ...).
        """
        if not name:
            return "Unknown"
        words = re.split(r"[^a-zA-Z0-9]+", str(name).strip())
        pascal = "".join(w[:1].upper() + w[1:].lower() for w in words if w)
        return pascal or "Unknown"