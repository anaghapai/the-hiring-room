import io
import os
import re
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/voice", tags=["voice"])

FILLER_WORDS = ["um", "uh", "like", "you know", "sort of", "kind of"]


def count_fillers(transcript: str) -> dict:
    text = transcript.lower()
    counts = {}
    for word in FILLER_WORDS:
        counts[word] = len(re.findall(r"\b" + re.escape(word) + r"\b", text))
    return counts


def total_fillers(transcript: str) -> int:
    return sum(count_fillers(transcript).values())


def summarize_signal(wpm: Optional[float], filler_count: Optional[int], pause_count: Optional[int]) -> str:
    """One short line for injecting into a persona prompt. Never invents a number —
    only reports fields that were actually provided."""
    parts = []
    if wpm:
        parts.append(f"{wpm:.0f} words/min")
    if filler_count is not None:
        parts.append(f"{filler_count} filler word(s)")
    if pause_count is not None:
        parts.append(f"{pause_count} pause(s)")
    return "; ".join(parts)


@router.post("/analyze")
def analyze_voice(
    file: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    # Lazy import: librosa/numpy only load when this endpoint is actually hit.
    import librosa
    import numpy as np

    raw = file.file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()  # release the handle so librosa can open it too (Windows locks open files)
        try:
            y, sr = librosa.load(tmp.name, sr=None, mono=True)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not decode the audio file. This endpoint needs ffmpeg installed "
                    "and on your PATH to read webm/opus recordings from the browser. "
                    f"Original error: {exc}"
                ),
            )
    finally:
        os.unlink(tmp.name)

    duration_sec = float(len(y) / sr) if sr else 0.0

    # Pitch estimate via pyin (robust-ish, no training data needed)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6")
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    avg_pitch_hz = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_variance = float(np.var(voiced_f0)) if voiced_f0.size else 0.0

    # Pause detection: only count significant silences (>500ms between speech)
    # Use higher top_db to avoid false positives on breath/background noise
    intervals = librosa.effects.split(y, top_db=40)

    # Background hum / mic pop can register as several tiny "speech" blips a
    # few milliseconds long — each gap between them then got counted as a
    # real pause, which is why near-silent clips were showing pause counts
    # that made no sense. Drop blips under 300ms before measuring anything.
    MIN_SEGMENT_SEC = 0.3
    real_intervals = [
        (start, end) for start, end in intervals
        if (end - start) / sr >= MIN_SEGMENT_SEC
    ]

    speech_time = sum((end - start) for start, end in real_intervals) / sr if sr else 0.0
    pause_time = max(duration_sec - speech_time, 0.0)

    # Only count gaps ≥500ms as "pauses" (exclude breaths & mic noise)
    significant_pauses = 0
    for i in range(len(real_intervals) - 1):
        gap_duration = (real_intervals[i + 1][0] - real_intervals[i][1]) / sr
        if gap_duration >= 0.5:  # 500ms minimum
            significant_pauses += 1

    # Under a second of real detected speech means there's essentially
    # nothing to measure — report 0 rather than a number derived from noise.
    pause_count = significant_pauses if speech_time >= 1.0 else 0

    result = {
        "duration_sec": round(duration_sec, 2),
        "avg_pitch_hz": round(avg_pitch_hz, 1),
        "pitch_variance": round(pitch_variance, 1),
        "estimated_pause_count": pause_count,
        "estimated_pause_time_sec": round(pause_time, 2),
    }

    if transcript:
        word_count = len(transcript.split())
        # Use actual speaking time, not total clip length, so pace reflects
        # how fast you talked while talking — not diluted by however long
        # you paused before/during/after. Falls back to clip duration only
        # if speech detection found basically nothing (avoids dividing by a
        # near-zero number and producing a wildly inflated WPM).
        speaking_seconds = speech_time if speech_time >= 1.0 else duration_sec
        minutes = speaking_seconds / 60 if speaking_seconds else 0
        wpm = round(word_count / minutes, 1) if minutes else 0
        result["words_per_minute"] = wpm
        filler_counts = count_fillers(transcript)
        result["filler_word_counts"] = filler_counts
        result["total_filler_count"] = sum(filler_counts.values())

    return result
