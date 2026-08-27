from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .core import EvidenceError, srt_timestamp


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None

    @property
    def uncertain(self) -> bool:
        return bool(
            (self.avg_logprob is not None and self.avg_logprob < -0.8)
            or (self.no_speech_prob is not None and self.no_speech_prob > 0.6)
        )


def write_srt(path: Path, segments: Iterable[TranscriptSegment]) -> None:
    blocks: list[str] = []
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    str(segment.index),
                    f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                    segment.text.strip() or "[无可辨识语音]",
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, segments: Iterable[TranscriptSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for segment in segments:
            handle.write(
                json.dumps(
                    {
                        "evidence_level": "ASR_RAW",
                        "index": segment.index,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                        "uncertain": segment.uncertain,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EvidenceError(
            "未安装 faster-whisper。请执行 python -m pip install -e \".[transcribe]\""
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = [
        TranscriptSegment(
            index=index,
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text.strip(),
            avg_logprob=float(segment.avg_logprob),
            no_speech_prob=float(segment.no_speech_prob),
        )
        for index, segment in enumerate(raw_segments, start=1)
    ]
    metadata = {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    return segments, metadata
