from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .core import (
    EvidenceError,
    append_jsonl,
    artifact_hashes,
    canonical_url,
    executable_version,
    extract_bvid,
    require_success,
    run_command,
    write_json,
)
from .transcribe import transcribe_audio, write_jsonl, write_srt


@dataclass(frozen=True)
class PipelineOptions:
    source: str
    output: Path
    model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    cookies_from_browser: str | None = None
    skip_video: bool = False
    skip_transcribe: bool = False
    skip_frames: bool = False
    scene_threshold: float = 0.35


def doctor_report() -> dict[str, object]:
    try:
        import faster_whisper  # noqa: F401

        whisper_available = True
    except ImportError:
        whisper_available = False
    tools = {
        "yt-dlp": executable_version("yt-dlp"),
        "ffmpeg": executable_version("ffmpeg"),
        "ffprobe": executable_version("ffprobe"),
        "faster_whisper": whisper_available,
    }
    return {
        "ready_for_collection": bool(tools["yt-dlp"] and tools["ffmpeg"]),
        "ready_for_transcription": whisper_available,
        "tools": tools,
    }


class EvidencePipeline:
    def __init__(self, options: PipelineOptions):
        self.options = options
        self.bvid = extract_bvid(options.source)
        self.source_url = canonical_url(options.source)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = options.output.resolve() / f"{self.bvid}_{timestamp}"
        self.raw_dir = self.run_dir / "raw"
        self.subtitle_dir = self.raw_dir / "subtitles"
        self.frames_dir = self.run_dir / "frames"
        self.logs_dir = self.run_dir / "logs"
        self.review_dir = self.run_dir / "review"
        self.analysis_dir = self.run_dir / "analysis"
        self.command_log = self.logs_dir / "commands.jsonl"
        self.stage_status: dict[str, object] = {}

    def _prepare(self) -> None:
        for path in (
            self.raw_dir,
            self.subtitle_dir,
            self.frames_dir,
            self.logs_dir,
            self.review_dir,
            self.analysis_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _cookie_args(self) -> list[str]:
        if not self.options.cookies_from_browser:
            return []
        return ["--cookies-from-browser", self.options.cookies_from_browser]

    def _run(self, stage: str, command: list[str], required: bool = True):
        result = run_command(command)
        append_jsonl(
            self.command_log,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "stage": stage,
                "command": result.command,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-4000:],
            },
        )
        if required:
            return require_success(result, stage)
        return result

    def _collect_metadata(self) -> dict[str, object]:
        command = [
            "yt-dlp",
            "--no-playlist",
            "--dump-single-json",
            "--skip-download",
            *self._cookie_args(),
            self.source_url,
        ]
        result = self._run("metadata", command)
        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EvidenceError("yt-dlp 元数据不是有效 JSON") from exc
        write_json(self.raw_dir / "metadata.json", metadata)
        self.stage_status["metadata"] = "collected"
        return metadata

    def _collect_subtitles(self) -> list[Path]:
        command = [
            "yt-dlp",
            "--no-playlist",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            "zh.*,ai-zh",
            "--sub-format",
            "srt/best",
            "--convert-subs",
            "srt",
            "--paths",
            str(self.subtitle_dir),
            "--output",
            "%(id)s.%(language)s.%(ext)s",
            *self._cookie_args(),
            self.source_url,
        ]
        result = self._run("subtitles", command, required=False)
        subtitles = sorted(self.subtitle_dir.glob("*.srt"))
        self.stage_status["subtitles"] = {
            "command_returncode": result.returncode,
            "files": [path.name for path in subtitles],
        }
        return subtitles

    def _download_video(self) -> Path | None:
        if self.options.skip_video:
            self.stage_status["video"] = "skipped"
            return None
        command = [
            "yt-dlp",
            "--no-playlist",
            "--format",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "--paths",
            str(self.raw_dir),
            "--output",
            "video.%(ext)s",
            *self._cookie_args(),
            self.source_url,
        ]
        self._run("video", command)
        candidates = sorted(
            path
            for path in self.raw_dir.glob("video.*")
            if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".flv"}
        )
        if not candidates:
            raise EvidenceError("yt-dlp 返回成功，但没有找到视频文件")
        self.stage_status["video"] = candidates[0].name
        return candidates[0]

    def _extract_audio(self, video_path: Path | None) -> Path | None:
        if video_path is None:
            self.stage_status["audio"] = "skipped_without_video"
            return None
        audio_path = self.raw_dir / "audio.wav"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
        self._run("audio", command)
        self.stage_status["audio"] = audio_path.name
        return audio_path

    def _extract_frames(self, video_path: Path | None) -> None:
        if video_path is None or self.options.skip_frames:
            self.stage_status["frames"] = "skipped"
            return
        threshold = self.options.scene_threshold
        filter_value = f"select='gt(scene,{threshold})',showinfo,scale='min(1280,iw)':-2"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_value,
            "-fps_mode",
            "vfr",
            str(self.frames_dir / "frame_%05d.jpg"),
        ]
        result = self._run("frames", command, required=False)
        frames = sorted(self.frames_dir.glob("*.jpg"))
        timestamps = [
            float(value)
            for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)
        ]
        frame_index: list[dict[str, object]] = []
        for index, frame in enumerate(frames):
            timestamp = timestamps[index] if index < len(timestamps) else None
            if timestamp is not None:
                renamed = frame.with_name(
                    f"t{round(timestamp * 1000):010d}ms_frame_{index + 1:05d}.jpg"
                )
                frame.replace(renamed)
                frame = renamed
            frame_index.append(
                {
                    "file": frame.name,
                    "timestamp_seconds": timestamp,
                    "source": "SCENE_CHANGE",
                }
            )
        write_json(self.frames_dir / "index.json", frame_index)
        self.stage_status["frames"] = {
            "command_returncode": result.returncode,
            "count": len(frames),
            "scene_threshold": threshold,
            "timestamps_resolved": sum(
                item["timestamp_seconds"] is not None for item in frame_index
            ),
        }

    def _prepare_transcript(self, subtitles: list[Path], audio_path: Path | None) -> None:
        review_path = self.review_dir / "transcript.to-review.srt"
        if subtitles:
            shutil.copy2(subtitles[0], review_path)
            self.stage_status["transcript"] = {
                "source": "PLATFORM_OR_AUTO_SUBTITLE",
                "file": subtitles[0].name,
                "review_status": "UNVERIFIED",
            }
            return
        if self.options.skip_transcribe:
            self.stage_status["transcript"] = "skipped"
            return
        if audio_path is None:
            raise EvidenceError("没有字幕，也没有可供转写的音轨")
        segments, transcription_metadata = transcribe_audio(
            audio_path,
            self.options.model,
            self.options.device,
            self.options.compute_type,
        )
        raw_srt = self.raw_dir / "transcript.asr.srt"
        raw_jsonl = self.raw_dir / "transcript.asr.jsonl"
        write_srt(raw_srt, segments)
        write_jsonl(raw_jsonl, segments)
        shutil.copy2(raw_srt, review_path)
        uncertain = [segment for segment in segments if segment.uncertain]
        self.stage_status["transcript"] = {
            "source": "ASR_RAW",
            "review_status": "UNVERIFIED",
            "segments": len(segments),
            "uncertain_segments": len(uncertain),
            **transcription_metadata,
        }
        lines = [
            "# 疑难转写片段",
            "",
            "以下条目由置信指标自动筛出，仍须人工回听；未列出的片段也不等于已经核实。",
            "",
        ]
        for segment in uncertain:
            lines.append(
                f"- `{segment.start:.3f}—{segment.end:.3f}`：{segment.text or '[无文本]'}"
            )
        (self.review_dir / "uncertain-segments.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )

    def _write_analysis_template(self, metadata: dict[str, object]) -> None:
        title = str(metadata.get("title") or self.bvid)
        duration = metadata.get("duration")
        template = f"""# 视频证据解读：{title}

证据状态：`E0_COLLECTED / E1_NOT_REVIEWED / E2_NOT_STARTED`

来源：{self.source_url}

BV号：`{self.bvid}`

时长：`{duration}` 秒

## 核对声明

- [ ] 已从头到尾回听；
- [ ] 专名、数字、否定词和逻辑连接词已复核；
- [ ] 已检查关键画面；
- [ ] 无法确认处保留 `[听不清]` 或 `[画面不清]`；
- [ ] 自动转写没有被直接称为视频原话。

## 时间轴

| 时间范围 | 音轨明确内容 | 画面明确内容 | 证据编号 | 状态 |
| --- | --- | --- | --- | --- |
| `00:00—00:00` | 待核对 | 待核对 | 待填写 | `UNRESOLVED` |

## 论证结构

| 编号 | 类型 | 主张或转述 | 直接证据 | 限定与反例 |
| --- | --- | --- | --- | --- |
| `C01` | `PARAPHRASE` | 待填写 | 时间戳／逐字稿段号 | 待填写 |

## 画面独立信息

| 编号 | 时间戳 | 关键帧 | 可见内容 | 是否需要外部核验 |
| --- | --- | --- | --- | --- |
| `V01` | 待填写 | `frames/frame_00001.jpg` | 待填写 | 是／否 |

## 分析者推断

| 编号 | 推断 | 所依赖的直接证据 | 推导过程 | 其他可能解释 |
| --- | --- | --- | --- | --- |
| `I01` | 待填写 | `C01`／`V01` | 待填写 | 待填写 |

## 未解决项

- 待填写。
"""
        (self.analysis_dir / "evidence-map.md").write_text(
            template, encoding="utf-8", newline="\n"
        )

    def _write_manifest(self, metadata: dict[str, object]) -> None:
        manifest = {
            "schema_version": "0.1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "url": self.source_url,
                "bvid": self.bvid,
                "title": metadata.get("title"),
                "uploader": metadata.get("uploader"),
                "duration": metadata.get("duration"),
            },
            "tools": doctor_report()["tools"],
            "stages": self.stage_status,
            "artifacts_sha256": artifact_hashes(
                self.run_dir,
                included_prefixes=("raw/", "frames/", "logs/"),
            ),
            "evidence_status": {
                "E0": "COLLECTED",
                "E1": "NOT_REVIEWED",
                "E2": "NOT_STARTED",
            },
        }
        write_json(self.run_dir / "manifest.json", manifest)

    def collect(self) -> Path:
        report = doctor_report()
        if not report["ready_for_collection"]:
            raise EvidenceError("缺少 yt-dlp 或 FFmpeg；请先运行 bvevidence doctor")
        self._prepare()
        metadata = self._collect_metadata()
        subtitles = self._collect_subtitles()
        video_path = self._download_video()
        audio_path = self._extract_audio(video_path)
        self._extract_frames(video_path)
        self._prepare_transcript(subtitles, audio_path)
        self._write_analysis_template(metadata)
        self._write_manifest(metadata)
        return self.run_dir


def verify_run(run_dir: Path) -> tuple[bool, list[str]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise EvidenceError(f"缺少 manifest.json：{run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts_sha256")
    if not isinstance(expected, dict):
        raise EvidenceError("manifest 缺少 artifacts_sha256")
    actual = artifact_hashes(run_dir, included_prefixes=("raw/", "frames/", "logs/"))
    problems: list[str] = []
    for relative, expected_hash in expected.items():
        if relative not in actual:
            problems.append(f"MISSING {relative}")
        elif actual[relative] != expected_hash:
            problems.append(f"CHANGED {relative}")
    for relative in sorted(set(actual) - set(expected)):
        problems.append(f"UNTRACKED {relative}")
    return not problems, problems
