from __future__ import annotations

import json
from pathlib import Path

import pytest

from bvevidence.core import artifact_hashes, canonical_url, extract_bvid, srt_timestamp
from bvevidence.pipeline import EvidencePipeline, verify_run
from bvevidence.transcribe import TranscriptSegment, write_srt


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("BV1vg8S6yEff", "BV1vg8S6yEff"),
        ("https://www.bilibili.com/video/BV1vg8S6yEff?p=1", "BV1vg8S6yEff"),
        ("prefix BV1vg8S6yEff suffix", "BV1vg8S6yEff"),
    ],
)
def test_extract_bvid(source: str, expected: str) -> None:
    assert extract_bvid(source) == expected


def test_extract_bvid_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        extract_bvid("not-a-video")


def test_canonical_url_discards_tracking_parameters() -> None:
    source = "https://www.bilibili.com/video/BV1vg8S6yEff?share_source=WEIXIN"
    assert canonical_url(source) == "https://www.bilibili.com/video/BV1vg8S6yEff"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://upos-sz-mirrorcos.bilivideo.com/path/video.m4s", True),
        ("https://api.bilibili.com/path/audio.m4s", True),
        ("http://upos-sz-mirrorcos.bilivideo.com/path/video.m4s", False),
        ("https://bilivideo.com.evil.example/path/video.m4s", False),
        ("https://example.com/video.m4s", False),
    ],
)
def test_allowed_media_url(url: str, expected: bool) -> None:
    assert EvidencePipeline._allowed_media_url(url) is expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (1.234, "00:00:01,234"),
        (61.001, "00:01:01,001"),
        (3661.999, "01:01:01,999"),
    ],
)
def test_srt_timestamp(seconds: float, expected: str) -> None:
    assert srt_timestamp(seconds) == expected


def test_write_srt_marks_empty_speech(tmp_path: Path) -> None:
    output = tmp_path / "transcript.srt"
    write_srt(
        output,
        [
            TranscriptSegment(
                index=1,
                start=0.0,
                end=1.5,
                text="",
                avg_logprob=-1.0,
                no_speech_prob=0.8,
            )
        ],
    )
    body = output.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in body
    assert "[无可辨识语音]" in body


def test_artifact_hashes_ignores_manifest(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.txt").write_text("evidence", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    hashes = artifact_hashes(tmp_path)
    assert list(hashes) == ["raw/a.txt"]


def test_verify_run_detects_changed_artifact(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    artifact = raw / "a.txt"
    artifact.write_text("first", encoding="utf-8")
    (tmp_path / "review").mkdir()
    review = tmp_path / "review" / "notes.md"
    review.write_text("editable", encoding="utf-8")
    manifest = {
        "artifacts_sha256": artifact_hashes(
            tmp_path, included_prefixes=("raw/", "frames/", "logs/")
        )
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ok, problems = verify_run(tmp_path)
    assert ok
    assert problems == []

    review.write_text("edited as expected", encoding="utf-8")
    ok, problems = verify_run(tmp_path)
    assert ok
    assert problems == []

    artifact.write_text("changed", encoding="utf-8")
    ok, problems = verify_run(tmp_path)
    assert not ok
    assert problems == ["CHANGED raw/a.txt"]
