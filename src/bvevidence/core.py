from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

BVID_PATTERN = re.compile(r"(?i)(BV[0-9A-Za-z]{10})")


class EvidenceError(RuntimeError):
    """Raised when an evidence stage cannot complete safely."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def extract_bvid(value: str) -> str:
    """Extract a canonical BV id from a BV id or Bilibili URL."""
    match = BVID_PATTERN.search(value.strip())
    if not match:
        raise ValueError(f"未找到有效 BV 号：{value}")
    raw = match.group(1)
    return "BV" + raw[2:]


def canonical_url(value: str) -> str:
    return f"https://www.bilibili.com/video/{extract_bvid(value)}"


def srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("时间戳不能为负数")
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(
    root: Path,
    ignored: Iterable[str] = ("manifest.json",),
    included_prefixes: Iterable[str] | None = None,
) -> dict[str, str]:
    ignored_set = set(ignored)
    prefixes = tuple(included_prefixes or ())
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ignored_set:
            continue
        if prefixes and not relative.startswith(prefixes):
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def executable_version(executable: str) -> str | None:
    resolved = shutil.which(executable)
    if not resolved:
        return None
    result = subprocess.run(
        [resolved, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else "unknown"


def run_command(command: Sequence[str], cwd: Path | None = None) -> CommandResult:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandResult(
        command=list(command),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def require_success(result: CommandResult, stage: str) -> CommandResult:
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-8:]
        detail = "\n".join(tail) or "命令未返回错误详情"
        raise EvidenceError(f"{stage}失败：\n{detail}")
    return result
