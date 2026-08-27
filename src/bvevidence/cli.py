from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import EvidenceError
from .pipeline import EvidencePipeline, PipelineOptions, doctor_report, verify_run


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")





def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bvevidence",
        description="可审计的 Bilibili 视频采集、转写与证据核对工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查外部工具和可选转写依赖")
    doctor.add_argument("--json", action="store_true", help="输出 JSON")

    collect = subparsers.add_parser("collect", help="生成一个新的视频证据包")
    collect.add_argument("source", help="BV号或 Bilibili 视频地址")
    collect.add_argument("--output", type=Path, default=Path("evidence"))
    collect.add_argument("--model", default="large-v3")
    collect.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    collect.add_argument("--compute-type", default="int8")
    collect.add_argument("--cookies-from-browser")
    collect.add_argument("--skip-video", action="store_true")
    collect.add_argument("--skip-transcribe", action="store_true")
    collect.add_argument("--skip-frames", action="store_true")
    collect.add_argument("--scene-threshold", type=float, default=0.35)

    verify = subparsers.add_parser("verify", help="校验证据包中的原始文件哈希")
    verify.add_argument("run_dir", type=Path)
    return parser


def _doctor(as_json: bool) -> int:
    report = doctor_report()
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name, value in report["tools"].items():
            status = value if value else "MISSING"
            print(f"{name}: {status}")
        print(f"collection: {'READY' if report['ready_for_collection'] else 'NOT READY'}")
        print(f"transcription: {'READY' if report['ready_for_transcription'] else 'NOT READY'}")
    return 0 if report["ready_for_collection"] else 1


def _collect(args: argparse.Namespace) -> int:
    options = PipelineOptions(
        source=args.source,
        output=args.output,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cookies_from_browser=args.cookies_from_browser,
        skip_video=args.skip_video,
        skip_transcribe=args.skip_transcribe,
        skip_frames=args.skip_frames,
        scene_threshold=args.scene_threshold,
    )
    run_dir = EvidencePipeline(options).collect()
    print(run_dir)
    return 0


def _verify(run_dir: Path) -> int:
    ok, problems = verify_run(run_dir.resolve())
    if ok:
        print("PASS: evidence artifacts match manifest")
        return 0
    for problem in problems:
        print(problem)
    return 1


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.json)
        if args.command == "collect":
            return _collect(args)
        if args.command == "verify":
            return _verify(args.run_dir)
    except (EvidenceError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error("未知命令")
    return 2
