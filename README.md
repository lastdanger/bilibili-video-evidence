# Bilibili Video Evidence

面向研究、事实核验和内容解读的 Bilibili 视频证据采集工具。

它不是“输入链接，自动生成一篇看起来合理的总结”。它把视频处理拆成可审计的证据链：

```text
公开视频 → 元数据／字幕／音视频 → 本地转写 → 关键帧 → 人工校正 → 带时间戳解读
```

自动转写始终标记为 `ASR_RAW`，不能直接当作视频原话。分析中的每项结论必须指向时间戳、字幕段或画面证据。

## 功能

- 使用 `yt-dlp` 获取公开视频、音轨、公开字幕和元数据；
- 视频没有字幕时，可选用 `faster-whisper` 本地转写；
- 使用 FFmpeg 提取场景变化关键帧和 16 kHz 单声道音轨；
- 保存命令、工具版本、原始元数据和文件 SHA-256；
- 生成逐字稿校正表、疑难片段表和证据解读模板；
- `verify` 命令检查证据文件是否在分析后被意外改动。

## 安装

要求：

- Python 3.11+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg](https://ffmpeg.org/)

基础安装：

```bash
python -m pip install -e .
```

需要本地转写：

```bash
python -m pip install -e ".[transcribe]"
```

开发与测试：

```bash
python -m pip install -e ".[dev]"
pytest
```

## 快速使用

先检查环境：

```bash
bvevidence doctor
```

采集完整证据包：

```bash
bvevidence collect "https://www.bilibili.com/video/BV1vg8S6yEff" --output evidence
```

CPU 转写可显式指定：

```bash
bvevidence collect "BV1vg8S6yEff" \
  --output evidence \
  --model large-v3 \
  --device cpu \
  --compute-type int8
```

若视频需要当前浏览器登录态：

```bash
bvevidence collect "BV..." --cookies-from-browser chrome
```

Cookie 只由 `yt-dlp` 在运行时读取，不会写入证据包或日志。

校验证据包：

```bash
bvevidence verify evidence/BV1vg8S6yEff_YYYYMMDDTHHMMSSZ
```

## 产物结构

```text
evidence/BV..._时间/
├─ manifest.json
├─ raw/
│  ├─ metadata.json
│  ├─ video.mp4
│  ├─ audio.wav
│  ├─ subtitles/
│  └─ transcript.asr.jsonl
├─ frames/
├─ logs/
├─ review/
│  ├─ transcript.to-review.srt
│  └─ uncertain-segments.md
└─ analysis/
   └─ evidence-map.md
```

`raw/` 是机器采集层；`review/` 是人工核对层；`analysis/` 才允许写解释。三层不得互相覆盖。

## 无猜测协议

详细规则见 [docs/evidence-protocol.md](docs/evidence-protocol.md)。最重要的四条是：

1. 没听清就写 `[听不清]`，不能按上下文补句。
2. ASR 文本只能称为“自动转写”，人工回听后才能称为“视频原话”。
3. 画面信息和音轨信息分开记录。
4. 推断必须标记为 `INFERENCE`，并列出它依赖的直接证据。

## 合规边界

- 只处理你有权访问和研究的内容；遵守网站规则及适用法律。
- 仓库不包含视频、字幕、Cookie、模型文件或第三方受版权保护文本。
- 证据包默认被 `.gitignore` 排除，不应提交到公开仓库。
- 请勿绕过付费、会员、地区或访问权限限制。

## License

[MIT](LICENSE)
