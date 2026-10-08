#!/usr/bin/env python3
"""
Clipper: video de YouTube -> 10 clips verticales con subtítulos.

Pipeline:
  1. yt-dlp descarga el video
  2. faster-whisper transcribe con tiempos por palabra
  3. Claude elige los N mejores momentos (JSON con inicio/fin/título)
  4. ffmpeg corta, recorta a 9:16 y quema los subtítulos

Requisitos:
  pip install yt-dlp faster-whisper anthropic
  ffmpeg instalado en el sistema
  export ANTHROPIC_API_KEY=...

Uso:
  python clipper.py "https://youtube.com/watch?v=XXXX" --clips 10 --lang es

Usa solo videos tuyos o con permiso del autor.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


def download(url: str, out_dir: Path):
    import yt_dlp

    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return out_dir / "source.mp4", info.get("title", "video")


def transcribe(video: Path, model_size: str, lang: str | None):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _ = model.transcribe(
        str(video), word_timestamps=True, language=lang, vad_filter=True
    )
    segs, words = [], []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    return segs, words


def pick_clips(segs, n: int, min_s: int, max_s: int, model: str):
    import anthropic

    client = anthropic.Anthropic()
    transcript = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs
    )
    prompt = f"""Eres editor de clips virales para redes (TikTok/Reels/Shorts).
De esta transcripción con tiempos en segundos, elige los {n} mejores momentos.
Cada clip debe durar entre {min_s} y {max_s} segundos, empezar con un gancho
fuerte, tener sentido por sí solo y no solaparse con otros clips.
Responde SOLO con un JSON: una lista de objetos con las claves
"start" (número), "end" (número), "title" (texto corto), "reason" (texto corto).

Transcripción:
{transcript}"""
    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    clips = json.loads(text)
    clips = [c for c in clips if c["end"] - c["start"] >= 5]
    return sorted(clips, key=lambda c: c["start"])[:n]


def ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(words, start: float, end: float, path: Path, group: int = 3):
    ws = [w for w in words if w["start"] >= start and w["end"] <= end]
    lines = []
    for i in range(0, len(ws), group):
        chunk = ws[i : i + group]
        a = chunk[0]["start"] - start
        b = chunk[-1]["end"] - start
        text = " ".join(w["word"] for w in chunk).upper()
        lines.append(f"{len(lines) + 1}\n{ts(a)} --> {ts(b)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_clip(source: Path, clip, words, out_dir: Path, idx: int, vertical: bool):
    srt_name = f"clip_{idx:02}.srt"
    out_name = f"clip_{idx:02}.mp4"
    write_srt(words, clip["start"], clip["end"], out_dir / srt_name)

    style = (
        "FontName=Arial,FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,Outline=2,Alignment=2,MarginV=80"
    )
    vf = []
    if vertical:
        vf.append("crop=ih*9/16:ih,scale=1080:1920")
    vf.append(f"subtitles={srt_name}:force_style='{style}'")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(clip["start"]), "-i", str(source.resolve()),
        "-t", str(clip["end"] - clip["start"]),
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", out_name,
    ]
    # cwd = carpeta de salida para evitar problemas de rutas en el filtro subtitles
    subprocess.run(cmd, check=True, cwd=out_dir)
    return out_dir / out_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--clips", type=int, default=10)
    p.add_argument("--min", type=int, default=20, help="duración mínima (s)")
    p.add_argument("--max", type=int, default=60, help="duración máxima (s)")
    p.add_argument("--lang", default=None, help="idioma del audio, ej. es, en")
    p.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--llm", default="claude-sonnet-5-5")
    p.add_argument("--horizontal", action="store_true", help="no recortar a 9:16")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/4 Descargando video...")
    source, title = download(args.url, out)

    print("2/4 Transcribiendo con IA...")
    segs, words = transcribe(source, args.whisper, args.lang)

    print(f"3/4 Eligiendo los {args.clips} mejores clips...")
    clips = pick_clips(segs, args.clips, args.min, args.max, args.llm)
    (out / "clips.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("4/4 Cortando y subtitulando...")
    for i, c in enumerate(clips, 1):
        f = render_clip(source, c, words, out, i, not args.horizontal)
        print(f"  {f.name}  [{c['start']:.0f}s-{c['end']:.0f}s]  {c['title']}")

    print(f"\nListo: {len(clips)} clips en {out.resolve()}")


if __name__ == "__main__":
    main()
