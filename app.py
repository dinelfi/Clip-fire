"""Servidor web de Clipper. Uso: python app.py  ->  http://localhost:5000"""
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import clipper  # el clipper.py de la misma carpeta

app = Flask(__name__)
BASE = Path("jobs")
BASE.mkdir(exist_ok=True)
JOBS = {}
LOCK = threading.Lock()  # un video a la vez para no saturar la máquina


def work(jid, url, n, lang, vertical):
    job = JOBS[jid]
    out = BASE / jid
    out.mkdir(exist_ok=True)
    with LOCK:
        try:
            job.update(step="Descargando video...")
            source, title = clipper.download(url, out)
            job.update(title=title, step="Transcribiendo con IA...")
            segs, words = clipper.transcribe(source, "small", lang)
            job.update(step=f"Eligiendo los {n} mejores clips...")
            clips = clipper.pick_clips(segs, n, 20, 60, "claude-sonnet-5-5")
            for i, c in enumerate(clips, 1):
                job.update(step=f"Cortando y subtitulando clip {i} de {len(clips)}...")
                f = clipper.render_clip(source, c, words, out, i, vertical)
                job["clips"].append(
                    {
                        "file": f.name,
                        "title": c.get("title", f"Clip {i}"),
                        "reason": c.get("reason", ""),
                        "start": round(c["start"]),
                        "end": round(c["end"]),
                    }
                )
            job.update(status="done", step="Listo")
        except Exception as e:  # noqa: BLE001
            job.update(status="error", step=f"Error: {e}")


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/api/jobs")
def create_job():
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not url.startswith("http"):
        return jsonify(error="URL inválida"), 400
    n = max(1, min(int(d.get("clips", 10)), 20))
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "step": "En cola...", "title": "", "clips": []}
    threading.Thread(
        target=work,
        args=(jid, url, n, d.get("lang") or None, bool(d.get("vertical", True))),
        daemon=True,
    ).start()
    return jsonify(id=jid)


@app.get("/api/jobs/<jid>")
def get_job(jid):
    return jsonify(JOBS.get(jid) or {"status": "error", "step": "No existe", "clips": []})


@app.get("/clips/<jid>/<name>")
def get_clip(jid, name):
    return send_from_directory(BASE / jid, name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
