import os
import uuid
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from detector import detect_clip, crop_video

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
UPLOAD = '/tmp/clips'
os.makedirs(UPLOAD, exist_ok=True)

ALLOWED = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'}


def cleanup(max_min=30):
    now = time.time()
    for f in Path(UPLOAD).glob('*'):
        if now - f.stat().st_mtime > max_min * 60:
            f.unlink(missing_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    """Upload + detect + extract in one call."""
    cleanup()

    if 'video' not in request.files:
        return jsonify({'error': 'No video'}), 400

    file = request.files['video']
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'mp4'
    if ext not in ALLOWED:
        return jsonify({'error': f'Formato .{ext} no soportado'}), 400

    job = str(uuid.uuid4())[:8]
    inp = os.path.join(UPLOAD, f'{job}_in.{ext}')
    out = os.path.join(UPLOAD, f'{job}_clip.mp4')
    file.save(inp)

    # Detect
    bounds = detect_clip(inp)
    if not bounds:
        os.unlink(inp)
        return jsonify({'error': 'No se detectaron bordes del clip. ¿El video tiene overlay/diseño?'}), 422

    top, bot, left, right, vid_w, vid_h = bounds

    # Crop
    ok = crop_video(inp, out, bounds)
    if not ok or not os.path.exists(out):
        if os.path.exists(inp): os.unlink(inp)
        return jsonify({'error': 'Error al recortar el video'}), 500

    orig_size = os.path.getsize(inp)
    clip_size = os.path.getsize(out)

    # Clean input
    os.unlink(inp)

    return jsonify({
        'job': job,
        'bounds': {'top': top, 'bottom': bot, 'left': left, 'right': right},
        'original': {'w': vid_w, 'h': vid_h, 'size': orig_size},
        'clip': {'w': right - left, 'h': bot - top, 'size': clip_size},
        'download': f'/api/download/{job}'
    })


@app.route('/api/download/<job>')
def download(job):
    out = os.path.join(UPLOAD, f'{job}_clip.mp4')
    if not os.path.exists(out):
        return jsonify({'error': 'Archivo no encontrado o expirado'}), 404
    return send_file(out, mimetype='video/mp4', as_attachment=True,
                     download_name=f'clip_{job}.mp4')


@app.route('/health')
def health():
    return 'ok'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
