"""
Clip detection engine - Claude Vision + algorithmic fallback.

Primary:  Claude Vision API  — single frame, adapts to any design.
Fallback: Pillow multi-pass edge detection (no API required).
"""

import subprocess
import os
import json
import re
import tempfile
import base64
from PIL import Image

try:
    import anthropic
    _CLAUDE = True
except ImportError:
    _CLAUDE = False

# Claude works best with images <= this dimension on the longest side
_MAX_DIM = 1568


# ── Video utilities ──────────────────────────────────────────────────────────

def get_video_info(path):
    r = subprocess.run([
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', path
    ], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    info = json.loads(r.stdout)
    vs = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not vs:
        return None
    return {
        'width':    int(vs['width']),
        'height':   int(vs['height']),
        'duration': float(info['format'].get('duration', 0)),
    }


def _extract_frame(video_path, time_sec, tmp_dir, grayscale=False):
    """Extract one frame; returns file path (color) or PIL Image (grayscale)."""
    out = os.path.join(tmp_dir, f'frame_{time_sec:.2f}.png')
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(time_sec), '-i', video_path,
        '-vframes', '1', '-v', 'quiet', out
    ], capture_output=True)
    if not os.path.exists(out):
        return None
    if grayscale:
        return Image.open(out).convert('L')
    return out


# ── Claude Vision detection ──────────────────────────────────────────────────

def _prepare_for_claude(img_path, tmp_dir):
    """
    Scale image so the longest side <= _MAX_DIM and convert to JPEG.
    Returns (jpeg_path, scale_factor).
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    scale = 1.0
    if max(w, h) > _MAX_DIM:
        scale = _MAX_DIM / max(w, h)
        nw = max(2, int(w * scale) & ~1)   # keep even
        nh = max(2, int(h * scale) & ~1)
        img = img.resize((nw, nh), Image.LANCZOS)
    out = os.path.join(tmp_dir, 'claude_input.jpg')
    img.save(out, 'JPEG', quality=92)
    return out, scale


def _detect_with_claude(frame_path, orig_w, orig_h, tmp_dir):
    """
    Ask Claude Vision for the exact bounding box of the inner video clip.
    Returns (top, bottom, left, right) in original pixel coordinates, or None.
    """
    if not _CLAUDE:
        return None

    try:
        jpeg_path, scale = _prepare_for_claude(frame_path, tmp_dir)
        img = Image.open(jpeg_path)
        sw, sh = img.size

        with open(jpeg_path, 'rb') as f:
            b64 = base64.standard_b64encode(f.read()).decode()

        prompt = (
            f"This is a single frame from a video. "
            f"The frame is {sw}×{sh} pixels (width × height).\n\n"
            "The frame contains a video clip embedded inside a designed layout. "
            "The actual footage is surrounded by black bars, colored overlays, "
            "borders, or decorative elements that belong to the design — NOT to the clip.\n\n"
            "Your task: find the EXACT pixel boundaries of the inner video content area "
            "(the rectangle that contains the real footage, excluding all surrounding design).\n\n"
            "Look for sharp edges, hard color transitions, or clear visual boundaries "
            "that separate the footage from the surrounding design/overlays.\n\n"
            "Respond with ONLY a raw JSON object — no markdown, no explanation:\n"
            '{"top": <y where content starts>, "bottom": <y where content ends>, '
            '"left": <x where content starts>, "right": <x where content ends>}'
        )

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model='claude-opus-4-5',
            max_tokens=128,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': b64,
                        }
                    },
                    {'type': 'text', 'text': prompt}
                ]
            }]
        )

        raw = resp.content[0].text.strip()
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if not m:
            print(f'[detector] Claude returned unexpected output: {raw[:120]}')
            return None

        c = json.loads(m.group())
        inv = 1.0 / scale
        top   = int(round(float(c['top'])    * inv))
        bot   = int(round(float(c['bottom']) * inv))
        left  = int(round(float(c['left'])   * inv))
        right = int(round(float(c['right'])  * inv))

        # Clamp to valid range
        top   = max(0, min(top,  orig_h - 4))
        bot   = max(top + 4, min(bot,  orig_h))
        left  = max(0, min(left, orig_w - 4))
        right = max(left + 4, min(right, orig_w))

        # Reject if the result covers nearly the entire frame (likely wrong)
        if (right - left) > orig_w * 0.98 and (bot - top) > orig_h * 0.98:
            print('[detector] Claude returned full-frame bounds — ignoring')
            return None

        # Reject if unrealistically small
        if (right - left) < orig_w * 0.05 or (bot - top) < orig_h * 0.05:
            print('[detector] Claude returned bounds too small — ignoring')
            return None

        return (top, bot, left, right)

    except Exception as e:
        print(f'[detector] Claude error: {e}')
        return None


# ── Algorithmic fallback ─────────────────────────────────────────────────────

def _row_diversity(row, w):
    s = sorted(row)
    med = s[w // 2]
    return sum(1 for v in row if abs(v - med) > 20) / w


def _row_gradient_cov(gray, y, w):
    if y <= 0 or y >= len(gray) - 1:
        return 0.0
    strong = sum(1 for x in range(w) if abs(-gray[y-1][x] + gray[y+1][x]) * 2 > 30)
    return strong / w


def _group_rows(rows, gap=5):
    if not rows:
        return []
    groups, cur = [], [rows[0]]
    for i in range(1, len(rows)):
        if rows[i] - rows[i-1] <= gap:
            cur.append(rows[i])
        else:
            groups.append(cur)
            cur = [rows[i]]
    groups.append(cur)
    return groups


def _smooth(arr, window):
    h = window // 2
    n = len(arr)
    return [
        sum(arr[max(0, i-h):min(n, i+h+1)]) / (min(n, i+h+1) - max(0, i-h))
        for i in range(n)
    ]


def _detect_bounds_gray(gray, w, h):
    """Multi-pass edge detection on a grayscale pixel matrix."""
    divs  = [_row_diversity(gray[y], w) for y in range(h)]
    grads = [_row_gradient_cov(gray, y, w) for y in range(h)]

    top_y = bottom_y = None

    # Pass 1 – full-width gradient edges
    fw = [y for y in range(h) if grads[y] > 0.80]
    for g in _group_rows(fw, 5):
        c  = g[len(g) // 2]
        ad = sum(divs[max(0, c-40):c])       / max(1, min(40, c))
        bd = sum(divs[c:min(h, c+40)])       / max(1, min(40, h-c))
        if bd > ad and top_y is None:
            top_y = g[0]
        elif ad > bd:
            bottom_y = g[-1]

    # Pass 2 – diversity transitions
    if top_y is None or bottom_y is None:
        sd = _smooth(divs, 30)
        best_s, best_e, best_l, cur_s = -1, -1, 0, -1
        for y in range(h):
            if sd[y] > 0.25 and cur_s < 0:
                cur_s = y
            if (sd[y] <= 0.25 or y == h-1) and cur_s >= 0:
                l = y - cur_s
                if l > best_l:
                    best_l, best_s, best_e = l, cur_s, y
                cur_s = -1
        if best_s >= 0:
            if top_y    is None: top_y    = best_s
            if bottom_y is None: bottom_y = best_e

    # Pass 3 – row variance
    if top_y is None or bottom_y is None:
        stds = []
        for y in range(h):
            m = sum(gray[y]) / w
            stds.append((sum((v - m) ** 2 for v in gray[y]) / w) ** 0.5)
        ss  = _smooth(stds, 30)
        thr = max(max(ss) * 0.2, 15)
        for y in range(h):
            if ss[y] > thr and top_y is None:
                top_y = y
                break
        for y in range(h-1, -1, -1):
            if ss[y] > thr and bottom_y is None:
                bottom_y = y
                break

    if (top_y is None or bottom_y is None
            or bottom_y <= top_y
            or (bottom_y - top_y) < h * 0.08):
        return None

    # Horizontal bounds
    lx, rx  = 0, w
    ch2     = bottom_y - top_y
    cdivs   = []
    for x in range(w):
        col = [gray[y][x] for y in range(top_y, bottom_y)]
        sc  = sorted(col)
        med = sc[len(col) // 2]
        cdivs.append(sum(1 for v in col if abs(v - med) > 20) / ch2)
    cc = [x for x in range(w) if cdivs[x] > 0.15]
    if cc and (cc[-1] - cc[0]) < w * 0.85:
        lx, rx = cc[0], cc[-1] + 1

    return (top_y, bottom_y, lx, rx)


def _detect_fallback(video_path, w, h, dur):
    """Analyse 5 frames and take the median bounds."""
    times      = [dur * p for p in [0.15, 0.35, 0.55, 0.75, 0.90]]
    all_bounds = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            img = _extract_frame(video_path, t, tmp, grayscale=True)
            if img is None:
                continue
            pixels = list(img.getdata())
            gray   = [pixels[y*w:(y+1)*w] for y in range(h)]
            b      = _detect_bounds_gray(gray, w, h)
            if b:
                all_bounds.append(b)
    if not all_bounds:
        return None
    mid   = len(all_bounds) // 2
    top   = sorted(b[0] for b in all_bounds)[mid]
    bot   = sorted(b[1] for b in all_bounds)[mid]
    left  = sorted(b[2] for b in all_bounds)[mid]
    right = sorted(b[3] for b in all_bounds)[mid]
    return (top, bot, left, right)


# ── Public API ───────────────────────────────────────────────────────────────

def detect_clip(video_path, debug=False):
    """
    Detect the inner clip boundary.

    Strategy:
      1. Extract a single frame at 15 % of the video duration.
      2. Send it to Claude Vision → precise, design-agnostic detection.
      3. If Claude is unavailable or returns unusable results, fall back to the
         algorithmic multi-frame approach.

    Returns (top, bot, left, right, vid_w, vid_h, method) or None.
    'method' is either 'claude-vision' or 'algorithmic'.
    """
    info = get_video_info(video_path)
    if not info:
        return None

    w, h, dur = info['width'], info['height'], info['duration']
    t          = max(0.5, dur * 0.15)
    method     = 'algorithmic'
    bounds     = None

    # ── Try Claude Vision first ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = _extract_frame(video_path, t, tmp, grayscale=False)
        if frame_path and _CLAUDE:
            bounds = _detect_with_claude(frame_path, w, h, tmp)
            if bounds:
                method = 'claude-vision'
                if debug:
                    top, bot, left, right = bounds
                    print(f'[Claude] ({left},{top}) → ({right},{bot})')

    # ── Algorithmic fallback ─────────────────────────────────────────────────
    if bounds is None:
        if debug:
            print('[detector] Falling back to algorithmic detection')
        bounds = _detect_fallback(video_path, w, h, dur)
        if not bounds:
            return None
        if debug:
            top, bot, left, right = bounds
            print(f'[Fallback] ({left},{top}) → ({right},{bot})')

    top, bot, left, right = bounds

    # Ensure even pixel values (required by most video codecs)
    top   += top   % 2
    bot   -= bot   % 2
    left  += left  % 2
    right -= right % 2

    return (top, bot, left, right, w, h, method)


def crop_video(input_path, output_path, bounds):
    """Crop input_path using the first four elements of bounds (top,bot,left,right)."""
    top, bot, left, right = bounds[0], bounds[1], bounds[2], bounds[3]
    cw = (right - left) - ((right - left) % 2)
    ch = (bot   - top)  - ((bot   - top)  % 2)

    r = subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'crop={cw}:{ch}:{left}:{top}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        r2 = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f'crop={cw}:{ch}:{left}:{top}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-an', '-movflags', '+faststart',
            output_path
        ], capture_output=True, text=True)
        return r2.returncode == 0

    return True
