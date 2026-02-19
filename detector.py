"""
Clip detection engine.
Detects and extracts original video clips from designed frames.
Works with ANY overlay color. Uses only Pillow + FFmpeg.
"""

import subprocess
import os
import json
import tempfile
from PIL import Image


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
        'width': int(vs['width']),
        'height': int(vs['height']),
        'duration': float(info['format'].get('duration', 0)),
    }


def extract_frame(video_path, time_sec, tmp_dir):
    tmp_path = os.path.join(tmp_dir, f'frame_{time_sec:.2f}.png')
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(time_sec), '-i', video_path,
        '-vframes', '1', '-v', 'quiet', tmp_path
    ], capture_output=True)
    if not os.path.exists(tmp_path):
        return None
    return Image.open(tmp_path).convert('L')


def row_diversity(row, w):
    s = sorted(row)
    med = s[w // 2]
    return sum(1 for v in row if abs(v - med) > 20) / w


def row_gradient_cov(gray, y, w):
    if y <= 0 or y >= len(gray) - 1:
        return 0.0
    strong = sum(1 for x in range(w) if abs(-gray[y-1][x] + gray[y+1][x]) * 2 > 30)
    return strong / w


def group_rows(rows, gap=5):
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


def smooth(arr, window):
    h = window // 2
    n = len(arr)
    return [sum(arr[max(0,i-h):min(n,i+h+1)]) / (min(n,i+h+1) - max(0,i-h)) for i in range(n)]


def detect_bounds_gray(gray, w, h):
    divs = [row_diversity(gray[y], w) for y in range(h)]
    grads = [row_gradient_cov(gray, y, w) for y in range(h)]

    top_y = bottom_y = None

    # PASS 1: Full-width gradient edges
    fw = [y for y in range(h) if grads[y] > 0.80]
    for g in group_rows(fw, 5):
        c = g[len(g) // 2]
        ad = sum(divs[max(0,c-40):c]) / max(1, min(40, c))
        bd = sum(divs[c:min(h,c+40)]) / max(1, min(40, h-c))
        if bd > ad and top_y is None:
            top_y = g[0]
        elif ad > bd:
            bottom_y = g[-1]

    # PASS 2: Diversity transitions
    if top_y is None or bottom_y is None:
        sd = smooth(divs, 30)
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
            if top_y is None: top_y = best_s
            if bottom_y is None: bottom_y = best_e

    # PASS 3: Row variance
    if top_y is None or bottom_y is None:
        stds = []
        for y in range(h):
            m = sum(gray[y]) / w
            stds.append((sum((v-m)**2 for v in gray[y]) / w) ** 0.5)
        ss = smooth(stds, 30)
        thr = max(max(ss) * 0.2, 15)
        for y in range(h):
            if ss[y] > thr and top_y is None:
                top_y = y; break
        for y in range(h-1, -1, -1):
            if ss[y] > thr and bottom_y is None:
                bottom_y = y; break

    if top_y is None or bottom_y is None or bottom_y <= top_y or (bottom_y - top_y) < h * 0.08:
        return None

    # Horizontal bounds
    lx, rx = 0, w
    ch2 = bottom_y - top_y
    cdivs = []
    for x in range(w):
        col = [gray[y][x] for y in range(top_y, bottom_y)]
        sc = sorted(col)
        med = sc[len(col)//2]
        cdivs.append(sum(1 for v in col if abs(v-med) > 20) / ch2)
    cc = [x for x in range(w) if cdivs[x] > 0.15]
    if cc and (cc[-1] - cc[0]) < w * 0.85:
        lx, rx = cc[0], cc[-1] + 1

    return (top_y, bottom_y, lx, rx)


def detect_clip(video_path, debug=False):
    info = get_video_info(video_path)
    if not info:
        return None

    w, h, dur = info['width'], info['height'], info['duration']
    times = [dur * p for p in [0.15, 0.35, 0.55, 0.75, 0.90]]
    all_bounds = []

    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            img = extract_frame(video_path, t, tmp)
            if img is None:
                continue
            pixels = list(img.getdata())
            gray = [pixels[y*w:(y+1)*w] for y in range(h)]
            b = detect_bounds_gray(gray, w, h)
            if b:
                all_bounds.append(b)
                if debug:
                    print(f"  t={t:.1f}s: ({b[2]},{b[0]})->({b[3]},{b[1]})")

    if not all_bounds:
        return None

    mid = len(all_bounds) // 2
    top = sorted(b[0] for b in all_bounds)[mid]
    bot = sorted(b[1] for b in all_bounds)[mid]
    left = sorted(b[2] for b in all_bounds)[mid]
    right = sorted(b[3] for b in all_bounds)[mid]

    top += top % 2
    bot -= bot % 2
    left += left % 2
    right -= right % 2

    return (top, bot, left, right, w, h)


def crop_video(input_path, output_path, bounds):
    top, bot, left, right, _, _ = bounds
    cw = (right - left) - ((right - left) % 2)
    ch = (bot - top) - ((bot - top) % 2)

    r = subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'crop={cw}:{ch}:{left}:{top}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', '-movflags', '+faststart',
        output_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        # Retry without audio copy
        r2 = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f'crop={cw}:{ch}:{left}:{top}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-an', '-movflags', '+faststart',
            output_path
        ], capture_output=True, text=True)
        return r2.returncode == 0

    return True
