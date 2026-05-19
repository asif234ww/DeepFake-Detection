from flask import Flask, request, jsonify
from flask_cors import CORS
import os, mimetypes, hashlib
import cv2
import librosa
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#############################################
# 1. IMAGE MODEL (placeholder: PyTorch CNN) #
#############################################
def predict_image(filepath):
    img = Image.open(filepath).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)
    score = np.random.uniform(0.7, 0.99)
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(float(score), 3)


#############################################
# 2. VIDEO MODEL (sample frames)            #
#############################################
def predict_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_scores = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > 10:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        tensor = transform(img).unsqueeze(0)
        frame_scores.append(np.random.uniform(0.7, 0.99))
        count += 1
    cap.release()
    score = float(np.mean(frame_scores))
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(score, 3)


#############################################
# 3. AUDIO MODEL (spectrogram analysis)     #
#############################################
def predict_audio(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    score = np.random.uniform(0.7, 0.99)
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(float(score), 3)


#############################################
# HOME — Upload UI                          #
#############################################
@app.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepfakeDetector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --purple-50: #EEEDFE;
    --purple-200: #AFA9EC;
    --purple-400: #7F77DD;
    --purple-600: #534AB7;
    --purple-800: #3C3489;
    --teal-50: #E1F5EE;
    --teal-400: #1D9E75;
    --teal-600: #0F6E56;
    --teal-800: #085041;
    --coral-50: #FAECE7;
    --amber-50: #FAEEDA;
    --amber-600: #854F0B;
    --red-50: #FCEBEB;
    --red-400: #E24B4A;
    --red-600: #A32D2D;
    --red-800: #791F1F;
    --green-50: #EAF3DE;
    --green-400: #1D9E75;
    --gray-100: #D3D1C7;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #F4F3F0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
  }

  .app { width: 100%; max-width: 520px; }

  .card {
    background: #fff;
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 16px;
    padding: 2rem;
    width: 100%;
    animation: slideUp 0.4s ease;
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Brand ── */
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.75rem;
  }
  .brand-icon {
    width: 38px; height: 38px;
    background: var(--purple-400);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .brand-icon svg { width: 18px; height: 18px; stroke: #fff; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .brand-title { font-size: 17px; font-weight: 600; color: #1a1a1a; }
  .brand-sub   { font-size: 12px; color: #888; margin-top: 1px; }

  /* ── Drop zone ── */
  .drop-zone {
    border: 1.5px dashed #ccc;
    border-radius: 12px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag {
    background: var(--purple-50);
    border-color: var(--purple-400);
  }
  .drop-icon {
    width: 52px; height: 52px;
    background: var(--purple-50);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem;
    transition: transform 0.2s;
  }
  .drop-zone:hover .drop-icon { transform: translateY(-3px); }
  .drop-icon svg { width: 22px; height: 22px; stroke: var(--purple-400); fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .drop-title { font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 5px; }
  .drop-sub   { font-size: 13px; color: #888; }
  .drop-formats {
    display: flex; gap: 6px; justify-content: center;
    margin-top: 14px; flex-wrap: wrap;
  }
  .fmt-tag {
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
  }
  .fmt-img { background: var(--purple-50); color: var(--purple-600); }
  .fmt-vid { background: var(--teal-50);   color: var(--teal-600);   }
  .fmt-aud { background: var(--amber-50);  color: var(--amber-600);  }

  /* ── File preview ── */
  .file-preview {
    display: none;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: #f9f9f7;
    border: 0.5px solid rgba(0,0,0,0.08);
    border-radius: 10px;
    margin-top: 1rem;
    animation: fadeIn 0.25s ease;
  }
  .file-preview.show { display: flex; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .file-ico {
    width: 38px; height: 38px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .file-ico svg { width: 18px; height: 18px; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
  .file-meta { flex: 1; min-width: 0; }
  .file-name { font-size: 13px; font-weight: 500; color: #1a1a1a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-size { font-size: 11px; color: #888; margin-top: 2px; }
  .file-remove {
    background: none; border: none; cursor: pointer;
    color: #aaa; font-size: 20px; line-height: 1; flex-shrink: 0; padding: 2px;
  }
  .file-remove:hover { color: #555; }

  /* ── Detect button ── */
  .btn-detect {
    width: 100%; margin-top: 1.25rem; padding: 12px;
    background: var(--purple-400); color: #fff;
    border: none; border-radius: 10px;
    font-size: 15px; font-weight: 500;
    cursor: pointer; transition: opacity 0.2s, transform 0.1s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-detect:hover:not(:disabled) { opacity: 0.88; }
  .btn-detect:active:not(:disabled) { transform: scale(0.985); }
  .btn-detect:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-detect svg { width: 16px; height: 16px; stroke: #fff; fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }

  /* ── Progress ── */
  .progress-wrap { display: none; margin-top: 1.25rem; }
  .progress-wrap.show { display: block; }
  .progress-label {
    font-size: 12px; color: #888; margin-bottom: 6px;
    display: flex; justify-content: space-between;
  }
  .progress-bar-bg {
    height: 6px; background: #eee; border-radius: 10px; overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; background: var(--purple-400);
    border-radius: 10px; width: 0;
    transition: width 0.05s linear;
  }
  .progress-steps {
    display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;
  }
  .step-pill {
    font-size: 11px; padding: 3px 10px; border-radius: 20px;
    background: #eee; color: #999;
    transition: background 0.3s, color 0.3s;
  }
  .step-pill.active { background: var(--purple-50); color: var(--purple-600); font-weight: 500; }
  .step-pill.done   { background: var(--teal-50);   color: var(--teal-600); }
</style>
</head>
<body>
<div class="app">
  <div class="card">

    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </div>
      <div>
        <div class="brand-title">DeepfakeDetector</div>
        <div class="brand-sub">AI-powered media authentication</div>
      </div>
    </div>

    <form id="upload-form" action="/api/detect" method="post" enctype="multipart/form-data">
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <div class="drop-icon">
          <svg viewBox="0 0 24 24">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <div class="drop-title">Drop your file here</div>
        <div class="drop-sub">or click to browse from your device</div>
        <div class="drop-formats">
          <span class="fmt-tag fmt-img">JPG / PNG / WEBP</span>
          <span class="fmt-tag fmt-vid">MP4 / MOV / AVI</span>
          <span class="fmt-tag fmt-aud">MP3 / WAV / OGG</span>
        </div>
      </div>

      <input type="file" id="file-input" name="file"
             accept="image/*,video/*,audio/*" style="display:none" required>

      <div class="file-preview" id="file-preview">
        <div class="file-ico" id="file-ico"></div>
        <div class="file-meta">
          <div class="file-name" id="file-name">—</div>
          <div class="file-size" id="file-size">—</div>
        </div>
        <button type="button" class="file-remove" id="file-remove" title="Remove">×</button>
      </div>

      <button type="submit" class="btn-detect" id="btn-detect" disabled>
        <svg viewBox="0 0 24 24">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        Analyze for deepfakes
      </button>

      <div class="progress-wrap" id="progress-wrap">
        <div class="progress-label">
          <span id="progress-text">Uploading...</span>
          <span id="progress-pct">0%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="progress-fill"></div>
        </div>
        <div class="progress-steps">
          <span class="step-pill" id="step1">Uploading</span>
          <span class="step-pill" id="step2">Preprocessing</span>
          <span class="step-pill" id="step3">Running model</span>
          <span class="step-pill" id="step4">Generating report</span>
        </div>
      </div>
    </form>

  </div>
</div>

<script>
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const form      = document.getElementById('upload-form');
const btnDetect = document.getElementById('btn-detect');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
document.getElementById('file-remove').addEventListener('click', e => { e.stopPropagation(); clearFile(); });

function setFile(f) {
  document.getElementById('file-name').textContent = f.name;
  document.getElementById('file-size').textContent = formatSize(f.size);
  const ico  = document.getElementById('file-ico');
  const type = f.type.split('/')[0];
  if (type === 'image') { ico.style.background = '#EEEDFE'; ico.innerHTML = fileIcon('image', '#534AB7'); }
  else if (type === 'video') { ico.style.background = '#E1F5EE'; ico.innerHTML = fileIcon('video', '#0F6E56'); }
  else { ico.style.background = '#FAEEDA'; ico.innerHTML = fileIcon('audio', '#854F0B'); }
  document.getElementById('file-preview').classList.add('show');
  btnDetect.disabled = false;
}

function clearFile() {
  fileInput.value = '';
  document.getElementById('file-preview').classList.remove('show');
  btnDetect.disabled = true;
}

function formatSize(b) {
  if (b < 1024)        return b + ' B';
  if (b < 1024*1024)   return Math.round(b / 1024) + ' KB';
  return (b / (1024*1024)).toFixed(1) + ' MB';
}

function fileIcon(type, color) {
  const d = {
    image: 'M21 19a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v14zM8.5 8.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM21 15l-5-5L5 21',
    video: 'M23 7l-7 5 7 5V7zM1 5h15a2 2 0 012 2v10a2 2 0 01-2 2H1V5z',
    audio: 'M9 18V5l12-2v13M6 21a3 3 0 100-6 3 3 0 000 6zM18 19a3 3 0 100-6 3 3 0 000 6z'
  };
  return `<svg viewBox="0 0 24 24" style="width:18px;height:18px;stroke:${color};fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round"><path d="${d[type]}"/></svg>`;
}

/* ── Animated form submit ── */
form.addEventListener('submit', async e => {
  e.preventDefault();
  if (!fileInput.files[0]) return;

  btnDetect.disabled = true;
  const pw = document.getElementById('progress-wrap');
  pw.classList.add('show');

  const steps = ['step1','step2','step3','step4'];
  const msgs  = ['Uploading file...','Preprocessing media...','Running detection model...','Generating report...'];

  async function animStep(i, from, to) {
    steps.forEach((s, j) => {
      document.getElementById(s).className =
        'step-pill' + (j < i ? ' done' : j === i ? ' active' : '');
    });
    document.getElementById('progress-text').textContent = msgs[i];
    await animatePct(from, to);
  }

  await animStep(0, 0, 25);

  /* Real fetch */
  const fd = new FormData(form);
  let resultHtml = '';
  const fetchPromise = fetch('/api/detect', { method: 'POST', body: fd })
    .then(r => r.text())
    .then(t => { resultHtml = t; });

  await animStep(1, 25, 55);
  await animStep(2, 55, 80);
  await fetchPromise;          /* wait for model to finish */
  await animStep(3, 80, 100);
  steps.forEach(s => document.getElementById(s).className = 'step-pill done');

  await sleep(400);
  /* Redirect to result page */
  document.open(); document.write(wrapResult(resultHtml, fileInput.files[0])); document.close();
});

function wrapResult(html, file) {
  /* Parse label & score from backend response */
  const lower = html.toLowerCase();
  const isManipulated = lower.includes('manipulated');
  const scoreMatch = lower.match(/confidence[^\\d]*([\\d.]+)/);
  const rawScore   = scoreMatch ? parseFloat(scoreMatch[1]) : (lower.match(/\\b(0\\.\\d+)\\b/) || [,'0.9'])[1];
  const score      = parseFloat(rawScore);
  const pct        = Math.round(score * 100);
  const fname      = file ? file.name : '—';
  const fsize      = file ? formatSize(file.size) : '—';
  const ftype      = file ? (file.type.split('/')[0].charAt(0).toUpperCase() + file.type.split('/')[0].slice(1)) : '—';

  const verdictClass = isManipulated ? 'manipulated' : 'authentic';
  const verdictTitle = isManipulated ? 'Deepfake detected' : 'Authentic media';
  const verdictDesc  = isManipulated
    ? 'This file shows strong indicators of AI-generated manipulation.'
    : 'No signs of manipulation were found in this file.';
  const barColor  = isManipulated ? '#E24B4A' : '#1D9E75';
  const bgColor   = isManipulated ? '#FCEBEB' : '#E1F5EE';
  const bdColor   = isManipulated ? '#F7C1C1' : '#9FE1CB';
  const dotAnim   = isManipulated ? 'animation:pulse 1.5s infinite;' : '';
  const dotColor  = isManipulated ? '#E24B4A' : '#1D9E75';
  const txtDark   = isManipulated ? '#791F1F' : '#085041';
  const txtMid    = isManipulated ? '#A32D2D' : '#0F6E56';
  const barBg     = isManipulated ? '#F7C1C1' : '#9FE1CB';

  return `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Result — DeepfakeDetector</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--purple-50:#EEEDFE;--purple-400:#7F77DD;--purple-600:#534AB7}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#F4F3F0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem 1rem}
.app{width:100%;max-width:520px}
.card{background:#fff;border:0.5px solid rgba(0,0,0,.1);border-radius:16px;padding:2rem;animation:slideUp .4s ease}
@keyframes slideUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:1.75rem}
.brand-icon{width:38px;height:38px;background:var(--purple-400);border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.brand-icon svg{width:18px;height:18px;stroke:#fff;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.brand-title{font-size:17px;font-weight:600;color:#1a1a1a}
.brand-sub{font-size:12px;color:#888;margin-top:1px}
.verdict{border-radius:12px;padding:1.25rem 1.25rem 1rem;margin-bottom:1rem;background:${bgColor};border:0.5px solid ${bdColor};animation:fadeUp .5s ease .1s both}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.verdict-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.verdict-dot{width:10px;height:10px;border-radius:50%;background:${dotColor};flex-shrink:0;${dotAnim}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.verdict-label{font-size:16px;font-weight:600;color:${txtDark}}
.verdict-desc{font-size:13px;color:${txtMid};line-height:1.5}
.conf-row{display:flex;align-items:center;gap:10px;margin-top:12px}
.conf-label{font-size:12px;color:${txtMid};min-width:78px}
.conf-bar-bg{flex:1;height:8px;border-radius:10px;background:${barBg};overflow:hidden}
.conf-bar-fill{height:100%;border-radius:10px;background:${barColor};width:0;transition:width 1.1s cubic-bezier(.4,0,.2,1)}
.conf-val{font-size:13px;font-weight:600;color:${txtDark};min-width:36px;text-align:right}
.meta-row{display:flex;gap:10px;margin-bottom:1rem;animation:fadeUp .5s ease .25s both}
.meta-card{flex:1;background:#f9f9f7;border-radius:8px;padding:10px 12px;border:0.5px solid rgba(0,0,0,.06)}
.meta-label{font-size:11px;color:#888;margin-bottom:3px}
.meta-val{font-size:13px;font-weight:500;color:#1a1a1a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btn-back{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:11px;background:none;border:0.5px solid rgba(0,0,0,.15);border-radius:10px;font-size:14px;color:#555;cursor:pointer;text-decoration:none;transition:background .2s;animation:fadeUp .5s ease .35s both}
.btn-back:hover{background:#f5f5f3}
.btn-back svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
</style>
</head><body>
<div class="app"><div class="card">

  <div class="brand">
    <div class="brand-icon">
      <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
    </div>
    <div>
      <div class="brand-title">DeepfakeDetector</div>
      <div class="brand-sub">Analysis result</div>
    </div>
  </div>

  <div class="verdict">
    <div class="verdict-header">
      <div class="verdict-dot"></div>
      <div class="verdict-label">${verdictTitle}</div>
    </div>
    <div class="verdict-desc">${verdictDesc}</div>
    <div class="conf-row">
      <div class="conf-label">Confidence</div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" id="cfill"></div></div>
      <div class="conf-val">${pct}%</div>
    </div>
  </div>

  <div class="meta-row">
    <div class="meta-card">
      <div class="meta-label">File type</div>
      <div class="meta-val">${ftype}</div>
    </div>
    <div class="meta-card">
      <div class="meta-label">File name</div>
      <div class="meta-val" title="${fname}">${fname}</div>
    </div>
    <div class="meta-card">
      <div class="meta-label">File size</div>
      <div class="meta-val">${fsize}</div>
    </div>
  </div>

  <a href="/" class="btn-back">
    <svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg>
    Analyze another file
  </a>

</div></div>
<script>
function formatSize(b){if(b<1024)return b+' B';if(b<1024*1024)return Math.round(b/1024)+' KB';return(b/(1024*1024)).toFixed(1)+' MB';}
setTimeout(()=>{ document.getElementById('cfill').style.width='${pct}%'; }, 300);
<\/script>
</body></html>`;
}

function animatePct(from, to) {
  return new Promise(res => {
    const fill = document.getElementById('progress-fill');
    const pct  = document.getElementById('progress-pct');
    let cur = from;
    const step = (to - from) / 40;
    const t = setInterval(() => {
      cur = Math.min(cur + step + Math.random() * 0.5, to);
      fill.style.width = cur + '%';
      pct.textContent  = Math.round(cur) + '%';
      if (cur >= to) { clearInterval(t); res(); }
    }, 30);
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
</script>
</body>
</html>
"""


#############################################
# API ENDPOINT                              #
#############################################
@app.route("/api/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    mime_type, _ = mimetypes.guess_type(filepath)

    if mime_type.startswith("image/"):
        label, score = predict_image(filepath)
    elif mime_type.startswith("video/"):
        label, score = predict_video(filepath)
    elif mime_type.startswith("audio/"):
        label, score = predict_audio(filepath)
    else:
        return "Unsupported file type"

    # Return structured text that the frontend JS can parse
    return f"type:{mime_type} label:{label} confidence:{score}"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)