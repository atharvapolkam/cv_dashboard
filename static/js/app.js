/* CV Dashboard frontend.
 * Live-preview architecture: every op declares a `cost` in the backend registry.
 *   cheap -> 40ms debounce, full res      medium -> 120ms
 *   expensive -> 260ms + backend renders a downscaled preview pass
 * Each request carries a generation id; stale responses are dropped and in-flight
 * requests are aborted, so dragging a slider never queues 100 round-trips.
 */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const S = {
  sid: localStorage.getItem('cvd.sid') || crypto.randomUUID().replace(/-/g, ''),
  ops: {}, quick: [], cats: {}, pipes: [], presets: {},
  op: null, params: {}, imgId: null, srcId: null, info: null,
  tool: 'none', pts: [], roi: null, gen: 0, ctl: null,
  favs: JSON.parse(localStorage.getItem('cvd.favs') || '[]'),
  recents: JSON.parse(localStorage.getItem('cvd.recents') || '[]'),
};
localStorage.setItem('cvd.sid', S.sid);
const DEBOUNCE = { cheap: 40, medium: 120, expensive: 260 };

const toast = (m, k = '') => { const d = document.createElement('div'); d.className = 'toast ' + k;
  d.textContent = m; $('#toasts').append(d); setTimeout(() => d.remove(), 4200); };
const api = async (url, opt = {}) => { const r = await fetch(url, opt);
  const j = r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(j.hint ? `${j.message} — ${j.hint}` : (j.message || r.statusText));
  return j; };
const post = (u, b, sig) => api(u, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: S.sid, ...b }), signal: sig });
const imgUrl = (id, max = 0) => `/api/img/${S.sid}/${id}${max ? '?max=' + max : ''}`;
const kv = (el, obj) => { el.innerHTML = Object.entries(obj || {}).map(([k, v]) =>
  `<span class="k">${k}</span><span class="v">${v ?? '—'}</span>`).join('') ||
  '<span class="empty-hint">—</span>'; };
const copy = t => { navigator.clipboard.writeText(t); toast('Copied', 'ok'); };

/* ------------------------------------------------------------------ boot */
(async function boot() {
  const r = await api('/api/ops');
  r.ops.forEach(o => S.ops[o.id] = o);
  S.quick = r.quick; S.cats = r.categories; S.pipes = r.pipelines; S.presets = r.hsv_presets;
  renderQuick(); renderCats(); renderPipes(); renderChips();
  try { const st = await api(`/api/state/${S.sid}`); if (!st.empty) adoptState(st); } catch (e) {}
})();

/* --------------------------------------------------------------- upload */
$('#file').onchange = e => e.target.files[0] && upload(e.target.files[0]);
const stage = $('#stage');
['dragover', 'dragenter'].forEach(t => stage.addEventListener(t, e => {
  e.preventDefault(); stage.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(t => stage.addEventListener(t, e => {
  e.preventDefault(); stage.classList.remove('dragover'); }));
stage.addEventListener('drop', e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

async function upload(f) {
  const fd = new FormData(); fd.append('file', f);
  const t0 = performance.now();
  try {
    const r = await api(`/api/upload?session_id=${S.sid}`, { method: 'POST', body: fd });
    S.srcId = r.image_id; setImage(r.image_id, r.info);
    S.uploadMs = Math.round(performance.now() - t0); S.uploadBytes = r.upload_bytes;
    S.pts = []; S.roi = null; drawOverlay(); refreshHistory(); refreshCode();
    kv($('#what'), { Operation: 'upload', Source: r.filename, Size: `${r.info.width} × ${r.info.height}` });
    toast(`Loaded ${r.info.width}×${r.info.height}`, 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

function adoptState(st) {
  S.srcId = st.history[0].image_id; setImage(st.image_id, st.info);
  refreshHistory(); refreshCode();
}

function setImage(id, info) {
  S.imgId = id; S.info = info;
  const v = $('#view'); v.src = imgUrl(id); v.classList.add('on'); stage.classList.add('has');
  v.onload = () => { sizeOverlay(); drawOverlay(); };
  kv($('#inspect'), {
    Image: `${info.width} × ${info.height}`, Channels: info.channels, dtype: info.dtype,
    'Aspect ratio': info.aspect_ratio, Megapixels: info.megapixels,
    Min: info.min, Max: info.max, Mean: info.mean, 'Std dev': info.std,
    'Array bytes': info.nbytes.toLocaleString(), shape: `(${info.shape.join(', ')})`,
  });
}

/* -------------------------------------------------------- op list / search */
function opBtn(o, cls = '') {
  const b = document.createElement('button'); b.className = cls;
  b.innerHTML = `<span class="tierdot"></span><span>${o.label}</span>`;
  b.title = `${o.cv} — ${o.summary}`; b.onclick = () => selectOp(o.id); return b;
}
function renderQuick() {
  $('#quick').innerHTML = '';
  S.quick.forEach(id => { const o = S.ops[id]; if (!o) return;
    const b = document.createElement('button'); b.textContent = o.label;
    b.title = `${o.cv} — ${o.summary}`; b.onclick = () => selectOp(id); $('#quick').append(b); });
}
function renderCats() {
  const box = $('#cats'); box.innerHTML = '';
  Object.entries(S.cats).forEach(([cat, ids]) => {
    const d = document.createElement('details');
    d.innerHTML = `<summary>${cat} <span class="meta">${ids.length}</span></summary>`;
    const st = document.createElement('div'); st.className = 'stack';
    ids.map(i => S.ops[i]).sort((a, b) => a.tier - b.tier)
      .forEach(o => st.append(opBtn(o, 'tier' + o.tier)));
    d.append(st); box.append(d);
  });
}
function renderPipes() {
  $('#pipelines').innerHTML = '';
  S.pipes.forEach(p => { const b = document.createElement('button'); b.className = 'btn sm';
    b.style.textAlign = 'left'; b.innerHTML = `<b>${p.label}</b><br><span class="meta">${p.desc}</span>`;
    b.onclick = () => runPipeline(p.id); $('#pipelines').append(b); });
}
function renderChips() {
  const mk = (host, ids, empty) => { host.innerHTML = ''; if (!ids.length) {
      host.innerHTML = `<span class="empty-hint">${empty}</span>`; return; }
    ids.forEach(id => { const o = S.ops[id]; if (!o) return;
      const b = document.createElement('button'); b.textContent = o.label;
      b.onclick = () => selectOp(id); host.append(b); }); };
  mk($('#favs'), S.favs, 'Click ☆ on any operation.');
  mk($('#recents'), S.recents, 'Nothing yet.');
}
const search = $('#search'), results = $('#results');
search.oninput = () => {
  const q = search.value.trim().toLowerCase();
  if (!q) return results.classList.remove('on');
  const hits = Object.values(S.ops).filter(o => o.search.includes(q))
    .sort((a, b) => a.tier - b.tier).slice(0, 24);
  results.innerHTML = hits.map(o => `<div class="hit" data-id="${o.id}"><code>${o.cv}</code>
    <b>${o.label}</b><span class="meta">${o.category}</span></div>`).join('') ||
    '<div class="hit"><span class="meta">no match</span></div>';
  results.classList.add('on');
  $$('.hit', results).forEach(h => h.onclick = () => { selectOp(h.dataset.id);
    results.classList.remove('on'); search.value = ''; });
};
document.addEventListener('click', e => { if (!e.target.closest('.srch')) results.classList.remove('on'); });

/* ------------------------------------------------------------ param form */
function selectOp(id) {
  const o = S.ops[id]; if (!o) return;
  S.op = o; S.params = {};
  o.params.forEach(p => S.params[p.name] = structuredClone(p.default));
  if (o.id === 'crop_rect' && S.info) Object.assign(S.params,
    { x1: 0, y1: 0, x2: S.info.width, y2: S.info.height });
  $('#paramCard').hidden = false;
  $('#opTitle').textContent = o.label; $('#opCv').textContent = o.cv;
  $('#opCost').textContent = o.cost; $('#opCost').className = 'tag ' + o.cost;
  $('#opSummary').textContent = o.summary;
  $('#favBtn').textContent = S.favs.includes(id) ? '★' : '☆';
  $('#cmpWrap').hidden = true;
  buildForm(); preview();
}
function buildForm() {
  const host = $('#params'); host.innerHTML = ''; let grp = null;
  S.op.params.forEach(p => {
    if (p.visible_when && !Object.entries(p.visible_when)
      .every(([k, vals]) => vals.map(String).includes(String(S.params[k])))) return;
    if (p.group && p.group !== grp) { grp = p.group;
      const g = document.createElement('div'); g.className = 'grp'; g.textContent = grp; host.append(g); }
    host.append(widget(p));
  });
  if (S.op.inputs === 2) { const d = document.createElement('div'); d.className = 'p';
    d.innerHTML = `<label>Second image</label><span class="meta">Uses the source frame.</span>`;
    host.append(d); }
  if (S.op.id === 'in_range') host.append(presetRow());
}
function presetRow() {
  const d = document.createElement('div'); d.className = 'p'; d.style.gridColumn = '1/-1';
  d.innerHTML = '<label>HSV presets</label>';
  const c = document.createElement('div'); c.className = 'chips';
  Object.entries(S.presets).forEach(([k, v]) => { const b = document.createElement('button');
    b.textContent = v.label; b.onclick = () => {
      const [l0, l1, l2] = v.lower, [u0, u1, u2] = v.upper;
      Object.assign(S.params, { space: 'hsv', l0, l1, l2, u0, u1, u2, wrap_hue: l0 > u0 });
      buildForm(); preview(); }; c.append(b); });
  d.append(c); return d;
}
function widget(p) {
  const d = document.createElement('div'); d.className = 'p';
  const val = S.params[p.name];
  const set = (v, live = true) => { S.params[p.name] = v;
    if (p.visible_when || S.op.params.some(q => q.visible_when && p.name in q.visible_when)) buildForm();
    if (live) preview(); };
  const lab = (extra = '') => `<label>${p.label}${p.help ? ` <span title="${p.help}">ⓘ</span>` : ''}
    <b>${extra}</b></label>`;
  if (p.type === 'int' || p.type === 'float') {
    let max = p.max;
    if (p.bind_dim && S.info) max = p.bind_dim === 'width' ? S.info.width : S.info.height;
    d.innerHTML = lab(val);
    const wrap = document.createElement('div'); wrap.className = 'pair';
    const r = document.createElement('input'); r.type = 'range';
    r.min = p.min ?? 0; r.max = max ?? 255; r.step = p.step || (p.type === 'int' ? 1 : 0.01); r.value = val;
    const n = document.createElement('input'); n.type = 'number'; n.value = val;
    n.style.maxWidth = '70px'; n.step = r.step;
    r.oninput = () => { n.value = r.value; $('b', d).textContent = r.value;
      set(p.type === 'int' ? +r.value : parseFloat(r.value)); };
    n.onchange = () => { r.value = n.value; $('b', d).textContent = n.value;
      set(p.type === 'int' ? +n.value : parseFloat(n.value)); };
    wrap.append(r, n); d.append(wrap);
  } else if (p.type === 'bool') {
    d.innerHTML = `<label class="chk"><input type="checkbox" ${val ? 'checked' : ''}> ${p.label}</label>`;
    $('input', d).onchange = e => set(e.target.checked);
  } else if (p.type === 'enum') {
    d.innerHTML = lab();
    const s = document.createElement('select');
    p.choices.forEach(c => { const o = document.createElement('option'); o.value = c.value;
      o.textContent = c.label; o.title = c.help || ''; if (c.value === String(val)) o.selected = true;
      s.append(o); });
    s.onchange = () => set(s.value); d.append(s);
  } else if (p.type === 'kernel') {
    d.innerHTML = lab(`${val[0]}×${val[1]}`);
    const wrap = document.createElement('div'); wrap.className = 'pair';
    [0, 1].forEach(i => { const n = document.createElement('input'); n.type = 'number';
      n.min = p.min ?? 1; n.max = p.max ?? 99; n.step = p.odd ? 2 : 1; n.value = val[i];
      n.oninput = () => { const v = [...S.params[p.name]]; v[i] = +n.value;
        $('b', d).textContent = `${v[0]}×${v[1]}`; set(v); }; wrap.append(n); });
    d.append(wrap);
  } else if (p.type === 'color') {
    const hex = '#' + [val[2], val[1], val[0]].map(c => (+c).toString(16).padStart(2, '0')).join('');
    d.innerHTML = lab(); const c = document.createElement('input'); c.type = 'color'; c.value = hex;
    c.onchange = () => set(c.value); d.append(c);
  } else {
    d.innerHTML = lab(); const t = document.createElement('input'); t.type = 'text';
    t.value = Array.isArray(val) ? JSON.stringify(val) : (val ?? '');
    t.onchange = () => { try { set(JSON.parse(t.value)); } catch { set(t.value); } }; d.append(t);
  }
  return d;
}

/* ------------------------------------------------- preview (debounced) */
let timer = null;
function preview() {
  if (!S.op || !S.imgId) return;
  clearTimeout(timer);
  timer = setTimeout(() => run(true), DEBOUNCE[S.op.cost] ?? 120);
}
async function run(isPreview) {
  if (!S.op || !S.imgId) return toast('Upload an image first', 'err');
  const gen = ++S.gen;
  S.ctl?.abort(); S.ctl = new AbortController();
  try {
    const r = await post('/api/execute', {
      op_id: S.op.id, params: S.params, preview: isPreview,
      image_id: isPreview ? S.imgId : undefined,
      image2_id: S.op.inputs === 2 ? S.srcId : undefined,
    }, S.ctl.signal);
    if (gen !== S.gen) return;                       // stale response — drop
    showResult(r, isPreview);
  } catch (e) { if (e.name !== 'AbortError') toast(e.message, 'err'); }
}
function showResult(r, isPreview) {
  const v = $('#view'); v.src = imgUrl(r.image_id); v.classList.add('on');
  if (!isPreview) { S.imgId = r.image_id; S.info = r.output; setImage(r.image_id, r.output);
    pushRecent(S.op.id); refreshHistory(); refreshCode(); }
  kv($('#what'), { Operation: r.op.cv, Input: `${r.input.width} × ${r.input.height} × ${r.input.channels}`,
    Output: `${r.output.width} × ${r.output.height} × ${r.output.channels}`,
    Execution: `${r.metrics.process} ms`, ...r.summary,
    Mode: isPreview ? 'preview (not committed)' : 'applied' });
  $('#notes').innerHTML = (r.notes || []).map(n => `<div class="notes">⚠ ${n}</div>`).join('');
  kv($('#perf'), { Upload: (S.uploadMs ?? '—') + ' ms', Process: r.metrics.process + ' ms',
    Load: r.metrics.load + ' ms', Store: r.metrics.store + ' ms', Total: r.metrics.total + ' ms',
    'Preview scale': r.scale, 'Input px': r.metrics.input_px?.toLocaleString(),
    'Upload bytes': (S.uploadBytes ?? 0).toLocaleString() });
  $('#code').textContent = r.code;
  const ex = $('#extras'); ex.innerHTML = '';
  Object.entries(r.extras || {}).forEach(([n, id]) => { const f = document.createElement('figure');
    f.innerHTML = `<img src="${imgUrl(id, 300)}" title="${n}"><figcaption>${n}</figcaption>`;
    $('img', f).onclick = () => { $('#view').src = imgUrl(id); }; ex.append(f); });
  if (r.data?.contours) renderContours(r.data.contours);
}
function renderContours() {/* contour table hook — data arrives in r.data */}

$('#applyBtn').onclick = () => run(false);
$('#favBtn').onclick = () => { const id = S.op.id;
  S.favs = S.favs.includes(id) ? S.favs.filter(x => x !== id) : [...S.favs, id];
  localStorage.setItem('cvd.favs', JSON.stringify(S.favs));
  $('#favBtn').textContent = S.favs.includes(id) ? '★' : '☆'; renderChips(); };
function pushRecent(id) { S.recents = [id, ...S.recents.filter(x => x !== id)].slice(0, 12);
  localStorage.setItem('cvd.recents', JSON.stringify(S.recents)); renderChips(); }

/* --------------------------------------------------------- A/B compare */
$('#cmpBtn').onclick = async () => {
  const a = structuredClone(S.params); const b = structuredClone(S.params);
  const k = S.op.params.find(p => p.type === 'kernel');
  const num = S.op.params.find(p => p.type === 'int' || p.type === 'float');
  if (k) { b[k.name] = [Math.min(99, a[k.name][0] * 2 + 1), Math.min(99, a[k.name][1] * 2 + 1)]; }
  else if (num) b[num.name] = Math.round((a[num.name] || 1) * 2);
  try {
    const r = await post('/api/compare', { op_id: S.op.id, params_a: a, params_b: b });
    const w = $('#cmpWrap'); w.hidden = false;
    w.innerHTML = `<figure><img src="${imgUrl(r.a.image_id, 400)}"><figcaption>A ${JSON.stringify(a).slice(0, 60)}</figcaption></figure>
      <figure><img src="${imgUrl(r.b.image_id, 400)}"><figcaption>B ${JSON.stringify(b).slice(0, 60)}</figcaption></figure>
      ${r.diff_image_id ? `<figure><img src="${imgUrl(r.diff_image_id, 400)}"><figcaption>Diff — ${r.diff.changed_px?.toLocaleString()} px changed, max Δ ${r.diff.max_delta}</figcaption></figure>` : ''}`;
  } catch (e) { toast(e.message, 'err'); }
};

/* --------------------------------------------------------- pipelines */
async function runPipeline(id) {
  try { const r = await post(`/api/pipeline/${id}`, {});
    const last = r.steps[r.steps.length - 1];
    S.op = S.ops[last.op.id]; showResult(last, false);
    toast(`${r.pipeline.label}: ${r.steps.length} steps applied`, 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

/* -------------------------------------------------------- history/code */
async function refreshHistory() {
  const st = await api(`/api/state/${S.sid}`); if (st.empty) return;
  $('#history').innerHTML = st.history.map(h =>
    `<li class="${h.i === st.cursor ? 'now' : ''}" data-i="${h.i}">${h.i}. ${h.label}
      <span class="spacer"></span><span class="meta">${h.metrics?.process ?? ''}</span></li>`).join('');
}
async function refreshCode() { $('#code').textContent = await api(`/api/code/${S.sid}`); }
const hist = async a => { try { const r = await post(`/api/history/${a}`, {});
  setImage(r.image_id, r.info); refreshHistory(); refreshCode();
  toast(`${a} → ${r.label}`); } catch (e) { toast(e.message, 'err'); } };
$('#undo').onclick = () => hist('undo'); $('#redo').onclick = () => hist('redo');
$('#reset').onclick = () => hist('reset');
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'z') { e.preventDefault(); hist('undo'); }
  if (e.ctrlKey && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) { e.preventDefault(); hist('redo'); }
  if (e.key === 'Enter' && S.op && !e.target.matches('input,select')) run(false);
});
$('#copyCode').onclick = () => copy($('#code').textContent);
$('#dlCode').onclick = () => dl(`/api/code/${S.sid}`, 'pipeline.py');
$('#dlImg').onclick = () => dl(imgUrl(S.imgId), 'output.webp');
$('#dlJson').onclick = () => { const blob = new Blob([JSON.stringify(
  { image: S.info, roi: S.roi, params: S.params, op: S.op?.id }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'coordinates.json'; a.click(); };
const dl = (url, name) => { const a = document.createElement('a'); a.href = url; a.download = name; a.click(); };
$$('.fold').forEach(f => f.onclick = () => f.parentElement.classList.toggle('closed'));
$('#showOrig').onchange = e => { $('#view').src = imgUrl(e.target.checked ? S.srcId : S.imgId); };

/* ------------------------------------------------------------ ROI canvas */
const cv = $('#overlay'), ctx = cv.getContext('2d');
function sizeOverlay() { const v = $('#view'), r = v.getBoundingClientRect(),
  s = stage.getBoundingClientRect();
  cv.width = r.width; cv.height = r.height; cv.style.left = (r.left - s.left) + 'px';
  cv.style.top = (r.top - s.top) + 'px'; cv.style.width = r.width + 'px'; cv.style.height = r.height + 'px'; }
addEventListener('resize', () => { sizeOverlay(); drawOverlay(); });
const toImg = e => { const r = cv.getBoundingClientRect();
  return [Math.round((e.clientX - r.left) / r.width * S.info.width),
          Math.round((e.clientY - r.top) / r.height * S.info.height)]; };
const toCv = ([x, y]) => [x / S.info.width * cv.width, y / S.info.height * cv.height];

$$('#roitools button').forEach(b => b.onclick = () => {
  $$('#roitools button').forEach(x => x.classList.remove('on')); b.classList.add('on');
  S.tool = b.dataset.tool; S.pts = []; drawOverlay(); });

let dragging = false;
cv.addEventListener('mousemove', e => { if (!S.info) return;
  const [x, y] = toImg(e);
  $('#cursorPos').textContent = `x ${x}  y ${y}   ·   norm ${(x / S.info.width).toFixed(4)}, ${(y / S.info.height).toFixed(4)}`;
  if (dragging && ['rect', 'line', 'circle'].includes(S.tool)) { S.pts[1] = [x, y]; drawOverlay(); } });
cv.addEventListener('mousedown', e => { if (!S.info || S.tool === 'none') return;
  const p = toImg(e);
  if (S.tool === 'pick') return pickPixel(p);
  if (S.tool === 'poly') { S.pts.push(p); drawOverlay(); return; }
  S.pts = [p, p]; dragging = true; });
cv.addEventListener('mouseup', () => { if (!dragging) return; dragging = false;
  if (['rect', 'line', 'circle'].includes(S.tool)) analyzeRoi(S.tool); });
cv.addEventListener('dblclick', () => S.tool === 'poly' && analyzeRoi('polygon'));
$('#finishPoly').onclick = () => analyzeRoi('polygon');
$('#clearRoi').onclick = () => { S.pts = []; S.roi = null; drawOverlay();
  $('#roiInfo').innerHTML = '<span class="empty-hint">Draw a rect, polygon, line or circle.</span>';
  $('#roiCode').textContent = ''; };

function drawOverlay() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.info || !S.pts.length) return;
  ctx.lineWidth = 2; ctx.strokeStyle = '#4da3ff'; ctx.fillStyle = 'rgba(77,163,255,.16)';
  const P = S.pts.map(toCv);
  if (S.tool === 'rect' && P.length > 1) {
    const [a, b] = P; ctx.beginPath();
    ctx.rect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]));
    ctx.fill(); ctx.stroke();
  } else if (S.tool === 'line' && P.length > 1) {
    ctx.beginPath(); ctx.moveTo(...P[0]); ctx.lineTo(...P[1]); ctx.stroke();
  } else if (S.tool === 'circle' && P.length > 1) {
    const r = Math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]);
    ctx.beginPath(); ctx.arc(P[0][0], P[0][1], r, 0, 7); ctx.fill(); ctx.stroke();
  } else {
    ctx.beginPath(); P.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p));
    if (P.length > 2) { ctx.closePath(); ctx.fill(); } ctx.stroke();
  }
  ctx.fillStyle = '#ffb454';
  P.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 3.5, 0, 7); ctx.fill(); });
}

async function analyzeRoi(kind) {
  if (!S.info || S.pts.length < 2) return;
  try {
    const r = await post('/api/roi/analyze', { width: S.info.width, height: S.info.height,
      kind, points: S.pts });
    S.roi = r;
    const o = r.rect || r.line || r.circle || r.polygon || {};
    const flat = {};
    Object.entries(o).forEach(([k, v]) => flat[k] = typeof v === 'object' ? JSON.stringify(v) : v);
    if (r.rect_norm) Object.entries(r.rect_norm).forEach(([k, v]) => flat['norm ' + k] = v);
    if (r.line_norm) { flat['norm start'] = JSON.stringify(r.line_norm.start);
      flat['norm end'] = JSON.stringify(r.line_norm.end); }
    if (kind === 'polygon') flat['normalised'] = r.normalized.length + ' pts';
    kv($('#roiInfo'), flat);
    $('#roiCode').textContent = Object.values(r.code || {})[0] || '';
  } catch (e) { toast(e.message, 'err'); }
}
$$('#roiActions [data-copy]').forEach(b => b.onclick = () => {
  const c = S.roi?.code?.[b.dataset.copy] ?? S.roi?.code?.draw;
  c ? copy(c) : toast('Draw an ROI first', 'err'); });
$('#roiCrop').onclick = () => {
  const r = S.roi?.rect || S.roi?.polygon?.bounding_rect; if (!r) return toast('Draw a rect first', 'err');
  selectOp('crop_rect');
  Object.assign(S.params, { x1: r.x, y1: r.y, x2: r.x2 ?? r.x + r.w, y2: r.y2 ?? r.y + r.h });
  buildForm(); run(false); };
$('#roiMask').onclick = async () => {
  if (S.pts.length < 3 && S.tool === 'poly') return toast('Need 3+ points', 'err');
  try { const r = await post('/api/roi/mask', { kind: S.tool === 'poly' ? 'polygon' : S.tool,
      points: S.pts, image_id: S.imgId, commit: true });
    setImage(r.masked_image_id, S.info);
    const ex = $('#extras'); ex.innerHTML = '';
    [['ROI mask', r.mask_image_id], ['Masked image', r.masked_image_id]].forEach(([n, id]) => {
      const f = document.createElement('figure');
      f.innerHTML = `<img src="${imgUrl(id, 300)}"><figcaption>${n}</figcaption>`; ex.append(f); });
    kv($('#what'), { Operation: 'ROI mask (fillPoly + bitwise_and)', Kind: S.tool,
      Points: S.pts.length, 'ROI coverage': r.coverage_pct + ' %' });
    refreshHistory(); refreshCode(); toast(`Mask applied — ${r.coverage_pct}% kept`, 'ok');
  } catch (e) { toast(e.message, 'err'); } };

/* ------------------------------------------------------- pixel inspector */
async function pickPixel([x, y]) {
  try { const r = await api(`/api/pixel/${S.sid}/${S.imgId}?x=${x}&y=${y}&n=5`);
    kv($('#pixel'), { X: r.x, Y: r.y, 'norm X': r.norm_x, 'norm Y': r.norm_y,
      B: r.b, G: r.g, R: r.r, RGB: r.rgb ? `(${r.rgb.join(', ')})` : undefined,
      HEX: r.hex, HSV: r.hsv ? `(${r.hsv.join(', ')})` : undefined,
      LAB: r.lab ? `(${r.lab.join(', ')})` : undefined,
      GRAY: r.gray, Intensity: r.intensity });
    const n = r.neighbourhood, g = $('#patch');
    g.style.gridTemplateColumns = `repeat(${n.values[0].length},1fr)`;
    g.innerHTML = n.values.map((row, i) => row.map((v, j) =>
      `<div style="background:${n.colors ? n.colors[i][j] : `rgb(${v},${v},${v})`}">${v}</div>`)
      .join('')).join('');
  } catch (e) { toast(e.message, 'err'); }
}

/* ---------------------------------------------------- coord converter */
$('#toNorm').onclick = async () => { const r = await post('/api/coords', {
    width: S.info?.width || 1920, height: S.info?.height || 1080,
    x: +$('#ccx').value, y: +$('#ccy').value, direction: 'to_norm' });
  $('#ccOut').textContent = `Pixel:\n${r.pixel.join(', ')}\n\nNormalized:\n${r.normalized.join(', ')}`;
  $('#ccnx').value = r.normalized[0]; $('#ccny').value = r.normalized[1]; };
$('#toPix').onclick = async () => { const r = await post('/api/coords', {
    width: S.info?.width || 1920, height: S.info?.height || 1080,
    nx: +$('#ccnx').value, ny: +$('#ccny').value, direction: 'to_pixel' });
  $('#ccOut').textContent = `Normalized:\n${r.normalized.join(', ')}\n\nPixel:\n${r.pixel.join(', ')}`;
  $('#ccx').value = r.pixel[0]; $('#ccy').value = r.pixel[1]; };
$('#ccOut').onclick = () => copy($('#ccOut').textContent);
