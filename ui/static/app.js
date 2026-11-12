// Cloud Map Generator local UI: one ES module, no framework, no build step.
// Talks to the JSON API served next to this page (see ui/README or the server module).
// Every storage access is wrapped: the page must work with localStorage blocked.

const STORE_KEY = 'cloud-map-generator.ui.v1';
const THEME_KEY = 'cloud-map-generator.theme';
const REFERENCE_URL = '/api/reference/coverage.png';
const DEG = Math.PI / 180;

const TABS = {
  coverage: { kind: 'coverage', caption: 'White is cloudy, black is clear.' },
  type: { kind: 'type', caption: 'White is convective, black is stratiform.' },
  high_share: { kind: 'high_share', caption: 'Share of high cloud. Black is none, white is all; clear sky has no value.' },
  wind: { kind: 'wind', caption: 'Red channel is eastward wind, green is northward, 128 is calm.' },
  earth: { kind: 'coverage', caption: 'Blue Marble cloud fraction, the reference the metrics compare against.' }
};

const FIELD_KEYS = ['rotation_period_days', 'radius_km', 'gravity_mps2', 'scale_height_km'];
const OPTION_KEYS = ['match_earth', 'isotropic', 'surface', 'experiment'];
const DEFAULT_FORM = {
  preset: 'earth',
  cloud_regime: 'water',
  fields: { rotation_period_days: '0.99727', radius_km: '6371', gravity_mps2: '9.81', scale_height_km: '8', seed: '42' },
  width: 1024,
  compute: 'auto',
  options: { match_earth: false, isotropic: false, surface: false, experiment: false }
};
// Icon of each compute mode key the contract defines. A key the page does not know gets no icon.
const COMPUTE_ICONS = { auto: 'wand-sparkles', cpu: 'cpu', gpu: 'zap', hybrid: 'layers' };
const GLOBE_HOME = { lon: 0, lat: 15 };
const SPHERE_BG = [34, 46, 64];
// The globe's ground under the clouds, by preset key (the page's presets and the service's): a hint of the body,
// kept dark enough that white cloud still reads on it. A custom body or a key not listed keeps SPHERE_BG.
const GROUND_BY_PRESET = {
  earth: [22, 50, 86],
  'earth-retrograde': [22, 50, 86],
  fast: [22, 50, 86],
  'fast-rotator': [22, 50, 86],
  'slow-rotator': [22, 50, 86],
  'super-earth': [24, 62, 78],
  mars: [142, 60, 32],
  titan: [160, 90, 26],
  venus: [172, 146, 88],
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function storeRead(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storeWrite(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage blocked: nothing to remember */ }
}

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function icon(name, cls = 'size-4') {
  return `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

const els = {
  form: $('#controls'), preset: $('#preset'), presetNote: $('#presetNote'), regime: $('#regime'),
  seed: $('#f-seed'), randomSeed: $('#randomSeed'), widths: $('#widths'), widthHint: $('#widthHint'),
  computeRow: $('#computeRow'), computeTip: $('#computeTip'), compute: $('#compute'),
  computeHint: $('#computeHint'), computeNote: $('#computeNote'),
  generate: $('#generate'), generateIcon: $('#generateIcon'), generateLabel: $('#generateLabel'),
  blockReason: $('#blockReason'), blockText: $('#blockText'), status: $('#status'),
  version: $('#version'), theme: $('#theme'),
  stats: $('#stats'), hadley: $('#st-hadley'), storm: $('#st-storm'), cyclones: $('#st-cyclones'),
  cyclonesDesc: $('#st-cyclones-desc'), texel: $('#st-texel'), texelDesc: $('#st-texel-desc'), warnings: $('#warnings'),
  tabs: $('#tabs'), downloadBox: $('#download'), downloadButton: $('#downloadButton'), downloadMenu: $('#downloadMenu'),
  panel: $('#panel'), empty: $('#empty'), emptyTitle: $('#emptyTitle'), emptyText: $('#emptyText'), emptyAction: $('#emptyAction'),
  stage: $('#stage'), mapImg: $('#mapImg'), globe: $('#globe'), globeGpu: $('#globeGpu'), loadingBox: $('#loadingBox'),
  loadingText: $('#loadingText'), loadingHint: $('#loadingHint'),
  readoutRow: $('#readoutRow'), readout: $('#readout'), caption: $('#caption'),
  details: $('#details'), metricsSub: $('#metricsSub'), metricsNote: $('#metricsNote'), metricsNoteText: $('#metricsNoteText'),
  runWarnings: $('#runWarnings'), metricsBody: $('#metricsBody'),
  zonal: $('#zonal'), zonalReadout: $('#zonalReadout'),
  flowList: $('#flowList'), vortexBody: $('#vortexBody'), vortexNone: $('#vortexNone'), toasts: $('#toasts')
};
const fieldInputs = Object.fromEntries($$('[data-field]').map((el) => [el.dataset.field, el]));
const optionInputs = Object.fromEntries($$('[data-option]').map((el) => [el.dataset.option, el]));

const state = {
  defaults: null,
  loadError: null,
  form: structuredClone(DEFAULT_FORM),
  tab: 'coverage',
  view: 'map',
  derive: null,
  deriveBusy: false,
  running: false,
  startedAt: 0,
  result: null,
  resultPreset: '', // the preset key the result was made with (the globe's ground colour)
  product: null,
  stageError: null,
  emptyAction: 'earth',
  globe: { ...GLOBE_HOME },
  // Compute selector: an unavailable mode the user tapped to read its reason, and the one-time
  // sentence saying a saved mode is gone. Both clear on the next interaction with the control.
  computeInspect: null,
  computeFallback: ''
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, body, signal) {
  let res;
  try {
    res = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal
    });
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError(0, 'Cannot reach the server. Check that it is still running.');
  }
  let data = null;
  try { data = await res.json(); } catch { /* an empty or non-JSON body */ }
  if (!res.ok) {
    const fallback = { 400: 'The server refused the input.', 429: 'A generation is already running.', 500: 'The server failed.' };
    throw new ApiError(res.status, (data && data.error) || fallback[res.status] || `Request failed (HTTP ${res.status}).`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Toasts and status
// ---------------------------------------------------------------------------

function sentence(text) {
  const t = String(text).trim();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

// kind: 'error' (red) or 'warning' (amber, for a retry-later condition such as a busy server).
function toast(message, kind = 'error') {
  const el = document.createElement('div');
  el.className = kind === 'warning' ? 'alert alert-warning' : 'alert alert-error';
  el.setAttribute('role', 'alert');
  el.innerHTML = `${icon(kind === 'warning' ? 'triangle-alert' : 'circle-alert', 'size-5')}<span class="text-sm"></span>
    <button type="button" class="btn btn-ghost btn-xs btn-square" aria-label="Dismiss">${icon('x')}</button>`;
  $('span', el).textContent = sentence(message);
  const dismiss = () => el.remove();
  $('button', el).addEventListener('click', dismiss);
  els.toasts.appendChild(el);
  setTimeout(dismiss, 8000);
}

// The status block under the map: one sentence, then (after a run) a badge per device that worked
// and the engine's note. `compute` is the "compute" object of a generate result, absent on an
// older backend.
function setStatus(text, compute) {
  const box = els.status;
  box.textContent = '';
  if (!text) return;
  const line = document.createElement('p');
  line.textContent = text;
  box.append(line);
  const devices = computeDevices(compute);
  if (devices.length > 1) {
    const row = document.createElement('div');
    row.className = 'flex flex-wrap items-center gap-1.5';
    row.setAttribute('aria-label', 'Share of the work by device');
    row.innerHTML = [...devices].sort((a, b) => (b.share || 0) - (a.share || 0)).map((d) => deviceBadge(d)).join('');
    box.append(row);
  }
  if (compute && typeof compute.note === 'string' && compute.note.trim()) {
    const note = document.createElement('p');
    note.className = 'flex items-start gap-1.5';
    fillInfo(note, sentence(compute.note));
    box.append(note);
  }
}

// A small info line: the icon, then the sentence. An empty sentence empties the element (it hides on :empty).
function fillInfo(el, text) {
  el.innerHTML = text ? `${icon('info', 'mt-0.5 size-4 shrink-0')}<span></span>` : '';
  if (text) $('span', el).textContent = text;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function decimalsOf(format) {
  const m = /\.(0+)/.exec(format || '');
  return m ? m[1].length : 0;
}

// The server defines each row's format; this reads the pattern ("0.000", "0.0%", "0.00e0").
function fmtValue(v, format) {
  if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '—';
  if (typeof v !== 'number') return String(v);
  const f = format || '';
  const d = decimalsOf(f);
  if (f.includes('%')) return `${(v * 100).toFixed(d)} %`;
  if (/e/i.test(f)) return v.toExponential(d);
  return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}

function relGap(generated, earth) {
  if (typeof generated !== 'number' || typeof earth !== 'number' || !Number.isFinite(generated) || !Number.isFinite(earth)) return '—';
  if (earth === 0) return generated === 0 ? '0 %' : '—';
  const pct = ((generated - earth) / Math.abs(earth)) * 100;
  const digits = Math.abs(pct) < 10 ? 1 : 0;
  return `${pct > 0 ? '+' : ''}${pct.toFixed(digits)} %`;
}

function fmtLatLon(lat, lon) {
  return `${Math.abs(lat).toFixed(1)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(1)}° ${lon >= 0 ? 'E' : 'W'}`;
}

// ---------------------------------------------------------------------------
// Form: fields, validation, persistence
// ---------------------------------------------------------------------------

function limitsOf(key) {
  return (state.defaults && state.defaults.limits[key]) || [-Infinity, Infinity];
}

function fieldError(key) {
  const input = fieldInputs[key];
  const name = input.dataset.name;
  const raw = input.value.trim();
  if (raw === '' || input.validity.badInput) return `${name} must be a number.`;
  const n = Number(raw);
  if (!Number.isFinite(n)) return `${name} must be a number.`;
  if (input.dataset.integer && !Number.isInteger(n)) return `${name} must be a whole number.`;
  const [lo, hi] = limitsOf(key);
  if (n < lo || n > hi) return `${name} must be between ${lo} and ${hi}.`;
  return null;
}

function showFieldError(key, message) {
  const input = fieldInputs[key];
  const hint = $(`[data-hint-for="${key}"]`);
  input.closest('.input').classList.toggle('input-error', Boolean(message));
  if (message) {
    input.setAttribute('aria-invalid', 'true');
    hint.textContent = message;
    input.setAttribute('aria-describedby', `hint-${key}`);
    hint.id = `hint-${key}`;
  } else {
    input.removeAttribute('aria-invalid');
    input.removeAttribute('aria-describedby');
    hint.textContent = '';
  }
}

function validationErrors() {
  return Object.keys(fieldInputs).map(fieldError).filter(Boolean);
}

function physicsBody() {
  if (FIELD_KEYS.some((k) => fieldError(k))) return null;
  const body = { cloud_regime: state.form.cloud_regime };
  for (const k of FIELD_KEYS) body[k] = Number(fieldInputs[k].value);
  return body;
}

function persist() {
  const fields = {};
  for (const k of Object.keys(fieldInputs)) fields[k] = fieldInputs[k].value;
  storeWrite(STORE_KEY, JSON.stringify({
    preset: state.form.preset, cloud_regime: state.form.cloud_regime, fields, width: state.form.width,
    compute: state.form.compute, options: state.form.options, tab: state.tab, view: state.view
  }));
}

function restore() {
  let saved = null;
  try { saved = JSON.parse(storeRead(STORE_KEY) || 'null'); } catch { saved = null; }
  let savedCompute = '';
  const form = structuredClone(DEFAULT_FORM);
  // The server owns the preset numbers: start from its copy of the default preset, not a hard-coded one.
  const base = presetByKey(form.preset);
  if (base) {
    form.cloud_regime = base.cloud_regime;
    for (const k of FIELD_KEYS) form.fields[k] = String(base[k]);
  }
  if (saved && typeof saved === 'object') {
    if (typeof saved.preset === 'string') form.preset = saved.preset;
    if (typeof saved.cloud_regime === 'string') form.cloud_regime = saved.cloud_regime;
    if (saved.fields && typeof saved.fields === 'object') {
      for (const k of Object.keys(form.fields)) if (typeof saved.fields[k] === 'string') form.fields[k] = saved.fields[k];
    }
    if (state.defaults.widths.includes(saved.width)) form.width = saved.width;
    // Kept as saved even without a compute block: the mode survives a visit to an older backend.
    if (typeof saved.compute === 'string' && saved.compute) {
      form.compute = saved.compute;
      savedCompute = saved.compute;
    }
    if (saved.options && typeof saved.options === 'object') {
      for (const k of OPTION_KEYS) if (typeof saved.options[k] === 'boolean') form.options[k] = saved.options[k];
    }
    if (saved.tab in TABS) state.tab = saved.tab;
    if (saved.view === 'map' || saved.view === 'globe') state.view = saved.view;
  }
  if (form.preset && !state.defaults.presets.some((p) => p.key === form.preset)) form.preset = '';
  state.form = form;
  reconcileCompute(savedCompute);
}

function presetByKey(key) {
  return state.defaults && state.defaults.presets.find((p) => p.key === key);
}

function fieldsMatchPreset(p) {
  return FIELD_KEYS.every((k) => Number(fieldInputs[k].value) === p[k]) && state.form.cloud_regime === p.cloud_regime;
}

function applyFormToDom() {
  for (const k of Object.keys(fieldInputs)) fieldInputs[k].value = state.form.fields[k];
  for (const k of OPTION_KEYS) optionInputs[k].checked = state.form.options[k];
  els.preset.value = state.form.preset;
  renderWidths();
  for (const k of Object.keys(fieldInputs)) showFieldError(k, null);
}

function applyPreset(key) {
  const p = presetByKey(key);
  if (!p) return;
  state.form.preset = key;
  state.form.cloud_regime = p.cloud_regime;
  for (const k of FIELD_KEYS) fieldInputs[k].value = String(p[k]);
  for (const k of FIELD_KEYS) showFieldError(k, null);
  afterPhysicsChange();
}

function afterPhysicsChange() {
  // The select follows the numbers: a matching preset is selected, anything else reads "Custom".
  const match = state.defaults.presets.find(fieldsMatchPreset);
  state.form.preset = match ? match.key : '';
  els.preset.value = state.form.preset;
  renderPresetMeta();
  persist();
  scheduleDerive();
  renderBlock();
}

function renderPresetMeta() {
  const p = presetByKey(state.form.preset);
  let note = 'Custom values. Pick a preset to start from a known body.';
  if (p) note = p.note + (p.experiment_required && !/experiment/i.test(p.note) ? ' Needs the Experiment option.' : '');
  els.presetNote.textContent = note;
  els.regime.textContent = state.form.cloud_regime;
}

function randomSeed() {
  const [lo, hi] = limitsOf('seed');
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

// ---------------------------------------------------------------------------
// Derived physics (POST /api/derive, debounced)
// ---------------------------------------------------------------------------

let deriveTimer = 0;
let deriveSeq = 0;
let deriveAbort = null;
let lastDeriveError = '';

function scheduleDerive() {
  clearTimeout(deriveTimer);
  if (deriveAbort) deriveAbort.abort();
  if (!physicsBody()) {
    deriveSeq++;
    state.derive = null;
    state.deriveBusy = false;
    renderDerive();
    return;
  }
  state.deriveBusy = true;
  els.stats.classList.add('opacity-60');
  deriveTimer = setTimeout(runDerive, 300);
}

async function runDerive() {
  const body = physicsBody();
  if (!body) return;
  const seq = ++deriveSeq;
  deriveAbort = new AbortController();
  try {
    const data = await api('/api/derive', { body }, deriveAbort.signal);
    if (seq !== deriveSeq) return;
    state.derive = data;
    lastDeriveError = '';
  } catch (err) {
    if (err.name === 'AbortError' || seq !== deriveSeq) return;
    state.derive = null;
    if (err.message !== lastDeriveError) toast(err.message);
    lastDeriveError = err.message;
  }
  state.deriveBusy = false;
  renderDerive();
}

function renderDerive() {
  const d = state.derive;
  const dash = '—';
  els.stats.classList.toggle('opacity-60', state.deriveBusy);
  els.hadley.textContent = d ? `${d.hadley_edge_deg.toFixed(1)}°` : dash;
  els.storm.textContent = d ? `${d.storm_track_deg.toFixed(1)}°` : dash;
  if (d) {
    els.cyclones.textContent = d.cyclones_per_hemisphere == null ? 'none' : String(d.cyclones_per_hemisphere);
    els.cyclonesDesc.textContent = d.has_eddies
      ? `Deformation radius ${Math.round(d.deformation_radius_km).toLocaleString('en-US')} km`
      : 'No baroclinic eddies';
  } else {
    els.cyclones.textContent = dash;
    els.cyclonesDesc.textContent = 'Baroclinic eddies';
  }
  const texel = d && d.texel_km ? d.texel_km[String(state.form.width)] : undefined;
  els.texel.textContent = typeof texel === 'number' ? `${texel.toFixed(1)} km` : dash;
  els.texelDesc.textContent = `At ${state.form.width} px wide`;

  els.warnings.innerHTML = '';
  for (const w of (d && d.warnings) || []) {
    const el = document.createElement('div');
    el.className = 'alert alert-soft alert-warning';
    el.setAttribute('role', 'alert');
    el.innerHTML = `${icon('triangle-alert', 'size-5')}<span class="text-sm"></span>`;
    $('span', el).textContent = w;
    els.warnings.appendChild(el);
  }
  renderBlock();
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

function blockReason() {
  if (!state.defaults) return 'Waiting for the server.';
  const errors = validationErrors();
  if (errors.length) return errors[0];
  const cap = state.defaults.max_cyclones_per_hemisphere;
  const eddies = state.derive && state.derive.cyclones_per_hemisphere;
  if (typeof cap === 'number' && typeof eddies === 'number' && eddies > cap) {
    return `Too fast for this tool: ${eddies} cyclones per hemisphere, the limit is ${cap}. Lengthen the rotation period.`;
  }
  if (state.form.width > widthLimit()) {
    return `${state.form.width} px is more than ${modeLabel(computeMode(state.form.compute))} offers (up to ${widthLimit()} px): pick a narrower map or another mode.`;
  }
  const bakeable = state.derive ? state.derive.bakeable : state.form.cloud_regime === 'water';
  if (!bakeable && !state.form.options.experiment) {
    return `Clouds of ${state.form.cloud_regime} are not water. Turn on Experiment to generate anyway.`;
  }
  return null;
}

function renderBlock() {
  const reason = state.running ? null : blockReason();
  els.generate.disabled = state.running || Boolean(reason);
  els.blockReason.classList.toggle('hidden', !reason);
  els.blockReason.classList.toggle('flex', Boolean(reason));
  els.blockText.textContent = reason || '';
}

let tickTimer = 0;

// The selected mode's own measured table (modes[].estimate_seconds); the legacy CPU table only for a backend without it.
function estimateSeconds() {
  const sel = computeMode(state.form.compute);
  if (sel && sel.estimate_seconds) return sel.estimate_seconds[String(state.form.width)];
  if (computeModes().length && (state.form.compute === 'gpu' || state.form.compute === 'hybrid')) return undefined;
  const table = (state.defaults && state.defaults.estimate_seconds) || {};
  return table[String(state.form.width)];
}

function tick() {
  const s = Math.floor((performance.now() - state.startedAt) / 1000);
  els.generateLabel.textContent = `Generating… ${s} s`;
  els.loadingText.textContent = `Generating… ${s} s`;
  const est = estimateSeconds();
  if (typeof est === 'number') {
    els.loadingHint.textContent = s > est * 1.5
      ? 'Taking longer than the estimate. The estimate assumes an idle machine.'
      : `About ${est} s expected at ${state.form.width} px.`;
  } else {
    els.loadingHint.textContent = '';
  }
}

function setRunning(running) {
  state.running = running;
  clearInterval(tickTimer);
  els.generateIcon.innerHTML = running
    ? '<span class="loading loading-spinner loading-sm"></span>'
    : icon('sparkles', 'size-5');
  if (running) {
    state.startedAt = performance.now();
    tick();
    tickTimer = setInterval(tick, 250);
  } else {
    els.generateLabel.textContent = 'Generate';
  }
  els.form.setAttribute('aria-busy', String(running));
  renderBlock();
  renderStage();
}

function payload() {
  const body = physicsBody();
  const o = state.form.options;
  const options = { match_earth: o.match_earth, isotropic: o.isotropic, experiment: o.experiment, surface: o.surface ? 'synthetic' : 'none' };
  // An older backend has no compute block and would not know the option: send it only when offered.
  if (computeModes().length) options.compute = state.form.compute;
  return { body, seed: Number(els.seed.value), width: state.form.width, options };
}

// On a phone the result sits below the whole form: bring it into view.
function revealResult() {
  if (!window.matchMedia('(max-width: 1023px)').matches) return;
  const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Start at the tabs: the map is right under them, the status block (what ran where) under the map.
  $('#viewBar').scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'start' });
}

async function generate() {
  if (state.running) return;
  const reason = blockReason();
  if (reason) {
    setStatus(reason);
    return;
  }
  const request = payload();
  const preset = presetByKey(state.form.preset);
  const label = `${preset ? preset.label : 'Custom body'}, seed ${request.seed}`;
  setStatus('');
  setRunning(true);
  try {
    const data = await api('/api/generate', request);
    releaseProducts();
    state.result = data;
    state.resultPreset = state.form.preset;
    state.stageError = null;
    if (state.tab === 'earth') state.tab = 'coverage';
    const where = computeUsedText(data.compute);
    setStatus(`Showing ${label}: ${data.width} x ${data.height} in ${Number(data.seconds).toFixed(1)} s${where ? ` ${where}` : ''}.`, data.compute);
    persist();
    renderResult();
    revealResult();
  } catch (err) {
    if (err.status === 429) toast('Another generation is already running. Try again when it has finished.', 'warning');
    else toast(err.message);
    setStatus('Last run failed.');
  } finally {
    setRunning(false);
  }
}

// ---------------------------------------------------------------------------
// Products: fetch once, keep the decoded image and (lazily) its exact bytes
// ---------------------------------------------------------------------------

const products = new Map();

function loadProduct(url) {
  if (!products.has(url)) {
    const pending = (async () => {
      const res = await fetch(url);
      if (!res.ok) {
        throw new ApiError(res.status, res.status === 404
          ? 'This result is no longer on the server. Generate it again.'
          : `Could not load the image (HTTP ${res.status}).`);
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const img = new Image();
      img.src = objectUrl;
      await img.decode();
      return { url, blob, objectUrl, img, w: img.naturalWidth, h: img.naturalHeight, pixels: null, pixelsPending: null };
    })();
    products.set(url, pending);
    pending.catch(() => products.delete(url));
  }
  return products.get(url);
}

// Exact bytes for readouts and the globe: no color management, no resampling.
function ensurePixels(p) {
  if (p.pixels) return Promise.resolve(p);
  if (!p.pixelsPending) {
    p.pixelsPending = createImageBitmap(p.blob, { colorSpaceConversion: 'none' }).then((bitmap) => {
      const c = document.createElement('canvas');
      c.width = p.w;
      c.height = p.h;
      const ctx = c.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      p.pixels = ctx.getImageData(0, 0, p.w, p.h).data;
      return p;
    });
  }
  return p.pixelsPending;
}

function releaseProducts() {
  for (const [url, pending] of Array.from(products)) {
    if (url === REFERENCE_URL) continue;
    products.delete(url);
    pending.then((p) => URL.revokeObjectURL(p.objectUrl)).catch(() => {});
  }
  state.product = null;
}

// ---------------------------------------------------------------------------
// Tabs, view mode, stage
// ---------------------------------------------------------------------------

function currentSource() {
  if (state.tab === 'earth') return { kind: 'coverage', url: REFERENCE_URL };
  if (state.result && state.result.images && state.result.images[state.tab]) {
    return { kind: TABS[state.tab].kind, url: state.result.images[state.tab] };
  }
  return null;
}

function setTab(tab, focus) {
  state.tab = tab;
  state.stageError = null;
  persist();
  renderTabs();
  if (focus) $(`#tab-${tab}`).focus();
  renderStage();
}

function setView(view) {
  state.view = view;
  persist();
  renderViewButtons();
  renderStage();
}

function renderTabs() {
  for (const btn of $$('[role="tab"]', els.tabs)) {
    const active = btn.dataset.tab === state.tab;
    btn.setAttribute('aria-selected', String(active));
    btn.tabIndex = active ? 0 : -1;
    btn.classList.toggle('tab-active', active);
  }
  els.panel.setAttribute('aria-labelledby', `tab-${state.tab}`);
}

function renderViewButtons() {
  renderSelected($$('[data-view]'), (b) => b.dataset.view === state.view, 'aria-pressed');
}

function renderDownloads() {
  const list = (state.result && state.result.downloads) || [];
  // A large map is never sent to the page: its big files are on the server's disk, listed by path.
  const onDisk = (state.result && state.result.files) || [];
  els.downloadMenu.innerHTML = list
    // `filename` is optional: the in-browser engine hands out blob: links, which have no name of their own.
    .map((d) => `<li><a href="${esc(d.href)}" download${d.filename ? `="${esc(d.filename)}"` : ''}>${esc(d.label)}</a></li>`)
    .join('')
    + onDisk.map((f) => `<li class="menu-title">${esc(f.label)}<br><span class="font-mono break-all">${esc(f.path)}</span> (${(f.bytes / 1048576).toFixed(0)} MiB)</li>`).join('');
  const enabled = list.length > 0;
  els.downloadButton.classList.toggle('btn-disabled', !enabled);
  els.downloadButton.setAttribute('aria-disabled', String(!enabled));
  if (!enabled) els.downloadBox.removeAttribute('open');
}

function showEmpty(title, text, action) {
  els.emptyTitle.textContent = title;
  els.emptyText.textContent = text;
  state.emptyAction = action;
  els.emptyAction.classList.toggle('hidden', !action);
  els.emptyAction.textContent = action === 'retry' ? 'Retry' : 'Load Earth reference';
  els.empty.classList.remove('hidden');
  els.empty.classList.add('flex');
  els.stage.classList.add('hidden');
  els.readoutRow.classList.add('hidden');
  els.readoutRow.classList.remove('flex');
}

function setLoadingBox(on, text, hint) {
  els.loadingBox.classList.toggle('hidden', !on);
  if (text !== undefined) els.loadingText.textContent = text;
  if (hint !== undefined) els.loadingHint.textContent = hint;
}

let stageSeq = 0;

async function renderStage() {
  const seq = ++stageSeq;
  const globe = state.view === 'globe';
  els.stage.classList.toggle('aspect-square', globe);
  els.stage.classList.toggle('sm:aspect-[2/1]', globe);
  els.stage.classList.toggle('aspect-[2/1]', !globe);
  els.caption.textContent = TABS[state.tab].caption;

  if (state.running && state.tab !== 'earth') {
    els.empty.classList.add('hidden');
    els.empty.classList.remove('flex');
    els.stage.classList.remove('hidden');
    els.mapImg.classList.add('hidden');
    els.globe.classList.add('hidden');
    setLoadingBox(true);
    tick();
    els.readoutRow.classList.add('hidden');
    els.readoutRow.classList.remove('flex');
    return;
  }

  if (state.loadError) {
    showEmpty('Cannot reach the server', state.loadError, 'retry');
    return;
  }

  const src = currentSource();
  if (!src) {
    if (state.stageError) showEmpty('Map unavailable', state.stageError, null);
    else showEmpty('No map yet', 'Pick a body on the left and press Generate. The four products appear here: coverage, type, high clouds and wind.', 'earth');
    return;
  }

  els.empty.classList.add('hidden');
  els.empty.classList.remove('flex');
  els.stage.classList.remove('hidden');
  els.readoutRow.classList.remove('hidden');
  els.readoutRow.classList.add('flex');
  els.readout.textContent = globe ? 'Drag the globe to rotate it. Double click to reset.' : 'Hover or touch the map to read values.';

  let product;
  try {
    const pending = loadProduct(src.url);
    setLoadingBox(true, 'Loading map…', '');
    els.mapImg.classList.add('hidden');
    els.globe.classList.add('hidden');
    product = await pending;
  } catch (err) {
    if (seq !== stageSeq) return;
    setLoadingBox(false);
    state.stageError = err.message;
    toast(err.message);
    if (err.status === 404 && src.url !== REFERENCE_URL) {
      // The server evicted this result: drop it and fall back to the empty state.
      state.result = null;
      renderResult();
    } else {
      showEmpty('Map unavailable', err.message, null);
    }
    return;
  }
  if (seq !== stageSeq) return;
  product.kind = src.kind;
  state.product = product;
  setLoadingBox(false);

  if (globe) {
    els.mapImg.classList.add('hidden');
    els.globe.classList.remove('hidden');
    await ensurePixels(product);
    if (seq !== stageSeq) return;
    drawGlobe(1);
  } else {
    els.globe.classList.add('hidden');
    if (els.mapImg.getAttribute('src') !== product.objectUrl) els.mapImg.src = product.objectUrl;
    els.mapImg.alt = `${TABS[state.tab].kind} map`;
    els.mapImg.classList.remove('hidden');
    ensurePixels(product);
  }
}

// ---------------------------------------------------------------------------
// Pixel readouts (map and globe share them)
// ---------------------------------------------------------------------------

function samplePixel(p, kind, u, v) {
  const x = Math.min(p.w - 1, Math.max(0, Math.floor(u * p.w)));
  const y = Math.min(p.h - 1, Math.max(0, Math.floor(v * p.h)));
  const i = (y * p.w + x) * 4;
  const d = p.pixels;
  switch (kind) {
    case 'coverage':
      return `Cloud fraction ${Math.round((d[i] / 255) * 100)} %`;
    case 'type':
      return `Convective share ${Math.round((d[i] / 255) * 100)} %`;
    case 'high_share':
      return d[i] === 255 ? 'High-cloud share none' : `High-cloud share ${Math.round((d[i] / 254) * 100)} %`;
    default: {
      const east = (d[i] - 128) * 0.5;
      const north = (d[i + 1] - 128) * 0.5;
      return `Wind east ${east.toFixed(1)}, north ${north.toFixed(1)} m/s (speed ${Math.hypot(east, north).toFixed(1)})`;
    }
  }
}

function setReadout(text) {
  els.readout.textContent = text;
}

function onMapMove(e) {
  const p = state.product;
  if (!p || state.view !== 'map') return;
  if (!p.pixels) {
    ensurePixels(p);
    return;
  }
  const box = els.mapImg.getBoundingClientRect();
  const scale = Math.min(box.width / p.w, box.height / p.h);
  const dw = p.w * scale;
  const dh = p.h * scale;
  const u = (e.clientX - (box.left + (box.width - dw) / 2)) / dw;
  const v = (e.clientY - (box.top + (box.height - dh) / 2)) / dh;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) {
    setReadout(' ');
    return;
  }
  setReadout(`${fmtLatLon(90 - v * 180, u * 360 - 180)}   ${samplePixel(p, p.kind, u, v)}`);
}

// ---------------------------------------------------------------------------
// Globe: orthographic projection drawn in JS from the same image
// ---------------------------------------------------------------------------

let globeImage = null;

// The ground of the body on show: the Earth tab is Earth's; a run keeps the preset it was made with, so editing
// the form afterwards does not repaint a map of another body.
function globeGround() {
  const key = state.tab === 'earth' ? 'earth' : state.result ? state.resultPreset : state.form.preset;
  return GROUND_BY_PRESET[key] || SPHERE_BG;
}

function globeGeometry(canvas, quality) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.max(2, Math.round(rect.width * dpr * quality));
  const H = Math.max(2, Math.round(rect.height * dpr * quality));
  return { rect, W, H, R: Math.min(W, H) * 0.47, cx: W / 2, cy: H / 2 };
}

function drawGlobe(quality) {
  const p = state.product;
  const canvas = els.globe;
  if (!p || !p.pixels || state.view !== 'globe') return;
  if (gpuGlobe.device && p.w <= gpuGlobe.maxSize) {
    const g = globeGeometry(canvas, 1);
    if (g.rect.width < 2) return;
    // The 2D canvas stays underneath, empty, for the pointer and the keys.
    if (!gpuGlobe.cleared) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      gpuGlobe.cleared = true;
    }
    drawGlobeGpu(p, g);
    return;
  }
  // Without WebGPU (or for a map wider than its largest texture): the CPU projection below.
  els.globeGpu.classList.add('invisible');
  gpuGlobe.cleared = false;
  const g = globeGeometry(canvas, quality);
  if (g.rect.width < 2) return;
  if (canvas.width !== g.W || canvas.height !== g.H) {
    canvas.width = g.W;
    canvas.height = g.H;
    globeImage = null;
  }
  const ctx = canvas.getContext('2d');
  if (!globeImage || globeImage.width !== g.W || globeImage.height !== g.H) globeImage = ctx.createImageData(g.W, g.H);
  const out = globeImage.data;
  const src = p.pixels;
  const w = p.w;
  const h = p.h;
  const kind = p.kind;
  const lon0 = state.globe.lon * DEG;
  const lat0 = state.globe.lat * DEG;
  const sinL = Math.sin(lat0);
  const cosL = Math.cos(lat0);
  const { W, H, R, cx, cy } = g;
  const x0 = Math.max(0, Math.floor(cx - R - 2));
  const x1 = Math.min(W, Math.ceil(cx + R + 2));
  const y0 = Math.max(0, Math.floor(cy - R - 2));
  const y1 = Math.min(H, Math.ceil(cy + R + 2));
  const [bgR, bgG, bgB] = globeGround();
  const TWO_PI = 2 * Math.PI;
  const mode = kind === 'coverage' ? 0 : kind === 'type' ? 1 : kind === 'high_share' ? 2 : 3;

  for (let y = y0; y < y1; y++) {
    const ny = (cy - y - 0.5) / R;
    for (let x = x0; x < x1; x++) {
      const nx = (x + 0.5 - cx) / R;
      const d2 = nx * nx + ny * ny;
      const o = (y * W + x) * 4;
      const r = Math.sqrt(d2);
      const cover = Math.min(1, Math.max(0, (1 - r) * R + 0.5));
      if (cover <= 0) {
        out[o + 3] = 0;
        continue;
      }
      const nz = Math.sqrt(Math.max(0, 1 - d2));
      let sinLat = nz * sinL + ny * cosL;
      sinLat = sinLat > 1 ? 1 : sinLat < -1 ? -1 : sinLat;
      const lat = Math.asin(sinLat);
      const lon = lon0 + Math.atan2(nx, nz * cosL - ny * sinL);
      let u = lon / TWO_PI + 0.5;
      u -= Math.floor(u);
      const v = 0.5 - lat / Math.PI;

      const fx = u * w - 0.5;
      const fy = v * h - 0.5;
      const ix = Math.floor(fx);
      const iy = Math.floor(fy);
      const tx = fx - ix;
      const ty = fy - iy;
      const xa = ix < 0 ? w - 1 : ix;
      const xb = xa + 1 >= w ? 0 : xa + 1;
      const ya = iy < 0 ? 0 : iy >= h ? h - 1 : iy;
      const yb = iy + 1 >= h ? h - 1 : iy + 1;
      const i00 = (ya * w + xa) * 4;
      const i10 = (ya * w + xb) * 4;
      const i01 = (yb * w + xa) * 4;
      const i11 = (yb * w + xb) * 4;
      const w00 = (1 - tx) * (1 - ty);
      const w10 = tx * (1 - ty);
      const w01 = (1 - tx) * ty;
      const w11 = tx * ty;

      let cr;
      let cg;
      let cb;
      if (mode === 2) {
        // High-cloud share: 255 means "none", so weight only the texels that carry a value.
        let sum = 0;
        let acc = 0;
        let v0 = src[i00];
        if (v0 !== 255) { sum += w00; acc += w00 * v0; }
        v0 = src[i10];
        if (v0 !== 255) { sum += w10; acc += w10 * v0; }
        v0 = src[i01];
        if (v0 !== 255) { sum += w01; acc += w01 * v0; }
        v0 = src[i11];
        if (v0 !== 255) { sum += w11; acc += w11 * v0; }
        if (sum < 1e-6) {
          cr = bgR; cg = bgG; cb = bgB;
        } else {
          cr = cg = cb = acc / sum;
        }
      } else if (mode === 3) {
        const east = (src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11 - 128) * 0.5;
        const north = (src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11 - 128) * 0.5;
        const t = Math.min(1, Math.hypot(east, north) / 35);
        cr = bgR + (255 - bgR) * t;
        cg = bgG + (255 - bgG) * t;
        cb = bgB + (255 - bgB) * t;
      } else {
        const gray = src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11;
        if (mode === 1) {
          cr = cg = cb = gray;
        } else {
          const t = gray / 255;
          cr = bgR + (255 - bgR) * t;
          cg = bgG + (255 - bgG) * t;
          cb = bgB + (255 - bgB) * t;
        }
      }
      // Thin limb darkening: only the last few percent of the radius dim.
      const q = 1 - nz;
      const k = 1 - 0.55 * q * q * q;
      out[o] = cr * k;
      out[o + 1] = cg * k;
      out[o + 2] = cb * k;
      out[o + 3] = 255 * cover;
    }
  }
  ctx.putImageData(globeImage, 0, 0);
}

// ---------------------------------------------------------------------------
// Globe on the GPU: raw WebGPU, one full-screen triangle, the sphere in the fragment shader
// ---------------------------------------------------------------------------
// Same orthographic view, radius and sampling as drawGlobe (so the pointer readout and the drag stay exact), plus
// light: a fixed sun up and to the left, a soft terminator, a dim night side, haze at the limb and a thin halo.

const GLOBE_WGSL = /* wgsl */ `
struct U {
  size: vec2f,    // canvas, device pixels
  tex: vec2f,     // map, texels
  ground: vec3f,  // 0..1
  R: f32,         // sphere radius, device pixels
  lon0: f32,
  lat0: f32,
  mode: f32,      // 0 coverage, 1 type, 2 high-cloud share, 3 wind
  pad: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var img: texture_2d<f32>;

const PI = 3.14159265;
const SKY = vec3f(0.55, 0.72, 1.0);

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn at(x: i32, y: i32) -> vec4f { return textureLoad(img, vec2i(x, y), 0) * 255.0; }

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let sun = normalize(vec3f(-0.42, 0.38, 0.82));
  let c = u.size * 0.5;
  let n2 = vec2f(pos.x - c.x, c.y - pos.y) / u.R;
  let r = length(n2);
  let atmo = mix(u.ground, SKY, 0.55);
  let t = clamp((r - 1.0) / 0.05, 0.0, 1.0);
  let side = 0.15 + 0.85 * smoothstep(-0.4, 0.6, dot(n2 / max(r, 1e-4), sun.xy));
  let haloA = (1.0 - t) * (1.0 - t) * 0.55 * side;
  let cover = clamp((1.0 - r) * u.R + 0.5, 0.0, 1.0);
  if (cover <= 0.0) { return vec4f(atmo * haloA, haloA); }

  // The point of the sphere under this pixel, then its texel (bilinear by hand: wrap in x, clamp in y).
  let nz = sqrt(max(0.0, 1.0 - dot(n2, n2)));
  let sinL = sin(u.lat0);
  let cosL = cos(u.lat0);
  let lat = asin(clamp(nz * sinL + n2.y * cosL, -1.0, 1.0));
  let lon = u.lon0 + atan2(n2.x, nz * cosL - n2.y * sinL);
  var uu = lon / (2.0 * PI) + 0.5;
  uu = uu - floor(uu);
  let vv = 0.5 - lat / PI;
  let w = i32(u.tex.x);
  let h = i32(u.tex.y);
  let fx = uu * u.tex.x - 0.5;
  let fy = vv * u.tex.y - 0.5;
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let tx = fx - floor(fx);
  let ty = fy - floor(fy);
  let xa = select(ix, w - 1, ix < 0);
  let xb = select(xa + 1, 0, xa + 1 >= w);
  let ya = clamp(iy, 0, h - 1);
  let yb = clamp(iy + 1, 0, h - 1);
  let s00 = at(xa, ya);
  let s10 = at(xb, ya);
  let s01 = at(xa, yb);
  let s11 = at(xb, yb);
  let w00 = (1.0 - tx) * (1.0 - ty);
  let w10 = tx * (1.0 - ty);
  let w01 = (1.0 - tx) * ty;
  let w11 = tx * ty;

  var col: vec3f;
  let mode = i32(u.mode);
  if (mode == 2) {
    // High-cloud share: 255 means "none", so weight only the texels that carry a value.
    var sum = 0.0;
    var acc = 0.0;
    if (s00.r < 254.5) { sum += w00; acc += w00 * s00.r; }
    if (s10.r < 254.5) { sum += w10; acc += w10 * s10.r; }
    if (s01.r < 254.5) { sum += w01; acc += w01 * s01.r; }
    if (s11.r < 254.5) { sum += w11; acc += w11 * s11.r; }
    col = select(vec3f(acc / max(sum, 1e-6) / 255.0), u.ground, sum < 1e-6);
  } else if (mode == 3) {
    let e = (s00.r * w00 + s10.r * w10 + s01.r * w01 + s11.r * w11 - 128.0) * 0.5;
    let n = (s00.g * w00 + s10.g * w10 + s01.g * w01 + s11.g * w11 - 128.0) * 0.5;
    col = mix(u.ground, vec3f(1.0), min(1.0, length(vec2f(e, n)) / 35.0));
  } else {
    let g = (s00.r * w00 + s10.r * w10 + s01.r * w01 + s11.r * w11) / 255.0;
    col = select(mix(u.ground, vec3f(1.0), g), vec3f(g), mode == 1);
  }

  let day = smoothstep(-0.15, 0.45, dot(vec3f(n2, nz), sun));
  col = col * mix(0.22, 1.0, day);
  col = mix(col, atmo * (0.25 + 0.75 * day), pow(1.0 - nz, 3.0) * 0.6);
  // Premultiplied: the sphere over its own halo at the antialiased edge.
  return vec4f(col * cover + atmo * haloA * (1.0 - cover), cover + haloA * (1.0 - cover));
}
`;

const gpuGlobe = { device: null, ctx: null, pipeline: null, ubo: null, tex: null, texFor: null, bind: null, maxSize: 0, cleared: false };

// Quietly absent without WebGPU (or on any error): the 2D projection stays.
async function initGpuGlobe() {
  try {
    if (!navigator.gpu) return;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    const device = await adapter.requestDevice();
    const ctx = els.globeGpu.getContext('webgpu');
    if (!ctx) return;
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: 'premultiplied' });
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: GLOBE_WGSL });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    const ubo = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const error = await device.popErrorScope();
    if (error) {
      console.warn('WebGPU globe disabled:', error.message);
      device.destroy();
      return;
    }
    Object.assign(gpuGlobe, { device, ctx, pipeline, ubo, maxSize: device.limits.maxTextureDimension2D });
    device.lost.then(() => {
      Object.assign(gpuGlobe, { device: null, tex: null, texFor: null, bind: null });
      drawGlobe(1);
    });
    drawGlobe(1);
  } catch (err) {
    console.warn('WebGPU globe disabled:', err);
  }
}

function drawGlobeGpu(p, g) {
  const G = gpuGlobe;
  const canvas = els.globeGpu;
  if (canvas.width !== g.W || canvas.height !== g.H) {
    canvas.width = g.W;
    canvas.height = g.H;
  }
  // The texture follows the product on show: uploaded once per product, not per frame.
  if (G.texFor !== p) {
    if (G.tex) G.tex.destroy();
    G.tex = G.device.createTexture({ size: [p.w, p.h], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    G.device.queue.writeTexture({ texture: G.tex }, p.pixels, { bytesPerRow: p.w * 4 }, [p.w, p.h]);
    G.bind = G.device.createBindGroup({
      layout: G.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: G.ubo } }, { binding: 1, resource: G.tex.createView() }],
    });
    G.texFor = p;
  }
  const kind = p.kind;
  const mode = kind === 'coverage' ? 0 : kind === 'type' ? 1 : kind === 'high_share' ? 2 : 3;
  const [gr, gg, gb] = globeGround();
  G.device.queue.writeBuffer(G.ubo, 0, new Float32Array([
    g.W, g.H, p.w, p.h, gr / 255, gg / 255, gb / 255, g.R, state.globe.lon * DEG, state.globe.lat * DEG, mode, 0,
  ]));
  const encoder = G.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: G.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
  });
  pass.setPipeline(G.pipeline);
  pass.setBindGroup(0, G.bind);
  pass.draw(3);
  pass.end();
  G.device.queue.submit([encoder.finish()]);
  canvas.classList.remove('invisible');
}

let globeRaf = 0;
let globeIdle = 0;

// While the user drags, draw at half resolution, then settle with a full-resolution pass.
function requestGlobe(fast) {
  clearTimeout(globeIdle);
  if (!fast) {
    drawGlobe(1);
    return;
  }
  if (!globeRaf) {
    globeRaf = requestAnimationFrame(() => {
      globeRaf = 0;
      drawGlobe(0.5);
    });
  }
  globeIdle = setTimeout(() => drawGlobe(1), 180);
}

function globePoint(e) {
  const rect = els.globe.getBoundingClientRect();
  const R = Math.min(rect.width, rect.height) * 0.47;
  const nx = (e.clientX - rect.left - rect.width / 2) / R;
  const ny = (rect.top + rect.height / 2 - e.clientY) / R;
  const d2 = nx * nx + ny * ny;
  if (d2 > 1) return null;
  const nz = Math.sqrt(1 - d2);
  const lat0 = state.globe.lat * DEG;
  const lat = Math.asin(Math.max(-1, Math.min(1, nz * Math.sin(lat0) + ny * Math.cos(lat0))));
  const lon = state.globe.lon * DEG + Math.atan2(nx, nz * Math.cos(lat0) - ny * Math.sin(lat0));
  let u = lon / (2 * Math.PI) + 0.5;
  u -= Math.floor(u);
  return { lat: lat / DEG, lon: u * 360 - 180, u, v: 0.5 - lat / Math.PI };
}

let drag = null;

function onGlobeDown(e) {
  if (e.button !== undefined && e.button !== 0) return;
  els.globe.setPointerCapture(e.pointerId);
  els.globe.classList.replace('cursor-grab', 'cursor-grabbing');
  drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
}

function onGlobeMove(e) {
  const p = state.product;
  if (drag && drag.id === e.pointerId) {
    const R = Math.min(els.globe.clientWidth, els.globe.clientHeight) * 0.47;
    const k = 1 / R / DEG;
    state.globe.lon -= (e.clientX - drag.x) * k;
    state.globe.lat = Math.max(-90, Math.min(90, state.globe.lat + (e.clientY - drag.y) * k));
    state.globe.lon = ((state.globe.lon + 540) % 360) - 180;
    drag.x = e.clientX;
    drag.y = e.clientY;
    requestGlobe(true);
    return;
  }
  if (!p || !p.pixels) return;
  const pt = globePoint(e);
  setReadout(pt ? `${fmtLatLon(pt.lat, pt.lon)}   ${samplePixel(p, p.kind, pt.u, pt.v)}` : ' ');
}

function onGlobeUp(e) {
  if (!drag || drag.id !== e.pointerId) return;
  drag = null;
  els.globe.classList.replace('cursor-grabbing', 'cursor-grab');
  requestGlobe(false);
}

function onGlobeKey(e) {
  const step = e.shiftKey ? 30 : 10;
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
  if (e.key === 'Home') {
    state.globe = { ...GLOBE_HOME };
  } else if (moves[e.key]) {
    state.globe.lon = ((state.globe.lon + moves[e.key][0] + 540) % 360) - 180;
    state.globe.lat = Math.max(-90, Math.min(90, state.globe.lat + moves[e.key][1]));
  } else {
    return;
  }
  e.preventDefault();
  requestGlobe(false);
}

// ---------------------------------------------------------------------------
// Result panels: metrics, zonal chart, flow
// ---------------------------------------------------------------------------

function renderResult() {
  renderDownloads();
  renderTabs();
  const r = state.result;
  els.details.classList.toggle('hidden', !r);
  els.details.classList.toggle('flex', Boolean(r));
  if (r) {
    renderMetrics(r);
    renderFlow(r);
    drawZonal();
  }
  renderStage();
}

function renderMetrics(r) {
  const rows = (r.metrics && r.metrics.table) || [];
  els.metricsSub.textContent = `Compared with ${(r.metrics && r.metrics.earth_label) || 'Earth'}.`;
  const note = r.metrics && r.metrics.note;
  els.metricsNote.classList.toggle('hidden', !note);
  els.metricsNote.classList.toggle('flex', Boolean(note));
  els.metricsNoteText.textContent = note || '';
  els.runWarnings.innerHTML = '';
  const known = (state.derive && state.derive.warnings) || [];
  for (const w of (Array.isArray(r.warnings) ? r.warnings : []).filter((x) => !known.includes(x))) {
    const el = document.createElement('div');
    el.className = 'alert alert-soft alert-warning';
    el.setAttribute('role', 'alert');
    el.innerHTML = `${icon('triangle-alert', 'size-5')}<span class="text-sm"></span>`;
    $('span', el).textContent = w;
    els.runWarnings.appendChild(el);
  }
  els.metricsBody.innerHTML = rows.map((row) => {
    const label = row.hint
      ? `<span class="tooltip tooltip-right text-left" tabindex="0" data-tip="${esc(row.hint)}">${esc(row.label)}</span>`
      : esc(row.label);
    return `<tr>
      <td>${label}</td>
      <td class="px-1.5 text-right whitespace-nowrap tabular-nums">${esc(fmtValue(row.generated, row.format))}</td>
      <td class="px-1.5 text-right whitespace-nowrap tabular-nums">${esc(fmtValue(row.earth, row.format))}</td>
      <td class="px-1.5 text-right whitespace-nowrap"><span class="badge badge-soft badge-sm tabular-nums">${esc(relGap(row.generated, row.earth))}</span></td>
    </tr>`;
  }).join('');
}

function renderFlow(r) {
  const meta = r.meta || {};
  const flow = meta.flow || {};
  const items = [
    ['Jet', fmtValue(flow.jet, '0.00')],
    ['Trades', fmtValue(flow.trades, '0.00')],
    ['Polar easterlies', fmtValue(flow.polar_easterlies, '0.00')],
    ['Hadley edge', typeof meta.hadley_edge_deg === 'number' ? `${meta.hadley_edge_deg.toFixed(1)}°` : '—'],
    ['Fine octaves', meta.fine_octaves == null ? '—' : String(meta.fine_octaves)]
  ];
  if (meta.surface) {
    items.push(['Land fraction', fmtValue(meta.surface.land_fraction, '0.00')]);
    items.push(['ITCZ shift', `${fmtValue(meta.surface.itcz_shift_deg, '0.0')}°`]);
  }
  els.flowList.innerHTML = items
    .map(([k, v]) => `<div><dt class="text-xs text-base-content/70">${esc(k)}</dt><dd class="tabular-nums">${esc(v)}</dd></div>`)
    .join('');
  const vortices = flow.vortices || [];
  els.vortexNone.classList.toggle('hidden', vortices.length > 0);
  els.vortexBody.innerHTML = vortices.map((v) => `<tr>
    <td>${esc(v.kind)}</td>
    <td class="text-right tabular-nums">${esc(fmtValue(v.lon_deg, '0.0'))}</td>
    <td class="text-right tabular-nums">${esc(fmtValue(v.lat_deg, '0.0'))}</td>
    <td class="text-right tabular-nums">${esc(fmtValue(v.radius, '0.000'))}</td>
    <td class="text-right tabular-nums">${esc(fmtValue(v.omega, '0.00'))}</td>
    <td class="text-right tabular-nums">${esc(fmtValue(v.age, '0.00'))}</td>
  </tr>`).join('');
}

let zonalHover = -1;

function drawZonal() {
  const z = state.result && state.result.zonal;
  const canvas = els.zonal;
  const rect = canvas.getBoundingClientRect();
  if (!z || rect.width < 2) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(`--color-${name}`).trim();
  const ink = color('base-content');
  const left = 36;
  const right = 8;
  const top = 8;
  const bottom = 24;
  const pw = rect.width - left - right;
  const ph = rect.height - top - bottom;
  const X = (lat) => left + ((lat + 90) / 180) * pw;
  const Y = (v) => top + (1 - v) * ph;

  ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`;
  ctx.lineWidth = 1;
  ctx.fillStyle = ink;
  ctx.strokeStyle = ink;
  for (let v = 0; v <= 1.0001; v += 0.25) {
    ctx.globalAlpha = 0.14;
    ctx.beginPath();
    ctx.moveTo(left, Math.round(Y(v)) + 0.5);
    ctx.lineTo(left + pw, Math.round(Y(v)) + 0.5);
    ctx.stroke();
    ctx.globalAlpha = 0.7;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(v.toFixed(2), left - 6, Y(v));
  }
  for (let lat = -90; lat <= 90; lat += 30) {
    ctx.globalAlpha = 0.14;
    ctx.beginPath();
    ctx.moveTo(Math.round(X(lat)) + 0.5, top);
    ctx.lineTo(Math.round(X(lat)) + 0.5, top + ph);
    ctx.stroke();
    ctx.globalAlpha = 0.7;
    ctx.textAlign = lat === -90 ? 'left' : lat === 90 ? 'right' : 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`${lat}°`, X(lat), top + ph + 6);
  }
  ctx.globalAlpha = 0.5;
  ctx.strokeRect(left + 0.5, top + 0.5, pw, ph);

  const series = [
    { data: z.target, stroke: ink, width: 1.5, dash: [5, 4], alpha: 0.75 },
    { data: z.earth, stroke: color('info'), width: 2, dash: [], alpha: 1 },
    { data: z.generated, stroke: color('success'), width: 2, dash: [], alpha: 1 }
  ];
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, pw, ph);
  ctx.clip();
  for (const s of series) {
    if (!s.data) continue;
    ctx.globalAlpha = s.alpha;
    ctx.strokeStyle = s.stroke;
    ctx.lineWidth = s.width;
    ctx.setLineDash(s.dash);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    s.data.forEach((val, i) => {
      const px = X(z.lat_deg[i]);
      const py = Y(val);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  ctx.restore();
  ctx.setLineDash([]);

  if (zonalHover >= 0 && zonalHover < z.lat_deg.length) {
    const hx = X(z.lat_deg[zonalHover]);
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(hx) + 0.5, top);
    ctx.lineTo(Math.round(hx) + 0.5, top + ph);
    ctx.stroke();
    ctx.globalAlpha = 1;
    for (const s of series) {
      if (!s.data) continue;
      ctx.fillStyle = s.stroke;
      ctx.beginPath();
      ctx.arc(hx, Y(s.data[zonalHover]), 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function onZonalMove(e) {
  const z = state.result && state.result.zonal;
  if (!z) return;
  const rect = els.zonal.getBoundingClientRect();
  const lat = ((e.clientX - rect.left - 36) / (rect.width - 44)) * 180 - 90;
  let best = 0;
  z.lat_deg.forEach((v, i) => { if (Math.abs(v - lat) < Math.abs(z.lat_deg[best] - lat)) best = i; });
  zonalHover = best;
  const f = (arr) => (arr ? arr[best].toFixed(3) : '—');
  els.zonalReadout.textContent = `Lat ${z.lat_deg[best].toFixed(1)}°   generated ${f(z.generated)}   Earth ${f(z.earth)}   target ${f(z.target)}`;
  drawZonal();
}

function onZonalLeave() {
  zonalHover = -1;
  els.zonalReadout.textContent = ' ';
  drawZonal();
}

// ---------------------------------------------------------------------------
// Controls wiring
// ---------------------------------------------------------------------------

function buildPresets() {
  els.preset.innerHTML = '<option value="">Custom</option>'
    + state.defaults.presets.map((p) => `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join('');
}

function buildWidths() {
  els.widths.innerHTML = state.defaults.widths.map((w) =>
    `<button type="button" role="radio" data-width="${w}" class="btn btn-sm join-item flex-1">${w}</button>`).join('');
}

// One selected look for every segmented choice: the inverted base color (dark on light,
// light on dark), never primary. daisyUI paints [aria-pressed] and [aria-checked] as active.
function renderSelected(buttons, isActive, attr) {
  for (const btn of buttons) {
    const on = isActive(btn);
    btn.classList.toggle('[--btn-color:var(--color-base-content)]', on);
    btn.classList.toggle('[--btn-fg:var(--color-base-100)]', on);
    btn.setAttribute(attr, String(on));
  }
}

// The widest map the selected mode offers (modes[].max_width; no limit for a backend that does not say).
function widthLimit() {
  const m = computeMode(state.form.compute);
  return m && Number.isFinite(m.max_width) ? m.max_width : Infinity;
}

function renderWidths() {
  const buttons = $$('[data-width]', els.widths);
  const limit = widthLimit();
  for (const b of buttons) {
    const over = Number(b.dataset.width) > limit;
    b.classList.toggle('opacity-40', over);
    b.title = over ? `More than ${modeLabel(computeMode(state.form.compute))} offers (up to ${limit} px).` : '';
  }
  renderSelected(buttons, (b) => Number(b.dataset.width) === state.form.width, 'aria-checked');
  for (const b of buttons) b.tabIndex = Number(b.dataset.width) === state.form.width ? 0 : -1;
}

function setWidth(width, focus) {
  state.form.width = width;
  persist();
  renderWidths();
  renderWidthHint();
  if (focus) $(`[data-width="${width}"]`, els.widths).focus();
}

function renderWidthHint() {
  const w = state.form.width;
  const est = estimateSeconds();
  const mach = state.defaults.compute && state.defaults.compute.machine;
  const disk = mach && mach.estimate_disk_bytes && mach.estimate_disk_bytes[String(w)];
  els.widthHint.textContent = `${w} x ${w / 2} px`
    + (typeof est === 'number' ? `, about ${est} s on an idle machine` : '')
    + (typeof disk === 'number' && w >= 8192 ? `, up to ${(disk / 1073741824).toFixed(1)} GiB of files` : '') + '.';
  renderDerive();
}

// ---------------------------------------------------------------------------
// Compute mode: the "compute" block of GET /api/defaults decides everything shown here
// ---------------------------------------------------------------------------

// The modes the server offers, in its order. An empty list (an older backend, or a block that is
// not the contract's) hides the whole selector and sends no compute option.
function computeModes() {
  const c = state.defaults && state.defaults.compute;
  if (!c || !Array.isArray(c.modes)) return [];
  return c.modes.filter((m) => m && typeof m.key === 'string' && m.key);
}

function computeMode(key) {
  return computeModes().find((m) => m.key === key);
}

function modeLabel(m) {
  return m.label || m.key;
}

function modeReason(m) {
  return m.reason || 'It is not available here.';
}

// Called once per load, after the saved form is read: the first load takes the server's default;
// a saved mode that is gone or unavailable falls back to Auto and says so in the info line.
function reconcileCompute(savedKey) {
  state.computeInspect = null;
  state.computeFallback = '';
  const modes = computeModes();
  if (!modes.length) return;
  const usable = (key) => modes.some((m) => m.key === key && m.available);
  const fallback = modes.find((m) => m.key === 'auto' && m.available) || modes.find((m) => m.available) || modes[0];
  if (!savedKey) {
    state.form.compute = usable(state.defaults.compute.default) ? state.defaults.compute.default : fallback.key;
    return;
  }
  if (usable(savedKey)) return;
  const gone = computeMode(savedKey);
  state.form.compute = fallback.key;
  // Short on purpose: the reason is one tap or hover away on the unavailable option itself.
  state.computeFallback = gone
    ? `Your saved mode, ${modeLabel(gone)}, is not available here. ${modeLabel(fallback)} is selected.`
    : `Your saved mode, ${savedKey}, does not exist here. ${modeLabel(fallback)} is selected.`;
}

function buildCompute() {
  const modes = computeModes();
  els.computeRow.hidden = modes.length === 0;
  delete els.computeTip.dataset.tip;
  els.compute.innerHTML = modes.map((m) => {
    const ic = COMPUTE_ICONS[m.key];
    return `<button type="button" role="radio" data-compute="${esc(m.key)}" class="btn btn-sm join-item flex-auto gap-1 px-2">`
      + `${ic ? icon(ic) : ''}<span class="whitespace-nowrap">${esc(modeLabel(m))}</span></button>`;
  }).join('');
}

function renderCompute() {
  const buttons = $$('[data-compute]', els.compute);
  renderSelected(buttons, (b) => b.dataset.compute === state.form.compute, 'aria-checked');
  for (const b of buttons) {
    const m = computeMode(b.dataset.compute);
    const ok = Boolean(m && m.available);
    b.tabIndex = b.dataset.compute === state.form.compute ? 0 : -1;
    // Not the disabled attribute: the option stays hoverable, focusable and tappable, so its reason can be read.
    b.classList.toggle('btn-disabled', !ok);
    b.classList.toggle('pointer-events-auto!', !ok);
    b.classList.toggle('cursor-not-allowed', !ok);
    if (ok) b.removeAttribute('aria-disabled');
    else b.setAttribute('aria-disabled', 'true');
  }
  renderComputeHint();
  renderComputeNote();
}

// What the chosen mode does: its detail ("16 threads", the GPU and its backend), then its description.
function renderComputeHint() {
  const m = computeMode(state.form.compute);
  els.computeHint.textContent = '';
  if (!m) return;
  if (m.detail) {
    const d = document.createElement('span');
    d.className = 'font-medium text-base-content';
    d.textContent = m.detail;
    els.computeHint.append(d, m.description ? ' · ' : '');
  }
  els.computeHint.append(m.description || '');
}

// The reason of an unavailable mode the user tapped or arrowed to, or the one-time fallback sentence.
function renderComputeNote() {
  const m = state.computeInspect && computeMode(state.computeInspect);
  let text = '';
  if (m && !m.available) text = `${modeLabel(m)} is unavailable. ${modeReason(m)}`;
  else if (state.computeFallback) text = state.computeFallback;
  fillInfo(els.computeNote, text);
}

// One tooltip for the whole control, its text following the option under the pointer (or the focus).
// Only an unavailable option has one: the server's reason. What an available mode does is the hint
// line under the control, which can run to several lines and would make a poor tooltip.
function setComputeTip(key) {
  const m = key && computeMode(key);
  const text = m && !m.available ? modeReason(m) : '';
  if (text) els.computeTip.dataset.tip = text;
  else delete els.computeTip.dataset.tip;
}

// An available mode becomes the choice. An unavailable one never does: it only shows why (the same
// line a touch screen needs, where there is no hover).
function chooseCompute(key) {
  const m = computeMode(key);
  if (!m) return;
  state.computeFallback = '';
  if (m.available) {
    state.computeInspect = null;
    state.form.compute = key;
    persist();
    renderWidthHint();
  } else {
    state.computeInspect = key;
  }
  renderCompute();
  setComputeTip(key);
}

function onComputeKey(e) {
  const keys = computeModes().map((m) => m.key);
  const focused = e.target.closest('[data-compute]');
  let i = keys.indexOf(focused ? focused.dataset.compute : state.form.compute);
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') i = Math.min(keys.length - 1, i + 1);
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') i = Math.max(0, i - 1);
  else if (e.key === 'Home') i = 0;
  else if (e.key === 'End') i = keys.length - 1;
  else return;
  e.preventDefault();
  chooseCompute(keys[i]);
  $(`[data-compute="${keys[i]}"]`, els.compute).focus();
}

function deviceLabel(kind) {
  return kind === 'cpu' ? 'CPU' : kind === 'gpu' ? 'GPU' : String(kind).toUpperCase();
}

// "NVIDIA GeForce RTX 5080" reads as "RTX 5080" in a sentence.
function shortGpuName(name) {
  return String(name).replace(/^(NVIDIA|AMD|Intel(\(R\))?)\s+/i, '').replace(/^GeForce\s+/i, '').trim();
}

function computeDevices(compute) {
  const list = compute && Array.isArray(compute.devices) ? compute.devices : [];
  return list.filter((d) => d && typeof d.kind === 'string');
}

// "on the CPU", "on the GPU (RTX 5080)", "on the CPU and the GPU (RTX 5080)"; empty when the server said nothing.
function computeUsedText(compute) {
  if (!compute || typeof compute !== 'object') return '';
  const gpu = computeDevices(compute).find((d) => d.kind === 'gpu' && d.name);
  const named = gpu ? ` (${shortGpuName(gpu.name)})` : '';
  if (compute.used === 'cpu') return 'on the CPU';
  if (compute.used === 'gpu') return `on the GPU${named}`;
  if (compute.used === 'hybrid') return `on the CPU and the GPU${named}`;
  return '';
}

// One soft badge per device that worked: "GPU 84 %". The device name and its seconds sit in the title.
function deviceBadge(d) {
  const share = typeof d.share === 'number' && Number.isFinite(d.share) ? d.share : null;
  const pct = share === null ? '' : ` ${share > 0 && share < 0.005 ? '<1' : Math.round(share * 100)} %`;
  const tip = [d.name, typeof d.seconds === 'number' && Number.isFinite(d.seconds) ? `${d.seconds.toFixed(1)} s of work` : '']
    .filter(Boolean).join(', ');
  const ic = d.kind === 'gpu' ? 'zap' : d.kind === 'cpu' ? 'cpu' : '';
  return `<span class="badge badge-soft badge-sm gap-1"${tip ? ` title="${esc(tip)}"` : ''}>`
    + `${ic ? icon(ic, 'size-3') : ''}${esc(deviceLabel(d.kind) + pct)}</span>`;
}

function applyLimits() {
  for (const [key, input] of Object.entries(fieldInputs)) {
    const [lo, hi] = limitsOf(key);
    input.min = String(lo);
    input.max = String(hi);
  }
}

function onFieldInput(e) {
  const key = e.target.dataset.field;
  // While typing, only flag a complete number that is out of range; "-" or "" waits for the change event.
  const err = fieldError(key);
  const typing = e.type === 'input';
  const complete = e.target.value.trim() !== '' && !e.target.validity.badInput;
  showFieldError(key, err && (!typing || complete) ? err : null);
  if (FIELD_KEYS.includes(key)) afterPhysicsChange();
  else {
    persist();
    renderBlock();
  }
}

async function init() {
  els.theme.addEventListener('change', () => {
    storeWrite(THEME_KEY, els.theme.checked ? 'dark' : 'light');
    requestAnimationFrame(drawZonal);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (storeRead(THEME_KEY)) return;
    els.theme.checked = e.matches;
    requestAnimationFrame(drawZonal);
  });

  els.form.addEventListener('submit', (e) => {
    e.preventDefault();
    generate();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      generate();
    }
  });

  for (const input of Object.values(fieldInputs)) {
    input.addEventListener('input', onFieldInput);
    input.addEventListener('change', onFieldInput);
  }
  for (const [key, input] of Object.entries(optionInputs)) {
    input.addEventListener('change', () => {
      state.form.options[key] = input.checked;
      persist();
      renderBlock();
    });
  }
  els.preset.addEventListener('change', () => {
    if (els.preset.value) applyPreset(els.preset.value);
    else {
      state.form.preset = '';
      renderPresetMeta();
      persist();
    }
  });
  els.randomSeed.addEventListener('click', () => {
    els.seed.value = String(randomSeed());
    showFieldError('seed', null);
    persist();
    renderBlock();
    els.seed.focus();
  });
  els.widths.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-width]');
    if (btn) setWidth(Number(btn.dataset.width), false);
  });
  els.widths.addEventListener('keydown', (e) => {
    const list = state.defaults.widths;
    let i = list.indexOf(state.form.width);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') i = Math.min(list.length - 1, i + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') i = Math.max(0, i - 1);
    else return;
    e.preventDefault();
    setWidth(list[i], true);
  });

  els.compute.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-compute]');
    if (btn) chooseCompute(btn.dataset.compute);
  });
  els.compute.addEventListener('keydown', onComputeKey);
  for (const ev of ['pointerover', 'focusin']) {
    els.compute.addEventListener(ev, (e) => {
      const btn = e.target.closest('[data-compute]');
      if (btn) setComputeTip(btn.dataset.compute);
    });
  }

  for (const btn of $$('[role="tab"]', els.tabs)) btn.addEventListener('click', () => setTab(btn.dataset.tab, false));
  els.tabs.addEventListener('keydown', (e) => {
    const keys = Object.keys(TABS);
    let i = keys.indexOf(state.tab);
    if (e.key === 'ArrowRight') i = (i + 1) % keys.length;
    else if (e.key === 'ArrowLeft') i = (i - 1 + keys.length) % keys.length;
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = keys.length - 1;
    else return;
    e.preventDefault();
    setTab(keys[i], true);
  });
  for (const btn of $$('[data-view]')) btn.addEventListener('click', () => setView(btn.dataset.view));

  els.downloadButton.addEventListener('click', (e) => {
    if (!(state.result && state.result.downloads && state.result.downloads.length)) e.preventDefault();
  });
  document.addEventListener('click', (e) => {
    if (els.downloadBox.hasAttribute('open') && (!els.downloadBox.contains(e.target) || e.target.closest('a'))) {
      els.downloadBox.removeAttribute('open');
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && els.downloadBox.hasAttribute('open')) {
      els.downloadBox.removeAttribute('open');
      els.downloadButton.focus();
    }
  });

  els.emptyAction.addEventListener('click', () => {
    if (state.emptyAction === 'retry') start();
    else setTab('earth', false);
  });

  els.mapImg.addEventListener('pointermove', onMapMove);
  els.mapImg.addEventListener('pointerleave', () => setReadout(' '));
  els.globe.addEventListener('pointerdown', onGlobeDown);
  els.globe.addEventListener('pointermove', onGlobeMove);
  els.globe.addEventListener('pointerup', onGlobeUp);
  els.globe.addEventListener('pointercancel', onGlobeUp);
  els.globe.addEventListener('pointerleave', () => { if (!drag) setReadout(' '); });
  els.globe.addEventListener('dblclick', () => {
    state.globe = { ...GLOBE_HOME };
    requestGlobe(false);
  });
  els.globe.addEventListener('keydown', onGlobeKey);
  els.zonal.addEventListener('pointermove', onZonalMove);
  els.zonal.addEventListener('pointerleave', onZonalLeave);

  new ResizeObserver(() => {
    if (state.view === 'globe') requestGlobe(false);
  }).observe(els.stage);
  new ResizeObserver(() => drawZonal()).observe(els.zonal);

  renderTabs();
  renderViewButtons();
  initGpuGlobe(); // in the background: the globe is lit once it is ready, drawn in 2D until then
  await start();
}

async function start() {
  try {
    state.defaults = await api('/api/defaults');
    state.loadError = null;
  } catch (err) {
    state.loadError = err.message;
    toast(err.message);
    renderStage();
    return;
  }
  const d = state.defaults;
  els.version.textContent = d.version ? `v${d.version}` : '';
  buildPresets();
  buildWidths();
  buildCompute();
  applyLimits();
  restore();
  applyFormToDom();
  renderCompute();
  // A saved mode that is gone was replaced by Auto: remember that, so the sentence shows once.
  if (state.computeFallback) persist();
  for (const key of Object.keys(fieldInputs)) {
    const err = fieldError(key);
    if (err) showFieldError(key, err);
  }
  renderPresetMeta();
  renderWidthHint();
  renderTabs();
  renderViewButtons();
  scheduleDerive();
  renderBlock();
  renderStage();
}

init();
