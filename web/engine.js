// Cloud Map Generator in the browser: the engine adapter.
//
// The page (ui/static/app.js, unchanged) talks to `/api/*` with fetch. `cloud-map-generator serve` answers those calls over HTTP;
// on a static host there is no server, so this script, loaded before the page, takes the `/api/*` calls itself and
// answers them from the in-browser engine, with the same JSON, the same status codes and the same messages:
//
//   GET  /api/defaults                      presets, widths, limits, estimates, the `compute` capabilities
//   POST /api/derive                        the physics numbers of a body
//   POST /api/generate                      one map: the maps, the files, the metrics against Earth, the `compute` block
//   GET  /api/result/<id>/<name>            a file of a kept result (the last 8)
//   GET  /api/reference/coverage.png        Earth's reference map
//
// The engine: a pool of Web Workers, each with its own WebAssembly instance (rust/cloud-map-generator-wasm). A static host
// cannot set the headers that give a page SharedArrayBuffer, so there are no threads inside WebAssembly: a map is cut in
// bands of rows, one band per worker, and the workers exchange only small vectors and finished rows by message passing
// (see rust/cloud-map-generator-wasm/src/band.rs for the passes and the barriers). The GPU path is in gpu.js.
//
// Nothing here makes a network request apart from loading this site's own files: probing and generating are local.

(function () {
  'use strict';

  const BASE = new URL('.', document.currentScript.src);
  // The build stamp of this script's URL (?v=..., scripts/build-web.mjs): on the worker and the module too, so one visit
  // never mixes files of two releases from the browser cache.
  const STAMP = new URL(document.currentScript.src).search;
  const KEEP = 8;
  const MAX_WORKERS = 16;
  const realFetch = window.fetch.bind(window);

  const jsonResponse = (status, payload) =>
    new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } });
  const failure = (status, message) => jsonResponse(status, { error: message });

  // 12 lowercase hex digits, like the id the server gives a result.
  function newId() {
    const bytes = new Uint8Array(6);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  }

  function concat(Type, parts) {
    const out = new Type(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------------------------
  // The worker pool

  class Pool {
    constructor(size, module) {
      this.size = size;
      this.nextId = 1;
      this.pending = new Map();
      this.workers = Array.from({ length: size }, (_, i) => {
        const w = new Worker(new URL(`worker.js${STAMP}`, BASE), { type: 'module', name: `cloud-map-generator-${i}` });
        w.onmessage = (e) => this.settle(e.data);
        w.onerror = (e) => this.crash(w, e.message || 'a worker failed to start');
        return w;
      });
      this.module = module;
    }

    settle({ id, ok, result, error }) {
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      if (ok) p.resolve(result);
      else p.reject(new Error(error));
    }

    crash(worker, message) {
      for (const [id, p] of this.pending) {
        if (p.worker === worker) {
          this.pending.delete(id);
          p.reject(new Error(`A worker failed: ${message}`));
        }
      }
    }

    call(index, op, args, transfer) {
      return new Promise((resolve, reject) => {
        const id = this.nextId++;
        const worker = this.workers[index];
        this.pending.set(id, { resolve, reject, worker });
        worker.postMessage({ id, op, args }, transfer || []);
      });
    }

    all(op, argsOf, transferOf) {
      return Promise.all(this.workers.map((_, i) => this.call(i, op, argsOf ? argsOf(i) : undefined, transferOf ? transferOf(i) : undefined)));
    }

    async start() {
      await this.all('init', () => ({ module: this.module }));
    }
  }

  // ---------------------------------------------------------------------------------------------------------
  // The engine

  class Engine {
    constructor() {
      this.busy = false;
      this.results = new Map();
      this.ready = this.start();
    }

    // The page's folder for the big files of banded maps, on the origin private file system: one per open page, held by a
    // Web Lock for the page's life, so a page removes the folders of pages that are gone and never another open page's.
    diskSession() {
      this.diskDir = `session-${newId()}`;
      if (!navigator.storage || !navigator.storage.getDirectory || !navigator.locks) return;
      const mine = this.diskDir;
      navigator.locks.request(`cloud-map-generator:${mine}`, () => new Promise(() => {})).catch(() => {});
      (async () => {
        const top = await (await navigator.storage.getDirectory()).getDirectoryHandle('cloud-map-generator', { create: true });
        for await (const name of top.keys()) {
          if (name === mine) continue;
          await navigator.locks.request(`cloud-map-generator:${name}`, { ifAvailable: true }, async (lock) => {
            if (lock) await top.removeEntry(name, { recursive: true }).catch(() => {});
          });
        }
      })().catch(() => {});
    }

    async diskRemove(names) {
      if (!names || !names.length) return;
      try {
        const top = await (await navigator.storage.getDirectory()).getDirectoryHandle('cloud-map-generator');
        const dir = await top.getDirectoryHandle(this.diskDir);
        for (const n of names) await dir.removeEntry(n).catch(() => {});
      } catch {}
    }

    async start() {
      this.diskSession();
      const threads = Math.max(1, navigator.hardwareConcurrency || 4);
      // The page's own thread keeps the interface alive: one hardware thread less for the workers (at least one).
      const size = Math.max(1, Math.min(threads - 1, MAX_WORKERS));
      const wasmUrl = new URL(`pkg/cloud_map_generator_wasm_bg.wasm${STAMP}`, BASE);
      let module;
      try {
        module = await WebAssembly.compileStreaming(realFetch(wasmUrl));
      } catch (err) {
        const bytes = await (await realFetch(wasmUrl)).arrayBuffer();
        module = await WebAssembly.compile(bytes);
      }
      this.pool = new Pool(size, module);
      await this.pool.start();
      // The last worker does the CPU-side stages after the maps (encodings, metrics against Earth): it computes Earth's
      // reference measures now, in the background, while the page is being read. With one worker the first map pays.
      this.finalizer = size - 1;
      const gpu = await this.probeGpu();
      // Earth's reference measures (about two seconds in WebAssembly), ahead of the first map.
      if (size > 1) this.pool.call(this.finalizer, 'warm').catch(() => {});
      this.caps = { threads, workers: size, gpu: gpu.info, gpu_reason: gpu.reason };
      this.capsText = JSON.stringify(this.caps);
    }

    // The GPU: WebGPU through wgpu, in the last worker (rust/cloud-map-generator-wasm/src/gpu.rs). A mode is offered only when the probe has
    // just run: the adapter opened, the double-single probe done, the kernels compiled and a small map made. Otherwise the mode
    // is listed with the reason, which is a sentence for the visitor.
    async probeGpu() {
      if (!navigator.gpu) {
        return {
          info: null,
          reason: 'This browser does not expose WebGPU (navigator.gpu is missing). Use a Chromium-based browser or a recent Safari, or the native engine (cloud-map-generator serve).'
        };
      }
      try {
        const facts = JSON.parse(await this.pool.call(this.finalizer, 'gpu_probe'));
        this.gpuFacts = facts;
        return { info: { name: facts.name, backend: facts.backend, df64: facts.df64, float64: false }, reason: null };
      } catch (err) {
        return { info: null, reason: String((err && err.message) || err) };
      }
    }

    // --- routes

    async handle(method, path, bodyText, signal) {
      try {
        await this.ready;
      } catch (err) {
        return failure(500, `The in-browser engine could not start: ${err.message || err}`);
      }
      const response = await this.route(method, path, bodyText);
      if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
      return response;
    }

    async route(method, path, bodyText) {
      if (method === 'GET' && path === '/api/defaults') {
        return new Response(await this.pool.call(0, 'defaults', { caps: this.capsText }), { headers: { 'Content-Type': 'application/json' } });
      }
      if (method === 'POST' && path === '/api/derive') {
        try {
          return new Response(await this.pool.call(0, 'derive', { request: bodyText }), { headers: { 'Content-Type': 'application/json' } });
        } catch (err) {
          return failure(400, err.message);
        }
      }
      if (method === 'POST' && path === '/api/generate') return this.generate(bodyText);
      if (method === 'GET' && path === '/api/reference/coverage.png') {
        const bytes = await this.pool.call(0, 'reference_png');
        return new Response(new Blob([bytes], { type: 'image/png' }));
      }
      const m = /^\/api\/result\/([0-9a-f]{12})\/([A-Za-z0-9_.]+)$/.exec(path);
      if (method === 'GET' && m) {
        const kept = this.results.get(m[1]);
        const blob = kept && kept.files.get(m[2]);
        if (!blob) return failure(404, `no such result (results are kept in memory, the last ${KEEP}): generate again`);
        return new Response(blob);
      }
      return failure(404, `not found: ${path}`);
    }

    // --- one generation

    async generate(requestText) {
      if (this.busy) return failure(429, 'a generation is already running');
      this.busy = true;
      try {
        const t0 = performance.now();
        const since = () => (performance.now() - t0) / 1000;
        // Validate first: a refused request costs nothing and keeps nothing.
        let plan;
        try {
          plan = JSON.parse(await this.pool.call(0, 'plan', { request: requestText, caps: this.capsText }));
        } catch (err) {
          return failure(400, err.message);
        }
        const id = newId();
        let done;
        try {
          if (plan.large) done = await this.generateLarge(plan, requestText, id, since);
          else done = plan.used === 'gpu' ? await this.generateGpu(plan, requestText, id, since) : await this.generateCpu(plan, requestText, id, since);
        } catch (err) {
          await this.pool.all('band_close').catch(() => {});
          await this.pool.all('large_close').catch(() => {});
          await this.pool.all('large_gpu_close').catch(() => {});
          return failure(500, err.message || String(err));
        }
        return jsonResponse(200, this.keep(id, done));
      } finally {
        this.busy = false;
      }
    }

    // Stores the files of a result and answers its JSON: images stay /api/result URLs (the page fetches them back), the
    // downloads become blob: links (an anchor cannot be intercepted) that carry their file name.
    keep(id, done) {
      const answer = JSON.parse(done.json);
      this.lastEngineSeconds = done.engine_seconds;
      // A file on disk (a banded map's master and R8) stays there: the Blob only refers to it.
      const files = new Map(done.files.map((f) => [f.name, new Blob([f.file || f.bytes], { type: f.type })]));
      const urls = [];
      for (const d of answer.downloads) {
        const name = d.href.split('/').pop();
        d.href = URL.createObjectURL(files.get(name));
        urls.push(d.href);
      }
      this.results.set(id, { files, urls, disk: done.disk || [] });
      while (this.results.size > KEEP) {
        const [oldest] = this.results.keys();
        const gone = this.results.get(oldest);
        for (const u of gone.urls) URL.revokeObjectURL(u);
        this.results.delete(oldest);
        this.diskRemove(gone.disk);
      }
      return answer;
    }

    async generateCpu(plan, request, id, since) {
      const pool = this.pool;
      const caps = this.capsText;
      const device = (seconds) => JSON.stringify([{ kind: 'cpu', name: null, seconds, share: 1 }]);
      if (!plan.banded) {
        // The surface terms couple the whole map: one worker makes it (rust/cloud-map-generator-wasm/src/band.rs, last paragraph).
        const done = await pool.call(this.finalizer, 'run_whole', { request, caps, id, note: plan.note, elapsed_before: since() });
        return done;
      }
      const { coverage, kind, highShare, phases } = await this.makeBands(plan.width, request, pool.size, since);
      const engineSeconds = since();
      this.lastTimings = phases;
      return pool.call(
        this.finalizer,
        'finalize',
        {
          request,
          caps,
          id,
          coverage,
          kind,
          high_share: highShare,
          used: 'cpu',
          devices: device(engineSeconds),
          note: plan.note,
          engine_seconds: engineSeconds,
          elapsed_before: engineSeconds
        },
        [coverage.buffer, kind.buffer, highShare.buffer]
      );
    }


    // The banded CPU pipeline (rust/cloud-map-generator-wasm/src/band.rs): the three maps of `width` made by `bands` workers, one band each.
    // `since` is the request's clock. Also the check that the band count changes nothing but float rounding: call it with 1.
    async makeBands(width, request, bands, since) {
      const pool = this.pool;
      const caps = this.capsText;
      const height = width / 2;
      bands = Math.min(bands, pool.size, height);
      const cuts = Array.from({ length: bands }, (_, i) => [Math.floor((i * height) / bands), Math.floor(((i + 1) * height) / bands)]);
      const at = (i) => cuts[i];
      const bandWorkers = Array.from({ length: bands }, (_, i) => i);
      const each = (op, argsOf, transferOf) =>
        Promise.all(bandWorkers.map((i) => pool.call(i, op, argsOf ? argsOf(i) : undefined, transferOf ? transferOf(i) : undefined)));

      // The seconds of each phase of the last run, for whoever measures the engine (window.cloudgenEngine.lastTimings).
      const phases = {};
      let mark = since();
      const lap = (name) => {
        const now = since();
        phases[name] = Math.round((now - mark) * 1000) / 1000;
        mark = now;
      };
      // Pass A: every band draws the plans, back-traces, and makes the raw fields; barrier 1 merges their moments.
      await each('band_open', (i) => ({ request, caps, r0: at(i)[0], r1: at(i)[1] }));
      lap('open');
      const moments = await each('band_prepare');
      lap('prepare');
      const constants = await pool.call(0, 'merge_moments', { flat: concat(Float64Array, moments) });
      lap('merge');
      // Pass B: standardize and solve the thresholds of the rows; barrier 2 smooths them in latitude.
      const rawParts = await each('band_solve', () => ({ constants }));
      lap('solve');
      const thr = await pool.call(0, 'smooth', { thr_raw: concat(Float64Array, rawParts) });
      lap('smooth');
      // Pass C: the maps.
      const parts = await each('band_compose', () => ({ thr }));
      lap('compose');
      const coverage = concat(Float32Array, parts.map((p) => p.coverage));
      const kind = concat(Float32Array, parts.map((p) => p.kind));
      const highShare = concat(Float32Array, parts.map((p) => p.high_share));
      lap('assemble');
      return { coverage, kind, highShare, phases };
    }

    // The server's banded engine (rust/cloud-map-generator-wasm/src/large.rs): 8192 wide, or CPU + GPU at any width. Every
    // unit (a CPU worker, the GPU worker) takes the next band of a pass when it is free, as the server's scheduler does with
    // its threads; between the passes every worker gets the moments, then the thresholds, of all bands; the finished bands
    // go north to south to the last worker, which writes the files and the answer.
    async generateLarge(plan, request, id, since) {
      const pool = this.pool;
      const caps = this.capsText;
      const bands = plan.bands;
      const useGpu = plan.used === 'gpu' || plan.used === 'hybrid';
      if (useGpu && !this.caps.gpu) throw new Error('The GPU is not available in this page.');
      const gpuWorker = useGpu ? this.finalizer : -1;
      const sink = this.finalizer;
      const units = [];
      if (plan.used !== 'gpu') for (let i = 0; i < pool.size; i++) if (i !== gpuWorker) units.push({ worker: i, gpu: false });
      if (useGpu) units.push({ worker: gpuWorker, gpu: true });
      const open = [...new Set([...units.map((u) => u.worker), sink])];
      await Promise.all(open.map((i) => pool.call(i, 'large_open', { request, caps, band_rows: plan.band_rows })));
      if (useGpu) await pool.call(gpuWorker, 'large_gpu_open');
      const t0 = since();

      const rowsOf = (b) => Math.min(plan.width / 2, (b + 1) * plan.band_rows) - b * plan.band_rows;
      const work = new Map(units.map((u) => [u, { rows: 0, seconds: 0 }]));
      // One pass: each unit pulls the next band until none is left; `onBand` takes each result as it comes.
      const pass = async (op, onBand) => {
        let next = 0;
        await Promise.all(
          units.map(async (u) => {
            for (let b = next++; b < bands; b = next++) {
              const start = performance.now();
              const out = await pool.call(u.worker, u.gpu ? `large_gpu_${op}` : `large_${op}`, { band: b });
              const w = work.get(u);
              w.seconds += (performance.now() - start) / 1000;
              w.rows += rowsOf(b);
              await onBand(b, out);
            }
          })
        );
      };

      const moments = new Array(bands);
      await pass('pass_a', (b, m) => { moments[b] = m; });
      const flat = concat(Float64Array, moments);
      await Promise.all(open.map((i) => pool.call(i, 'large_set_moments', { flat })));

      const parts = new Array(bands);
      await pass('pass_b', (b, p) => { parts[b] = p; });
      let lo = Infinity;
      let hi = -Infinity;
      const thr = [];
      const sens = [];
      for (const p of parts) {
        const n = p[2];
        const ns = p[3];
        if (Number.isFinite(p[0])) lo = Math.min(lo, p[0]);
        if (Number.isFinite(p[1])) hi = Math.max(hi, p[1]);
        thr.push(p.subarray(4, 4 + n));
        sens.push(p.subarray(4 + n, 4 + n + ns));
      }
      const thrRaw = concat(Float64Array, thr);
      const sensitivity = concat(Float64Array, sens);
      await Promise.all(open.map((i) => pool.call(i, 'large_set_thresholds', { thr_raw: thrRaw, sensitivity })));

      // Pass C: the bands reach the sink in order, whatever order they are made in.
      await pool.call(sink, 'large_sink_begin', { dir: this.diskDir, id });
      const ready = new Map();
      let toSink = 0;
      let chain = Promise.resolve();
      await pass('pass_c', (b, data) => {
        ready.set(b, data);
        while (ready.has(toSink)) {
          const band = toSink++;
          const rows = ready.get(band);
          ready.delete(band);
          chain = chain.then(() => pool.call(sink, 'large_sink_band', { band, data: rows }, [rows.buffer]));
        }
      });
      await chain;
      const engineSeconds = since() - t0;

      // What each kind of device did: its share of the rows of the three passes, and how long it worked.
      const total = [...work.values()].reduce((n, w) => n + w.rows, 0) || 1;
      const devices = [];
      for (const kind of ['gpu', 'cpu']) {
        const mine = units.filter((u) => u.gpu === (kind === 'gpu')).map((u) => work.get(u));
        if (!mine.length) continue;
        devices.push({
          kind,
          name: kind === 'gpu' ? (this.caps.gpu && this.caps.gpu.name) || null : null,
          seconds: Math.max(...mine.map((w) => w.seconds)),
          share: mine.reduce((n, w) => n + w.rows, 0) / total
        });
      }
      const done = await pool.call(sink, 'large_finish', {
        id,
        used: plan.used,
        devices: JSON.stringify(devices),
        note: plan.note,
        anomaly_min: Number.isFinite(lo) ? lo : NaN,
        anomaly_max: Number.isFinite(hi) ? hi : NaN,
        engine_seconds: engineSeconds,
        elapsed_before: since()
      });
      await Promise.all(open.filter((i) => i !== sink).map((i) => pool.call(i, 'large_close')));
      if (useGpu) await pool.call(gpuWorker, 'large_gpu_close');
      return done;
    }

    async generateGpu(plan, request, id, since) {
      if (!this.caps.gpu) throw new Error('The GPU is not available in this page.');
      // The worker that opened the device makes the map and the answer: nothing is copied between workers.
      return this.pool.call(this.finalizer, 'run_gpu', { request, caps: this.capsText, id, note: plan.note, elapsed_before: since() });
    }
  }

  // ---------------------------------------------------------------------------------------------------------
  // Take over `/api/*`

  const engine = new Engine();
  window.cloudgenEngine = engine;

  window.fetch = function (input, init) {
    const request = typeof input === 'string' || input instanceof URL ? null : input;
    const url = new URL(request ? request.url : String(input), location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return realFetch(input, init);
    const method = ((init && init.method) || (request && request.method) || 'GET').toUpperCase();
    const signal = (init && init.signal) || (request && request.signal) || null;
    const body = init && typeof init.body === 'string' ? init.body : '';
    return engine.handle(method, url.pathname, body, signal);
  };
})();
