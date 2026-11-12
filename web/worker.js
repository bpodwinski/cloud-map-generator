// Cloud Map Generator in the browser: one Web Worker of the CPU pool.
//
// Every worker has its own instance of the WebAssembly module (no shared memory on a static host) and answers the
// engine adapter (engine.js) with one message per request: { id, op, args } in, { id, ok, result } or
// { id, ok: false, error } out. Big arrays travel as transferable buffers, never copied.
//
// The ops are the functions of rust/cloud-map-generator-wasm/src/lib.rs plus the band lifecycle: a worker keeps the band it was
// opened with between the passes (band_open, band_prepare, band_solve, band_compose), so only moments, thresholds and
// finished rows cross between workers.

// The glue of the same build as the module the adapter compiled: the build stamp of this worker's URL (?v=..., set by
// scripts/build-web.mjs) goes on the import too, so a glue cached from an older release never meets a newer module (the
// names of their imports differ, and instantiating fails with "function import requires a callable").
const glue = import(new URL(`./pkg/cloud_map_generator_wasm.js${new URL(import.meta.url).search}`, import.meta.url).href);
let wasm = null;

let job = null;

// A browser tells a page little about the GPU: the vendor and the architecture (GPUAdapterInfo), and a description only on
// some. That is the adapter's name here ("NVIDIA Blackwell"), as the contract asks.
async function adapterName() {
  try {
    const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter && adapter.info;
    if (!info) return undefined;
    if (info.description) return info.description;
    const vendors = { nvidia: 'NVIDIA', amd: 'AMD', intel: 'Intel', apple: 'Apple', qualcomm: 'Qualcomm', arm: 'Arm', 'mesa': 'Mesa' };
    const word = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : '');
    const name = [vendors[info.vendor] || word(info.vendor), word(info.architecture)].filter(Boolean).join(' ');
    return name || undefined;
  } catch {
    return undefined;
  }
}

function takeJob() {
  if (!job) throw new Error('no band is open in this worker');
  return job;
}

const ops = {
  // args: { module } - the compiled WebAssembly.Module the adapter made once for every worker.
  init({ module }) {
    wasm.initSync({ module });
    wasm.init();
    return wasm.version();
  },
  defaults: ({ caps }) => wasm.defaults(caps),
  derive: ({ request }) => wasm.derive(request),
  plan: ({ request, caps }) => wasm.plan(request, caps),
  // The GPU: opens the WebGPU device in this worker, runs the double-single probe, compiles the pipeline and checks it with
  // a small map. Resolves to the JSON of the adapter facts, or rejects with the reason (a sentence for the visitor).
  async gpu_probe() {
    if (!self.navigator || !self.navigator.gpu) {
      throw new Error('This browser does not expose WebGPU to Web Workers (navigator.gpu is missing in them). Use a recent Chromium-based browser or Safari, or the native engine (cloud-map-generator serve).');
    }
    return wasm.gpu_probe(await adapterName());
  },
  // A whole map on the GPU, then the answer; needs gpu_probe to have succeeded in this worker.
  async run_gpu(a) {
    return finished(await wasm.run_gpu(a.request, a.caps, a.id, a.note ?? undefined, a.elapsed_before));
  },
  // Earth's reference measures, computed ahead of the first map.
  warm: () => {
    wasm.warm();
    return true;
  },
  reference_png: () => {
    const bytes = wasm.reference_png();
    return { result: bytes, transfer: [bytes.buffer] };
  },

  // Band lifecycle.
  band_open({ request, caps, r0, r1 }) {
    if (job) job.free();
    job = new wasm.BandJob(request, caps, r0, r1);
    return true;
  },
  band_prepare() {
    const moments = takeJob().prepare();
    return { result: moments, transfer: [moments.buffer] };
  },
  band_solve({ constants }) {
    const thr = takeJob().solve(constants);
    return { result: thr, transfer: [thr.buffer] };
  },
  band_compose({ thr }) {
    const j = takeJob();
    j.compose(thr);
    const out = { coverage: j.coverage(), kind: j.kind(), high_share: j.high_share() };
    j.free();
    job = null;
    return { result: out, transfer: [out.coverage.buffer, out.kind.buffer, out.high_share.buffer] };
  },
  band_close() {
    if (job) job.free();
    job = null;
    return true;
  },

  // The server's banded engine (src/large.rs): a run opened in each worker that takes part, the three passes per band on
  // this worker's CPU or its GPU, the two barriers, and the sink (the last worker) that writes the files.
  large_open({ request, caps, band_rows }) {
    wasm.large_open(request, caps, band_rows);
    return true;
  },
  async large_close() {
    wasm.large_close();
    await diskDrop();
    return true;
  },
  large_pass_a({ band }) {
    const m = wasm.large_pass_a(band);
    return { result: m, transfer: [m.buffer] };
  },
  large_pass_b({ band }) {
    const t = wasm.large_pass_b(band);
    return { result: t, transfer: [t.buffer] };
  },
  large_pass_c({ band }) {
    const rows = wasm.large_pass_c(band);
    return { result: rows, transfer: [rows.buffer] };
  },
  large_gpu_open() {
    wasm.large_gpu_open();
    return true;
  },
  large_gpu_close() {
    wasm.large_gpu_close();
    return true;
  },
  async large_gpu_pass_a({ band }) {
    const m = await wasm.large_gpu_pass_a(band);
    return { result: m, transfer: [m.buffer] };
  },
  async large_gpu_pass_b({ band }) {
    const t = await wasm.large_gpu_pass_b(band);
    return { result: t, transfer: [t.buffer] };
  },
  async large_gpu_pass_c({ band }) {
    const rows = await wasm.large_gpu_pass_c(band);
    return { result: rows, transfer: [rows.buffer] };
  },
  large_set_moments: ({ flat }) => {
    wasm.large_set_moments(flat);
    return true;
  },
  large_set_thresholds: ({ thr_raw, sensitivity }) => {
    wasm.large_set_thresholds(thr_raw, sensitivity);
    return true;
  },
  // The sink writes the two big files to disk when it can (see diskOpen); `dir` is the page's session folder.
  async large_sink_begin({ dir, id }) {
    await diskDrop();
    disk = await diskOpen(dir, id);
    wasm.large_sink_begin(!!disk);
    sinkNext = 0;
    sinkWaiting.clear();
    return { disk: !!disk };
  },
  // A unit's pass C: the band goes straight to the sink worker over this worker's port (never through the page's thread),
  // and the call returns once the sink has written it, so no unit runs more than one band ahead of the files.
  large_pass_c_send: ({ band }) => toSink(band, wasm.large_pass_c(band)),
  async large_gpu_pass_c_send({ band }) {
    return toSink(band, await wasm.large_gpu_pass_c(band));
  },
  // The page links the workers once: every unit gets a port to the sink, the sink the other end of each.
  connect({ port, role }) {
    if (role === 'self') {
      sinkPort = null;
      return true;
    }
    if (role === 'sink') {
      port.onmessage = (e) => sinkReceive(e.data, (msg) => port.postMessage(msg));
      return true;
    }
    sinkPort = port;
    port.onmessage = (e) => {
      const p = acks.get(e.data.reply);
      acks.delete(e.data.reply);
      if (p) (e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(true));
    };
    return true;
  },
  async large_finish(a) {
    const files = disk ? await diskFinish() : [];
    const out = finished(wasm.large_finish(a.id, a.used, a.devices, a.note ?? undefined, a.anomaly_min, a.anomaly_max, a.engine_seconds, a.elapsed_before));
    out.result.files.push(...files);
    out.result.disk = files.map((f) => f.path);
    return out;
  },

  // Barriers and the answer.
  merge_moments: ({ flat }) => wasm.merge_moments(flat),
  smooth: ({ thr_raw }) => {
    const out = wasm.smooth_thresholds(thr_raw);
    return { result: out, transfer: [out.buffer] };
  },
  finalize: (a) => finished(wasm.finalize_maps(a.request, a.caps, a.id, a.coverage, a.kind, a.high_share, a.used, a.devices, a.note ?? undefined, a.engine_seconds, a.elapsed_before)),
  run_whole: (a) => finished(wasm.run_whole(a.request, a.caps, a.id, a.note ?? undefined, a.elapsed_before))
};

// ---------------------------------------------------------------------------------------------------------
// Pass C between the workers: units post their bands to the sink, which writes them north to south as they complete.

let sinkPort = null; // a unit's port to the sink; null in the sink itself
let nextAck = 1;
const acks = new Map();
let sinkNext = 0;
const sinkWaiting = new Map();

function toSink(band, data) {
  return new Promise((resolve, reject) => {
    if (!sinkPort) {
      sinkReceive({ band, data }, (msg) => (msg.error ? reject(new Error(msg.error)) : resolve(true)));
      return;
    }
    const reply = nextAck++;
    acks.set(reply, { resolve, reject });
    sinkPort.postMessage({ band, data, reply }, [data.buffer]);
  });
}

// The sink: a band waits until every band north of it is written; each unit hears back when its own band is.
function sinkReceive({ band, data, reply }, respond) {
  sinkWaiting.set(band, { data, reply, respond });
  while (sinkWaiting.has(sinkNext)) {
    const w = sinkWaiting.get(sinkNext);
    sinkWaiting.delete(sinkNext);
    try {
      writeBand(sinkNext, w.data);
      w.respond({ reply: w.reply });
    } catch (err) {
      w.respond({ reply: w.reply, error: (err && err.message) || String(err) });
    }
    sinkNext++;
  }
}

function writeBand(band, data) {
  wasm.large_sink_band(band, data);
  if (disk) {
    const master = wasm.large_sink_take_master();
    disk.master.h.write(master, { at: disk.masterAt });
    disk.masterAt += master.length;
    const at = wasm.large_sink_r8_offset();
    disk.r8.h.write(wasm.large_sink_take_r8(), { at });
  }
}

// ---------------------------------------------------------------------------------------------------------
// The two big files of a banded map on disk: the origin private file system (no permission asked, a share of the disk),
// written through synchronous access handles, which only a worker has. The master is appended band by band; the R8 rows
// land at their place counted from the south (a seek, as the server's temporary file), then the R8 is read back in chunks
// through the module's gzip. The page gets the files as disk-backed File objects: a 1 GB master is never in memory.
// Without that file system (an old browser), the sink keeps them in memory.

let disk = null;
const CHUNK = 8 << 20;

async function diskOpen(dirName, id) {
  const opened = [];
  try {
    if (!self.navigator || !self.navigator.storage || !self.navigator.storage.getDirectory) return null;
    const root = await self.navigator.storage.getDirectory();
    const top = await root.getDirectoryHandle('cloud-map-generator', { create: true });
    const dir = await top.getDirectoryHandle(dirName, { create: true });
    const open = async (name) => {
      const f = await dir.getFileHandle(name, { create: true });
      if (!f.createSyncAccessHandle) throw new Error('no synchronous access handles');
      const h = await f.createSyncAccessHandle();
      h.truncate(0);
      const file = { f, h, name };
      opened.push(file);
      return file;
    };
    return { dir, master: await open(`${id}_master16.png`), r8: await open(`${id}_coverage.r8`), masterAt: 0, id };
  } catch (err) {
    for (const o of opened) o.h.close();
    return null;
  }
}

// After the last band: the master's last bytes, the R8 compressed into its .gz, the raw R8 removed. Returns the files.
async function diskFinish() {
  const d = disk;
  const tail = wasm.large_sink_end_master();
  d.master.h.write(tail, { at: d.masterAt });
  d.master.h.flush();
  d.master.h.close();
  const gzName = `${d.id}_coverage.r8.gz`;
  const gzHandle = await d.dir.getFileHandle(gzName, { create: true });
  const gz = await gzHandle.createSyncAccessHandle();
  gz.truncate(0);
  let gzAt = 0;
  const put = (bytes) => {
    gz.write(bytes, { at: gzAt });
    gzAt += bytes.length;
  };
  wasm.large_gz_begin();
  const size = d.r8.h.getSize();
  const buf = new Uint8Array(CHUNK);
  for (let at = 0; at < size; ) {
    const n = d.r8.h.read(buf, { at });
    if (n <= 0) throw new Error('the raw R8 file is short');
    put(wasm.large_gz_feed(buf.subarray(0, n)));
    at += n;
  }
  put(wasm.large_gz_finish());
  gz.flush();
  gz.close();
  d.r8.h.close();
  await d.dir.removeEntry(d.r8.name).catch(() => {});
  disk = null;
  return [
    { name: 'master16.png', type: 'image/png', file: await d.master.f.getFile(), path: d.master.name },
    { name: 'coverage.r8.gz', type: 'application/gzip', file: await gzHandle.getFile(), path: gzName }
  ];
}

// A run that failed or was replaced: its handles closed, its files removed.
async function diskDrop() {
  const d = disk;
  disk = null;
  if (!d) return;
  for (const o of [d.master, d.r8]) {
    try {
      o.h.close();
    } catch {}
    await d.dir.removeEntry(o.name).catch(() => {});
  }
}

// A wasm `Finished` -> { json, files: [{ name, type, bytes }] }, with the bytes transferred.
function finished(done) {
  const files = [];
  for (let i = 0; i < done.file_count(); i++) {
    files.push({ name: done.file_name(i), type: done.file_type(i), bytes: done.file_bytes(i) });
  }
  const json = done.json();
  const engine_seconds = done.engine_seconds();
  done.free();
  return { result: { json, engine_seconds, files }, transfer: files.map((f) => f.bytes.buffer) };
}

self.onmessage = async (event) => {
  const { id, op, args } = event.data;
  try {
    if (!wasm) wasm = await glue;
    const out = await ops[op](args || {});
    if (out && typeof out === 'object' && 'transfer' in out && 'result' in out) {
      self.postMessage({ id, ok: true, result: out.result }, out.transfer);
    } else {
      self.postMessage({ id, ok: true, result: out });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: typeof err === 'string' ? err : (err && err.message) || String(err) });
  }
};
