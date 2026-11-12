// Cloud Map Generator in the browser: one Web Worker of the CPU pool.
//
// Every worker has its own instance of the WebAssembly module (no shared memory on a static host) and answers the
// engine adapter (engine.js) with one message per request: { id, op, args } in, { id, ok, result } or
// { id, ok: false, error } out. Big arrays travel as transferable buffers, never copied.
//
// The ops are the functions of rust/cloud-map-generator-wasm/src/lib.rs plus the band lifecycle: a worker keeps the band it was
// opened with between the passes (band_open, band_prepare, band_solve, band_compose), so only moments, thresholds and
// finished rows cross between workers.

import * as wasm from './pkg/cloud_map_generator_wasm.js';

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

  // Barriers and the answer.
  merge_moments: ({ flat }) => wasm.merge_moments(flat),
  smooth: ({ thr_raw }) => {
    const out = wasm.smooth_thresholds(thr_raw);
    return { result: out, transfer: [out.buffer] };
  },
  finalize: (a) => finished(wasm.finalize_maps(a.request, a.caps, a.id, a.coverage, a.kind, a.high_share, a.used, a.devices, a.note ?? undefined, a.engine_seconds, a.elapsed_before)),
  run_whole: (a) => finished(wasm.run_whole(a.request, a.caps, a.id, a.note ?? undefined, a.elapsed_before))
};

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
