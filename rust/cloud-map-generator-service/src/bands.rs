//! The band scheduler: the three passes of `cloud_map_generator_core::banded` run by one or several workers (a CPU worker, a GPU
//! worker), each pulling the next band when it is free, with the throughput each worker measured rebalancing the tail.
//!
//! The contract is `docs/compute-modes.html` (hybrid): the units pull from one queue of bands (work pulling, no central
//! assignment); a band is a pure function of its position, so a band made by either device joins its neighbors without a
//! seam; a worker that fails hands its band back and the others finish the map (the note says so); the shares are the rows
//! each device completed in passes A to C over the total.
//!
//! The rebalancing rule: a worker takes the next band only when its own measured time per band is not longer than what the
//! other workers, at their measured rates, need to finish every band still unassigned. Without it the slower device would
//! take the last band of a pass and the whole run would wait for it.

use cloud_map_generator_core::banded::{
    BandMoments, BandResult, BandSink, BandThresholds, Banded, BandedDone, BandedError, Pass, Progress,
};
use cloud_map_generator_core::generate::NormStats;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

/// One device's way of running the passes on a band. `Err` is a device failure: the scheduler gives the band to
/// another worker (and stops using this one), or fails the run when there is none.
pub trait BandCompute: Send + Sync {
    fn kind(&self) -> crate::engine::DeviceKind;
    fn name(&self) -> Option<String>;
    fn pass_a(&self, band: usize) -> Result<BandMoments, String>;
    fn pass_b(&self, band: usize, norm: &NormStats) -> Result<BandThresholds, String>;
    fn pass_c(&self, band: usize, norm: &NormStats, thr: &[f64], sens: &[f64]) -> Result<BandResult, String>;
}

/// The CPU worker: `cloud_map_generator_core::banded` on every thread of the pool.
/// The CPU worker. `threads`: run each band in a private pool of that many threads (hybrid leaves cores to the GPU's
/// driver thread and to the file writer, which share the process-wide pool); `None`: the process-wide pool.
pub struct CpuBands<'a, 'b>(pub &'a Banded<'b>, pub Option<usize>);

impl CpuBands<'_, '_> {
    fn on_pool<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        match self.1 {
            Some(n) => cloud_map_generator_core::par::with_threads(n, f),
            None => f(),
        }
    }
}

impl BandCompute for CpuBands<'_, '_> {
    fn kind(&self) -> crate::engine::DeviceKind {
        crate::engine::DeviceKind::Cpu
    }
    fn name(&self) -> Option<String> {
        crate::engine::cpu_name()
    }
    fn pass_a(&self, band: usize) -> Result<BandMoments, String> {
        Ok(self.on_pool(|| self.0.pass_a(band)))
    }
    fn pass_b(&self, band: usize, norm: &NormStats) -> Result<BandThresholds, String> {
        Ok(self.on_pool(|| self.0.pass_b(band, norm)))
    }
    fn pass_c(&self, band: usize, norm: &NormStats, thr: &[f64], sens: &[f64]) -> Result<BandResult, String> {
        Ok(self.on_pool(|| self.0.pass_c(band, norm, thr, sens)))
    }
}

/// What a worker did in a run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkerReport {
    /// Wall seconds the worker spent inside its band computations.
    pub seconds: f64,
    /// Rows it completed, summed over the three passes.
    pub rows: usize,
    /// Why it stopped early (a device failure), when it did.
    pub lost: Option<String>,
}

struct State<T> {
    next: usize,
    requeue: VecDeque<usize>,
    pending: BTreeMap<usize, T>,
    delivered: usize,
    done: usize,
    alive: Vec<bool>,
    /// Per worker, in this pass: (seconds, bands) completed.
    rate: Vec<(f64, usize)>,
    error: Option<BandedError>,
}

/// Runs one pass over `bands` bands. `work(worker, band)` computes a band; `deliver(band, value)` receives the values:
/// in band order when `ordered` (a sink), as they come otherwise.
#[allow(clippy::too_many_arguments)]
fn run_pass<T: Send>(
    pass: Pass,
    bands: usize,
    workers: &[&dyn BandCompute],
    reports: &Mutex<Vec<WorkerReport>>,
    rates: &Mutex<Vec<(f64, usize)>>,
    rows_of: &(dyn Fn(usize) -> usize + Sync),
    work: &(dyn Fn(&dyn BandCompute, usize) -> Result<T, String> + Sync),
    ordered: bool,
    deliver: &mut (dyn FnMut(usize, T) -> Result<(), String> + Send),
    progress: &(dyn Fn(Progress) -> bool + Sync),
) -> Result<(), BandedError> {
    let state = Mutex::new(State::<T> {
        next: 0,
        requeue: VecDeque::new(),
        pending: BTreeMap::new(),
        delivered: 0,
        done: 0,
        alive: vec![true; workers.len()],
        // The throughput measured in the earlier passes is the starting point (a device keeps its speed from pass to pass).
        rate: rates.lock().unwrap_or_else(|p| p.into_inner()).clone(),
        error: None,
    });
    let wake = Condvar::new();
    let deliver = Mutex::new(deliver);
    let window = workers.len() * 2 + 1;
    if !progress(Progress { pass, band: 0, bands }) {
        return Err(BandedError::Cancelled);
    }
    std::thread::scope(|scope| {
        for (w, &worker) in workers.iter().enumerate() {
            let (state, wake, deliver) = (&state, &wake, &deliver);
            scope.spawn(move || loop {
                // Take the next band, or stop.
                let band = {
                    let mut st = state.lock().unwrap_or_else(|p| p.into_inner());
                    loop {
                        if st.error.is_some() || st.done >= bands || !st.alive[w] {
                            wake.notify_all();
                            return;
                        }
                        let unassigned = (bands - st.next) + st.requeue.len();
                        if unassigned == 0 {
                            // Everything is taken; a failure elsewhere may hand a band back: wait for that or the end.
                            st = wake.wait(st).unwrap_or_else(|p| p.into_inner());
                            continue;
                        }
                        if ordered && st.next - st.delivered >= window && st.requeue.is_empty() {
                            st = wake.wait(st).unwrap_or_else(|p| p.into_inner());
                            continue;
                        }
                        // The measured-throughput rule. A worker with no measure yet takes a band only while the others
                        // could not finish what is left without it (a cold slow device must not get the tail of a pass).
                        let (secs, count) = st.rate[w];
                        if count == 0 && workers.len() > 1 && unassigned <= workers.len() {
                            let warm_peer = st.rate.iter().enumerate().any(|(o, &(_, c))| o != w && st.alive[o] && c > 0);
                            if warm_peer {
                                return;
                            }
                        }
                        if count > 0 {
                            let mut others = 0.0_f64;
                            let mut unknown = false;
                            for (o, &(s, c)) in st.rate.iter().enumerate() {
                                if o != w && st.alive[o] {
                                    if c == 0 {
                                        unknown = true;
                                    } else {
                                        others += c as f64 / s.max(1e-9);
                                    }
                                }
                            }
                            if !unknown && others > 0.0 && secs / count as f64 > unassigned as f64 / others {
                                return;
                            }
                        }
                        break st.requeue.pop_front().unwrap_or_else(|| {
                            let b = st.next;
                            st.next += 1;
                            b
                        });
                    }
                };
                let t0 = Instant::now();
                let result = work(worker, band);
                let spent = t0.elapsed().as_secs_f64();
                let mut st = state.lock().unwrap_or_else(|p| p.into_inner());
                match result {
                    Ok(value) => {
                        st.rate[w].0 += spent;
                        st.rate[w].1 += 1;
                        {
                            let mut r = reports.lock().unwrap_or_else(|p| p.into_inner());
                            r[w].seconds += spent;
                            r[w].rows += rows_of(band);
                        }
                        st.pending.insert(band, value);
                        // Deliver what is ready: in order for a sink, at once otherwise.
                        loop {
                            let key = if ordered { Some(st.delivered) } else { st.pending.keys().next().copied() };
                            let Some(k) = key else { break };
                            let Some(v) = st.pending.remove(&k) else { break };
                            if ordered {
                                st.delivered += 1;
                            }
                            let outcome = (deliver.lock().unwrap_or_else(|p| p.into_inner()))(k, v);
                            if let Err(e) = outcome {
                                st.error = Some(BandedError::Sink(e));
                                break;
                            }
                        }
                        st.done += 1;
                        let done = st.done;
                        if st.error.is_none() && !progress(Progress { pass, band: done, bands }) {
                            st.error = Some(BandedError::Cancelled);
                        }
                    }
                    Err(why) => {
                        st.alive[w] = false;
                        st.requeue.push_back(band);
                        let mut r = reports.lock().unwrap_or_else(|p| p.into_inner());
                        r[w].lost = Some(why);
                        if !st.alive.iter().any(|&a| a) {
                            let why = r[w].lost.clone().unwrap_or_default();
                            st.error = Some(BandedError::Sink(format!("every compute device failed: {why}")));
                        }
                    }
                }
                wake.notify_all();
            });
        }
    });
    let st = state.into_inner().unwrap_or_else(|p| p.into_inner());
    *rates.lock().unwrap_or_else(|p| p.into_inner()) = st.rate.clone();
    match st.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The whole banded run over `workers`, the bands of pass C handed to `sink` in order. Returns the sidecar numbers
/// and what each worker did.
pub fn run_bands(
    banded: &Banded<'_>,
    workers: &[&dyn BandCompute],
    sink: &mut dyn BandSink,
    progress: &(dyn Fn(Progress) -> bool + Sync),
) -> Result<(BandedDone, Vec<WorkerReport>), BandedError> {
    assert!(!workers.is_empty(), "a run needs at least one worker");
    let bands = banded.bands();
    let reports = Mutex::new(vec![WorkerReport::default(); workers.len()]);
    let rates = Mutex::new(vec![(0.0_f64, 0usize); workers.len()]);
    let rows_of = |band: usize| {
        let (a, b) = banded.band_range(band);
        b - a
    };

    // Pass A.
    let mut moments: Vec<Option<BandMoments>> = (0..bands).map(|_| None).collect();
    run_pass(
        Pass::A,
        bands,
        workers,
        &reports,
        &rates,
        &rows_of,
        &|w, i| w.pass_a(i),
        false,
        &mut |i, m| {
            moments[i] = Some(m);
            Ok(())
        },
        progress,
    )?;
    let norm = banded.norm_from(&moments.into_iter().map(|m| m.expect("pass A ran every band")).collect::<Vec<_>>());

    // Pass B.
    let mut results: Vec<Option<BandThresholds>> = (0..bands).map(|_| None).collect();
    run_pass(
        Pass::B,
        bands,
        workers,
        &reports,
        &rates,
        &rows_of,
        &|w, i| w.pass_b(i, &norm),
        false,
        &mut |i, b| {
            results[i] = Some(b);
            Ok(())
        },
        progress,
    )?;
    let (mut thr_raw, mut sens) = (Vec::with_capacity(banded.height), Vec::with_capacity(banded.height));
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for b in results.into_iter().map(|b| b.expect("pass B ran every band")) {
        thr_raw.extend_from_slice(&b.thr_raw);
        sens.extend_from_slice(&b.sensitivity);
        lo = lo.min(b.anomaly_min);
        hi = hi.max(b.anomaly_max);
    }
    let thr = banded.smooth(&thr_raw);

    // Pass C. The files are written by a thread of their own, so a worker that has finished a band starts the next
    // one at once instead of waiting for the PNG and the gzip: the bands reach the sink in order through a bounded queue.
    sink.begin(&banded.info()).map_err(BandedError::Sink)?;
    {
        let (tx, rx) = std::sync::mpsc::sync_channel::<BandResult>(workers.len() * 2 + 2);
        let writer_sink = &mut *sink;
        let (pass, wrote) = std::thread::scope(|scope| {
            let writer = scope.spawn(move || -> Result<(), String> {
                for r in rx {
                    writer_sink.band(&r.as_out())?;
                }
                Ok(())
            });
            let pass = run_pass(
                Pass::C,
                bands,
                workers,
                &reports,
                &rates,
                &rows_of,
                &|w, i| w.pass_c(i, &norm, &thr, &sens),
                true,
                &mut |_, r: BandResult| tx.send(r).map_err(|_| "the file writer stopped".to_string()),
                progress,
            );
            drop(tx);
            (pass, writer.join().unwrap_or_else(|_| Err("the file writer panicked".to_string())))
        });
        wrote.map_err(BandedError::Sink)?;
        pass?;
    }
    let meta = banded.meta(banded.uses_surface().then_some([lo, hi]));
    let done = BandedDone { meta, thresholds: thr, norm };
    sink.finish(&done).map_err(BandedError::Sink)?;
    let reports = reports.into_inner().unwrap_or_else(|p| p.into_inner());
    Ok((done, reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::banded::SubNoise;
    use cloud_map_generator_core::climate::EARTH;
    use cloud_map_generator_core::generate::GenParams;

    struct Collect(Vec<f32>);
    impl BandSink for Collect {
        fn band(&mut self, out: &cloud_map_generator_core::banded::BandOut<'_>) -> Result<(), String> {
            self.0.extend_from_slice(out.coverage);
            Ok(())
        }
    }

    /// A worker that fails on its second band.
    struct Flaky<'a, 'b> {
        inner: CpuBands<'a, 'b>,
        calls: Mutex<usize>,
    }
    impl BandCompute for Flaky<'_, '_> {
        fn kind(&self) -> crate::engine::DeviceKind {
            crate::engine::DeviceKind::Gpu
        }
        fn name(&self) -> Option<String> {
            Some("flaky".into())
        }
        fn pass_a(&self, band: usize) -> Result<BandMoments, String> {
            let mut c = self.calls.lock().unwrap();
            *c += 1;
            if *c == 2 {
                return Err("device lost".to_string());
            }
            drop(c);
            self.inner.pass_a(band)
        }
        fn pass_b(&self, band: usize, norm: &NormStats) -> Result<BandThresholds, String> {
            self.inner.pass_b(band, norm)
        }
        fn pass_c(&self, band: usize, norm: &NormStats, thr: &[f64], sens: &[f64]) -> Result<BandResult, String> {
            self.inner.pass_c(band, norm, thr, sens)
        }
    }

    #[test]
    fn two_workers_make_the_one_workers_map_and_a_lost_worker_hands_its_band_back() {
        let params = GenParams::default();
        let body = EARTH;
        let banded = Banded::new(&body, 3, 256, &params, None, SubNoise::Counter, 16, false).unwrap();
        let cpu = CpuBands(&banded, None);
        let mut solo = Collect(vec![]);
        let (_, r) = run_bands(&banded, &[&cpu], &mut solo, &|_| true).unwrap();
        assert_eq!(r[0].rows, 128 * 3);
        // Two identical CPU workers: the same bytes (a band is a pure function of its position).
        let cpu2 = CpuBands(&banded, None);
        let mut pair = Collect(vec![]);
        let (_, r) = run_bands(&banded, &[&cpu, &cpu2], &mut pair, &|_| true).unwrap();
        assert_eq!(r.iter().map(|w| w.rows).sum::<usize>(), 128 * 3);
        assert_eq!(solo.0, pair.0);
        // A worker that dies: the other finishes, and the dead one says why.
        let flaky = Flaky { inner: CpuBands(&banded, None), calls: Mutex::new(0) };
        let mut lost = Collect(vec![]);
        let (_, r) = run_bands(&banded, &[&cpu, &flaky], &mut lost, &|_| true).unwrap();
        assert_eq!(solo.0, lost.0);
        assert_eq!(r[1].lost.as_deref(), Some("device lost"));
        assert_eq!(r.iter().map(|w| w.rows).sum::<usize>(), 128 * 3, "every row of every pass was completed once");
    }

    #[test]
    fn cancel_stops_the_run() {
        let params = GenParams::default();
        let body = EARTH;
        let banded = Banded::new(&body, 3, 256, &params, None, SubNoise::Counter, 16, false).unwrap();
        let cpu = CpuBands(&banded, None);
        let n = std::sync::atomic::AtomicUsize::new(0);
        let mut sink = Collect(vec![]);
        let r = run_bands(&banded, &[&cpu], &mut sink, &|_| n.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 4);
        assert!(matches!(r, Err(BandedError::Cancelled)));
    }
}
