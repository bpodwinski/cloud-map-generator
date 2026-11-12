//! One heavy job at a time: a small FIFO queue with a single worker thread.
//!
//! Generation and baking are CPU-bound and take seconds to minutes. The async runtime must stay free for pings,
//! cancellations and the next request, so the work runs on one worker thread, strictly one job at a time (they
//! would only slow each other down). At most [`MAX_QUEUED`] jobs wait; the next submit is refused with a message
//! that says what to do. The last [`KEEP_JOBS`] finished jobs stay readable.

use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::errors::AppError;
use crate::logging::{log, Level};
use crate::pyfmt::{is_printable, py_repr, token_hex};

pub const MAX_QUEUED: usize = 4;
pub const KEEP_JOBS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }

    pub fn is_final(self) -> bool {
        matches!(self, Status::Done | Status::Failed | Status::Cancelled)
    }
}

struct State<T> {
    status: Status,
    progress: f64,
    message: String,
    result: Option<T>,
    error: Option<String>,
    cancel_requested: bool,
    started: Option<Instant>,
    finished: Option<Instant>,
    /// Every distinct progress value reported, in order, for a caller that forwards them all.
    history: Vec<(f64, String)>,
}

/// One job: its status and progress while it waits and runs, its result or error when it ends.
pub struct Job<T> {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub estimate_seconds: Option<f64>,
    state: Mutex<State<T>>,
    cv: Condvar,
    notify: tokio::sync::Notify,
}

fn lock<S>(m: &Mutex<S>) -> MutexGuard<'_, S> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl<T: Clone> Job<T> {
    fn new(kind: &str, label: &str, estimate_seconds: Option<f64>) -> Arc<Self> {
        Arc::new(Job {
            id: token_hex(6),
            kind: kind.to_string(),
            label: label.to_string(),
            estimate_seconds,
            state: Mutex::new(State {
                status: Status::Queued,
                progress: 0.0,
                message: "waiting in the queue".to_string(),
                result: None,
                error: None,
                cancel_requested: false,
                started: None,
                finished: None,
                history: Vec::new(),
            }),
            cv: Condvar::new(),
            notify: tokio::sync::Notify::new(),
        })
    }

    pub fn status(&self) -> Status {
        lock(&self.state).status
    }

    pub fn is_final(&self) -> bool {
        self.status().is_final()
    }

    pub fn progress(&self) -> f64 {
        lock(&self.state).progress
    }

    pub fn message(&self) -> String {
        lock(&self.state).message.clone()
    }

    pub fn result(&self) -> Option<T> {
        lock(&self.state).result.clone()
    }

    pub fn error(&self) -> Option<String> {
        lock(&self.state).error.clone()
    }

    pub fn cancel_requested(&self) -> bool {
        lock(&self.state).cancel_requested
    }

    /// Seconds since the job started running (0 while queued); frozen once it is final.
    pub fn elapsed(&self) -> f64 {
        let st = lock(&self.state);
        match st.started {
            None => 0.0,
            Some(s) => st.finished.unwrap_or_else(Instant::now).duration_since(s).as_secs_f64(),
        }
    }

    /// Block until the job is final or `timeout` passes; true when it is final.
    pub fn wait_done(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.state);
        while !st.status.is_final() {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            st = self.cv.wait_timeout(st, deadline - now).unwrap_or_else(|p| p.into_inner()).0;
        }
        true
    }

    /// The progress values reported after the first `from` ones.
    pub fn history_since(&self, from: usize) -> Vec<(f64, String)> {
        let st = lock(&self.state);
        st.history.get(from..).map(<[_]>::to_vec).unwrap_or_default()
    }

    /// Resolves when the job's progress or status changed (a wake-up is never lost: it is stored).
    pub async fn changed(&self) {
        self.notify.notified().await;
    }

    fn set_progress(&self, fraction: f64, message: &str) {
        if fraction.is_nan() {
            return;
        }
        {
            let mut st = lock(&self.state);
            let f = fraction.clamp(0.0, 1.0);
            let before = st.progress;
            st.progress = st.progress.max(f); // never goes backwards
            if !message.is_empty() {
                st.message = message.to_string();
            }
            if st.progress > before {
                let entry = (st.progress, st.message.clone());
                st.history.push(entry);
            }
        }
        self.notify.notify_one();
    }

    fn finish(&self, status: Status, message: Option<&str>, error: Option<String>, result: Option<T>) {
        {
            let mut st = lock(&self.state);
            st.status = status;
            let now = Instant::now();
            st.finished = Some(now);
            if st.started.is_none() {
                st.started = Some(now);
            }
            if status == Status::Done {
                let before = st.progress;
                st.progress = 1.0;
                if message.is_some() {
                    st.message = message.unwrap_or_default().to_string();
                }
                if before < 1.0 {
                    let entry = (1.0, st.message.clone());
                    st.history.push(entry);
                }
            } else if let Some(m) = message {
                st.message = m.to_string();
            }
            st.error = error;
            st.result = result;
        }
        self.cv.notify_all();
        self.notify.notify_one();
    }
}

/// What a job function receives: a way to report progress and a way to ask whether to stop.
pub struct JobControl<T> {
    job: Arc<Job<T>>,
}

impl<T: Clone> JobControl<T> {
    pub fn progress(&self, fraction: f64, message: &str) {
        self.job.set_progress(fraction, message);
    }

    pub fn cancelled(&self) -> bool {
        self.job.cancel_requested()
    }

    /// `Err(Cancelled)` when a stop was requested.
    pub fn check(&self) -> Result<(), AppError> {
        if self.cancelled() {
            Err(AppError::Cancelled)
        } else {
            Ok(())
        }
    }
}

pub type JobFn<T> = Box<dyn FnOnce(&JobControl<T>) -> Result<T, AppError> + Send>;

struct MState<T> {
    jobs: HashMap<String, Arc<Job<T>>>,
    order: Vec<String>,
    queue: VecDeque<(Arc<Job<T>>, JobFn<T>)>,
    running: Option<Arc<Job<T>>>,
    worker_started: bool,
    closed: bool,
}

struct Inner<T> {
    format_error: Box<dyn Fn(&AppError) -> String + Send + Sync>,
    max_queued: usize,
    keep: usize,
    cv: Condvar,
    st: Mutex<MState<T>>,
}

/// A FIFO queue drained by one daemon worker thread, started on the first submit.
pub struct JobManager<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Clone + Send + 'static> JobManager<T> {
    /// `format_error` turns the error of a failed job into the text its status carries.
    pub fn new(format_error: impl Fn(&AppError) -> String + Send + Sync + 'static) -> Self {
        Self::with_limits(format_error, MAX_QUEUED, KEEP_JOBS)
    }

    pub fn with_limits(
        format_error: impl Fn(&AppError) -> String + Send + Sync + 'static,
        max_queued: usize,
        keep: usize,
    ) -> Self {
        JobManager {
            inner: Arc::new(Inner {
                format_error: Box::new(format_error),
                max_queued,
                keep,
                cv: Condvar::new(),
                st: Mutex::new(MState {
                    jobs: HashMap::new(),
                    order: Vec::new(),
                    queue: VecDeque::new(),
                    running: None,
                    worker_started: false,
                    closed: false,
                }),
            }),
        }
    }

    pub fn keep(&self) -> usize {
        self.inner.keep
    }

    /// How many jobs the manager remembers (queued, running and the last finished ones).
    pub fn job_count(&self) -> usize {
        lock(&self.inner.st).jobs.len()
    }

    /// Queue a job. Refused with a message that says what to do when `max_queued` jobs already wait.
    pub fn submit(
        &self,
        kind: &str,
        label: &str,
        f: impl FnOnce(&JobControl<T>) -> Result<T, AppError> + Send + 'static,
        estimate_seconds: Option<f64>,
    ) -> Result<Arc<Job<T>>, AppError> {
        let mut st = lock(&self.inner.st);
        if st.closed {
            return Err(AppError::user("the server is shutting down"));
        }
        if st.queue.len() >= self.inner.max_queued {
            return Err(AppError::user(format!(
                "The job queue is full ({} jobs are already waiting behind the running one). Wait for one to \
                 finish (get_generation) or cancel a waiting job (cancel_generation) and try again.",
                self.inner.max_queued
            )));
        }
        let job = Job::new(kind, label, estimate_seconds);
        st.jobs.insert(job.id.clone(), job.clone());
        st.order.push(job.id.clone());
        st.queue.push_back((job.clone(), Box::new(f)));
        prune(&mut st, self.inner.keep);
        if !st.worker_started {
            st.worker_started = true;
            let inner = self.inner.clone();
            std::thread::Builder::new()
                .name("cloud-map-generator-worker".to_string())
                .spawn(move || run_forever(inner))
                .map_err(|e| AppError::internal("IoError", e.to_string()))?;
        }
        self.inner.cv.notify_all();
        Ok(job)
    }

    pub fn get(&self, job_id: &str) -> Result<Arc<Job<T>>, AppError> {
        let found = lock(&self.inner.st).jobs.get(job_id).cloned();
        found.ok_or_else(|| {
            let shown: String = job_id.chars().take(40).map(|c| if is_printable(c) { c } else { '?' }).collect();
            AppError::user(format!(
                "Unknown job_id {}. Job ids come from start_generation; only the last {} jobs of this server \
                 session are kept.",
                py_repr(&shown),
                self.inner.keep
            ))
        })
    }

    /// How many jobs run or wait before this one (0 when it is running or next in line).
    pub fn queue_ahead(&self, job: &Arc<Job<T>>) -> usize {
        let st = lock(&self.inner.st);
        if job.status() != Status::Queued {
            return 0;
        }
        let mut ahead = usize::from(st.running.is_some());
        for (j, _) in st.queue.iter() {
            if Arc::ptr_eq(j, job) {
                return ahead;
            }
            ahead += 1;
        }
        ahead
    }

    /// Cancel a queued job at once, or ask a running one to stop at its next stage boundary.
    pub fn cancel(&self, job_id: &str) -> Result<Arc<Job<T>>, AppError> {
        let job = self.get(job_id)?;
        let mut st = lock(&self.inner.st);
        if job.is_final() {
            return Ok(job);
        }
        if job.status() == Status::Queued {
            st.queue.retain(|(j, _)| !Arc::ptr_eq(j, &job));
            lock(&job.state).cancel_requested = true;
            job.finish(Status::Cancelled, Some("cancelled before it started"), None, None);
        } else {
            let mut js = lock(&job.state);
            js.cancel_requested = true;
            js.message = "cancellation requested; stopping at the next stage boundary".to_string();
        }
        drop(st);
        job.notify.notify_one();
        Ok(job)
    }

    /// Stop accepting work and let the worker exit after the current job.
    pub fn close(&self) {
        let mut st = lock(&self.inner.st);
        st.closed = true;
        let waiting: Vec<_> = st.queue.drain(..).collect();
        for (job, _) in waiting {
            lock(&job.state).cancel_requested = true;
            job.finish(Status::Cancelled, Some("the server is shutting down"), None, None);
        }
        if let Some(running) = &st.running {
            lock(&running.state).cancel_requested = true;
        }
        self.inner.cv.notify_all();
    }
}

impl<T> Drop for JobManager<T> {
    fn drop(&mut self) {
        // the worker holds its own Arc: tell it to stop once the current job ends
        let mut st = lock(&self.inner.st);
        st.closed = true;
        for (job, _) in st.queue.iter() {
            lock(&job.state).cancel_requested = true;
        }
        if let Some(running) = &st.running {
            lock(&running.state).cancel_requested = true;
        }
        self.inner.cv.notify_all();
    }
}

fn prune<T: Clone>(st: &mut MState<T>, keep: usize) {
    if st.order.len() <= keep {
        return;
    }
    let mut extra = st.order.len() - keep;
    let mut kept = Vec::with_capacity(keep);
    let order = std::mem::take(&mut st.order);
    for id in order {
        let final_job = st.jobs.get(&id).is_some_and(|j| j.status().is_final());
        if extra > 0 && final_job {
            st.jobs.remove(&id);
            extra -= 1;
        } else {
            kept.push(id);
        }
    }
    st.order = kept;
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic".to_string()
    }
}

fn run_forever<T: Clone + Send + 'static>(inner: Arc<Inner<T>>) {
    loop {
        let (job, f) = {
            let mut st = lock(&inner.st);
            loop {
                if let Some(next) = st.queue.pop_front() {
                    next.0.set_running();
                    st.running = Some(next.0.clone());
                    break next;
                }
                if st.closed {
                    return;
                }
                st = inner.cv.wait(st).unwrap_or_else(|p| p.into_inner());
            }
        };
        let control = JobControl { job: job.clone() };
        let outcome = catch_unwind(AssertUnwindSafe(|| f(&control)));
        let outcome = outcome.unwrap_or_else(|payload| Err(AppError::internal("Panic", panic_text(payload.as_ref()))));
        match outcome {
            Ok(value) => job.finish(Status::Done, Some("done"), None, Some(value)),
            Err(AppError::Cancelled) => job.finish(Status::Cancelled, Some("cancelled"), None, None),
            Err(e) => {
                match &e {
                    AppError::Internal { kind, detail } => {
                        log(Level::Error, "cloud_map_generator_mcp.jobs", format_args!("job {} ({}) failed: {kind}: {detail}", job.id, job.label))
                    }
                    other => log(Level::Info, "cloud_map_generator_mcp.jobs", format_args!("job {} ({}) refused: {other}", job.id, job.label)),
                }
                let text = (inner.format_error)(&e);
                job.finish(Status::Failed, Some("failed"), Some(text), None);
            }
        }
        let mut st = lock(&inner.st);
        st.running = None;
        prune(&mut st, inner.keep);
    }
}

impl<T: Clone> Job<T> {
    fn set_running(&self) {
        let mut st = lock(&self.state);
        st.status = Status::Running;
        st.started = Some(Instant::now());
        st.message = "starting".to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn manager() -> JobManager<String> {
        JobManager::new(|e| format!("failed: {e}"))
    }

    fn wait_until(cond: impl Fn() -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs(5);
        while Instant::now() < end {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    fn gate() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn wait_gate(g: &AtomicBool) {
        let end = Instant::now() + Duration::from_secs(5);
        while !g.load(Ordering::SeqCst) && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn a_job_runs_reports_progress_and_keeps_its_result() {
        let jm = manager();
        let g = gate();
        let g2 = g.clone();
        let job = jm
            .submit(
                "generate",
                "g",
                move |c| {
                    c.progress(0.4, "half");
                    c.progress(0.2, "never goes backwards");
                    wait_gate(&g2);
                    Ok("the result".to_string())
                },
                Some(3.0),
            )
            .unwrap();
        assert!(wait_until(|| job.status() == Status::Running && job.message() == "never goes backwards"));
        assert_eq!(job.progress(), 0.4);
        assert_eq!(jm.queue_ahead(&job), 0);
        g.store(true, Ordering::SeqCst);
        assert!(job.wait_done(Duration::from_secs(5)));
        assert_eq!(job.status(), Status::Done);
        assert_eq!(job.result().as_deref(), Some("the result"));
        assert_eq!(job.progress(), 1.0);
        assert!(job.error().is_none());
        assert!(Arc::ptr_eq(&jm.get(&job.id).unwrap(), &job));
        let h = job.history_since(0);
        assert_eq!(h.first().map(|x| x.0), Some(0.4));
        assert_eq!(h.last().map(|x| x.0), Some(1.0));
        jm.close();
    }

    #[test]
    fn one_job_at_a_time_in_submission_order_and_a_queue_of_four() {
        let jm = manager();
        let g = gate();
        let order = Arc::new(Mutex::new(Vec::new()));
        let make = |name: &str| {
            let (g, order, name) = (g.clone(), order.clone(), name.to_string());
            move |_: &JobControl<String>| {
                order.lock().unwrap().push(name.clone());
                wait_gate(&g);
                Ok(String::new())
            }
        };
        let first = jm.submit("g", "first", make("first"), None).unwrap();
        assert!(wait_until(|| first.status() == Status::Running));
        let waiting: Vec<_> = (0..4).map(|i| jm.submit("g", &format!("w{i}"), make(&format!("w{i}")), None).unwrap()).collect();
        let ahead: Vec<usize> = waiting.iter().map(|j| jm.queue_ahead(j)).collect();
        assert_eq!(ahead, vec![1, 2, 3, 4]);
        let e = jm.submit("g", "fifth", make("fifth"), None).err().unwrap().to_string();
        assert!(e.contains("queue is full") && e.contains("4 jobs") && e.contains("cancel_generation"), "{e}");
        g.store(true, Ordering::SeqCst);
        assert!(waiting.iter().all(|j| j.wait_done(Duration::from_secs(5))));
        assert_eq!(*order.lock().unwrap(), vec!["first", "w0", "w1", "w2", "w3"]);
        jm.close();
    }

    #[test]
    fn cancel_a_queued_job_at_once_and_a_running_job_cooperatively() {
        let jm = manager();
        let started = gate();
        let s2 = started.clone();
        let running = jm
            .submit(
                "g",
                "run",
                move |c| {
                    s2.store(true, Ordering::SeqCst);
                    while !c.cancelled() {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    c.check()?;
                    Ok(String::new())
                },
                None,
            )
            .unwrap();
        assert!(wait_until(|| started.load(Ordering::SeqCst)));
        let queued = jm.submit("g", "wait", |_| Ok("never".to_string()), None).unwrap();
        assert_eq!(jm.cancel(&queued.id).unwrap().status(), Status::Cancelled);
        assert!(queued.is_final() && queued.result().is_none());
        jm.cancel(&running.id).unwrap();
        assert!(running.cancel_requested()); // a running job is only asked: it ends when it notices
        assert!(running.wait_done(Duration::from_secs(5)));
        assert_eq!(running.status(), Status::Cancelled);
        assert_eq!(jm.cancel(&running.id).unwrap().status(), Status::Cancelled); // a finished job is left alone
        jm.close();
    }

    #[test]
    fn a_failing_job_reports_a_formatted_error_and_the_worker_survives() {
        let jm = manager();
        let bad = jm.submit("g", "bad", |_| Err(AppError::user("width: 3 is not supported")), None).unwrap();
        assert!(bad.wait_done(Duration::from_secs(5)));
        assert_eq!(bad.status(), Status::Failed);
        assert_eq!(bad.error().as_deref(), Some("failed: width: 3 is not supported"));
        let good = jm.submit("g", "good", |_| Ok("7".to_string()), None).unwrap();
        assert!(good.wait_done(Duration::from_secs(5)));
        assert_eq!((good.status(), good.result().as_deref()), (Status::Done, Some("7")));
        jm.close();
    }

    #[test]
    fn a_panicking_job_is_an_internal_error_and_the_worker_survives() {
        let jm: JobManager<String> = JobManager::new(|e| match e {
            AppError::Internal { kind, .. } => format!("internal {kind}"),
            other => other.to_string(),
        });
        let bad = jm.submit("g", "boom", |_| panic!("secret detail"), None).unwrap();
        assert!(bad.wait_done(Duration::from_secs(5)));
        assert_eq!(bad.error().as_deref(), Some("internal Panic"));
        let good = jm.submit("g", "good", |_| Ok("7".to_string()), None).unwrap();
        assert!(good.wait_done(Duration::from_secs(5)) && good.status() == Status::Done);
        jm.close();
    }

    #[test]
    fn a_backend_cancellation_counts_as_cancelled() {
        let jm = manager();
        let job = jm
            .submit(
                "g",
                "stop",
                |c| {
                    c.progress(0.1, "x");
                    Err(AppError::Cancelled)
                },
                None,
            )
            .unwrap();
        assert!(job.wait_done(Duration::from_secs(5)));
        assert_eq!(job.status(), Status::Cancelled);
        assert!(job.error().is_none());
        jm.close();
    }

    #[test]
    fn only_the_last_sixteen_finished_jobs_are_kept() {
        let jm = manager();
        let mut jobs = Vec::new();
        for i in 0..20 {
            let j = jm.submit("g", &format!("j{i}"), |_| Ok("1".to_string()), None).unwrap();
            assert!(j.wait_done(Duration::from_secs(5)));
            jobs.push(j);
        }
        let known = |j: &Arc<Job<String>>| jm.get(&j.id).is_ok();
        assert!(wait_until(|| jobs.iter().filter(|j| known(j)).count() == 16));
        assert!(!known(&jobs[0]) && known(&jobs[19]));
        let e = jm.get(&jobs[0].id).err().unwrap().to_string();
        assert!(e.contains("Unknown job_id") && e.contains("last 16"), "{e}");
        let e = jm.get(&"x\u{1b}[31m".repeat(20)).err().unwrap().to_string();
        assert!(e.contains("Unknown job_id") && !e.contains('\u{1b}'), "{e:?}"); // echoed safely
        jm.close();
    }
}
