//! Data parallelism for the CPU engine, behind the `rayon` cargo feature.
//!
//! The rule that keeps the output bit-identical to the single-thread result for ANY thread count: a parallel loop
//! is only ever an order-preserving map over items that do not depend on each other (a texel, a row), and every
//! reduction (a sum, a mean, a standard deviation) is a serial one or runs over blocks of a FIXED size that are
//! combined in index order. Nothing here splits work by the thread count, and nothing accumulates across threads.
//!
//! Without the `rayon` feature (the default of this crate, so that it builds for wasm32) every helper runs on the
//! calling thread and [`threads`] is 1.

/// Texels per task of the per-texel maps: big enough to hide the scheduling, small enough to balance the load.
pub const BLOCK: usize = 2048;

/// The environment variable that sets the thread count of the CPU engine.
pub const THREADS_ENV: &str = "CLOUD_MAP_GENERATOR_THREADS";

/// The number of logical cores of the machine (1 when the system does not say).
pub fn hardware_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Parses a thread count: a whole number of at least 1 (`what` names the source in the message).
pub fn parse_threads(text: &str, what: &str) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!("{what} must be a whole number of threads, at least 1 (got {text:?})")),
    }
}

/// The thread count asked for by the environment (`CLOUD_MAP_GENERATOR_THREADS`), `None` when it is not set or empty.
pub fn env_threads() -> Result<Option<usize>, String> {
    match std::env::var(THREADS_ENV) {
        Ok(v) if !v.trim().is_empty() => parse_threads(&v, THREADS_ENV).map(Some),
        _ => Ok(None),
    }
}

#[cfg(feature = "rayon")]
mod imp {
    use super::*;
    use rayon::prelude::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Builds the process-wide pool once: the `requested` count, else `CLOUD_MAP_GENERATOR_THREADS`, else every logical core.
    pub fn init_threads(requested: Option<usize>) -> Result<usize, String> {
        let want = match requested {
            Some(n) if n >= 1 => n,
            Some(_) => return Err("the thread count must be at least 1".to_string()),
            None => env_threads()?.unwrap_or_else(hardware_threads),
        };
        INIT.call_once(|| {
            // An already-initialized global pool (an embedder's own) is left as it is: `threads()` reports the truth.
            let _ = rayon::ThreadPoolBuilder::new().num_threads(want).build_global();
        });
        Ok(rayon::current_num_threads())
    }

    /// The threads the parallel loops use now: inside [`with_threads`] that pool's, else the process-wide pool's.
    pub fn threads() -> usize {
        // A bad `CLOUD_MAP_GENERATOR_THREADS` is an error of `init_threads` (the front ends call it); here it falls back.
        let _ = init_threads(None);
        rayon::current_num_threads()
    }

    /// Runs `f` with a private pool of `n` threads (a benchmark or a test that compares thread counts).
    pub fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).build().expect("a thread pool");
        pool.install(f)
    }

    pub fn map<T: Sync, U: Send>(src: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
        src.par_iter().with_min_len(BLOCK).map(f).collect()
    }

    pub fn map_index<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
        (0..n).into_par_iter().with_min_len(BLOCK).map(f).collect()
    }

    pub fn map_rows<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
        (0..n).into_par_iter().map(f).collect()
    }

    pub fn map_chunks<T: Sync, U: Send>(src: &[T], block: usize, f: impl Fn(&[T]) -> Vec<U> + Sync + Send) -> Vec<U> {
        let parts: Vec<Vec<U>> = src.par_chunks(block.max(1)).map(f).collect();
        let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        for p in parts {
            out.extend(p);
        }
        out
    }

    pub fn for_each_chunk_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
        data.par_chunks_mut(chunk.max(1)).enumerate().for_each(|(i, c)| f(i, c));
    }

    pub fn for_each_chunk2_mut<T: Send, U: Send>(
        a: &mut [T],
        b: &mut [U],
        chunk: usize,
        f: impl Fn(usize, &mut [T], &mut [U]) + Sync + Send,
    ) {
        a.par_chunks_mut(chunk.max(1)).zip(b.par_chunks_mut(chunk.max(1))).enumerate().for_each(|(i, (x, y))| f(i, x, y));
    }
}

#[cfg(not(feature = "rayon"))]
mod imp {
    use super::*;

    pub fn init_threads(requested: Option<usize>) -> Result<usize, String> {
        match requested {
            Some(0) => Err("the thread count must be at least 1".to_string()),
            _ => env_threads().map(|_| 1),
        }
    }

    pub fn threads() -> usize {
        1
    }

    pub fn with_threads<R: Send>(_n: usize, f: impl FnOnce() -> R + Send) -> R {
        f()
    }

    pub fn map<T: Sync, U: Send>(src: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
        src.iter().map(f).collect()
    }

    pub fn map_index<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
        (0..n).map(f).collect()
    }

    pub fn map_rows<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
        (0..n).map(f).collect()
    }

    pub fn map_chunks<T: Sync, U: Send>(src: &[T], block: usize, f: impl Fn(&[T]) -> Vec<U> + Sync + Send) -> Vec<U> {
        let mut out = Vec::with_capacity(src.len());
        for c in src.chunks(block.max(1)) {
            out.extend(f(c));
        }
        out
    }

    pub fn for_each_chunk_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
        for (i, c) in data.chunks_mut(chunk.max(1)).enumerate() {
            f(i, c);
        }
    }

    pub fn for_each_chunk2_mut<T: Send, U: Send>(
        a: &mut [T],
        b: &mut [U],
        chunk: usize,
        f: impl Fn(usize, &mut [T], &mut [U]) + Sync + Send,
    ) {
        for (i, (x, y)) in a.chunks_mut(chunk.max(1)).zip(b.chunks_mut(chunk.max(1))).enumerate() {
            f(i, x, y);
        }
    }
}

/// Fixes the thread count of the process-wide pool: `requested` (a `--threads` flag), else `CLOUD_MAP_GENERATOR_THREADS`, else
/// every logical core. Returns the count in force. Call it once, before the first map; later calls change nothing.
/// Without the `rayon` feature the count is always 1.
pub fn init_threads(requested: Option<usize>) -> Result<usize, String> {
    imp::init_threads(requested)
}

/// The threads the parallel maps use (1 without the `rayon` feature).
pub fn threads() -> usize {
    imp::threads()
}

/// Runs `f` with a private pool of `n` threads: the maps inside use exactly `n` (a benchmark, or a test that
/// compares thread counts). Without the `rayon` feature it just runs `f`.
pub fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    imp::with_threads(n, f)
}

/// `src.iter().map(f).collect()`, in parallel and in order.
pub fn map<T: Sync, U: Send>(src: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    imp::map(src, f)
}

/// `(0..n).map(f).collect()`, in parallel and in order.
pub fn map_index<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
    imp::map_index(n, f)
}

/// `(0..n).map(f).collect()` in parallel and in order, one task per index: for work that is big per item (a row of a
/// field), where [`map_index`]'s minimum block would leave most cores idle.
pub fn map_rows<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync + Send) -> Vec<U> {
    imp::map_rows(n, f)
}

/// Splits `src` in blocks of `block` items, maps each block to a `Vec` and joins them in order. `f` must be
/// element-wise (the result of an item must not depend on the block it fell in).
pub fn map_chunks<T: Sync, U: Send>(src: &[T], block: usize, f: impl Fn(&[T]) -> Vec<U> + Sync + Send) -> Vec<U> {
    imp::map_chunks(src, block, f)
}

/// Calls `f(chunk_index, chunk)` on every `chunk`-long piece of `data` (the last may be shorter): the rows of a field.
pub fn for_each_chunk_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    imp::for_each_chunk_mut(data, chunk, f)
}

/// [`for_each_chunk_mut`] over two slices cut at the same item boundaries (`chunk` items of each).
pub fn for_each_chunk2_mut<T: Send, U: Send>(
    a: &mut [T],
    b: &mut [U],
    chunk: usize,
    f: impl Fn(usize, &mut [T], &mut [U]) + Sync + Send,
) {
    imp::for_each_chunk2_mut(a, b, chunk, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_keep_the_order() {
        let v: Vec<u32> = (0..10_000).collect();
        assert_eq!(map(&v, |x| x * 2), v.iter().map(|x| x * 2).collect::<Vec<_>>());
        assert_eq!(map_index(10_000, |i| i as u32), v);
        assert_eq!(map_chunks(&v, 77, |c| c.to_vec()), v);
        let mut w = vec![0usize; 1000];
        for_each_chunk_mut(&mut w, 64, |i, c| c.iter_mut().for_each(|x| *x = i));
        assert_eq!(w[0], 0);
        assert_eq!(w[999], 999 / 64);
    }

    #[test]
    fn thread_counts_parse() {
        assert_eq!(parse_threads(" 4 ", "x"), Ok(4));
        assert!(parse_threads("0", "x").is_err());
        assert!(parse_threads("many", "x").is_err());
    }

    #[test]
    fn a_private_pool_has_the_asked_threads() {
        let n = with_threads(3, threads);
        assert_eq!(n, if cfg!(feature = "rayon") { 3 } else { 1 });
    }
}
