//! The results the server keeps: the last `keep` generations, as the bytes of their files, evicted oldest first
//! (`ResultStore` of `ui/server.py`). They live in memory: restarting the server forgets them.

use std::collections::hash_map::RandomState;
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// The files `GET /api/result/<id>/<name>` serves: name, content type, whether it is a download.
pub const RESULT_FILES: [(&str, &str, bool); 7] = [
    ("coverage.png", "image/png", false),
    ("type.png", "image/png", false),
    ("high_share.png", "image/png", false),
    ("wind.png", "image/png", false),
    ("master16.png", "image/png", true),
    ("coverage.r8.gz", "application/gzip", true),
    ("meta.json", "application/json; charset=utf-8", true),
];

/// One kept generation.
#[derive(Debug)]
pub struct StoredResult {
    /// File name (a [`RESULT_FILES`] name) and its bytes.
    pub files: HashMap<&'static str, Vec<u8>>,
    /// The `Content-Disposition` file name of each download (the seed, the size and the id are in it).
    pub names: HashMap<&'static str, String>,
}

#[derive(Default)]
struct Inner {
    order: VecDeque<String>,
    items: HashMap<String, Arc<StoredResult>>,
}

pub struct ResultStore {
    pub keep: usize,
    inner: Mutex<Inner>,
    counter: AtomicU64,
    hasher: RandomState,
}

impl ResultStore {
    pub fn new(keep: usize) -> ResultStore {
        ResultStore { keep, inner: Mutex::new(Inner::default()), counter: AtomicU64::new(0), hasher: RandomState::new() }
    }

    /// A new id: 12 lowercase hex digits (48 bits from a per-process random key, a counter and the clock).
    pub fn new_id(&self) -> String {
        let mut h = self.hasher.build_hasher();
        h.write_u64(self.counter.fetch_add(1, Ordering::Relaxed));
        h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
        format!("{:012x}", h.finish() & 0xffff_ffff_ffff)
    }

    pub fn add(&self, id: &str, item: StoredResult) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.items.insert(id.to_string(), Arc::new(item));
        inner.order.push_back(id.to_string());
        while inner.order.len() > self.keep {
            if let Some(old) = inner.order.pop_front() {
                inner.items.remove(&old);
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<StoredResult>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).items.get(id).cloned()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> StoredResult {
        StoredResult { files: HashMap::new(), names: HashMap::new() }
    }

    #[test]
    fn the_store_keeps_the_last_n() {
        let s = ResultStore::new(3);
        let ids: Vec<String> = (0..5).map(|_| s.new_id()).collect();
        for id in &ids {
            s.add(id, item());
        }
        assert_eq!(s.len(), 3);
        assert_eq!(ids.iter().map(|i| s.get(i).is_some()).collect::<Vec<_>>(), [false, false, true, true, true]);
    }

    #[test]
    fn ids_are_twelve_hex_digits_and_distinct() {
        let s = ResultStore::new(8);
        let ids: Vec<String> = (0..200).map(|_| s.new_id()).collect();
        assert!(ids.iter().all(|i| i.len() == 12 && i.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))), "{ids:?}");
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
    }

    #[test]
    fn the_served_files_are_the_seven_of_the_contract() {
        let names: Vec<&str> = RESULT_FILES.iter().map(|f| f.0).collect();
        assert_eq!(names, ["coverage.png", "type.png", "high_share.png", "wind.png", "master16.png", "coverage.r8.gz", "meta.json"]);
        assert_eq!(RESULT_FILES.iter().filter(|f| f.2).count(), 3);
    }
}
