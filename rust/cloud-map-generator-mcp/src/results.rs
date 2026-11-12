//! The cache of generation results this server session holds, owned by the server (the engine keeps none).
//!
//! The last [`MAX_RESULTS`] results stay; the pixels of the oldest are dropped first when the total passes a
//! memory budget (a 4096 map holds tens of megabytes of planes), keeping the summary, the preview and the files.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::Value;

use crate::backend::Plane;

pub const MAX_RESULTS: usize = 16;
pub const ARRAY_BUDGET_BYTES: usize = 768 * 1024 * 1024;

/// The pixels of a result: the coverage and the other products it can be looked at as.
#[derive(Debug, Clone)]
pub struct Planes {
    pub coverage: Plane,
    pub extra: BTreeMap<String, Plane>,
}

impl Planes {
    pub fn nbytes(&self) -> usize {
        self.coverage.nbytes() + self.extra.values().map(Plane::nbytes).sum::<usize>()
    }

    /// The product names a result can be drawn as, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::iter::once("coverage".to_string()).chain(self.extra.keys().cloned()).collect();
        v.sort();
        v
    }

    pub fn get(&self, product: &str) -> Option<&Plane> {
        if product == "coverage" {
            Some(&self.coverage)
        } else {
            self.extra.get(product)
        }
    }
}

pub struct StoredResult {
    pub id: String,
    pub summary: Value,
    pub preview: Vec<u8>,
    planes: Mutex<Option<Arc<Planes>>>,
    pub folder: Option<PathBuf>,
    pub files: BTreeMap<String, PathBuf>,
}

impl StoredResult {
    pub fn new(
        id: String,
        summary: Value,
        preview: Vec<u8>,
        planes: Option<Planes>,
        folder: Option<PathBuf>,
        files: BTreeMap<String, PathBuf>,
    ) -> Self {
        StoredResult { id, summary, preview, planes: Mutex::new(planes.map(Arc::new)), folder, files }
    }

    /// The pixels, or `None` when they were dropped to save memory.
    pub fn planes(&self) -> Option<Arc<Planes>> {
        lock(&self.planes).clone()
    }

    pub fn array_bytes(&self) -> usize {
        self.planes().map_or(0, |p| p.nbytes())
    }

    fn drop_planes(&self) {
        *lock(&self.planes) = None;
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

pub struct ResultStore {
    max_results: usize,
    array_budget: usize,
    items: Mutex<Vec<Arc<StoredResult>>>, // oldest first
}

impl Default for ResultStore {
    fn default() -> Self {
        Self::new(MAX_RESULTS, ARRAY_BUDGET_BYTES)
    }
}

impl ResultStore {
    pub fn new(max_results: usize, array_budget: usize) -> Self {
        ResultStore { max_results, array_budget, items: Mutex::new(Vec::new()) }
    }

    pub fn max_results(&self) -> usize {
        self.max_results
    }

    pub fn add(&self, stored: Arc<StoredResult>) {
        let mut items = lock(&self.items);
        items.retain(|s| s.id != stored.id);
        items.push(stored.clone());
        while items.len() > self.max_results {
            items.remove(0);
        }
        let mut total: usize = items.iter().map(|s| s.array_bytes()).sum();
        for s in items.iter() {
            // oldest first; the newest keeps its pixels whatever the budget
            if total <= self.array_budget || Arc::ptr_eq(s, &stored) {
                break;
            }
            total -= s.array_bytes();
            s.drop_planes();
        }
    }

    pub fn get(&self, result_id: &str) -> Option<Arc<StoredResult>> {
        lock(&self.items).iter().find(|s| s.id == result_id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        lock(&self.items).iter().map(|s| s.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(i: usize, nbytes: usize) -> Arc<StoredResult> {
        let planes = Planes { coverage: Plane::new(nbytes, 1, vec![0; nbytes]), extra: BTreeMap::new() };
        Arc::new(StoredResult::new(format!("{i:012x}"), Value::Null, Vec::new(), Some(planes), None, BTreeMap::new()))
    }

    #[test]
    fn the_store_keeps_the_last_sixteen() {
        let store = ResultStore::new(16, 1_000_000_000);
        for i in 0..20 {
            store.add(stored(i, 100));
        }
        assert!(store.get(&format!("{:012x}", 0)).is_none());
        assert!(store.get(&format!("{:012x}", 19)).is_some());
        assert_eq!(store.ids().len(), 16);
        assert!(store.get("../x").is_none());
    }

    #[test]
    fn the_pixels_of_the_oldest_go_first_over_the_memory_budget() {
        let store = ResultStore::new(16, 250);
        for i in 0..3 {
            store.add(stored(i, 100));
        }
        assert!(store.get(&format!("{:012x}", 0)).unwrap().planes().is_none());
        assert!(store.get(&format!("{:012x}", 1)).unwrap().planes().is_some());
        store.add(stored(3, 1000)); // the newest always keeps its pixels
        assert!(store.get(&format!("{:012x}", 3)).unwrap().planes().is_some());
        assert!(store.get(&format!("{:012x}", 1)).unwrap().planes().is_none());
    }
}
