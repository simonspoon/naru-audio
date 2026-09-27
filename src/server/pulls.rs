//! §2.5 in-flight `/api/pull` tracking: one entry per model currently
//! streaming, so a second `POST /api/pull` for the same model is refused
//! (409 `pull_in_progress`), `GET /api/pulls` can report live progress, and
//! `DELETE /api/pulls/{name}` can cancel one.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde_json::{Value, json};

/// One pull's live progress, shared between the streaming handler (which
/// updates it as `Progress` comes in) and `GET /api/pulls` (which only
/// reads it).
pub struct PullEntry {
    model: String,
    file: Mutex<Option<String>>,
    completed: AtomicU64,
    total: Mutex<Option<u64>>,
    model_completed: AtomicU64,
    model_total: AtomicU64,
    started_at: SystemTime,
    /// Checked by `Registry::pull_cancellable`'s chunk callback.
    pub cancel: Arc<AtomicBool>,
}

impl PullEntry {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            file: Mutex::new(None),
            completed: AtomicU64::new(0),
            total: Mutex::new(None),
            model_completed: AtomicU64::new(0),
            model_total: AtomicU64::new(0),
            started_at: SystemTime::now(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Records a `Progress::Model` boundary.
    pub fn set_model_progress(&self, index: usize, total: usize) {
        self.model_completed.store(index as u64, Ordering::Relaxed);
        self.model_total.store(total as u64, Ordering::Relaxed);
    }

    /// Records a `Progress::Downloading` update.
    pub fn set_file_progress(&self, file: &str, completed: u64, total: Option<u64>) {
        *self.file.lock().unwrap_or_else(|e| e.into_inner()) = Some(file.to_string());
        self.completed.store(completed, Ordering::Relaxed);
        *self.total.lock().unwrap_or_else(|e| e.into_inner()) = total;
    }

    fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "file": *self.file.lock().unwrap_or_else(|e| e.into_inner()),
            "completed": self.completed.load(Ordering::Relaxed),
            "total": *self.total.lock().unwrap_or_else(|e| e.into_inner()),
            "model_completed": self.model_completed.load(Ordering::Relaxed),
            "model_total": self.model_total.load(Ordering::Relaxed),
            "status": "running",
            "started_at": humantime::format_rfc3339_seconds(self.started_at).to_string(),
        })
    }
}

/// `POST /api/pull` already has a pull running for that model.
pub struct PullInProgress;

/// `DELETE /api/pulls/{name}`: no pull is running for that model.
pub struct PullNotFound;

/// Pulls currently streaming, keyed by model name.
#[derive(Default)]
pub struct PullTracker {
    inner: Mutex<HashMap<String, Arc<PullEntry>>>,
}

impl PullTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `name` as pulling, or `Err` if it already is.
    pub fn start(&self, name: &str) -> Result<Arc<PullEntry>, PullInProgress> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.contains_key(name) {
            return Err(PullInProgress);
        }
        let entry = Arc::new(PullEntry::new(name));
        inner.insert(name.to_string(), entry.clone());
        Ok(entry)
    }

    /// Ends tracking for `name`, on success, failure or cancellation alike.
    pub fn finish(&self, name: &str) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
    }

    /// Live snapshots, sorted by model name.
    pub fn list(&self) -> Vec<Value> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<Value> = inner.values().map(|e| e.to_json()).collect();
        rows.sort_by(|a, b| a["model"].as_str().cmp(&b["model"].as_str()));
        rows
    }

    /// Signals cancellation of `name`'s pull; `Err` if none is running.
    pub fn cancel(&self, name: &str) -> Result<(), PullNotFound> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match inner.get(name) {
            Some(entry) => {
                entry.cancel.store(true, Ordering::Relaxed);
                Ok(())
            }
            None => Err(PullNotFound),
        }
    }
}
