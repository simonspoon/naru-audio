//! §3.3–3.5 model manager: default resolution, keep-alive, the memory
//! budget, LRU eviction of idle models, and measured resident bytes.
//!
//! One `Mutex` guards the bookkeeping and is only ever held for map updates:
//! never across a load, a decode or file I/O, so `/health` and `/api/ps`
//! never wait behind either (§4.3). A load reserves its bytes in a `loading`
//! slot first, so concurrent loads cannot overshoot the budget together and
//! a second request for the same model waits for the first load instead of
//! loading it again.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant, SystemTime};

use tokio::sync::watch;

use crate::log::Logger;
use crate::profile::Profile;
use crate::registry::Registry;
use crate::registry::manifest::{Kind, Manifest};
use crate::stt::{SttError, SttModel};
use crate::tts::{TtsError, TtsModel};

/// §3.4 default when neither the request nor `NARU_AUDIO_KEEP_ALIVE` sets one.
const DEFAULT_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);

/// §3.4 `keep_alive`: how long a model stays loaded once idle. `For(0)`
/// unloads as soon as the model is idle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeepAlive {
    Forever,
    For(Duration),
}

impl KeepAlive {
    /// Seconds (`"300"`, `"-1"`) or a duration string (`"5m"`, `"1h30m"`);
    /// negative means never unload.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if let Ok(secs) = s.parse::<f64>() {
            return Self::from_secs(secs);
        }
        let (negative, rest) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let d = humantime::parse_duration(rest)
            .map_err(|e| format!("keep_alive {s:?} is not a duration or seconds: {e}"))?;
        Ok(if negative {
            KeepAlive::Forever
        } else {
            KeepAlive::For(d)
        })
    }

    pub fn from_secs(secs: f64) -> Result<Self, String> {
        if secs.is_nan() {
            return Err("keep_alive must be a number of seconds, not NaN".to_string());
        }
        if secs < 0.0 {
            return Ok(KeepAlive::Forever);
        }
        Duration::try_from_secs_f64(secs)
            .map(KeepAlive::For)
            .map_err(|_| format!("keep_alive {secs} is too large"))
    }

    pub fn is_zero(self) -> bool {
        self == KeepAlive::For(Duration::ZERO)
    }
}

/// What the manager is configured with: §3.3 defaults, §3.4 keep-alive and
/// the §3.5 budget.
#[derive(Debug, Clone)]
pub struct Settings {
    pub profile: Profile,
    pub budget_bytes: u64,
    pub keep_alive: KeepAlive,
    pub stt_default: String,
    pub tts_default: String,
}

impl Settings {
    /// `NARU_AUDIO_STT_MODEL`, `NARU_AUDIO_TTS_MODEL` and
    /// `NARU_AUDIO_KEEP_ALIVE` over the profile's defaults.
    pub fn from_env(profile: Profile) -> Result<Self, String> {
        let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let keep_alive = match env("NARU_AUDIO_KEEP_ALIVE") {
            Some(v) => KeepAlive::parse(&v).map_err(|e| format!("NARU_AUDIO_KEEP_ALIVE: {e}"))?,
            None => KeepAlive::For(DEFAULT_KEEP_ALIVE),
        };
        Ok(Self {
            budget_bytes: profile.budget_bytes(),
            keep_alive,
            stt_default: env("NARU_AUDIO_STT_MODEL")
                .unwrap_or_else(|| profile.default_stt().to_string()),
            tts_default: env("NARU_AUDIO_TTS_MODEL")
                .unwrap_or_else(|| profile.default_tts().to_string()),
            profile,
        })
    }
}

/// A loaded model.
#[derive(Clone)]
pub enum Resident {
    Stt(Arc<dyn SttModel>),
    Tts(Arc<dyn TtsModel>),
}

pub struct LoadedModel {
    pub model: Resident,
    /// The RSS the load added (§3.5), when it could be measured.
    pub measured_bytes: Option<u64>,
}

/// Why a model did not load, by its kind.
#[derive(Debug)]
pub enum LoadError {
    Stt(SttError),
    Tts(TtsError),
}

impl LoadError {
    /// §3.6: the backend cannot run here (§2.6 `backend_unavailable`).
    pub fn is_backend_unavailable(&self) -> bool {
        matches!(
            self,
            LoadError::Stt(SttError::BackendUnavailable { .. })
                | LoadError::Tts(TtsError::BackendUnavailable { .. })
        )
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Stt(e) => e.fmt(f),
            LoadError::Tts(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<SttError> for LoadError {
    fn from(e: SttError) -> Self {
        LoadError::Stt(e)
    }
}

impl From<TtsError> for LoadError {
    fn from(e: TtsError) -> Self {
        LoadError::Tts(e)
    }
}

/// Loads a pulled model from its directory. Tests inject a fake.
pub trait Loader: Send + Sync {
    fn load(&self, manifest: &Manifest, dir: &Path) -> Result<LoadedModel, LoadError>;
}

/// The real loader: `backend::load_stt` or `backend::load_tts` by the
/// manifest's kind, measured by the RSS delta or by what the model reports
/// ([`SttModel::resident_bytes`]).
pub struct BackendLoader;

/// One load at a time, so no other load's allocations land in a delta.
static LOADS: Mutex<()> = Mutex::new(());

impl Loader for BackendLoader {
    fn load(&self, manifest: &Manifest, dir: &Path) -> Result<LoadedModel, LoadError> {
        let _one_at_a_time = LOADS.lock().unwrap_or_else(|e| e.into_inner());
        let before = crate::profile::rss();
        let (model, reported) = if manifest.model.kind == Kind::Tts {
            let model = crate::backend::load_tts(manifest, dir)?;
            (Resident::Tts(Arc::from(model)), None)
        } else {
            let model = crate::backend::load_stt(manifest, dir)?;
            // An MLX model's bytes are in the sidecar, not in this delta.
            let reported = model.resident_bytes();
            (Resident::Stt(Arc::from(model)), reported)
        };
        let after = crate::profile::rss();
        let measured_bytes = reported.or_else(|| {
            before
                .zip(after)
                .map(|(b, a)| a.saturating_sub(b))
                .filter(|&d| d > 0)
        });
        Ok(LoadedModel {
            model,
            measured_bytes,
        })
    }
}

#[derive(Debug)]
pub enum ManagerError {
    /// §3.5: the model does not fit even after evicting every idle model.
    InsufficientMemory(String),
    Load(LoadError),
    /// The load panicked.
    Internal(String),
}

/// `DELETE` or unload of a model with requests in flight, or still loading.
#[derive(Debug)]
pub struct Busy;

/// One row of `/api/ps`.
#[derive(Debug, Clone)]
pub struct Status {
    pub name: String,
    pub kind: Kind,
    pub backend: String,
    pub resident_bytes: u64,
    /// When an idle model unloads; `None` while busy, loading, or kept forever.
    pub expires_at: Option<SystemTime>,
    pub busy: bool,
    pub loading: bool,
}

pub struct ModelManager {
    registry: Arc<Registry>,
    log: Arc<Logger>,
    loader: Arc<dyn Loader>,
    settings: Settings,
    measured_path: PathBuf,
    inner: Mutex<Inner>,
    /// Serialises `measured.json` writes: one tmp file, newest snapshot last.
    measured_write: Mutex<()>,
}

struct Inner {
    slots: HashMap<String, Slot>,
    /// `state/measured.json`: model name, then backend, to resident bytes.
    measured: BTreeMap<String, BTreeMap<String, u64>>,
    /// Models unloaded or evicted so far, counted as their drop starts. A
    /// load that sees it change has an RSS delta that is not its own.
    unloads: u64,
    /// Drops still running ([`ModelManager::dispose`]).
    disposing: usize,
}

struct Slot {
    kind: Kind,
    backend: String,
    resident_bytes: u64,
    state: State,
    /// Requests holding a [`Guard`]; the loading request counts from the start.
    in_flight: usize,
    last_used: Instant,
    keep_alive: KeepAlive,
    expires_at: Option<SystemTime>,
    /// Bumped on every acquire and release, so a stale idle timer does nothing.
    generation: u64,
}

enum State {
    /// Waiters wake when the loading task drops the sender.
    Loading(watch::Receiver<()>),
    Ready(Resident),
}

impl Slot {
    fn is_idle(&self) -> bool {
        self.in_flight == 0 && matches!(self.state, State::Ready(_))
    }
}

enum Step {
    Ready(Guard),
    Wait(watch::Receiver<()>),
    /// This caller loads; `tx` wakes the waiters when dropped.
    Load {
        tx: watch::Sender<()>,
        need: u64,
        evicted: Vec<(String, Slot)>,
    },
}

/// A model in use. Dropping it ends the request, and the last one out
/// starts the keep-alive timer (§3.4).
pub struct Guard {
    manager: Arc<ModelManager>,
    name: String,
    model: Resident,
}

impl Guard {
    /// Panics on a TTS model: callers acquire with an STT manifest.
    pub fn stt(&self) -> &Arc<dyn SttModel> {
        match &self.model {
            Resident::Stt(model) => model,
            Resident::Tts(_) => panic!("\"{}\" is not a speech-to-text model", self.name),
        }
    }

    /// Panics on an STT model: callers acquire with a TTS manifest.
    pub fn tts(&self) -> &Arc<dyn TtsModel> {
        match &self.model {
            Resident::Tts(model) => model,
            Resident::Stt(_) => panic!("\"{}\" is not a text-to-speech model", self.name),
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.manager.release(&self.name);
    }
}

impl ModelManager {
    pub fn new(
        registry: Arc<Registry>,
        log: Arc<Logger>,
        settings: Settings,
        loader: Arc<dyn Loader>,
    ) -> Self {
        let measured_path = registry.home().join("state").join("measured.json");
        let measured = match std::fs::read(&measured_path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                log.line(
                    "warn",
                    None,
                    &format!("measured_unreadable path={} {e}", measured_path.display()),
                );
                BTreeMap::new()
            }),
            Err(_) => BTreeMap::new(),
        };
        Self {
            registry,
            log,
            loader,
            settings,
            measured_path,
            inner: Mutex::new(Inner {
                slots: HashMap::new(),
                measured,
                unloads: 0,
                disposing: 0,
            }),
            measured_write: Mutex::new(()),
        }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The configured default for `kind`; VAD models have none.
    pub fn default_for(&self, kind: Kind) -> Option<&str> {
        match kind {
            Kind::Stt => Some(&self.settings.stt_default),
            Kind::Tts => Some(&self.settings.tts_default),
            Kind::Vad => None,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Loaded and loading models, by name.
    pub fn ps(&self) -> Vec<Status> {
        let inner = self.lock();
        let mut rows: Vec<Status> = inner
            .slots
            .iter()
            .map(|(name, s)| Status {
                name: name.clone(),
                kind: s.kind,
                backend: s.backend.clone(),
                resident_bytes: s.resident_bytes,
                expires_at: s.expires_at,
                busy: s.in_flight > 0,
                loading: matches!(s.state, State::Loading(_)),
            })
            .collect();
        drop(inner);
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        rows
    }

    /// The model, loaded if need be (§3.4), with `keep_alive` (or the
    /// default) applying once it is idle again.
    pub async fn acquire(
        self: &Arc<Self>,
        manifest: Manifest,
        keep_alive: Option<KeepAlive>,
    ) -> Result<Guard, ManagerError> {
        let name = manifest.model.name.clone();
        let keep_alive = keep_alive.unwrap_or(self.settings.keep_alive);
        let (tx, need, evicted) = loop {
            match self.try_acquire(&manifest, keep_alive)? {
                Step::Ready(guard) => return Ok(guard),
                // Err once the loader is done, either way.
                Step::Wait(mut rx) => drop(rx.changed().await),
                Step::Load { tx, need, evicted } => break (tx, need, evicted),
            }
        };
        // The evictions and the load run in their own task so a caller
        // that goes away mid-load cannot strand the `loading` slot; its
        // guard is then dropped with the task's output.
        let this = self.clone();
        let task = tokio::spawn(async move {
            let _wake_waiters = tx;
            for (evicted, slot) in evicted {
                if let Some(dropped) = this.dispose(&evicted, "evict", Some(slot)) {
                    let _ = dropped.await;
                }
            }
            // After the evictions are dropped: they free memory before the
            // load. A drop still running elsewhere could free memory during
            // it, so its delta would not be its own.
            let unloads = {
                let inner = this.lock();
                (inner.disposing == 0).then_some(inner.unloads)
            };
            let started = Instant::now();
            let loader = this.loader.clone();
            let dir = this.registry.model_dir(&manifest.model.name);
            let result = tokio::task::spawn_blocking(move || loader.load(&manifest, &dir)).await;
            this.finish_load(name, need, unloads, started, result)
        });
        task.await
            .map_err(|e| ManagerError::Internal(e.to_string()))?
    }

    /// The locked part of [`ModelManager::acquire`]: take a loaded model,
    /// wait on a loading one, or reserve room and a `loading` slot.
    fn try_acquire(
        self: &Arc<Self>,
        manifest: &Manifest,
        keep_alive: KeepAlive,
    ) -> Result<Step, ManagerError> {
        let name = &manifest.model.name;
        let mut inner = self.lock();
        if let Some(slot) = inner.slots.get_mut(name) {
            return Ok(match &slot.state {
                State::Ready(model) => {
                    let model = model.clone();
                    slot.in_flight += 1;
                    slot.generation += 1;
                    slot.keep_alive = keep_alive;
                    slot.expires_at = None;
                    Step::Ready(Guard {
                        manager: self.clone(),
                        name: name.clone(),
                        model,
                    })
                }
                State::Loading(rx) => Step::Wait(rx.clone()),
            });
        }
        let need = inner.need(manifest);
        let evicted = inner.make_room(name, need, self.settings.budget_bytes)?;
        let (tx, rx) = watch::channel(());
        inner.slots.insert(
            name.clone(),
            Slot {
                kind: manifest.model.kind,
                backend: manifest.model.backend.clone(),
                resident_bytes: need,
                state: State::Loading(rx),
                in_flight: 1,
                last_used: Instant::now(),
                keep_alive,
                expires_at: None,
                generation: 0,
            },
        );
        Ok(Step::Load { tx, need, evicted })
    }

    fn finish_load(
        self: &Arc<Self>,
        name: String,
        need: u64,
        unloads: Option<u64>,
        started: Instant,
        result: Result<Result<LoadedModel, LoadError>, tokio::task::JoinError>,
    ) -> Result<Guard, ManagerError> {
        let loaded = match result {
            Ok(Ok(loaded)) => loaded,
            Ok(Err(e)) => {
                self.lock().slots.remove(&name);
                return Err(ManagerError::Load(e));
            }
            Err(e) => {
                self.lock().slots.remove(&name);
                return Err(ManagerError::Internal(format!("the load failed: {e}")));
            }
        };
        let mut inner = self.lock();
        // Memory freed during the load would shrink the delta: keep the estimate.
        let measured = loaded
            .measured_bytes
            .filter(|_| unloads == Some(inner.unloads));
        let Some(slot) = inner.slots.get_mut(&name) else {
            unreachable!("a loading slot is never removed by anyone else");
        };
        slot.state = State::Ready(loaded.model.clone());
        slot.generation += 1;
        slot.resident_bytes = measured.unwrap_or(need);
        let backend = slot.backend.clone();
        if let Some(bytes) = measured {
            inner
                .measured
                .entry(name.clone())
                .or_default()
                .insert(backend, bytes);
        }
        drop(inner);
        if let (Some(bytes), None) = (loaded.measured_bytes, measured) {
            self.log.line(
                "warn",
                None,
                &format!("measure_discarded model={name} rss={bytes} reason=unload_during_load"),
            );
        }

        self.log.info(
            None,
            &format!(
                "load model={name} ms={} rss={}",
                started.elapsed().as_millis(),
                loaded
                    .measured_bytes
                    .map_or("-".to_string(), |b| b.to_string())
            ),
        );
        if measured.is_some() {
            self.write_measured();
        }
        Ok(Guard {
            manager: self.clone(),
            name,
            model: loaded.model,
        })
    }

    fn write_measured(&self) {
        let _one_writer = self
            .measured_write
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Taken under the write lock, so a later write never has an older
        // snapshot.
        let json = serde_json::to_vec_pretty(&self.lock().measured).unwrap_or_default();
        let path = &self.measured_path;
        let tmp = path.with_extension("json.tmp");
        let result = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&tmp, &json))
            .and_then(|()| std::fs::rename(&tmp, path));
        if let Err(e) = result {
            self.log.line(
                "warn",
                None,
                &format!("measured_write_failed path={} {e}", path.display()),
            );
        }
    }

    /// A request ends; the last one out starts the keep-alive (§3.4).
    fn release(self: &Arc<Self>, name: &str) {
        let mut inner = self.lock();
        let Some(slot) = inner.slots.get_mut(name) else {
            return;
        };
        slot.in_flight -= 1;
        if slot.in_flight > 0 {
            return;
        }
        slot.last_used = Instant::now();
        slot.generation += 1;
        let generation = slot.generation;
        match slot.keep_alive {
            KeepAlive::For(d) if d.is_zero() => {
                let slot = inner.slots.remove(name);
                drop(inner);
                self.dispose(name, "idle", slot);
            }
            KeepAlive::For(d) => {
                // Too far away to represent: in practice never.
                slot.expires_at = SystemTime::now().checked_add(d);
                if slot.expires_at.is_none() {
                    return;
                }
                drop(inner);
                // A guard is only dropped on the runtime's threads (a handler
                // or its `spawn_blocking`), so there is always a runtime.
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    let this = Arc::downgrade(self);
                    let name = name.to_string();
                    rt.spawn(async move {
                        tokio::time::sleep(d).await;
                        if let Some(this) = Weak::upgrade(&this) {
                            this.expire(&name, generation);
                        }
                    });
                }
            }
            KeepAlive::Forever => slot.expires_at = None,
        }
    }

    /// The idle timer fired: unload unless the model was used since.
    fn expire(self: &Arc<Self>, name: &str, generation: u64) {
        let mut inner = self.lock();
        if inner
            .slots
            .get(name)
            .is_some_and(|s| s.generation == generation && s.is_idle())
        {
            let slot = inner.slots.remove(name);
            drop(inner);
            self.dispose(name, "idle", slot);
        }
    }

    /// `keep_alive: 0` (§2.5 `POST /api/load`): unload now if idle,
    /// otherwise when the last request ends. Whether it is still loaded.
    pub fn unload_when_idle(self: &Arc<Self>, name: &str) -> bool {
        match self.unload(name, "idle") {
            Ok(()) => false,
            Err(Busy) => {
                if let Some(slot) = self.lock().slots.get_mut(name) {
                    slot.keep_alive = KeepAlive::For(Duration::ZERO);
                }
                true
            }
        }
    }

    /// Unloads an idle model; `Busy` while it has requests or is loading.
    /// A model that is not loaded is fine.
    pub fn unload(self: &Arc<Self>, name: &str, reason: &str) -> Result<(), Busy> {
        let mut inner = self.lock();
        match inner.slots.get(name) {
            None => return Ok(()),
            Some(s) if !s.is_idle() => return Err(Busy),
            Some(_) => {}
        }
        let slot = inner.slots.remove(name);
        drop(inner);
        self.dispose(name, reason, slot);
        Ok(())
    }

    /// Counts an unloaded or evicted model, then drops it outside the lock.
    /// A drop can block (an MLX model waits for its sidecar), so on the
    /// runtime it runs on the blocking pool, and the task returned ends
    /// once the model is gone.
    fn dispose(
        self: &Arc<Self>,
        name: &str,
        reason: &str,
        slot: Option<Slot>,
    ) -> Option<tokio::task::JoinHandle<()>> {
        self.log
            .info(None, &format!("unload model={name} reason={reason}"));
        {
            let mut inner = self.lock();
            inner.unloads += 1;
            inner.disposing += 1;
        }
        let this = self.clone();
        let drop_slot = move || {
            drop(slot);
            this.lock().disposing -= 1;
        };
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => Some(rt.spawn_blocking(drop_slot)),
            Err(_) => {
                drop_slot();
                None
            }
        }
    }
}

impl Inner {
    /// §3.5 `need = measured(X) ?? manifest.resident_bytes`.
    fn need(&self, manifest: &Manifest) -> u64 {
        let m = &manifest.model;
        self.measured
            .get(&m.name)
            .and_then(|by_backend| by_backend.get(&m.backend))
            .copied()
            .or_else(|| {
                manifest
                    .raw
                    .get("model")
                    .and_then(|v| v.get("resident_bytes"))
                    .and_then(|v| v.as_integer())
                    .and_then(|n| u64::try_from(n).ok())
            })
            .unwrap_or(0)
    }

    /// §3.5: removes idle models, least recently used first, until `need`
    /// fits; removes nothing if even evicting every idle model would not do.
    fn make_room(
        &mut self,
        name: &str,
        need: u64,
        budget: u64,
    ) -> Result<Vec<(String, Slot)>, ManagerError> {
        let used: u64 = self.slots.values().map(|s| s.resident_bytes).sum();
        let mut idle: Vec<(&String, &Slot)> =
            self.slots.iter().filter(|(_, s)| s.is_idle()).collect();
        idle.sort_by_key(|(_, s)| s.last_used);
        let mut left = used;
        let mut victims = Vec::new();
        for (victim, slot) in idle {
            if left.saturating_add(need) <= budget {
                break;
            }
            left -= slot.resident_bytes;
            victims.push(victim.clone());
        }
        if left.saturating_add(need) > budget {
            let mut busy: Vec<&str> = self
                .slots
                .iter()
                .filter(|(_, s)| !s.is_idle())
                .map(|(n, _)| n.as_str())
                .collect();
            busy.sort();
            let busy = if busy.is_empty() {
                String::new()
            } else {
                format!("; busy: {}", busy.join(", "))
            };
            return Err(ManagerError::InsufficientMemory(format!(
                "the model \"{name}\" needs {need} bytes, but only {} of the {budget}-byte \
                 budget can be freed{busy}",
                budget.saturating_sub(left)
            )));
        }
        Ok(victims
            .into_iter()
            .filter_map(|n| self.slots.remove(&n).map(|s| (n, s)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_alive_parses_durations_and_seconds() {
        let secs = |s| KeepAlive::For(Duration::from_secs(s));
        assert_eq!(KeepAlive::parse("5m").unwrap(), secs(300));
        assert_eq!(KeepAlive::parse("300").unwrap(), secs(300));
        assert_eq!(KeepAlive::parse(" 1h 30m ").unwrap(), secs(5400));
        assert_eq!(
            KeepAlive::parse("1.5").unwrap(),
            KeepAlive::For(Duration::from_millis(1500))
        );
        assert!(KeepAlive::parse("0").unwrap().is_zero());
        assert!(KeepAlive::parse("0s").unwrap().is_zero());
        assert_eq!(KeepAlive::parse("-1").unwrap(), KeepAlive::Forever);
        assert_eq!(KeepAlive::parse("-5m").unwrap(), KeepAlive::Forever);
        for bad in ["soon", "", "NaN", "1e300", "5 parsecs"] {
            assert!(KeepAlive::parse(bad).is_err(), "{bad}");
        }
    }
}
