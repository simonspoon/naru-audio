//! `GET`/`PUT /api/settings` (naru task 2869): the admin's Settings tab over
//! the `config.toml` keys outside `[defaults]`' models and voices (those are
//! `/api/defaults`). A `PUT` validates every key it names before writing any,
//! then edits only those keys, keeping every other table, key and comment.

use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Extension, State};
use serde_json::{Map, Value, json};

use super::live::checkin::DEFAULT_NARU_URL;
use super::{AppState, RequestId, transcriptions};
use crate::error::ApiError;
use crate::manager::{DEFAULT_KEEP_ALIVE, KeepAlive, Settings, edit_config, parse_size};

/// The keys, as `<table>.<key>`.
const KEEP_ALIVE: &str = "defaults.keep_alive";
const MAX_RESIDENT: &str = "memory.max_resident";
const MIC_DEVICE: &str = "mic.device";
const NARU_URL: &str = "naru.url";
const DEPTH_CACHE: &str = "mlx.breeze_depth_cache";
const KEYS: [&str; 5] = [KEEP_ALIVE, MAX_RESIDENT, MIC_DEVICE, NARU_URL, DEPTH_CACHE];

/// How long a request waits for the input device list.
const MIC_LIST_WAIT: Duration = Duration::from_secs(2);
static MIC_LIST: LazyLock<InputList> = LazyLock::new(|| InputList::new(crate::mic::input_names));

/// A device lister run on its own thread, one run at a time, and waited on
/// for at most a set time (naru task 2883): a CoreAudio call can block for
/// good, and a blocked run's thread never returns, so it is never doubled.
struct InputList {
    list: fn() -> Vec<String>,
    /// When the run in flight started; `None` when none is.
    running: Mutex<Option<Instant>>,
    done: tokio::sync::watch::Sender<Vec<String>>,
}

impl InputList {
    fn new(list: fn() -> Vec<String>) -> Self {
        Self {
            list,
            running: Mutex::new(None),
            done: tokio::sync::watch::Sender::new(Vec::new()),
        }
    }

    /// The names, from a fresh run or one already in flight. A run older
    /// than `wait` is reported at once instead of waited on again.
    async fn names(&'static self, wait: Duration) -> Result<Vec<String>, String> {
        let mut done = self.done.subscribe();
        {
            let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
            match *running {
                Some(since) if since.elapsed() >= wait => {
                    return Err(format!(
                        "could not list input devices: CoreAudio has not answered for {}s",
                        since.elapsed().as_secs()
                    ));
                }
                Some(_) => {}
                None => {
                    *running = Some(Instant::now());
                    let spawned =
                        std::thread::Builder::new()
                            .name("input-list".into())
                            .spawn(move || {
                                let names = (self.list)();
                                *self.running.lock().unwrap_or_else(|e| e.into_inner()) = None;
                                self.done.send_replace(names);
                            });
                    if let Err(e) = spawned {
                        *running = None;
                        return Err(format!("could not list input devices: {e}"));
                    }
                }
            }
        }
        match tokio::time::timeout(wait, done.changed()).await {
            Ok(Ok(())) => Ok(done.borrow().clone()),
            _ => Err(format!(
                "could not list input devices: CoreAudio did not answer within {}s",
                wait.as_secs()
            )),
        }
    }
}

/// `GET /api/settings`: per key its `configured` value in `config.toml`
/// (`null` when unset), the `effective` value in force, its `default`, the
/// `env` var that overrides it and whether that is set (`env_override`), and
/// `restart_needed` when the file says something the running server is not
/// using. Also `mic_devices` (input device names; empty with a
/// `mic_devices_error` when they could not be listed in time) and
/// `keep_recent`.
pub(super) async fn get(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(settings_json(&st, MIC_LIST.names(MIC_LIST_WAIT).await))
}

/// `PUT /api/settings` `{"<table>.<key>": value | null, ...}`: `null` removes
/// the key (back to its default). Any unknown key or invalid value is a 400
/// naming it, and nothing is written.
pub(super) async fn put(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let body: Map<String, Value> = serde_json::from_slice(&body).map_err(|_| {
        transcriptions::bad_request(
            "body",
            "invalid_request",
            "the body must be a JSON object of setting to value",
        )
    })?;
    let mut edits = Vec::new();
    for (key, value) in &body {
        let Some(key) = KEYS.iter().find(|k| *k == key) else {
            return Err(transcriptions::bad_request(
                "body",
                "invalid_request",
                format!("unknown setting {key:?}; known: {}", KEYS.join(", ")),
            ));
        };
        let value = validate(key, value)
            .map_err(|e| transcriptions::bad_request(key, "invalid_setting", e))?;
        edits.push((*key, value));
    }
    write(st.registry.home(), &edits).map_err(|e| transcriptions::internal(&st, &req_id, e))?;
    Ok(Json(settings_json(
        &st,
        MIC_LIST.names(MIC_LIST_WAIT).await,
    )))
}

/// The value to write for `key`, `None` to remove it.
fn validate(key: &str, value: &Value) -> Result<Option<toml_edit::Value>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let text = || {
        value
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{key} must be a non-empty string or null"))
    };
    let v = match key {
        KEEP_ALIVE => {
            let s = text()?;
            KeepAlive::parse(s).map_err(|e| {
                format!(
                    "{e}; use a duration like \"5m\" or \"1h\", 0 to unload once idle, -1 for never"
                )
            })?;
            s.into()
        }
        MAX_RESIDENT => {
            let s = text()?;
            match parse_size(s) {
                Ok(0) => return Err("max_resident must be more than 0".to_string()),
                Ok(_) => s.into(),
                Err(e) => return Err(e),
            }
        }
        MIC_DEVICE => text()?.into(),
        NARU_URL => {
            let s = text()?;
            let host = s
                .strip_prefix("http://")
                .or_else(|| s.strip_prefix("https://"))
                .unwrap_or("");
            if host.is_empty() || host.starts_with('/') {
                return Err(format!("{s:?} is not a URL like \"http://127.0.0.1:7777\""));
            }
            s.into()
        }
        DEPTH_CACHE => value
            .as_bool()
            .ok_or("breeze_depth_cache must be true or false")?
            .into(),
        _ => unreachable!("validate: unknown key {key}"),
    };
    Ok(Some(v))
}

/// Applies `edits` to `config.toml`. A replaced value keeps its comments; a
/// removed key takes the comment lines directly above it with it.
fn write(home: &Path, edits: &[(&str, Option<toml_edit::Value>)]) -> Result<(), String> {
    if edits.is_empty() {
        return Ok(());
    }
    edit_config(home, |config| {
        for (path, value) in edits {
            let (table, key) = path.split_once('.').expect("keys are <table>.<key>");
            match value {
                None => {
                    if let Some(t) = config.get_mut(table).and_then(|t| t.as_table_like_mut()) {
                        t.remove(key);
                    }
                }
                Some(value) => {
                    let t = config
                        .entry(table)
                        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
                    let Some(t) = t.as_table_like_mut() else {
                        return Err(format!("config.toml: `{table}` is not a table"));
                    };
                    let mut value = value.clone();
                    if let Some(old) = t.get(key).and_then(|i| i.as_value()) {
                        *value.decor_mut() = old.decor().clone();
                    }
                    t.insert(key, toml_edit::Item::Value(value));
                }
            }
        }
        Ok(())
    })
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn fmt_keep_alive(k: KeepAlive) -> String {
    match k {
        KeepAlive::Forever => "-1".to_string(),
        KeepAlive::For(d) if d == Duration::ZERO => "0".to_string(),
        KeepAlive::For(d) => humantime::format_duration(d).to_string(),
    }
}

fn settings_json(st: &AppState, mic_devices: Result<Vec<String>, String>) -> Value {
    let home = st.registry.home();
    let running = st.models.settings();
    let (config, config_error) = match crate::registry::read_config(home) {
        Ok(c) => (c, None),
        Err(e) => (toml::Table::new(), Some(e)),
    };
    let configured = |path: &str| -> Value {
        let (table, key) = path.split_once('.').unwrap();
        config
            .get(table)
            .and_then(|t| t.get(key))
            .map(|v| serde_json::to_value(v).unwrap_or(Value::Null))
            .unwrap_or(Value::Null)
    };
    // What a restart would use, env included, beside what is running.
    let reloaded = Settings::load(running.profile.clone(), home);
    let (ka_restart, budget_restart, settings_error) = match &reloaded {
        Ok(s) => (
            s.keep_alive != running.keep_alive,
            s.budget_bytes != running.budget_bytes,
            None,
        ),
        Err(e) => (false, false, Some(e.clone())),
    };
    let ka_env = env("NARU_AUDIO_KEEP_ALIVE");
    let mic_env = env("NARU_AUDIO_MIC_DEVICE");
    let url_env = env("NARU_AUDIO_NARU_URL");
    let url_now = super::naru_url(home);
    // The mlx module exists only on Apple-silicon macOS (src/lib.rs).
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let depth_cache = crate::mlx::breeze_depth_cache(home);
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    let depth_cache: Result<bool, String> = Ok(false);
    let mic_device = crate::mic::configured_device(home);
    // An unreadable file fails every reader the same way: report it once.
    let error = [
        config_error.clone(),
        settings_error,
        depth_cache.as_ref().err().cloned(),
        mic_device.as_ref().err().cloned(),
        url_now.as_ref().err().cloned(),
    ]
    .into_iter()
    .flatten()
    .take(if config_error.is_some() {
        1
    } else {
        usize::MAX
    })
    .collect::<Vec<_>>();
    json!({
        "config_path": crate::registry::config_path(home),
        "error": (!error.is_empty()).then(|| error.join("; ")),
        "settings": {
            KEEP_ALIVE: {
                "configured": configured(KEEP_ALIVE),
                "effective": fmt_keep_alive(running.keep_alive),
                "default": fmt_keep_alive(KeepAlive::For(DEFAULT_KEEP_ALIVE)),
                "env": "NARU_AUDIO_KEEP_ALIVE",
                "env_override": ka_env.is_some(),
                "applies": "restart",
                "restart_needed": ka_restart,
            },
            MAX_RESIDENT: {
                "configured": configured(MAX_RESIDENT),
                "effective": running.budget_bytes,
                "default": running.profile.budget_bytes(),
                "env": null,
                "env_override": false,
                "applies": "restart",
                "restart_needed": budget_restart,
            },
            MIC_DEVICE: {
                "configured": configured(MIC_DEVICE),
                "effective": mic_device.ok().flatten(),
                "default": null,
                "env": "NARU_AUDIO_MIC_DEVICE",
                "env_override": mic_env.is_some(),
                "applies": "listener_start",
                "restart_needed": false,
            },
            NARU_URL: {
                "configured": configured(NARU_URL),
                "effective": st.naru_url,
                "default": DEFAULT_NARU_URL,
                "env": "NARU_AUDIO_NARU_URL",
                "env_override": url_env.is_some(),
                "applies": "restart",
                "restart_needed": url_now.ok() != st.naru_url,
            },
            DEPTH_CACHE: {
                "configured": configured(DEPTH_CACHE),
                "effective": depth_cache.as_ref().ok(),
                "default": false,
                "available": cfg!(all(target_arch = "aarch64", target_os = "macos")),
                "env": null,
                "env_override": false,
                "applies": "model_load",
                "restart_needed": false,
            },
        },
        "mic_devices": mic_devices.as_deref().unwrap_or_default(),
        "mic_devices_error": mic_devices.err(),
        "keep_recent": {
            "on": st.keep_recent,
            "env": "NARU_AUDIO_KEEP_RECENT",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::config_path;

    const ORIGINAL: &str = "# my config\n[defaults]\nstt = \"x\"   # keep me\nkeep_alive = \"5m\"  # idle\n\n[mlx]\nbreeze_depth_cache = false # Breeze\nother = 1\n\n[extra]\nk = [1, 2]\n";

    fn home_with(text: &str) -> tempfile::TempDir {
        let home = tempfile::tempdir().unwrap();
        std::fs::write(config_path(home.path()), text).unwrap();
        home
    }

    fn edit(key: &'static str, value: Value) -> Result<String, String> {
        let home = home_with(ORIGINAL);
        let v = validate(key, &value)?;
        write(home.path(), &[(key, v)])?;
        Ok(std::fs::read_to_string(config_path(home.path())).unwrap())
    }

    #[test]
    fn toggling_depth_cache_changes_only_that_value() {
        let text = edit(DEPTH_CACHE, json!(true)).unwrap();
        assert_eq!(
            text,
            ORIGINAL.replace(
                "breeze_depth_cache = false # Breeze",
                "breeze_depth_cache = true # Breeze"
            )
        );
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        let home = home_with(&text);
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        assert_eq!(crate::mlx::breeze_depth_cache(home.path()), Ok(true));
    }

    #[test]
    fn a_new_table_is_appended_and_null_removes_a_key() {
        let text = edit(MIC_DEVICE, json!("BlackHole 2ch")).unwrap();
        assert!(text.starts_with(ORIGINAL), "{text}");
        assert!(
            text.ends_with("[mic]\ndevice = \"BlackHole 2ch\"\n"),
            "{text}"
        );
        let text = edit(KEEP_ALIVE, Value::Null).unwrap();
        assert_eq!(text, ORIGINAL.replace("keep_alive = \"5m\"  # idle\n", ""));
    }

    #[test]
    fn bad_values_are_refused() {
        for (key, value) in [
            (KEEP_ALIVE, json!("soon")),
            (KEEP_ALIVE, json!("")),
            (KEEP_ALIVE, json!(5)),
            (MAX_RESIDENT, json!("lots")),
            (MAX_RESIDENT, json!("0")),
            (MAX_RESIDENT, json!("8 TB please")),
            (NARU_URL, json!("127.0.0.1:7777")),
            (NARU_URL, json!("http://")),
            (DEPTH_CACHE, json!("yes")),
            (MIC_DEVICE, json!(" ")),
        ] {
            assert!(validate(key, &value).is_err(), "{key} {value}");
        }
        for (key, value) in [
            (KEEP_ALIVE, json!("1h")),
            (KEEP_ALIVE, json!("-1")),
            (KEEP_ALIVE, json!("0")),
            (MAX_RESIDENT, json!("8GiB")),
            (NARU_URL, json!("http://127.0.0.1:7777")),
        ] {
            assert!(validate(key, &value).is_ok(), "{key} {value}");
        }
    }

    fn state(home: &Path) -> Arc<AppState> {
        use crate::log::Logger;
        use crate::manager::{BackendLoader, ModelManager};
        use crate::registry::Registry;
        let log = Arc::new(Logger::stderr());
        let registry = Arc::new(Registry::open(home).unwrap());
        let settings = Settings::from_env(crate::profile::Profile::detect().unwrap()).unwrap();
        let models = ModelManager::new(
            registry.clone(),
            log.clone(),
            settings,
            Arc::new(BackendLoader),
        );
        Arc::new(AppState {
            port: 7870,
            allow_remote: false,
            started: Instant::now(),
            log,
            registry,
            models: Arc::new(models),
            pulls: Arc::new(super::super::PullTracker::new()),
            keep_recent: false,
            mic: None,
            naru_url: None,
        })
    }

    /// A device lister that never returns (a CoreAudio call stuck in
    /// coreaudiod): requests answer within the wait, start one run between
    /// them, and leave the server answering.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stuck_device_list_never_hangs_settings() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn stuck() -> Vec<String> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            loop {
                std::thread::park();
            }
        }
        let list: &'static InputList = Box::leak(Box::new(InputList::new(stuck)));
        let home = home_with(ORIGINAL);
        let st = state(home.path());
        for i in 0..10 {
            let t = Instant::now();
            let body = settings_json(&st, list.names(MIC_LIST_WAIT).await);
            let took = t.elapsed();
            assert!(
                took < MIC_LIST_WAIT + Duration::from_millis(500),
                "#{i} took {took:?}"
            );
            assert_eq!(body["mic_devices"], json!([]), "{body}");
            let err = body["mic_devices_error"].as_str().unwrap();
            assert!(err.contains("could not list input devices"), "{err}");
            let settings = body["settings"].as_object().unwrap();
            assert_eq!(settings.len(), 5, "{body}");
            assert_eq!(settings[KEEP_ALIVE]["configured"], "5m");
            assert_eq!(settings[DEPTH_CACHE]["configured"], false);
            assert_eq!(body["error"], Value::Null, "{body}");
        }
        assert_eq!(RUNS.load(Ordering::SeqCst), 1);

        use tower::ServiceExt;
        let resp = super::super::router(st)
            .oneshot(
                axum::http::Request::get("/health")
                    .header("host", "127.0.0.1:7870")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn a_device_list_that_answers_is_fresh_each_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn quick() -> Vec<String> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            vec!["BlackHole 2ch".to_string()]
        }
        let list: &'static InputList = Box::leak(Box::new(InputList::new(quick)));
        for _ in 0..2 {
            assert_eq!(
                list.names(MIC_LIST_WAIT).await,
                Ok(vec!["BlackHole 2ch".to_string()])
            );
        }
        assert_eq!(RUNS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn keep_alive_formats_as_config_accepts() {
        for k in [
            KeepAlive::Forever,
            KeepAlive::For(Duration::ZERO),
            KeepAlive::For(Duration::from_secs(300)),
            KeepAlive::For(Duration::from_secs(5400)),
        ] {
            assert_eq!(KeepAlive::parse(&fmt_keep_alive(k)), Ok(k));
        }
    }
}
