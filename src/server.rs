//! HTTP surface: §2.1 guards, §2.2 transcriptions, §2.3 speech, §2.4
//! streaming transcriptions, §2.5 `/health`, voices and registry routes,
//! §2.6 errors, `X-Request-Id`.

mod speech;
mod stream;
mod transcriptions;

use std::collections::HashMap;
use std::convert::Infallible;
use std::hash::{BuildHasher, RandomState};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Extension, Path, Query, Request, State};
use axum::http::header::{CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::log::Logger;
use crate::manager::{KeepAlive, ManagerError, ModelManager};
use crate::registry::manifest::Kind;
use crate::registry::{Progress, Registry, RegistryError};

pub const DEFAULT_LISTEN: &str = "127.0.0.1:7870";
const REQUEST_ID: &str = "x-request-id";

pub struct AppState {
    /// The bound port; the Host guard only accepts loopback names with it.
    pub port: u16,
    /// `--allow-remote`: skips the Host guard (§2.1).
    pub allow_remote: bool,
    pub started: Instant,
    pub log: Arc<Logger>,
    pub registry: Arc<Registry>,
    pub models: Arc<ModelManager>,
}

/// The request's `X-Request-Id`, for handlers' log lines.
#[derive(Clone)]
struct RequestId(String);

/// §2.1: a non-loopback bind needs `--allow-remote`.
pub fn check_listen(addr: SocketAddr, allow_remote: bool) -> Result<(), String> {
    if addr.ip().is_loopback() || allow_remote {
        Ok(())
    } else {
        Err(format!(
            "refusing to listen on non-loopback address {addr} without --allow-remote"
        ))
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route(
            "/v1/audio/transcriptions",
            post(transcriptions::transcriptions)
                .layer(DefaultBodyLimit::max(transcriptions::MAX_BODY_BYTES)),
        )
        .route("/v1/audio/transcriptions/stream", get(stream::stream))
        .route("/v1/audio/speech", post(speech::speech))
        .route(
            "/v1/audio/voices",
            get(speech::voices)
                .post(speech::add_voice)
                .layer(DefaultBodyLimit::max(transcriptions::MAX_BODY_BYTES)),
        )
        .route(
            "/v1/audio/voices/{name}",
            get(speech::voice_export).delete(speech::delete_voice),
        )
        .route("/api/pull", post(pull))
        .route("/api/models/{name}", delete(remove))
        .route("/api/ps", get(ps))
        .route("/api/load", post(load))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// §2.5 `/health`. The model manager's lock is never held across a load or
/// a decode, so this never waits behind either (§4.3). `ready` reads and
/// parses each default model's small `manifest.json` and `stat`s what it
/// requires, so a pull or `rm` by the CLI shows at once.
async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    let settings = st.models.settings();
    let profile = &settings.profile;
    let ps = st.models.ps();
    let stt = &settings.stt_default;
    let state = ps.iter().find(|m| &m.name == stt);
    let tts = &settings.tts_default;
    let tts_state = ps.iter().find(|m| &m.name == tts);
    let backends: Vec<Value> = ["sherpa-onnx", "mlx"]
        .iter()
        .map(|name| match crate::backend::available(name) {
            Ok(()) => json!({"name": name, "available": true}),
            Err(reason) => json!({"name": name, "available": false, "reason": reason}),
        })
        .collect();
    // §5.3: a crashed sidecar says so, until it is back.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let backends = {
        let mut backends = backends;
        if let Some(reason) = crate::mlx::sidecar::reason(st.registry.home()) {
            backends[1]["reason"] = json!(reason);
        }
        backends
    };
    let problems: Vec<Value> = profile
        .problem()
        .map(|message| json!({"code": "rosetta", "message": message}))
        .into_iter()
        .collect();
    let stt_problem = default_problem(&st.registry, stt);
    let tts_problem = default_problem(&st.registry, tts);
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "api": 1,
        "pid": std::process::id(),
        "uptime_s": st.started.elapsed().as_secs(),
        "profile": {
            "name": profile.name,
            "arch": profile.arch,
            "translated": profile.translated,
            "ram_bytes": profile.ram_bytes,
            "budget_bytes": settings.budget_bytes,
        },
        "stt": {
            "default": stt,
            "ready": stt_problem.is_none(),
            "problem": stt_problem,
            "loaded": state.is_some_and(|m| !m.loading),
            "loading": state.is_some_and(|m| m.loading),
        },
        "tts": {
            "default": tts,
            "ready": tts_problem.is_none(),
            "problem": tts_problem,
            "loaded": tts_state.is_some_and(|m| !m.loading),
            "loading": tts_state.is_some_and(|m| m.loading),
        },
        "backends": backends,
        "problems": problems,
    }))
}

/// Why a default model is not ready (§2.5: pulled and loadable), if it is
/// not: not pulled, a `requires` not pulled, or its backend.
fn default_problem(registry: &Registry, name: &str) -> Option<Value> {
    let problem = |code: &str, message: String| Some(json!({"code": code, "message": message}));
    let manifest = match registry.pulled_manifest(name) {
        Ok(m) => m,
        Err(e) => {
            let e = registry_error(e);
            return problem(e.code, e.message);
        }
    };
    if let Err(reason) = crate::backend::available(&manifest.model.backend) {
        return problem("backend_unavailable", reason);
    }
    manifest
        .model
        .requires
        .iter()
        .find(|r| !registry.is_installed(r))
        .and_then(|r| {
            problem(
                "model_not_pulled",
                format!("\"{name}\" requires \"{r}\"; run `naru-audio pull {r}`"),
            )
        })
}

/// §2.5 `GET /v1/models`, `?pulled=true|false` filtering on `x_pulled`.
async fn models(
    State(st): State<Arc<AppState>>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let pulled = match query.get("pulled").map(String::as_str) {
        None => None,
        Some("true") => Some(true),
        Some("false") => Some(false),
        Some(other) => {
            return Err(ApiError {
                param: Some("pulled"),
                ..ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "unsupported_value",
                    format!("pulled must be true or false, not {other:?}"),
                )
            });
        }
    };
    let registry = st.registry.clone();
    let loaded: Vec<String> = st
        .models
        .ps()
        .into_iter()
        .filter(|m| !m.loading)
        .map(|m| m.name)
        .collect();
    let data: Vec<Value> = tokio::task::spawn_blocking(move || registry.list())
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
        .map_err(registry_error)?
        .into_iter()
        .filter(|e| pulled.is_none_or(|p| p == e.pulled))
        .map(|e| {
            let name = &e.manifest.model.name;
            json!({
                "id": e.manifest.model.name,
                "object": "model",
                "created": e.created,
                "owned_by": "naru-audio",
                "x_kind": e.manifest.model.kind.as_str(),
                "x_backend": e.manifest.model.backend,
                "x_available": e.available.is_ok(),
                "x_unavailable_reason": e.available.err(),
                "x_pulled": e.pulled,
                "x_loaded": loaded.contains(name),
                "x_size_bytes": e.size_bytes,
                "x_default": st.models.default_for(e.manifest.model.kind) == Some(name.as_str()),
                "x_license": e.manifest.model.license,
                "x_license_url": e.manifest.model.license_url,
                "x_non_commercial": e.manifest.model.non_commercial,
                // §5.3: what a client picking a TTS model can do with it —
                // clone a reference recording, whether that clone needs a
                // transcript alongside it, and design a voice from a
                // description. Meaningless for STT/VAD, but sent for every
                // kind rather than only TTS, same as the other `x_` fields.
                "x_clone": e.manifest.clones(),
                "x_clone_requires_transcript": e.manifest.clone_requires_transcript(),
                "x_instruct": e.manifest.instructs(),
            })
        })
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

/// §2.5 `POST /api/pull`: NDJSON progress, then `success`. Unknown models
/// and unavailable backends fail before the stream starts; a later failure
/// ends the stream with an `{"error":…}` line.
async fn pull(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let name = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(str::to_string))
        .ok_or_else(|| ApiError {
            param: Some("model"),
            ..ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "the body must be JSON with a string \"model\"",
            )
        })?;
    st.registry
        .check_pull(&name, false)
        .map_err(registry_error)?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let registry = st.registry.clone();
    tokio::task::spawn_blocking(move || {
        // A send fails only once the client has gone; the pull still finishes.
        let send = |v: Value| {
            let _ = tx.send(format!("{v}\n"));
        };
        let result = registry.pull_with(&name, false, &mut |p| {
            send(match p {
                Progress::Downloading {
                    file,
                    completed,
                    total,
                } => json!({"status": "downloading", "file": file, "completed": completed, "total": total}),
                Progress::Verifying => json!({"status": "verifying"}),
            })
        });
        match result {
            Ok(_) => send(json!({"status": "success"})),
            Err(e) => send(json!({"error": {
                "message": e.to_string(),
                "type": "server_error",
                "code": "pull_failed",
                "param": null,
            }})),
        }
    });
    let lines = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|line| (Ok::<_, Infallible>(line), rx))
    });
    Ok((
        [(CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(lines),
    )
        .into_response())
}

/// §2.5 `DELETE /api/models/{name}`: 409 `model_in_use` while the model is
/// loaded and busy (or loading); an idle loaded model is unloaded first,
/// under the model's registry lock and only once no other pulled model
/// `requires` it. A catalog model that is not pulled is 404: there is
/// nothing to delete, and the usual `model_not_pulled` advice (pull it) is
/// wrong here.
async fn remove(
    State(st): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    let (registry, models) = (st.registry.clone(), st.models.clone());
    let removed = {
        let name = name.clone();
        // `remove` waits on the model's lock, which a pull may hold for
        // minutes. A load holds it too, so no reload can slip in between
        // the unload and the deletion (see `ModelManager::acquire`).
        tokio::task::spawn_blocking(move || {
            registry.remove_with(&name, false, &[], || models.unload(&name, "rm").is_ok())
        })
    }
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
    .map_err(|e| match e {
        RegistryError::NotPulled(name) => ApiError {
            param: Some("model"),
            ..ApiError::new(
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!("the model \"{name}\" is not pulled; see GET /v1/models?pulled=true"),
            )
        },
        e => registry_error(e),
    })?;
    if !removed {
        return Err(ApiError {
            param: Some("model"),
            ..ApiError::new(
                StatusCode::CONFLICT,
                "model_in_use",
                format!(
                    "the model \"{name}\" is loaded and busy; try again once its requests finish"
                ),
            )
        });
    }
    Ok(StatusCode::NO_CONTENT)
}

/// §2.5 `GET /api/ps`: loaded models, and those still loading (§3.4).
async fn ps(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(Value::Array(
        st.models
            .ps()
            .into_iter()
            .map(|m| {
                json!({
                    "name": m.name,
                    "kind": m.kind.as_str(),
                    "backend": m.backend,
                    "resident_bytes": m.resident_bytes,
                    "expires_at": m.expires_at.map(|t| humantime::format_rfc3339_seconds(t).to_string()),
                    "busy": m.busy,
                    "loading": m.loading,
                })
            })
            .collect(),
    ))
}

/// §2.5 `POST /api/load` `{"model":"default"|name,"kind":"stt"|"tts","keep_alive":"10m"}`
/// warms a model; `keep_alive: 0` unloads it (once idle, if busy).
async fn load(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let bad = |param: &'static str, code: &'static str, message: String| ApiError {
        param: Some(param),
        ..ApiError::new(StatusCode::BAD_REQUEST, code, message)
    };
    let body: Value = serde_json::from_slice(&body).unwrap_or_default();
    let Some(model) = body["model"].as_str() else {
        return Err(bad(
            "model",
            "invalid_request",
            "the body must be JSON with a string \"model\"".to_string(),
        ));
    };
    let keep_alive = match &body["keep_alive"] {
        Value::Null => None,
        Value::String(s) => Some(KeepAlive::parse(s)),
        Value::Number(n) => Some(KeepAlive::from_secs(n.as_f64().unwrap_or(f64::NAN))),
        _ => Some(Err(
            "keep_alive must be a duration string or a number of seconds".to_string(),
        )),
    }
    .transpose()
    .map_err(|e| bad("keep_alive", "invalid_request", e))?;

    let (name, kind) = match (model, body["kind"].as_str()) {
        ("default", Some("stt")) => (st.models.settings().stt_default.clone(), Kind::Stt),
        ("default", Some("tts")) => (st.models.settings().tts_default.clone(), Kind::Tts),
        ("default", None) => {
            return Err(bad(
                "kind",
                "invalid_request",
                "\"kind\" is required with model \"default\"".to_string(),
            ));
        }
        ("default", Some(other)) => {
            return Err(bad(
                "kind",
                "unsupported_value",
                format!("kind {other:?} has no default; use stt or tts"),
            ));
        }
        (name, _) => (name.to_string(), Kind::Stt),
    };

    let manifest = {
        let (st, name) = (st.clone(), name.clone());
        tokio::task::spawn_blocking(move || transcriptions::kind_manifest(&st, &name, kind))
    }
    .await
    .map_err(|e| transcriptions::internal(&st, &req_id, e.to_string()))??;
    if keep_alive.is_some_and(KeepAlive::is_zero) {
        let loaded = st.models.unload_when_idle(&name);
        return Ok(Json(json!({"model": name, "loaded": loaded})));
    }
    drop(
        st.models
            .acquire(manifest, keep_alive)
            .await
            .map_err(|e| manager_error(&st, &req_id, e))?,
    );
    Ok(Json(json!({"model": name, "loaded": true})))
}

/// §2.6 codes for model manager failures; a 500 is logged with the
/// request id.
fn manager_error(st: &AppState, req_id: &str, e: ManagerError) -> ApiError {
    match e {
        ManagerError::InsufficientMemory(message) => ApiError {
            param: Some("model"),
            ..ApiError::new(
                StatusCode::INSUFFICIENT_STORAGE,
                "insufficient_memory",
                message,
            )
        },
        ManagerError::Load(e) => {
            let code = if e.is_backend_unavailable() {
                "backend_unavailable"
            } else {
                "model_load_failed"
            };
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, code, e.to_string())
        }
        ManagerError::Registry(e) => registry_error(e),
        ManagerError::Internal(message) => transcriptions::internal(st, req_id, message),
    }
}

/// §2.6 codes for registry failures.
fn registry_error(e: RegistryError) -> ApiError {
    let (status, code, message) = match &e {
        RegistryError::UnknownModel(name) => (
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("the model \"{name}\" is not in the catalog; see GET /v1/models"),
        ),
        RegistryError::NotPulled(name) => (
            StatusCode::CONFLICT,
            "model_not_pulled",
            format!("the model \"{name}\" is not pulled; run `naru-audio pull {name}`"),
        ),
        RegistryError::RequiredBy { .. } => (StatusCode::CONFLICT, "model_required", e.to_string()),
        RegistryError::BackendUnavailable { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_unavailable",
            e.to_string(),
        ),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    };
    let param = matches!(
        e,
        RegistryError::UnknownModel(_) | RegistryError::NotPulled(_)
    )
    .then_some("model");
    ApiError {
        param,
        ..ApiError::new(status, code, message)
    }
}

async fn not_found(req: Request) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "not_found",
        format!("no route for {} {}", req.method(), req.uri().path()),
    )
}

async fn method_not_allowed(req: Request) -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("{} is not allowed on {}", req.method(), req.uri().path()),
    )
}

/// Request id, Origin and Host guards, and the `request` log line.
async fn guard(State(st): State<Arc<AppState>>, mut req: Request, next: Next) -> Response {
    let started = Instant::now();
    let req_id = request_id(&req);
    let route = req.uri().path().to_owned();
    req.extensions_mut().insert(RequestId(req_id.clone()));

    let mut resp = match rejection(&st, &req) {
        Some(err) => err.into_response(),
        None => next.run(req).await,
    };

    if let Ok(v) = HeaderValue::from_str(&req_id) {
        resp.headers_mut().insert(REQUEST_ID, v);
    }
    st.log.info(
        Some(&req_id),
        &format!(
            "request route={route} status={} ms={}",
            resp.status().as_u16(),
            started.elapsed().as_millis()
        ),
    );
    resp
}

fn rejection(st: &AppState, req: &Request) -> Option<ApiError> {
    // `allowed_origins` is empty by default and there is no config yet, so
    // any Origin is refused.
    if let Some(origin) = req.headers().get(ORIGIN) {
        return Some(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden_origin",
            format!(
                "origin {:?} is not allowed; browsers reach naru-audio through Naru",
                String::from_utf8_lossy(origin.as_bytes())
            ),
        ));
    }
    if st.allow_remote {
        return None;
    }
    // The URI authority stands in only when there is no Host header at all;
    // an unreadable or repeated Host is refused.
    let mut hosts = req.headers().get_all(HOST).iter();
    let host = match (hosts.next(), hosts.next()) {
        (None, _) => req.uri().authority().map(|a| a.as_str()),
        (Some(h), None) => h.to_str().ok(),
        (Some(_), Some(_)) => None,
    };
    match host {
        Some(h) if host_allowed(h, st.port) => None,
        _ => Some(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden_host",
            format!(
                "host {:?} is not allowed; use 127.0.0.1:{p}, localhost:{p} or [::1]:{p}, \
                 or start with --allow-remote",
                host.unwrap_or(""),
                p = st.port
            ),
        )),
    }
}

fn host_allowed(host: &str, port: u16) -> bool {
    ["127.0.0.1", "localhost", "[::1]"]
        .iter()
        .any(|name| host.eq_ignore_ascii_case(&format!("{name}:{port}")))
}

/// Echo a well-formed client `X-Request-Id`, otherwise generate one. The id
/// goes into the space-separated log line, so only `[A-Za-z0-9._-]{1,64}`
/// is echoed.
fn request_id(req: &Request) -> String {
    let incoming = req
        .headers()
        .get(REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .filter(|s| {
            (1..=64).contains(&s.len())
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        });
    match incoming {
        Some(id) => id.to_owned(),
        None => format!("{:016x}", RandomState::new().hash_one(Instant::now())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::{BackendLoader, Settings};
    use crate::profile::Profile;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    const PORT: u16 = 7870;

    fn app(allow_remote: bool) -> Router {
        let log = Arc::new(Logger::stderr());
        // These tests never touch the registry: a missing home is an empty
        // catalog.
        let registry = Arc::new(Registry::open("/nonexistent/naru-audio-home").unwrap());
        let settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
        let models = ModelManager::new(
            registry.clone(),
            log.clone(),
            settings,
            Arc::new(BackendLoader),
        );
        router(Arc::new(AppState {
            port: PORT,
            allow_remote,
            started: Instant::now(),
            log,
            registry,
            models: Arc::new(models),
        }))
    }

    async fn send(app: Router, req: Request<Body>) -> (StatusCode, Option<String>, Value) {
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let id = resp
            .headers()
            .get(REQUEST_ID)
            .map(|v| v.to_str().unwrap().to_owned());
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, id, serde_json::from_slice(&bytes).unwrap())
    }

    fn get_req(path: &str, host: &str) -> axum::http::request::Builder {
        Request::get(path).header(HOST, host)
    }

    fn assert_envelope(body: &Value, code: &str) {
        let err = &body["error"];
        assert_eq!(err["code"], code, "{body}");
        assert_eq!(err["type"], "invalid_request_error");
        assert!(err["message"].as_str().is_some_and(|m| !m.is_empty()));
        assert!(err.get("param").is_some());
    }

    #[tokio::test]
    async fn health_returns_api_1() {
        let req = get_req("/health", "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let (status, id, body) = send(app(false), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["api"], 1);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(body["pid"], std::process::id());
        assert!(body["uptime_s"].is_u64());
        assert_eq!(id.unwrap().len(), 16);
    }

    #[tokio::test]
    async fn loopback_hosts_are_allowed() {
        for host in ["localhost:7870", "[::1]:7870", "LOCALHOST:7870"] {
            let req = get_req("/health", host).body(Body::empty()).unwrap();
            assert_eq!(send(app(false), req).await.0, StatusCode::OK, "{host}");
        }
    }

    #[tokio::test]
    async fn foreign_origin_is_403_with_envelope() {
        for allow_remote in [false, true] {
            let req = get_req("/health", "127.0.0.1:7870")
                .header(ORIGIN, "http://evil.test")
                .body(Body::empty())
                .unwrap();
            let (status, id, body) = send(app(allow_remote), req).await;
            assert_eq!(status, StatusCode::FORBIDDEN);
            assert_envelope(&body, "forbidden_origin");
            assert!(id.is_some());
        }
    }

    #[tokio::test]
    async fn foreign_host_is_403_by_default() {
        for host in ["evil.test:7870", "127.0.0.1:9999", "192.168.1.5:7870"] {
            let req = get_req("/health", host).body(Body::empty()).unwrap();
            let (status, id, body) = send(app(false), req).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{host}");
            assert_envelope(&body, "forbidden_host");
            assert!(id.is_some());
        }
    }

    #[tokio::test]
    async fn non_utf8_host_is_403() {
        let req = Request::get("http://127.0.0.1:7870/health")
            .header(HOST, HeaderValue::from_bytes(b"evil\xff.test").unwrap())
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(app(false), req).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_envelope(&body, "forbidden_host");
    }

    #[tokio::test]
    async fn duplicate_host_is_403() {
        let req = get_req("/health", "127.0.0.1:7870")
            .header(HOST, "evil.test:7870")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(app(false), req).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_envelope(&body, "forbidden_host");
    }

    #[tokio::test]
    async fn missing_host_falls_back_to_uri_authority() {
        let req = Request::get("http://127.0.0.1:7870/health")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(app(false), req).await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn foreign_host_is_200_with_allow_remote() {
        let req = get_req("/health", "evil.test:7870")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(app(true), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["api"], 1);
    }

    #[tokio::test]
    async fn unknown_route_is_404_with_envelope() {
        let req = get_req("/nope", "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let (status, id, body) = send(app(false), req).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_envelope(&body, "not_found");
        assert!(id.is_some());
    }

    #[tokio::test]
    async fn wrong_method_is_405_with_envelope() {
        let req = Request::post("/health")
            .header(HOST, "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(app(false), req).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        assert_envelope(&body, "method_not_allowed");
    }

    #[tokio::test]
    async fn request_id_is_echoed_when_well_formed() {
        let req = get_req("/health", "127.0.0.1:7870")
            .header(REQUEST_ID, "naru-42.a_b")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(app(false), req).await.1.unwrap(), "naru-42.a_b");

        let req = get_req("/health", "127.0.0.1:7870")
            .header(REQUEST_ID, "has space")
            .body(Body::empty())
            .unwrap();
        let id = send(app(false), req).await.1.unwrap();
        assert_ne!(id, "has space");
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn non_loopback_listen_needs_allow_remote() {
        let any: SocketAddr = "0.0.0.0:7870".parse().unwrap();
        let lan: SocketAddr = "192.168.1.5:7870".parse().unwrap();
        let err = check_listen(any, false).unwrap_err();
        assert!(err.contains("--allow-remote"), "{err}");
        assert!(check_listen(lan, false).is_err());
        assert!(check_listen(any, true).is_ok());
        assert!(check_listen(DEFAULT_LISTEN.parse().unwrap(), false).is_ok());
        assert!(check_listen("[::1]:7870".parse().unwrap(), false).is_ok());
    }
}
