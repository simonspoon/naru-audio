//! HTTP surface: §2.1 guards, §2.5 `/health`, §2.6 errors, `X-Request-Id`.

use std::hash::{BuildHasher, RandomState};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::header::{HOST, ORIGIN};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::log::Logger;

pub const DEFAULT_LISTEN: &str = "127.0.0.1:7870";
const REQUEST_ID: &str = "x-request-id";

pub struct AppState {
    /// The bound port; the Host guard only accepts loopback names with it.
    pub port: u16,
    /// `--allow-remote`: skips the Host guard (§2.1).
    pub allow_remote: bool,
    pub started: Instant,
    pub log: Arc<Logger>,
}

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
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "api": 1,
        "pid": std::process::id(),
        "uptime_s": st.started.elapsed().as_secs(),
    }))
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
async fn guard(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let started = Instant::now();
    let req_id = request_id(&req);
    let route = req.uri().path().to_owned();

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
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    const PORT: u16 = 7870;

    fn app(allow_remote: bool) -> Router {
        router(Arc::new(AppState {
            port: PORT,
            allow_remote,
            started: Instant::now(),
            log: Arc::new(Logger::stderr()),
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
