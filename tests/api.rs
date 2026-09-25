//! §2.5 registry routes: `GET /v1/models`, `POST /api/pull`,
//! `DELETE /api/models/{name}`.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use common::{Stub, file_entry, home, model_header};
use http_body_util::BodyExt;
use naru_audio::log::Logger;
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::Registry;
use naru_audio::server::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

const HOSTPORT: &str = "127.0.0.1:7870";
/// Built in, and listed on macOS only.
const MLX: &[&str] = &[
    "parakeet-tdt-0.6b-v2-mlx",
    "qwen3-tts-0.6b-base-mlx",
    "qwen3-tts-0.6b-mlx",
];

fn app(home: &Path) -> Router {
    let log = Arc::new(Logger::stderr());
    let registry = Arc::new(Registry::open(home).unwrap());
    let settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
    let models = ModelManager::new(
        registry.clone(),
        log.clone(),
        settings,
        Arc::new(BackendLoader),
    );
    router(Arc::new(AppState {
        port: 7870,
        allow_remote: false,
        started: Instant::now(),
        log,
        registry,
        models: Arc::new(models),
    }))
}

/// Status, content type and raw body.
async fn send(app: Router, req: Request<Body>) -> (StatusCode, String, Vec<u8>) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(CONTENT_TYPE)
        .map_or(String::new(), |v| v.to_str().unwrap().to_owned());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ctype, bytes.to_vec())
}

async fn get_json(home: &Path, uri: &str) -> (StatusCode, Value) {
    let req = Request::get(uri)
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(home), req).await;
    (status, serde_json::from_slice(&body).unwrap())
}

async fn pull(home: &Path, body: &str) -> (StatusCode, String, Vec<u8>) {
    let req = Request::post("/api/pull")
        .header(HOST, HOSTPORT)
        .body(Body::from(body.to_owned()))
        .unwrap();
    send(app(home), req).await
}

async fn delete(home: &Path, name: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::delete(format!("/api/models/{name}"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(home), req).await;
    (status, body)
}

fn error_code(body: &[u8]) -> Value {
    serde_json::from_slice::<Value>(body).unwrap()["error"]["code"].clone()
}

/// `stt` requires `vad`; both are served by `stub`.
fn stt_and_vad(stub: &Stub) -> Vec<String> {
    vec![
        model_header("vad", &[]) + &file_entry("vad.onnx", &stub.url("/vad.onnx"), b"silero"),
        model_header("stt", &["vad"])
            + &file_entry("enc.onnx", &stub.url("/enc.onnx"), b"parakeet"),
    ]
}

fn stub() -> Stub {
    Stub::start(
        &[
            ("/vad.onnx", b"silero".to_vec()),
            ("/enc.onnx", b"parakeet".to_vec()),
        ],
        Duration::ZERO,
    )
}

#[tokio::test]
async fn v1_models_entries_have_exactly_the_2_5_fields() {
    let stub = stub();
    let mut manifests = stt_and_vad(&stub);
    // Listed, but not runnable: its backend is unavailable.
    manifests.push(model_header("odd", &[]).replace("sherpa-onnx", "nope"));
    // Not listed: no platform matches this machine.
    manifests.push(
        model_header("elsewhere", &[])
            .replace("requires", "platforms = [\"plan9-mips\"]\nrequires"),
    );
    let dir = home(&manifests);
    Registry::open(dir.path()).unwrap().pull("vad").unwrap();

    let (status, body) = get_json(dir.path(), "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    // The built-in catalog is listed too; the checks below are about the
    // rest. The MLX models are listed on macOS only, their platforms.
    let ids: Vec<&str> = data
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .filter(|id| !MLX.contains(id))
        .collect();
    assert_eq!(
        ids,
        [
            "kokoro-v1.0",
            "odd",
            "parakeet-tdt-0.6b-v2-int8",
            "pocket-tts-int8",
            "silero-vad",
            "stt",
            "vad"
        ]
    );
    let data: Vec<&Value> = data
        .iter()
        .filter(|m| {
            let id = m["id"].as_str().unwrap();
            ![
                "kokoro-v1.0",
                "parakeet-tdt-0.6b-v2-int8",
                "pocket-tts-int8",
                "silero-vad",
            ]
            .contains(&id)
                && !MLX.contains(&id)
        })
        .collect();

    let fields = [
        "created",
        "id",
        "object",
        "owned_by",
        "x_available",
        "x_backend",
        "x_default",
        "x_kind",
        "x_loaded",
        "x_pulled",
        "x_size_bytes",
        "x_unavailable_reason",
    ];
    for m in &data {
        let keys: Vec<&str> = m.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, fields, "{m}");
        assert_eq!(m["object"], "model");
        assert_eq!(m["owned_by"], "naru-audio");
        assert_eq!(m["x_kind"], "stt");
        assert_eq!(m["x_loaded"], false);
        assert_eq!(m["x_default"], false);
    }

    let vad = &data[2];
    assert_eq!(vad["x_pulled"], true);
    assert_eq!(vad["x_available"], true);
    assert_eq!(vad["x_unavailable_reason"], Value::Null);
    assert_eq!(vad["x_backend"], "sherpa-onnx");
    assert!(vad["created"].as_u64().unwrap() > 0, "{vad}");
    // vad.onnx plus manifest.json.
    assert!(vad["x_size_bytes"].as_u64().unwrap() > 6, "{vad}");

    let stt = &data[1];
    assert_eq!(stt["x_pulled"], false);
    assert_eq!(stt["created"], 0);
    assert_eq!(stt["x_size_bytes"], 8);

    let odd = &data[0];
    assert_eq!(odd["x_available"], false);
    assert_eq!(odd["x_unavailable_reason"], "unknown backend `nope`");

    let (_, pulled) = get_json(dir.path(), "/v1/models?pulled=true").await;
    let ids: Vec<&str> = pulled["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["vad"]);

    let (status, bad) = get_json(dir.path(), "/v1/models?pulled=yes").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(bad["error"]["code"], "unsupported_value");
    assert_eq!(bad["error"]["param"], "pulled");
}

#[tokio::test]
async fn api_pull_streams_ndjson_then_is_idempotent() {
    let stub = stub();
    let dir = home(&stt_and_vad(&stub));

    let (status, ctype, body) = pull(dir.path(), r#"{"model":"stt"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ctype, "application/x-ndjson");
    let lines: Vec<Value> = String::from_utf8(body)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let download = |file: &str, completed: u64, total: u64| json!({"status": "downloading", "file": file, "completed": completed, "total": total});
    assert_eq!(
        lines,
        [
            download("vad.onnx", 0, 6),
            download("vad.onnx", 6, 6),
            json!({"status": "verifying"}),
            download("enc.onnx", 0, 8),
            download("enc.onnx", 8, 8),
            json!({"status": "verifying"}),
            json!({"status": "success"}),
        ]
    );
    assert_eq!(stub.hits(), ["/vad.onnx", "/enc.onnx"]);

    let (status, _, body) = pull(dir.path(), r#"{"model":"stt"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        String::from_utf8(body).unwrap(),
        "{\"status\":\"success\"}\n"
    );
    assert_eq!(stub.hits().len(), 2);
}

#[tokio::test]
async fn api_pull_failure_mid_stream_ends_with_an_error_line() {
    let stub = Stub::start(&[("/vad.onnx", b"tampered".to_vec())], Duration::ZERO);
    let dir = home(&[
        model_header("vad", &[]) + &file_entry("vad.onnx", &stub.url("/vad.onnx"), b"silero")
    ]);
    let (status, _, body) = pull(dir.path(), r#"{"model":"vad"}"#).await;
    assert_eq!(status, StatusCode::OK);
    let body = String::from_utf8(body).unwrap();
    let last: Value = serde_json::from_str(body.lines().last().unwrap()).unwrap();
    assert_eq!(last["error"]["code"], "pull_failed", "{body}");
    assert!(!body.contains("success"), "{body}");
}

#[tokio::test]
async fn api_pull_errors_use_the_envelope() {
    let dir = home(&[model_header("odd", &[]).replace("sherpa-onnx", "nope")]);

    let (status, _, body) = pull(dir.path(), r#"{"model":"nope"}"#).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "model_not_found");

    let (status, _, body) = pull(dir.path(), r#"{"model":"odd"}"#).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(&body), "backend_unavailable");

    let (status, _, body) = pull(dir.path(), r#"{"name":"odd"}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");
}

#[tokio::test]
async fn delete_is_204_404_or_409() {
    let stub = stub();
    let dir = home(&stt_and_vad(&stub));
    let reg = Registry::open(dir.path()).unwrap();
    reg.pull("stt").unwrap();

    let (status, body) = delete(dir.path(), "vad").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(error_code(&body), "model_required");

    let (status, body) = delete(dir.path(), "nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "model_not_found");

    let (status, body) = delete(dir.path(), "stt").await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(body.is_empty());
    assert!(!reg.is_installed("stt"));
    assert_eq!(delete(dir.path(), "vad").await.0, StatusCode::NO_CONTENT);

    // Known, but no longer pulled: nothing to delete.
    let (status, body) = delete(dir.path(), "vad").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "model_not_found");
    let message = serde_json::from_slice::<Value>(&body).unwrap()["error"]["message"].clone();
    assert!(
        message.as_str().unwrap().contains("not pulled"),
        "{message}"
    );
    assert_eq!(reg.pulled().unwrap(), Vec::<String>::new());
}

#[tokio::test]
async fn unreadable_manifest_json_is_skipped() {
    let stub = stub();
    let dir = home(&stt_and_vad(&stub));
    let reg = Registry::open(dir.path()).unwrap();
    reg.pull("stt").unwrap();
    std::fs::write(dir.path().join("models/stt/manifest.json"), b"{not json").unwrap();

    let (status, body) = get_json(dir.path(), "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .filter(|id| !MLX.contains(id))
        .collect();
    assert_eq!(
        ids,
        [
            "kokoro-v1.0",
            "parakeet-tdt-0.6b-v2-int8",
            "pocket-tts-int8",
            "silero-vad",
            "vad"
        ]
    );

    // The unreadable stt no longer blocks removing what it requires.
    assert_eq!(delete(dir.path(), "vad").await.0, StatusCode::NO_CONTENT);
}
