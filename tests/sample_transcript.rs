//! naru task 1569 part 3: `POST /v1/audio/samples/{id}/transcribe` and
//! `GET .../transcript`. There is no fake STT model that returns words
//! (`fake-stt` has a manifest and no model files), so the decode and its
//! `transcript.txt` write are covered by `src/server/sample_transcript.rs`'s
//! unit test and a real-model run; here: the 404/409/400 paths, the request
//! reaching the model load, and the stored transcript read back.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use naru_audio::log::Logger;
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::Registry;
use naru_audio::server::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

const HOSTPORT: &str = "127.0.0.1:7870";
const SAMPLE: &str = "abc123";

/// A home with `fake-stt` "pulled" and a sample `abc123` holding a short
/// `clean.wav` and a `meta.json` (only its presence is checked).
fn home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("models").join("fake-stt");
    std::fs::create_dir_all(&model_dir).unwrap();
    let manifest = json!({"model": {"name": "fake-stt", "kind": "stt", "backend": "sherpa-onnx"}});
    std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();

    let sample = dir.path().join("prep").join("samples").join(SAMPLE);
    std::fs::create_dir_all(&sample).unwrap();
    std::fs::write(sample.join("meta.json"), "{}").unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(sample.join("clean.wav"), spec).unwrap();
    for i in 0..8_000 {
        let s = (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.2;
        writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    writer.finalize().unwrap();
    dir
}

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
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
    }))
}

async fn send(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn get(home: &Path, path: &str) -> (StatusCode, Value) {
    let req = Request::get(path)
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    send(app(home), req).await
}

async fn post_json(home: &Path, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::post(path)
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    send(app(home), req).await
}

#[tokio::test]
async fn unknown_sample_is_404_on_both_routes() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples/nonexistent/transcribe",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "sample_not_found");
    let (status, body) = get(home.path(), "/v1/audio/samples/nonexistent/transcript").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "sample_not_found");
}

#[tokio::test]
async fn transcript_before_transcribe_is_409() {
    let home = home();
    let (status, body) = get(
        home.path(),
        &format!("/v1/audio/samples/{SAMPLE}/transcript"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "transcript_not_run");
}

#[tokio::test]
async fn bad_body_is_400() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        &format!("/v1/audio/samples/{SAMPLE}/transcribe"),
        json!({"verbatim": "yes"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
}

/// Past every check (verbatim defaults on, no body keys needed) into
/// `ModelManager`, which cannot load `fake-stt`'s absent files.
#[tokio::test]
async fn transcribe_reaches_the_load_and_leaves_transcript_txt_alone() {
    let home = home();
    let txt = home
        .path()
        .join("prep")
        .join("samples")
        .join(SAMPLE)
        .join("transcript.txt");
    std::fs::write(&txt, "before").unwrap();
    let (status, body) = post_json(
        home.path(),
        &format!("/v1/audio/samples/{SAMPLE}/transcribe"),
        json!({"stt_model": "fake-stt"}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "model_load_failed");
    assert_eq!(std::fs::read_to_string(txt).unwrap(), "before");
}

#[tokio::test]
async fn stored_transcript_is_served() {
    let home = home();
    let stored = json!({
        "words": [{"start": 0.0, "end": 0.4, "text": "42", "spoken": "forty-two",
                   "differs": true, "filler": false}],
        "text": "42", "spoken_text": "forty-two", "language": "en",
        "stt_model": "m", "verbatim": true, "verbatim_supported": true,
    });
    std::fs::write(
        home.path()
            .join("prep")
            .join("samples")
            .join(SAMPLE)
            .join("transcript.json"),
        stored.to_string(),
    )
    .unwrap();
    let (status, body) = get(
        home.path(),
        &format!("/v1/audio/samples/{SAMPLE}/transcript"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, stored);
}
