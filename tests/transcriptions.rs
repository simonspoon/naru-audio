//! §2.2 `POST /v1/audio/transcriptions`: the checks that answer before a
//! model loads. None of these needs a pulled model; `fake-stt` is "pulled"
//! by writing its `manifest.json` alone.

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

const BOUNDARY: &str = "naru-audio-test-boundary";

/// A home with `fake-stt` (English only) and `fake-vad` pulled.
fn home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, kind, languages) in [("fake-stt", "stt", vec!["en"]), ("fake-vad", "vad", vec![])] {
        let model_dir = dir.path().join("models").join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        let manifest = json!({"model": {
            "name": name, "kind": kind, "backend": "sherpa-onnx", "languages": languages,
        }});
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();
    }
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
    }))
}

/// 0.1 s of 16 kHz silence.
fn wav() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for _ in 0..1600 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

/// POSTs `file` plus `fields` as multipart/form-data.
async fn post(home: &Path, file: &[u8], fields: &[(&str, &str)]) -> (StatusCode, Value) {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());

    let req = Request::post("/v1/audio/transcriptions")
        .header(HOST, "127.0.0.1:7870")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap();
    let resp = app(home).oneshot(req).await.unwrap();
    let status = resp.status();
    assert!(resp.headers().contains_key("x-request-id"));
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn assert_error(body: &Value, code: &str, param: &str) {
    assert_eq!(body["error"]["code"], code, "{body}");
    assert_eq!(body["error"]["param"], param, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty())
    );
}

#[tokio::test]
async fn non_wav_is_415() {
    let home = home();
    for file in [&b"OggS\0\0\0\0 not wav"[..], b"plain text"] {
        let (status, body) = post(home.path(), file, &[("model", "fake-stt")]).await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_error(&body, "unsupported_media_type", "file");
    }
}

#[tokio::test]
async fn unlisted_language_is_400() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[("model", "fake-stt"), ("language", "fr")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error(&body, "unsupported_value", "language");
}

/// `en` is listed, so the request gets past the language check to the load,
/// which fails: `fake-stt` has no model files.
#[tokio::test]
async fn listed_language_reaches_the_load() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[("model", "fake-stt"), ("language", "en")],
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "model_load_failed", "{body}");
}

#[tokio::test]
async fn srt_and_vtt_are_400() {
    let home = home();
    for format in ["srt", "vtt"] {
        let (status, body) = post(home.path(), &wav(), &[("response_format", format)]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{format}");
        assert_error(&body, "unsupported_value", "response_format");
    }
}

#[tokio::test]
async fn word_granularity_is_400() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error(&body, "unsupported_value", "timestamp_granularities[]");
}

#[tokio::test]
async fn segment_granularity_needs_verbose_json() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[("timestamp_granularities[]", "segment")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error(&body, "unsupported_value", "timestamp_granularities[]");
}

#[tokio::test]
async fn unknown_model_is_404() {
    let home = home();
    let (status, body) = post(home.path(), &wav(), &[("model", "nope")]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_error(&body, "model_not_found", "model");
}

/// A name that is neither in the catalog nor pulled never reaches the
/// filesystem: a manifest planted outside `models/` is not loaded.
#[tokio::test]
async fn path_like_model_names_are_404() {
    let home = home();
    let outside = tempfile::tempdir().unwrap();
    let manifest = json!({"model": {
        "name": "evil", "kind": "stt", "backend": "sherpa-onnx", "languages": ["en"],
    }});
    std::fs::write(outside.path().join("manifest.json"), manifest.to_string()).unwrap();
    // Also planted as a sibling of `models/`, reachable as `../evil`.
    let sibling = home.path().join("evil");
    std::fs::create_dir_all(&sibling).unwrap();
    std::fs::write(sibling.join("manifest.json"), manifest.to_string()).unwrap();

    let absolute = outside.path().to_str().unwrap().to_owned();
    for name in [
        "../evil",
        "../models/fake-stt",
        "fake-stt/",
        absolute.as_str(),
    ] {
        let (status, body) = post(home.path(), &wav(), &[("model", name)]).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{name}: {body}");
        assert_error(&body, "model_not_found", "model");
    }
}

#[tokio::test]
async fn non_stt_model_is_400() {
    let home = home();
    let (status, body) = post(home.path(), &wav(), &[("model", "fake-vad")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error(&body, "invalid_request", "model");
}

/// Every alias, and no `model` at all, is the default model, which is in
/// the built-in catalog but not pulled here: 409, never a pull.
#[tokio::test]
async fn aliases_resolve_to_the_unpulled_default() {
    let home = home();
    let mut cases: Vec<Vec<(&str, &str)>> = [
        "default",
        "whisper-1",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
    ]
    .iter()
    .map(|m| vec![("model", *m)])
    .collect();
    cases.push(vec![]);
    for fields in cases {
        let (status, body) = post(home.path(), &wav(), &fields).await;
        assert_eq!(status, StatusCode::CONFLICT, "{fields:?}");
        assert_error(&body, "model_not_pulled", "model");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("naru-audio pull parakeet-tdt-0.6b-v2-int8"),
            "{message}"
        );
    }
    assert!(
        !home
            .path()
            .join("models/parakeet-tdt-0.6b-v2-int8")
            .exists()
    );
}

#[tokio::test]
async fn bad_vad_values_are_400() {
    let home = home();
    for (param, value) in [
        ("vad_threshold", "1.5"),
        ("vad_threshold", "-0.1"),
        ("vad_threshold", "NaN"),
        ("vad_min_speech", "-1"),
        ("vad_min_speech", "inf"),
        ("vad_min_silence", "0"),
        ("vad_min_silence", "-0.5"),
        ("vad_min_silence", "inf"),
        ("vad_min_silence", "soon"),
        ("vad", "maybe"),
    ] {
        let (status, body) = post(
            home.path(),
            &wav(),
            &[("model", "fake-stt"), (param, value)],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{param}={value}");
        assert_error(&body, "invalid_request", param);
    }
}

#[tokio::test]
async fn bad_hotwords_are_400() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[("model", "fake-stt"), ("hotwords", "naru:3.0")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error(&body, "invalid_request", "hotwords");
}

/// The fields §2.2 ignores, and a valid `keep_alive`, do not fail the
/// request: it gets as far as the load.
#[tokio::test]
async fn ignored_fields_are_accepted() {
    let home = home();
    let (status, body) = post(
        home.path(),
        &wav(),
        &[
            ("model", "fake-stt"),
            ("prompt", "anything"),
            ("temperature", "not a number"),
            ("chunking_strategy", "auto"),
            ("include[]", "logprobs"),
            ("keep_alive", "5m"),
            ("vad", "false"),
            ("vad_threshold", "0.5"),
            ("vad_min_speech", "0"),
            ("vad_min_silence", "0.3"),
            ("hotwords", "naru\nhelios :4.0"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "model_load_failed", "{body}");
}

#[tokio::test]
async fn missing_file_is_400() {
    let home = home();
    let req = Request::post("/v1/audio/transcriptions")
        .header(HOST, "127.0.0.1:7870")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nfake-stt\r\n--{BOUNDARY}--\r\n"
        )))
        .unwrap();
    let resp = app(home.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert_error(
        &serde_json::from_slice(&bytes).unwrap(),
        "invalid_request",
        "file",
    );
}

#[tokio::test]
async fn non_multipart_body_is_400_with_envelope() {
    let home = home();
    let req = Request::post("/v1/audio/transcriptions")
        .header(HOST, "127.0.0.1:7870")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = app(home.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["code"], "invalid_request", "{body}");
}
