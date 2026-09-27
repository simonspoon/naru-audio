//! naru task 1461 §8 voice-prep: `/v1/audio/prep/clips` (upload,
//! transcript, raw/working audio) and `/v1/audio/samples` (crop + clean).
//! None of these tests needs a real speech-to-text, diarization or denoise
//! engine to load — like `tests/transcriptions.rs`, they exercise the
//! request-validation layer and, for `create_sample`'s happy path, the
//! pipeline's pure signal-processing steps (crop, normalise) with the model
//! steps turned off by their own flags.

mod common;

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

const BOUNDARY: &str = "naru-audio-prep-test-boundary";
const HOSTPORT: &str = "127.0.0.1:7870";

/// A home with `fake-stt`, `fake-diarization` and `fake-denoise` "pulled"
/// (a bare `manifest.json`, no real model files — enough to get past
/// `kind_manifest`, same as `tests/transcriptions.rs`'s `fake-stt`).
fn home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, kind) in [
        ("fake-stt", "stt"),
        ("fake-diarization", "diarization"),
        ("fake-denoise", "denoise"),
        ("fake-separation", "separation"),
    ] {
        let model_dir = dir.path().join("models").join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        let manifest = json!({"model": {"name": name, "kind": kind, "backend": "sherpa-onnx"}});
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
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
    }))
}

/// 0.5 s of a 440 Hz tone at 16 kHz mono 16-bit, so upload/crop/normalise
/// have a non-silent signal to work on.
fn wav() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for i in 0..8_000 {
        let t = i as f32 / 16_000.0;
        let s = (t * 440.0 * std::f32::consts::TAU).sin() * 0.2;
        writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

async fn send(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

async fn get(home: &Path, path: &str) -> (StatusCode, Value) {
    let req = Request::get(path)
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    send(app(home), req).await
}

async fn delete(home: &Path, path: &str) -> (StatusCode, Value) {
    let req = Request::delete(path)
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

async fn patch_json(home: &Path, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::patch(path)
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    send(app(home), req).await
}

async fn upload(home: &Path, file: &[u8]) -> (StatusCode, Value) {
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let req = Request::post("/v1/audio/prep/clips")
        .header(HOST, HOSTPORT)
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap();
    send(app(home), req).await
}

#[tokio::test]
async fn upload_with_no_file_field_is_400() {
    let home = home();
    let req = Request::post("/v1/audio/prep/clips")
        .header(HOST, HOSTPORT)
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(format!("--{BOUNDARY}--\r\n")))
        .unwrap();
    let (status, body) = send(app(home.path()), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "file");
}

#[tokio::test]
async fn upload_then_get_then_list_then_delete_a_clip() {
    let home = home();
    let (status, body) = upload(home.path(), &wav()).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().unwrap().to_string();
    assert!(body["duration_secs"].as_f64().unwrap() > 0.0);

    let (status, got) = get(home.path(), &format!("/v1/audio/prep/clips/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(got["id"], id);

    let (status, listed) = get(home.path(), "/v1/audio/prep/clips").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["clips"], json!([id]));

    let (status, _) = delete(home.path(), &format!("/v1/audio/prep/clips/{id}")).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (status, _) = get(home.path(), &format!("/v1/audio/prep/clips/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn unknown_clip_id_is_404() {
    let home = home();
    let (status, body) = get(home.path(), "/v1/audio/prep/clips/nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "clip_not_found");
}

#[tokio::test]
async fn clip_audio_returns_raw_and_working() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();

    let req = Request::get(format!("/v1/audio/prep/clips/{id}/audio"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let resp = app(home.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let working = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(!working.is_empty());

    let req = Request::get(format!("/v1/audio/prep/clips/{id}/audio?raw=true"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let resp = app(home.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let raw = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(raw.to_vec(), wav());
}

/// `fake-stt` has no real model files, so the load itself fails — the
/// same 503 `tests/transcriptions.rs::listed_language_reaches_the_load`
/// gets, reached the same way: past every check, into `ModelManager`.
#[tokio::test]
async fn transcribe_reaches_the_load() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = post_json(
        home.path(),
        &format!("/v1/audio/prep/clips/{id}/transcribe"),
        json!({"stt_model": "fake-stt", "diarize": false}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "model_load_failed");
}

#[tokio::test]
async fn transcribe_unknown_clip_is_404() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "clip_not_found");
}

#[tokio::test]
async fn transcript_before_transcribe_is_409() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = get(
        home.path(),
        &format!("/v1/audio/prep/clips/{id}/transcript"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "transcript_not_run");
}

/// `isolate` defaults to `false` (unlike `denoise`/`trim_silence`): no
/// separation engine ships pulled by default, so this is the "explicit
/// `isolate` defaults to `true`, the same as `denoise`/`trim_silence`/
/// `normalize`: it is a documented pipeline step, so leaving it out with
/// no separation model pulled must fail with a clear 409 naming the
/// model, never silently skip (naru task 1461 §8). `isolate:false` is
/// the explicit skip.
#[tokio::test]
async fn isolate_defaults_on_and_names_the_unpulled_model() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();

    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": id, "name": "s1", "start": 0.0, "end": 0.3,
            "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("source-separation-spleeter-2stems-int8")
    );

    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": id, "name": "s2", "start": 0.0, "end": 0.3,
            "isolate": false, "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "{body} (isolate:false, explicit skip)"
    );
}

/// `fake-separation` has no real model files, so the load itself fails —
/// the same 500 shape as any other backend construction failure, reached
/// past every validation check, once a model is actually pulled.
#[tokio::test]
async fn isolate_reaches_the_load() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": id, "name": "s1", "start": 0.0, "end": 0.3,
            "isolate": true, "isolation_model": "fake-separation",
            "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
}

#[tokio::test]
async fn create_sample_needs_a_range_or_a_speaker() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({"clip_id": id, "name": "s1", "denoise": false, "trim_silence": false}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "start");
}

#[tokio::test]
async fn create_sample_unknown_clip_is_404() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({"clip_id": "nonexistent", "name": "s1", "start": 0.0, "end": 0.1}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

/// `denoise` defaults to `true`; leaving it out with no denoise model
/// pulled must fail with a clear 409 naming the model, never silently
/// skip a pipeline stage (naru task 1461 §8).
#[tokio::test]
async fn denoise_defaults_on_and_names_the_unpulled_model() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({"clip_id": id, "name": "s1", "start": 0.0, "end": 0.3, "isolate": false, "trim_silence": false}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("speech-denoiser-gtcrn")
    );
}

/// `trim_silence` defaults to `true`; with `silero-vad` not pulled, the
/// same "fail loudly" rule applies.
#[tokio::test]
async fn trim_silence_defaults_on_and_names_the_unpulled_model() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({"clip_id": id, "name": "s1", "start": 0.0, "end": 0.3, "isolate": false, "denoise": false}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("silero-vad")
    );
}

/// The happy path with every model-backed stage turned off: crop and
/// normalise are pure signal processing, so this exercises them, plus
/// storage (both WAVs, `meta.json`) and the sample CRUD routes end to end.
#[tokio::test]
async fn create_list_get_rename_fetch_and_delete_a_sample() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let clip_id = body["id"].as_str().unwrap().to_string();

    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip_id, "name": "bria", "start": 0.1, "end": 0.4,
            "isolate": false, "denoise": false, "trim_silence": false, "normalize": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().unwrap().to_string();
    assert_eq!(body["name"], "bria");
    assert_eq!(body["source_clip_id"], clip_id);
    assert!((body["range"]["start"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((body["range"]["end"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    assert_eq!(body["warnings"], json!([]));

    let (status, listed) = get(home.path(), "/v1/audio/samples").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["samples"], json!([id.clone()]));

    let (status, got) = get(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(got["name"], "bria");

    let (status, renamed) = patch_json(
        home.path(),
        &format!("/v1/audio/samples/{id}"),
        json!({"name": "bria-2"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{renamed}");
    assert_eq!(renamed["name"], "bria-2");

    for variant in ["clean", "cropped"] {
        let req = Request::get(format!("/v1/audio/samples/{id}/audio?variant={variant}"))
            .header(HOST, HOSTPORT)
            .body(Body::empty())
            .unwrap();
        let resp = app(home.path()).oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{variant}");
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(!bytes.is_empty(), "{variant}");
    }

    let (status, _) = delete(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, _) = get(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// A `{id}` path segment can still decode to `..` or embed a `/`: axum
/// percent-decodes the captured value *after* matching one raw segment,
/// so `%2F` in the URL becomes a literal `/` in `id`. Without a check,
/// `DELETE /v1/audio/prep/clips/..%2Fsamples%2F<id>` reaches a sibling
/// directory outside `clips/` entirely (naru task 1461 review).
#[tokio::test]
async fn delete_clip_traversal_id_is_404_and_does_not_touch_the_target() {
    let home = home();
    let (_, clip_body) = upload(home.path(), &wav()).await;
    let clip_id = clip_body["id"].as_str().unwrap().to_string();
    let (status, sample_body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip_id, "name": "victim", "start": 0.0, "end": 0.3,
            "isolate": false, "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{sample_body}");
    let sample_id = sample_body["id"].as_str().unwrap().to_string();
    let victim_meta = home
        .path()
        .join("prep")
        .join("samples")
        .join(&sample_id)
        .join("meta.json");
    assert!(
        victim_meta.is_file(),
        "victim sample must exist before the attack"
    );

    let req = Request::delete(format!("/v1/audio/prep/clips/..%2Fsamples%2F{sample_id}"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app(home.path()), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(
        victim_meta.is_file(),
        "the traversal must not have deleted the sibling sample directory"
    );
}

/// [`delete_clip_traversal_id_is_404_and_does_not_touch_the_target`], the
/// other direction: `DELETE /v1/audio/samples/..%2Fclips%2F<id>` must not
/// reach a sibling clip directory.
#[tokio::test]
async fn delete_sample_traversal_id_is_404_and_does_not_touch_the_target() {
    let home = home();
    let (_, clip_body) = upload(home.path(), &wav()).await;
    let clip_id = clip_body["id"].as_str().unwrap().to_string();
    let victim_meta = home
        .path()
        .join("prep")
        .join("clips")
        .join(&clip_id)
        .join("meta.json");
    assert!(
        victim_meta.is_file(),
        "victim clip must exist before the attack"
    );

    let req = Request::delete(format!("/v1/audio/samples/..%2Fclips%2F{clip_id}"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app(home.path()), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(
        victim_meta.is_file(),
        "the traversal must not have deleted the sibling clip directory"
    );
}

/// A failed `create_sample` (here: `isolate`'s engine load fails, via
/// `fake-separation`'s missing model files) must not leave a half-written
/// sample directory behind (naru task 1461 review).
#[tokio::test]
async fn failed_create_sample_leaves_no_directory_behind() {
    let home = home();
    let (_, body) = upload(home.path(), &wav()).await;
    let id = body["id"].as_str().unwrap();
    let samples_dir = home.path().join("prep").join("samples");
    let before = std::fs::read_dir(&samples_dir)
        .map(|d| d.count())
        .unwrap_or(0);

    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": id, "name": "s1", "start": 0.0, "end": 0.3,
            "isolate": true, "isolation_model": "fake-separation",
            "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");

    let after = std::fs::read_dir(&samples_dir)
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(
        before, after,
        "a failed create must not leave a sample directory behind"
    );
}
