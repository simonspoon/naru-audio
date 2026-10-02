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

/// `num_speakers` must be validated before the clip lookup (like
/// `transcribe_unknown_clip_is_404`'s clip id), so a bad body never
/// reaches model load.
#[tokio::test]
async fn transcribe_num_speakers_zero_is_400() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({"num_speakers": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "num_speakers");
}

#[tokio::test]
async fn transcribe_num_speakers_negative_is_400() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({"num_speakers": -1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "num_speakers");
}

#[tokio::test]
async fn transcribe_num_speakers_overflow_is_400() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({"num_speakers": 3000000000i64}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "num_speakers");
}

#[tokio::test]
async fn transcribe_cluster_threshold_out_of_range_is_400() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({"cluster_threshold": 0.0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "cluster_threshold");

    let (status, body) = post_json(
        home.path(),
        "/v1/audio/prep/clips/nonexistent/transcribe",
        json!({"cluster_threshold": 2.5}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "cluster_threshold");
}

/// JSON has no `NaN`/`Infinity` literal, but a number too large for `f32`
/// (still valid JSON, still fits `f64`) parses to `f32::INFINITY` — that's
/// the non-finite case the handler's `is_finite()` check is for.
#[tokio::test]
async fn transcribe_cluster_threshold_non_finite_is_400() {
    let home = home();
    let req = Request::post("/v1/audio/prep/clips/nonexistent/transcribe")
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"cluster_threshold": 1e300}"#))
        .unwrap();
    let (status, body) = send(app(home.path()), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert_eq!(body["error"]["param"], "cluster_threshold");
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
    // The clip was never transcribed, so the cropped span has no words —
    // `transcript.txt` still gets written (empty), which is a `""`, not a
    // `null`; `null` is reserved for a sample whose `transcript.txt` is
    // absent entirely.
    assert_eq!(body["transcript"], "");

    let (status, listed) = get(home.path(), "/v1/audio/samples").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["samples"], json!([id.clone()]));

    let (status, got) = get(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(got["name"], "bria");
    assert_eq!(got["transcript"], "");

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

// ---- steps, render, catalogue, analysis ------------------------------------

async fn post_raw(
    home: &Path,
    path: &str,
    body: Value,
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let req = Request::post(path)
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app(home).oneshot(req).await.unwrap();
    let (status, headers) = (resp.status(), resp.headers().clone());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes.to_vec())
}

async fn uploaded_clip(home: &Path) -> String {
    let (_, body) = upload(home, &wav()).await;
    body["id"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn steps_catalogue_lists_types_params_and_the_default_chain() {
    let home = home();
    let (status, body) = get(home.path(), "/v1/audio/prep/steps").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let steps = body["steps"].as_array().unwrap();
    let hp = steps.iter().find(|s| s["type"] == "highpass").unwrap();
    let freq = hp["params"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "freq_hz")
        .unwrap();
    assert_eq!(freq["default"], 80.0);
    assert_eq!(freq["unit"], "Hz");
    assert!(freq["min"].is_number() && freq["max"].is_number());
    let types: Vec<&str> = steps.iter().map(|s| s["type"].as_str().unwrap()).collect();
    for t in [
        "isolate",
        "denoise",
        "trim_silence",
        "normalize",
        "eq",
        "deess",
        "loudness",
        "reverb",
    ] {
        assert!(types.contains(&t), "{t} missing from {types:?}");
    }
    let chain: Vec<&str> = body["default_chain"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["type"].as_str().unwrap())
        .collect();
    assert_eq!(chain, ["isolate", "denoise", "trim_silence", "normalize"]);
}

#[tokio::test]
async fn steps_array_defines_the_chain_and_ignores_the_flags() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    // The flags would default to isolate+denoise+trim (409 with no models);
    // `steps` replaces them wholesale.
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip, "name": "s", "start": 0.0, "end": 0.4,
            "isolate": true,
            "steps": [
                {"type": "highpass"},
                {"type": "loudness", "target_lufs": -20},
                {"type": "isolate", "enabled": false},
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let steps = body["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["freq_hz"], 80.0);
    assert_eq!(steps[1]["target_lufs"], -20.0);
    assert_eq!(steps[2]["enabled"], false);
    assert_eq!(body["engines"], json!([]));

    let a = &body["analysis"];
    assert!(
        (a["integrated_lufs"].as_f64().unwrap() + 20.0).abs() < 1.0,
        "{a}"
    );
    assert!(a["peak_dbfs"].as_f64().unwrap() <= -1.0 + 0.01);
    assert!(a["noise_floor_dbfs"].is_number());
    assert!(a["speech_secs"].is_null(), "no VAD model pulled");
    assert!((a["duration_secs"].as_f64().unwrap() - 0.4).abs() < 1e-3);

    // Persisted: a GET returns the same steps and analysis.
    let id = body["id"].as_str().unwrap();
    let (_, got) = get(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(got["steps"], body["steps"]);
    assert_eq!(got["analysis"], body["analysis"]);
}

#[tokio::test]
async fn create_sample_stores_a_corrected_transcript() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip, "name": "s", "start": 0.0, "end": 0.4,
            "steps": [{"type": "highpass"}],
            "transcript": "zero point fifteen um",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["transcript"], "zero point fifteen um");
    let id = body["id"].as_str().unwrap();
    let (_, got) = get(home.path(), &format!("/v1/audio/samples/{id}")).await;
    assert_eq!(got["transcript"], "zero point fifteen um");

    // A blank transcript is not a correction: the clip's words (none here) are used.
    let (status, blank) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip, "name": "s", "start": 0.0, "end": 0.4,
            "steps": [{"type": "highpass"}],
            "transcript": "  ",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{blank}");
    assert_eq!(blank["transcript"], "");
}

#[tokio::test]
async fn legacy_flags_build_the_default_chain_and_report_it() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip, "name": "s", "start": 0.0, "end": 0.4,
            "isolate": false, "denoise": false, "trim_silence": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let steps: Vec<(&str, bool)> = body["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["type"].as_str().unwrap(), s["enabled"].as_bool().unwrap()))
        .collect();
    assert_eq!(
        steps,
        [
            ("isolate", false),
            ("denoise", false),
            ("trim_silence", false),
            ("normalize", true)
        ]
    );
    assert_eq!(body["steps"][3]["peak_db"], -1.0);
    assert!((body["analysis"]["peak_dbfs"].as_f64().unwrap() + 1.0).abs() < 0.01);
}

#[tokio::test]
async fn steps_preflight_follows_the_enabled_steps() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    for (steps, model) in [
        (json!([{"type": "denoise"}]), "speech-denoiser-gtcrn"),
        (json!([{"type": "trim_silence"}]), "silero-vad"),
        (
            json!([{"type": "isolate"}]),
            "source-separation-spleeter-2stems-int8",
        ),
    ] {
        let (status, body) = post_json(
            home.path(),
            "/v1/audio/samples",
            json!({"clip_id": clip, "name": "s", "start": 0.0, "end": 0.3, "steps": steps}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert_eq!(body["error"]["code"], "model_not_pulled");
        assert!(body["error"]["message"].as_str().unwrap().contains(model));
    }
}

#[tokio::test]
async fn invalid_steps_are_400_naming_the_param() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    for (steps, needle) in [
        (
            json!([{"type": "normalize"}, {"type": "denoise", "amount": 2}]),
            "steps[1].amount",
        ),
        (json!([{"type": "speed", "rate": 0}]), "steps[0].rate"),
        (json!([{"type": "bogus"}]), "bogus"),
        (
            json!(vec![json!({"type": "speed", "rate": 0.5}); 6]),
            "speed",
        ),
        (json!([{"type": "highpass", "freq": 80}]), "freq"),
    ] {
        let (status, body) = post_json(
            home.path(),
            "/v1/audio/samples",
            json!({"clip_id": clip, "name": "s", "start": 0.0, "end": 0.3, "steps": steps}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"].as_str().unwrap().contains(needle),
            "{body}"
        );
    }
}

fn render_path(clip: &str) -> String {
    format!("/v1/audio/prep/clips/{clip}/render")
}

fn wav_len(bytes: &[u8]) -> usize {
    hound::WavReader::new(std::io::Cursor::new(bytes))
        .unwrap()
        .len() as usize
}

#[tokio::test]
async fn render_whole_chain_until_and_solo() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    let steps = json!([
        {"type": "normalize", "peak_db": -6},
        {"type": "speed", "rate": 2.0},
        {"type": "normalize", "peak_db": -20, "enabled": false},
    ]);
    let render = |extra: Value| {
        let mut body = json!({"start": 0.0, "end": 0.4, "steps": steps});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        post_raw(home.path(), render_path(&clip).leak(), body)
    };

    let (status, headers, bytes) = render(json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[CONTENT_TYPE], "audio/wav");
    assert_eq!(wav_len(&bytes), 3200, "0.4 s at 2x, disabled step skipped");
    let peak: f64 = headers["x-naru-peak-dbfs"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((peak + 6.0).abs() < 1.0, "{peak}");
    assert!(headers.contains_key("x-naru-loudness-lufs"));
    assert!(headers.contains_key("x-naru-noise-floor-dbfs"));
    assert!(
        !headers.contains_key("x-naru-speech-secs"),
        "null analysis is omitted"
    );
    let analysis: Value =
        serde_json::from_str(headers["x-naru-analysis"].to_str().unwrap()).unwrap();
    assert!(analysis["speech_secs"].is_null());
    assert!((analysis["duration_secs"].as_f64().unwrap() - 0.2).abs() < 1e-3);

    let (_, _, bytes) = render(json!({"until": 0})).await;
    assert_eq!(wav_len(&bytes), 6400, "until 0 stops before the speed step");
    let (_, _, bytes) = render(json!({"solo": 1})).await;
    assert_eq!(wav_len(&bytes), 3200);
    let (status, headers, _) = render(json!({"solo": 2})).await;
    assert_eq!(status, StatusCode::OK, "solo applies a disabled step");
    let peak: f64 = headers["x-naru-peak-dbfs"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((peak + 20.0).abs() < 1.0, "{peak}");
}

#[tokio::test]
async fn render_rejects_until_with_solo_and_bad_indices() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    let steps = json!([{"type": "normalize"}]);
    for extra in [
        json!({"until": 0, "solo": 0}),
        json!({"until": 1}),
        json!({"solo": 5}),
    ] {
        let mut body = json!({"start": 0.0, "end": 0.3, "steps": steps});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let (status, body) = post_json(home.path(), &render_path(&clip), body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test]
async fn render_unknown_clip_is_404_and_missing_model_is_409() {
    let home = home();
    let (status, body) = post_json(
        home.path(),
        &render_path("nonexistent"),
        json!({"start": 0.0, "end": 0.3, "steps": []}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

    let clip = uploaded_clip(home.path()).await;
    let (status, body) = post_json(
        home.path(),
        &render_path(&clip),
        json!({"start": 0.0, "end": 0.3, "steps": [{"type": "denoise"}, {"type": "normalize"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");

    // Only the steps that run need their models: solo on the model-free one.
    let (status, _, _) = post_raw(
        home.path(),
        &render_path(&clip),
        json!({"start": 0.0, "end": 0.3, "steps": [{"type": "denoise"}, {"type": "normalize"}], "solo": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

// ---- studio: long clips, peaks, segments, overlaps, takes, projects -----------

/// `secs` seconds of 16 kHz mono 16-bit audio: a 440 Hz tone at `amp`.
fn tone_wav(secs: usize, amp: f32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for i in 0..secs * 16_000 {
        let t = (i % 16_000) as f32 / 16_000.0;
        let s = (t * 440.0 * std::f32::consts::TAU).sin() * amp;
        writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

fn write_transcript(home: &Path, clip: &str, transcript: Value) {
    let path = home
        .join("prep")
        .join("clips")
        .join(clip)
        .join("transcript.json");
    std::fs::write(path, transcript.to_string()).unwrap();
}

/// Words every 0.4 s from 0 to `secs`, no confidence.
fn words_json(secs: f64) -> Vec<Value> {
    let mut out = Vec::new();
    let mut t = 0.0;
    while t + 0.3 <= secs {
        out.push(json!({"start": t, "end": t + 0.3, "text": "word"}));
        t += 0.4;
    }
    out
}

/// A clip past the STT endpoints' 10-minute cap uploads, and its peaks are
/// served from `working.wav` without a whole-clip decode.
#[tokio::test]
async fn long_clip_uploads_and_has_peaks() {
    let home = home();
    // 11 minutes, quiet, with a loud second at 600 s.
    let mut wav = tone_wav(11 * 60, 0.01);
    let loud = tone_wav(1, 0.8);
    let at = 44 + 600 * 16_000 * 2;
    wav[at..at + 32_000].copy_from_slice(&loud[44..44 + 32_000]);
    let (status, body) = upload(home.path(), &wav).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().unwrap();
    assert!((body["duration_secs"].as_f64().unwrap() - 660.0).abs() < 0.01);

    let (status, p) = get(home.path(), &format!("/v1/audio/prep/clips/{id}/peaks")).await;
    assert_eq!(status, StatusCode::OK, "{p}");
    assert_eq!(p["buckets"], 2000);
    assert_eq!(p["sample_rate"], 16_000);
    assert_eq!(p["min"].as_array().unwrap().len(), 2000);
    assert_eq!(p["max"].as_array().unwrap().len(), 2000);
    assert!((p["duration"].as_f64().unwrap() - 660.0).abs() < 0.01);
    // 600 s is bucket 1818 of 2000 (0.33 s each).
    assert!(p["max"][1819].as_f64().unwrap() > 0.5);
    assert!(p["max"][100].as_f64().unwrap() < 0.1);

    let (status, p) = get(
        home.path(),
        &format!("/v1/audio/prep/clips/{id}/peaks?start=599&end=602&buckets=6"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{p}");
    assert_eq!(
        (p["start"].as_f64(), p["end"].as_f64()),
        (Some(599.0), Some(602.0))
    );
    assert!(p["max"][2].as_f64().unwrap() > 0.5);
    assert!(p["max"][0].as_f64().unwrap() < 0.1);

    for bad in ["buckets=0", "buckets=8193", "start=x", "start=5&end=5"] {
        let (status, body) = get(
            home.path(),
            &format!("/v1/audio/prep/clips/{id}/peaks?{bad}"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
    let (status, _) = get(home.path(), "/v1/audio/prep/clips/nope/peaks").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn render_and_create_sample_take_segments() {
    let home = home();
    let clip = uploaded_clip(home.path()).await; // 0.5 s
    // 0.1 s + 0.1 s joined with a 10 ms crossfade (160 samples); the
    // overlapping first two merge into 0.0-0.12.
    let segments = json!([
        {"start": 0.2, "end": 0.3}, {"start": 0.0, "end": 0.1}, {"start": 0.05, "end": 0.12},
    ]);
    let (status, _, bytes) = post_raw(
        home.path(),
        &render_path(&clip),
        json!({"segments": segments, "steps": [{"type": "normalize", "peak_db": -6}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(wav_len(&bytes), 1920 + 1600 - 160);

    for (bad, param) in [
        (json!([]), "segments"),
        (json!([{"start": 0.3, "end": 0.3}]), "segments"),
        (json!([{"start": 0.1, "end": 9.0}]), "segments"),
        (json!([{"start": -1.0, "end": 0.2}]), "segments"),
    ] {
        let (status, body) = post_json(
            home.path(),
            &render_path(&clip),
            json!({"segments": bad, "steps": []}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["param"], param);
    }

    write_transcript(
        home.path(),
        &clip,
        json!({
            "words": [
                {"start": 0.0, "end": 0.1, "text": "one"},
                {"start": 0.1, "end": 0.2, "text": "two"},
                {"start": 0.2, "end": 0.3, "text": "three"},
            ],
            "speakers": [], "stt_model": "m", "diarization_model": null,
        }),
    );
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({
            "clip_id": clip, "name": "seg", "segments": segments,
            "steps": [{"type": "highpass"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["transcript"], "one three");
    assert_eq!(
        body["segments"],
        json!([{"start": 0.0, "end": 0.12}, {"start": 0.2, "end": 0.3}])
    );
    assert_eq!(body["range"], json!({"start": 0.0, "end": 0.3}));
    let id = body["id"].as_str().unwrap();
    let req = Request::get(format!("/v1/audio/samples/{id}/audio?variant=cropped"))
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let resp = app(home.path()).oneshot(req).await.unwrap();
    let cropped = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(wav_len(&cropped), 1920 + 1600 - 160);

    // A plain range still works and stores no segments.
    let (status, body) = post_json(
        home.path(),
        "/v1/audio/samples",
        json!({"clip_id": clip, "name": "r", "start": 0.0, "end": 0.2, "steps": []}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["segments"], json!([]));
}

#[tokio::test]
async fn transcript_without_overlaps_or_confidence_still_loads() {
    let home = home();
    let clip = uploaded_clip(home.path()).await;
    write_transcript(
        home.path(),
        &clip,
        json!({
            "words": [{"start": 0.0, "end": 0.1, "text": "hi"}],
            "speakers": [{"start": 0.0, "end": 0.4, "speaker": 0}],
            "stt_model": "m", "diarization_model": "d",
        }),
    );
    let (status, t) = get(
        home.path(),
        &format!("/v1/audio/prep/clips/{clip}/transcript"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{t}");
    assert_eq!(t["overlaps"], json!([]));
    assert!(t["words"][0].get("confidence").is_none());

    write_transcript(
        home.path(),
        &clip,
        json!({
            "words": [{"start": 0.0, "end": 0.1, "text": "hi", "confidence": 0.75}],
            "speakers": [], "stt_model": "m", "diarization_model": null,
            "overlaps": [{"start": 1.0, "end": 2.0}],
        }),
    );
    let (_, t) = get(
        home.path(),
        &format!("/v1/audio/prep/clips/{clip}/transcript"),
    )
    .await;
    assert_eq!(t["overlaps"], json!([{"start": 1.0, "end": 2.0}]));
    assert_eq!(t["words"][0]["confidence"], 0.75);
}

#[tokio::test]
async fn takes_need_a_transcript_then_pick_the_best() {
    let home = home();
    let (_, body) = upload(home.path(), &tone_wav(40, 0.1)).await;
    let clip = body["id"].as_str().unwrap().to_string();
    let path = format!("/v1/audio/prep/clips/{clip}/takes");

    let (status, body) = post_json(home.path(), &path, json!({})).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "transcript_not_run");

    // Speaker 0 for 0-20 s and speaker 1 for 20-40 s, crosstalk 8-12 s.
    write_transcript(
        home.path(),
        &clip,
        json!({
            "words": words_json(40.0),
            "speakers": [
                {"start": 0.0, "end": 20.0, "speaker": 0},
                {"start": 8.0, "end": 12.0, "speaker": 1},
                {"start": 20.0, "end": 40.0, "speaker": 1},
            ],
            "overlaps": [{"start": 8.0, "end": 12.0}],
            "stt_model": "m", "diarization_model": "d",
        }),
    );
    let (status, out) = post_json(home.path(), &path, json!({"speaker": 0})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    let takes = out["takes"].as_array().unwrap();
    assert!(!takes.is_empty());
    let picked: f64 = takes
        .iter()
        .filter(|t| t["picked"] == true)
        .map(|t| t["end"].as_f64().unwrap() - t["start"].as_f64().unwrap())
        .sum();
    assert!((10.0..=20.0).contains(&picked), "{picked}");
    assert!((out["picked_secs"].as_f64().unwrap() - picked).abs() < 1e-6);
    for t in takes {
        assert!(t["end"].as_f64().unwrap() <= 8.0 || t["start"].as_f64().unwrap() >= 12.0);
        assert!(t["end"].as_f64().unwrap() <= 20.0);
        let score = t["score"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&score));
        let m = &t["metrics"];
        assert!(m["noise_floor_dbfs"].is_number());
        assert_eq!(m["clipped_ratio"], 0.0);
        assert_eq!(m["overlap_ratio"], 0.0);
        assert!(m["confidence"].is_null());
        assert!(m["words_per_sec"].as_f64().unwrap() > 2.0);
        assert!(m["duration"].as_f64().unwrap() >= 1.0);
    }

    // Keeping the overlap leaves takes with crosstalk in them.
    let (_, out) = post_json(
        home.path(),
        &path,
        json!({"speaker": 0, "exclude_overlaps": false, "segments": [{"start": 6.0, "end": 14.0}]}),
    )
    .await;
    let takes = out["takes"].as_array().unwrap();
    assert!(
        takes
            .iter()
            .all(|t| t["start"].as_f64().unwrap() >= 6.0 && t["end"].as_f64().unwrap() <= 14.0)
    );
    assert!(
        takes
            .iter()
            .any(|t| t["metrics"]["overlap_ratio"].as_f64().unwrap() > 0.0)
    );

    let (status, body) = post_json(home.path(), &path, json!({"speaker": 9})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "speaker");
    let (status, _) = post_json(
        home.path(),
        &path,
        json!({"target_min": 30, "target_max": 20}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_json(home.path(), "/v1/audio/prep/clips/nope/takes", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

async fn put_json(home: &Path, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::put(path)
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    send(app(home), req).await
}

#[tokio::test]
async fn project_put_get_list_delete_and_clip_delete() {
    let home = home();
    let a = uploaded_clip(home.path()).await;
    let b = uploaded_clip(home.path()).await;
    let path = |c: &str| format!("/v1/audio/prep/clips/{c}/project");

    let (status, body) = get(home.path(), &path(&a)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "no_project");
    let (status, body) = get(home.path(), "/v1/audio/prep/projects").await;
    assert_eq!((status, &body["projects"]), (StatusCode::OK, &json!([])));

    let edit = json!({
        "name": " My take ", "speaker": 1,
        "segments": [{"start": 0.3, "end": 0.4}, {"start": 0.0, "end": 0.2}],
        "cuts": [{"start": 0.2, "end": 0.3}],
        "exclude_overlaps": true,
        "steps": [{"type": "normalize", "peak_db": -3}],
        "transcript": "fixed words",
        "takes": {"takes": [], "picked_secs": 0},
    });
    let (status, put) = put_json(home.path(), &path(&a), edit.clone()).await;
    assert_eq!(status, StatusCode::OK, "{put}");
    assert_eq!(put["clip_id"], a);
    assert_eq!(put["name"], "My take");
    assert_eq!(
        put["segments"],
        json!([{"start": 0.0, "end": 0.2}, {"start": 0.3, "end": 0.4}])
    );
    assert_eq!(put["cuts"], json!([{"start": 0.2, "end": 0.3}]));
    assert_eq!(put["transcript"], "fixed words");
    assert_eq!(put["takes"], json!({"takes": [], "picked_secs": 0}));
    assert_eq!(put["speaker"], 1);
    assert_eq!(put["exclude_overlaps"], true);
    let created = put["created_at"].clone();
    assert!(put["updated_at"].is_string());

    let (_, got) = get(home.path(), &path(&a)).await;
    assert_eq!(got, put);

    // Replacing keeps created_at.
    let (status, again) = put_json(
        home.path(),
        &path(&a),
        json!({"name": "renamed", "steps": []}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(again["created_at"], created);
    assert_eq!(again["segments"], json!([]));
    assert!(again["transcript"].is_null() && again["speaker"].is_null());

    // Validation.
    for bad in [
        json!({"name": " "}),
        json!({"name": "x", "steps": [{"type": "normalize", "peak_db": 99}]}),
        json!({"name": "x", "segments": [{"start": 0.0, "end": 99.0}]}),
        json!({"name": "x", "cuts": [{"start": 0.4, "end": 0.1}]}),
    ] {
        let (status, body) = put_json(home.path(), &path(&a), bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
    let (status, _) = put_json(home.path(), &path("nope"), json!({"name": "x"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let big = json!({"name": "x", "transcript": "a".repeat(1024 * 1024)});
    let (status, _) = put_json(home.path(), &path(&a), big).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

    // A second project, listed newest first.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    put_json(
        home.path(),
        &path(&b),
        json!({"name": "second", "segments": [{"start": 0.0, "end": 0.1}]}),
    )
    .await;
    write_transcript(
        home.path(),
        &b,
        json!({"words": [], "speakers": [], "stt_model": "m", "diarization_model": null}),
    );
    let (_, listed) = get(home.path(), "/v1/audio/prep/projects").await;
    let rows = listed["projects"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["clip_id"], b);
    assert_eq!(rows[0]["name"], "second");
    assert_eq!(rows[0]["segment_count"], 1);
    assert_eq!(rows[0]["has_transcript"], true);
    assert!((rows[0]["duration"].as_f64().unwrap() - 0.5).abs() < 0.01);
    assert!(rows[0]["updated_at"].is_string() && rows[0]["speaker"].is_null());
    assert_eq!(rows[1]["clip_id"], a);
    assert_eq!(rows[1]["has_transcript"], false);

    let (status, _) = delete(home.path(), &path(&a)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, body) = delete(home.path(), &path(&a)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "no_project");
    let (status, _) = get(home.path(), &path(&a)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Deleting the clip deletes its project.
    let (status, _) = delete(home.path(), &format!("/v1/audio/prep/clips/{b}")).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, listed) = get(home.path(), "/v1/audio/prep/projects").await;
    assert_eq!(listed["projects"], json!([]));
    assert!(!home.path().join("prep/clips").join(&b).exists());
}
