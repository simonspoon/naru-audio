//! §2.5 registry routes: `GET /v1/models`, `POST /api/pull`,
//! `DELETE /api/models/{name}`, and the cloned-voice export
//! `GET /v1/audio/voices/{name}`.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
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
    "breeze-tts-2-mlx",
    "chatterbox-tts-8bit-mlx",
    "indextts-1.5-mlx",
    "omnivoice-bf16-mlx",
    "parakeet-tdt-0.6b-v2-mlx",
    "qwen3-tts-0.6b-base-mlx",
    "qwen3-tts-0.6b-mlx",
    "qwen3-tts-1.7b-base-mlx",
    "qwen3-tts-1.7b-voicedesign-mlx",
    "voxcpm2-8bit-mlx",
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
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
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

/// `get_json`, but against an already-built `Router` (sharing its
/// `AppState`, and so its `pulls` tracker) rather than a fresh one per call.
async fn get_json_router(app: Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::get(uri)
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app, req).await;
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

async fn delete_voice(home: &Path, name: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::delete(format!("/v1/audio/voices/{name}"))
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
            "source-separation-spleeter-2stems-int8",
            "speaker-diarization-en",
            "speech-denoiser-gtcrn",
            "stt",
            "vad"
        ]
    );
    // Pocket TTS: CC-BY-4.0, but its README flags it non-commercial (§1441).
    let pocket = data.iter().find(|m| m["id"] == "pocket-tts-int8").unwrap();
    assert_eq!(pocket["x_license"], "CC-BY-4.0");
    assert_eq!(
        pocket["x_license_url"],
        "https://creativecommons.org/licenses/by/4.0/"
    );
    assert_eq!(pocket["x_non_commercial"], true);
    // Kokoro is Apache-2.0 and commercial-usable.
    let kokoro = data.iter().find(|m| m["id"] == "kokoro-v1.0").unwrap();
    assert_eq!(kokoro["x_license"], "Apache-2.0");
    assert_eq!(kokoro["x_non_commercial"], false);

    let data: Vec<&Value> = data
        .iter()
        .filter(|m| {
            let id = m["id"].as_str().unwrap();
            ![
                "kokoro-v1.0",
                "parakeet-tdt-0.6b-v2-int8",
                "pocket-tts-int8",
                "silero-vad",
                // The diarization/denoise/separation kinds are not "stt",
                // so they would fail this loop's `x_kind == "stt"` check
                // (naru task 1461 review).
                "source-separation-spleeter-2stems-int8",
                "speaker-diarization-en",
                "speech-denoiser-gtcrn",
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
        "x_clone",
        "x_clone_requires_transcript",
        "x_cloned_voice_count",
        "x_default",
        "x_design",
        "x_design_voice_model",
        "x_instruct",
        "x_kind",
        "x_languages",
        "x_license",
        "x_license_url",
        "x_loaded",
        "x_loaded_at",
        "x_non_commercial",
        "x_prompt_format",
        "x_pulled",
        "x_size_bytes",
        "x_source",
        "x_unavailable_reason",
        "x_voice_count",
    ];
    for m in &data {
        let keys: Vec<&str> = m.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, fields, "{m}");
        assert_eq!(m["object"], "model");
        assert_eq!(m["owned_by"], "naru-audio");
        assert_eq!(m["x_kind"], "stt");
        assert_eq!(m["x_loaded"], false);
        assert_eq!(m["x_default"], false);
        // None of these STT/VAD test fixtures clone or design a voice.
        assert_eq!(m["x_clone"], false);
        assert_eq!(m["x_clone_requires_transcript"], false);
        assert_eq!(m["x_instruct"], false);
        assert_eq!(m["x_design"], false);
        assert_eq!(m["x_design_voice_model"], Value::Null);
        assert_eq!(m["x_voice_count"], 0);
        assert_eq!(m["x_cloned_voice_count"], 0);
        assert_eq!(m["x_languages"], Value::Null);
        assert_eq!(m["x_source"], Value::Null);
        assert_eq!(m["x_loaded_at"], Value::Null);
        assert_eq!(m["x_prompt_format"], Value::Null);
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
    let download = |file: &str, completed: u64, total: u64, model_completed: u64| {
        json!({"status": "downloading", "file": file, "completed": completed, "total": total,
               "model_completed": model_completed, "model_total": 2})
    };
    let verifying = |model_completed: u64| json!({"status": "verifying", "model_completed": model_completed, "model_total": 2});
    assert_eq!(
        lines,
        [
            download("vad.onnx", 0, 6, 0),
            download("vad.onnx", 6, 6, 0),
            verifying(0),
            download("enc.onnx", 0, 8, 1),
            download("enc.onnx", 8, 8, 1),
            verifying(1),
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
            "source-separation-spleeter-2stems-int8",
            "speaker-diarization-en",
            "speech-denoiser-gtcrn",
            "vad"
        ]
    );

    // The unreadable stt no longer blocks removing what it requires.
    assert_eq!(delete(dir.path(), "vad").await.0, StatusCode::NO_CONTENT);
}

/// A cloned voice `name` in `home`, as `voice add` leaves it (always
/// `CLONE_MODEL`, since the CLI has no `--model`).
fn add_voice(home: &Path, name: &str, wav: &[u8], text: &str) {
    add_voice_for(home, name, wav, text, naru_audio::voices::CLONE_MODEL);
}

/// A cloned voice `name` in `home`, made for `model`.
fn add_voice_for(home: &Path, name: &str, wav: &[u8], text: &str, model: &str) {
    let voice = home.join("voices").join(name);
    std::fs::create_dir_all(&voice).unwrap();
    std::fs::write(voice.join("ref.wav"), wav).unwrap();
    std::fs::write(voice.join("ref.txt"), text).unwrap();
    std::fs::write(voice.join("model.txt"), model).unwrap();
}

#[tokio::test]
async fn a_cloned_voice_exports_its_clip_byte_for_byte() {
    let dir = home(&[]);
    // Every byte value, so a lossy encoding would show.
    let wav: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
    add_voice(dir.path(), "amy", &wav, "Hello there.\n");

    let (status, body) = get_json(dir.path(), "/v1/audio/voices/amy").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["name"], "amy");
    assert_eq!(body["text"], "Hello there.");
    assert_eq!(body["model"], naru_audio::voices::CLONE_MODEL);
    assert_eq!(body["description"], Value::Null);
    let decoded = STANDARD
        .decode(body["wav_base64"].as_str().unwrap())
        .unwrap();
    assert_eq!(
        decoded,
        std::fs::read(dir.path().join("voices/amy/ref.wav")).unwrap()
    );
    assert_eq!(body.as_object().unwrap().len(), 5);
}

#[tokio::test]
async fn an_unknown_or_built_in_voice_is_a_404() {
    let dir = home(&[]);
    // No `ref.txt`: not a complete cloned voice.
    std::fs::create_dir_all(dir.path().join("voices/half")).unwrap();
    std::fs::write(dir.path().join("voices/half/ref.wav"), b"RIFF").unwrap();
    for name in ["nope", "af_heart", "half"] {
        let (status, body) = get_json(dir.path(), &format!("/v1/audio/voices/{name}")).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{name}");
        assert_eq!(body["error"]["code"], "voice_not_found", "{name}");
    }
}

#[tokio::test]
async fn a_path_traversal_name_is_refused() {
    let dir = home(&[]);
    // A voice `..%2Fetc` could reach, were the name not checked.
    add_voice(dir.path(), "etc", b"RIFF", "Hi.");
    std::fs::create_dir_all(dir.path().join("voices/x")).unwrap();
    for uri in [
        "/v1/audio/voices/..%2Fvoices%2Fetc",
        "/v1/audio/voices/..%2Fetc",
        "/v1/audio/voices/.hidden",
        "/v1/audio/voices/a%5Cb",
    ] {
        let (status, body) = get_json(dir.path(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(body["error"]["code"], "invalid_request", "{uri}");
        assert_eq!(body["error"]["param"], "name", "{uri}");
    }
    // Unencoded, `../x` is two segments: no route.
    let (status, body) = get_json(dir.path(), "/v1/audio/voices/../x").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

/// `DELETE /v1/audio/voices/{name}` removes a cloned voice's directory;
/// the listing no longer has it. Unknown names are a 404, a bad name a
/// 400 (as `GET` and `POST` have it), and a built-in voice, never in
/// `voices/`, is refused rather than answered as merely unknown.
#[tokio::test]
async fn delete_removes_a_cloned_voice_or_refuses() {
    let dir = home(&[]);
    let model = dir.path().join("models/fake-clone");
    std::fs::create_dir_all(&model).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(model.join("manifest.json"), manifest.to_string()).unwrap();
    add_voice(dir.path(), "amy", b"RIFF", "Hi.");

    // Unknown name: 404, same code `GET` uses.
    let (status, body) = delete_voice(dir.path(), "nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "voice_not_found");

    // A built-in voice (kokoro's, from the catalog, never a directory
    // under `voices/`) is refused, not merely reported unknown.
    let (status, body) = delete_voice(dir.path(), "af_heart").await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "builtin_voice");

    // A bad name is a 400, as the other voice routes have it.
    let (status, body) = delete_voice(dir.path(), "..%2Fetc").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_request");
    assert!(dir.path().join("voices/amy").is_dir());

    // The real thing: removed, and gone from the listing.
    let (status, body) = delete_voice(dir.path(), "amy").await;
    assert_eq!(
        status,
        StatusCode::NO_CONTENT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(body.is_empty());
    assert!(!dir.path().join("voices/amy").exists());
    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=fake-clone").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["voices"], json!([]));

    // Already gone: a repeat is a 404, not success again.
    let (status, body) = delete_voice(dir.path(), "amy").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "voice_not_found");
}

async fn patch_voice(home: &Path, name: &str, body: Value) -> (StatusCode, Value) {
    send_json_router(
        app(home),
        "PATCH",
        &format!("/v1/audio/voices/{name}"),
        body,
    )
    .await
}

/// A JSON request with `method` and `body` against an already-built
/// `Router`, sharing its `AppState` — the live `ModelManager` defaults
/// among them — with whatever else was sent to the same `Router`.
async fn send_json_router(
    app: Router,
    method: &str,
    uri: &str,
    body: Value,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(HOST, HOSTPORT)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let (status, _, body) = send(app, req).await;
    (status, serde_json::from_slice(&body).unwrap())
}

fn fake_clone_manifest(dir: &Path) {
    let model = dir.join("models/fake-clone");
    std::fs::create_dir_all(&model).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(model.join("manifest.json"), manifest.to_string()).unwrap();
}

/// naru task 1458 §2.6: `PATCH` renames a cloned voice's directory and
/// updates its transcript, 409s on a name already taken, and 409s
/// `builtin_voice` for a built-in name rather than treating it as unknown
/// (as `DELETE` does).
#[tokio::test]
async fn patch_renames_a_cloned_voice_or_refuses() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");
    add_voice_for(dir.path(), "zed", b"RIFF", "Yo.", "fake-clone");

    // The happy path: renamed, its transcript updated, and the listing
    // shape ("origin" and friends) comes back in the response.
    let (status, body) = patch_voice(
        dir.path(),
        "amy",
        json!({"name": "amelia", "text": "Hello there."}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "amelia");
    assert_eq!(body["origin"], "cloned");
    assert_eq!(body["cloned"], true);
    assert!(!dir.path().join("voices/amy").exists());
    assert!(dir.path().join("voices/amelia").is_dir());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("voices/amelia/ref.txt")).unwrap(),
        "Hello there.\n"
    );

    // 409 `voice_exists`: renaming onto a name already taken.
    let (status, body) = patch_voice(dir.path(), "amelia", json!({"name": "zed"})).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "voice_exists");
    assert!(dir.path().join("voices/amelia").is_dir());

    // 404 `voice_not_found`: no such cloned voice.
    let (status, body) = patch_voice(dir.path(), "nope", json!({"text": "Hi."})).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "voice_not_found");

    // 409 `builtin_voice`: a catalog model's own voice, never in `voices/`.
    let (status, body) = patch_voice(dir.path(), "af_heart", json!({"text": "Hi."})).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "builtin_voice");
}

/// naru task 1458 §2.6: `description` on `POST /v1/audio/voices` writes
/// `design.txt`, whose mere presence the listing reports as `origin`
/// `"designed"` rather than `"cloned"`, with the description alongside it.
#[tokio::test]
async fn a_designed_voice_is_listed_with_its_description() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");
    std::fs::write(
        dir.path().join("voices/amy/design.txt"),
        "A warm, husky woman.\n",
    )
    .unwrap();

    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=fake-clone").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let amy = body["voices"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "amy")
        .unwrap();
    assert_eq!(amy["origin"], "designed");
    assert_eq!(amy["description"], "A warm, husky woman.");

    let (status, body) = get_json(dir.path(), "/v1/audio/voices/amy").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["description"], "A warm, husky woman.");
}

/// naru task 1458 §2.6: a configured default voice (`PUT /api/defaults`)
/// follows its own rename and is cleared once it is deleted, both in the
/// live `ModelManager` and in `config.toml [defaults.voices]` (checked
/// here by a fresh `app` re-reading it after each change).
#[tokio::test]
async fn a_configured_default_voice_follows_rename_and_is_cleared_on_delete() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");
    // One `Router`/`AppState` for the whole test: the live default voice
    // lives in the `ModelManager`'s `RwLock`, not only in `config.toml`, so
    // a fresh `app` per call (as `patch_voice`/`delete_voice` build) would
    // not see what an earlier call set.
    let app = app(dir.path());

    let (status, body) = send_json_router(
        app.clone(),
        "PUT",
        "/api/defaults",
        json!({"voices": {"fake-clone": "amy"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["voices"]["fake-clone"], "amy");

    let (status, body) = send_json_router(
        app.clone(),
        "PATCH",
        "/v1/audio/voices/amy",
        json!({"name": "amelia"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["default"], true);

    let (status, body) = get_json_router(app.clone(), "/api/defaults").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["voices"]["fake-clone"], "amelia");

    let req = Request::delete("/v1/audio/voices/amelia")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app.clone(), req).await;
    assert_eq!(
        status,
        StatusCode::NO_CONTENT,
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (status, body) = get_json_router(app, "/api/defaults").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["voices"].get("fake-clone").is_none(), "{body}");
}

/// naru task 1458: `PUT /api/defaults`, `PATCH /v1/audio/voices/{name}`
/// (rename) and `DELETE /v1/audio/voices/{name}` all persist to
/// `config.toml` before touching the live `ModelManager` default, so a
/// failed write leaves memory exactly as it was rather than ahead of disk.
/// An unparsable `config.toml` is the cheapest way to make the write fail.
#[tokio::test]
async fn a_default_voice_write_failure_never_applies_live() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");
    std::fs::write(dir.path().join("config.toml"), "not [ valid toml").unwrap();
    let app = app(dir.path());

    let (status, body) = send_json_router(
        app.clone(),
        "PUT",
        "/api/defaults",
        json!({"voices": {"fake-clone": "amy"}}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    let (status, body) = get_json_router(app, "/api/defaults").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["voices"].get("fake-clone").is_none(),
        "the failed write must not have applied the default live: {body}"
    );
}

/// naru task 1458 §2.6 `GET /api/voices/{model}/{voice}/sample`: a cloned
/// voice's own `ref.wav`, byte for byte; a built-in voice with nothing
/// cached and no `?generate=true` is 404 `preview_not_cached` rather than
/// synthesised unasked.
#[tokio::test]
async fn voice_sample_serves_a_clones_clip_and_refuses_an_uncached_builtin() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    let wav: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
    add_voice_for(dir.path(), "amy", &wav, "Hi.", "fake-clone");

    let req = Request::get("/api/voices/fake-clone/amy/sample")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let resp = app(dir.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Exactly one `Content-Type`, not `Vec<u8>`'s default
    // `application/octet-stream` plus an appended `audio/wav` (naru task
    // 1458): the admin page plays this straight into a browser `<audio>`
    // tag, which chokes on two.
    let ctypes: Vec<&str> = resp
        .headers()
        .get_all(CONTENT_TYPE)
        .iter()
        .map(|v| v.to_str().unwrap())
        .collect();
    assert_eq!(ctypes, ["audio/wav"]);
    let body = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    assert_eq!(body, wav);

    // A built-in voice of a model not even pulled here (kokoro's own
    // catalog manifest answers `GET`/`POST /v1/audio/voices` without a
    // pull too): nothing cached, and no `generate=true`.
    let req = Request::get("/api/voices/kokoro-v1.0/af_heart/sample")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(dir.path()), req).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "preview_not_cached");

    // `GET` never generates, even asked to with `?generate=true` (a
    // cross-site `<audio src>` sends no `Origin` header, naru task 1458
    // ST2 review): kokoro is not even pulled here, so an attempt to load
    // and synthesise it would fail as `model_not_pulled` (409), not this
    // 404 — the same `preview_not_cached` as with no query at all proves
    // `GET` never tried.
    let req = Request::get("/api/voices/kokoro-v1.0/af_heart/sample?generate=true")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(dir.path()), req).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "preview_not_cached");

    // An unknown voice of a real model is `voice_not_found`, not treated
    // as an uncached built-in.
    let req = Request::get("/api/voices/fake-clone/nope/sample")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(dir.path()), req).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "voice_not_found");
}

/// naru task 1458 ST2 review: generating moved from `GET
/// .../sample?generate=true` to `POST .../sample`, which the Host/Origin
/// guard (§2.1) actually covers. `POST` serves a cloned voice's clip the
/// same as `GET`; for a built-in voice it really does try to load and
/// synthesise (kokoro is not pulled here, so that is `model_not_pulled`,
/// not `preview_not_cached` — the opposite of `GET`, above).
#[tokio::test]
async fn post_sample_generates_where_get_refuses() {
    let dir = home(&[]);
    fake_clone_manifest(dir.path());
    let wav: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
    add_voice_for(dir.path(), "amy", &wav, "Hi.", "fake-clone");

    let req = Request::post("/api/voices/fake-clone/amy/sample")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let resp = app(dir.path()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    assert_eq!(body, wav);

    let req = Request::post("/api/voices/kokoro-v1.0/af_heart/sample")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(app(dir.path()), req).await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "model_not_pulled");
}

/// §5.3 "per-model voice capabilities": `/v1/models` names each TTS
/// model's cloning and voice-design capability, so a client can tell
/// before offering it. Uses the built-in catalog, not a fake manifest,
/// since it is these three flags' wiring from `Manifest` into the JSON
/// that is under test, not the catalog values themselves (`registry::manifest`
/// already covers those per model).
#[tokio::test]
async fn v1_models_names_cloning_and_voice_design_capability() {
    let dir = home(&[]);
    let (status, body) = get_json(dir.path(), "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().unwrap();
    let get = |id: &str| data.iter().find(|m| m["id"] == id).unwrap();

    // Kokoro: preset voices only, no cloning or voice design.
    let kokoro = get("kokoro-v1.0");
    assert_eq!(kokoro["x_clone"], false);
    assert_eq!(kokoro["x_clone_requires_transcript"], false);
    assert_eq!(kokoro["x_instruct"], false);
    // Speed is the only style control sherpa-onnx's Kokoro takes.
    assert_eq!(
        kokoro["x_prompt_format"],
        json!({"knobs": [{"name": "speed", "default": 1.0, "min": 0.5, "max": 2.0}]})
    );

    // Chatterbox, Qwen3-TTS Base, VoxCPM2 and IndexTTS are all
    // `platforms = ["macos-arm64", "macos-x86_64"]` in their catalog
    // entries (mlx-audio only runs on Apple silicon or macOS x86_64), so
    // `Registry::list`'s `runs_here()` filter (registry.rs) drops them from
    // an unpulled `/v1/models` off macOS — e.g. on the Linux CI runner.
    // Gate on the same condition so macOS still asserts them.
    if cfg!(target_os = "macos") {
        // Chatterbox: clones, but its transcript is ignored, so no
        // `x_clone_requires_transcript`. No `instruct`, but it still
        // declares its own knobs.
        let chatterbox = get("chatterbox-tts-8bit-mlx");
        assert_eq!(chatterbox["x_clone"], true);
        assert_eq!(chatterbox["x_clone_requires_transcript"], false);
        assert_eq!(chatterbox["x_instruct"], false);
        assert_eq!(chatterbox["x_prompt_format"]["style"], Value::Null);
        assert_eq!(
            chatterbox["x_prompt_format"]["knobs"],
            json!([
                {"name": "exaggeration", "default": 0.1, "min": 0.0, "max": 1.0},
                {"name": "cfg_weight", "default": 0.5, "min": 0.0, "max": 1.0},
            ])
        );

        // Qwen3-TTS Base: clones, and needs the transcript for in-context
        // cloning. No `instruct`, but `Model.generate`'s own sampling knobs
        // are still declared.
        let base = get("qwen3-tts-0.6b-base-mlx");
        assert_eq!(base["x_clone"], true);
        assert_eq!(base["x_clone_requires_transcript"], true);
        assert_eq!(base["x_instruct"], false);
        assert_eq!(base["x_prompt_format"]["style"], Value::Null);
        assert_eq!(
            base["x_prompt_format"]["knobs"],
            json!([
                {"name": "temperature", "default": 0.9, "min": 0.0, "max": 2.0},
                {"name": "top_p", "default": 1.0, "min": 0.0, "max": 1.0},
            ])
        );

        // VoxCPM2: clones (no transcript needed) and also designs a voice —
        // inline, as a `(description)` prefix mlx-audio's own `generate`
        // prepends onto the text (naru_1457).
        let voxcpm2 = get("voxcpm2-8bit-mlx");
        assert_eq!(voxcpm2["x_clone"], true);
        assert_eq!(voxcpm2["x_clone_requires_transcript"], false);
        assert_eq!(voxcpm2["x_instruct"], true);
        assert_eq!(voxcpm2["x_prompt_format"]["style"], "inline_prefix");
        assert_eq!(
            voxcpm2["x_prompt_format"]["inline"],
            json!({"syntax": "(description)text"})
        );

        // IndexTTS has no style control at all, but declares that
        // explicitly as an empty object — distinct from `null`, which is
        // reserved for a non-TTS model (asserted below).
        assert_eq!(get("indextts-1.5-mlx")["x_prompt_format"], json!({}));
    }

    // Pocket TTS runs on Linux too (`platforms` includes `linux-*`), so it
    // stays unconditional. Same "no style control" shape as IndexTTS, above.
    assert_eq!(get("pocket-tts-int8")["x_prompt_format"], json!({}));

    // A non-TTS model has no `prompt_format` table at all: `null`.
    assert_eq!(get("silero-vad")["x_prompt_format"], Value::Null);
}

#[tokio::test]
async fn the_voice_listing_marks_cloned_voices() {
    let dir = home(&[]);
    let model = dir.path().join("models/fake-clone");
    std::fs::create_dir_all(&model).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
        "voice": [{"id": "af_heart", "sid": 0, "default": true}],
    });
    std::fs::write(model.join("manifest.json"), manifest.to_string()).unwrap();
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");

    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=fake-clone").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        json!({"model": "fake-clone", "voices": [
            {"id": "af_heart", "accent": null, "gender": null, "default": true, "cloned": false,
             "origin": "builtin", "description": null, "duration": null, "has_transcript": false},
            {"id": "amy", "accent": null, "gender": null, "default": false, "cloned": true,
             "origin": "cloned", "description": null, "duration": null, "has_transcript": true},
        ]})
    );
}

/// §5.3 "voices keyed by model": a clone made for one model does not show
/// up under another cloning model's listing, only its own; `?model=clones`
/// still lists every clone, each naming its own `model`; a legacy voice
/// (`add_voice`, no `model.txt`) is attributed to `CLONE_MODEL`.
#[tokio::test]
async fn voices_are_listed_only_under_the_model_they_were_made_for() {
    let dir = home(&[]);
    for name in ["fake-clone", "other-clone"] {
        let model = dir.path().join("models").join(name);
        std::fs::create_dir_all(&model).unwrap();
        let manifest = json!({
            "model": {"name": name, "kind": "tts", "backend": "sherpa-onnx"},
            "backend": {"sherpa-onnx": {"clone": true}},
        });
        std::fs::write(model.join("manifest.json"), manifest.to_string()).unwrap();
    }
    add_voice_for(dir.path(), "amy", b"RIFF", "Hi.", "fake-clone");
    add_voice_for(dir.path(), "zed", b"RIFF", "Yo.", "other-clone");
    // Predates per-model voices: attributed to CLONE_MODEL, not either.
    add_voice(dir.path(), "legacy", b"RIFF", "Old.");

    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=fake-clone").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["voices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["amy"]);

    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=other-clone").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<&str> = body["voices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["zed"]);

    let (status, body) = get_json(dir.path(), "/v1/audio/voices?model=clones").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "clones");
    let mut got: Vec<(String, String)> = body["voices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v["id"].as_str().unwrap().to_string(),
                v["model"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        [
            ("amy".to_string(), "fake-clone".to_string()),
            (
                "legacy".to_string(),
                naru_audio::voices::CLONE_MODEL.to_string()
            ),
            ("zed".to_string(), "other-clone".to_string()),
        ]
    );
}

/// Serves one `GET` request's body in `chunk`-sized pieces, pausing `delay`
/// between them, so a pull's download loop has time to be observed (or
/// cancelled) mid-transfer. Never touches the network: a plain loopback
/// TCP listener, torn down when the test's `home` (and so the pull) is
/// done with it.
fn slow_server(total: usize, chunk: usize, delay: Duration) -> (String, Vec<u8>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bytes: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
    let body = bytes.clone();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            for piece in body.chunks(chunk) {
                if s.write_all(piece).is_err() {
                    break;
                }
                let _ = s.flush();
                std::thread::sleep(delay);
            }
        }
    });
    (format!("http://{addr}/big.bin"), bytes)
}

/// naru task 1458: `DELETE /api/pulls/{name}` cancels a running pull; its
/// `POST /api/pull` stream ends with `pull_cancelled` instead of `success`,
/// and no staging directory is left under `tmp/`.
#[tokio::test]
async fn cancelling_a_pull_ends_the_stream_with_pull_cancelled_and_leaves_no_staging_dir() {
    let (url, bytes) = slow_server(2_000_000, 100_000, Duration::from_millis(30));
    let dir = home(&[model_header("slow", &[]) + &file_entry("big.bin", &url, &bytes)]);
    let router = app(dir.path());

    let post = {
        let router = router.clone();
        tokio::spawn(async move {
            let req = Request::post("/api/pull")
                .header(HOST, HOSTPORT)
                .body(Body::from(r#"{"model":"slow"}"#))
                .unwrap();
            send(router, req).await
        })
    };

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, list) = get_json_router(router.clone(), "/api/pulls").await;
        assert_eq!(status, StatusCode::OK);
        if list
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["model"] == "slow")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the pull never showed up in /api/pulls"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let req = Request::delete("/api/pulls/slow")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = send(router.clone(), req).await;
    assert_eq!(status, StatusCode::ACCEPTED);

    let (status, ctype, body) = post.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ctype, "application/x-ndjson");
    let body = String::from_utf8(body).unwrap();
    let last: Value = serde_json::from_str(body.lines().last().unwrap()).unwrap();
    assert_eq!(last["error"]["code"], "pull_cancelled", "{body}");
    assert!(!body.contains("success"), "{body}");

    assert!(!dir.path().join("tmp/slow").exists(), "staging left behind");
    assert!(!dir.path().join("models/slow").exists());

    let (_, list) = get_json_router(router.clone(), "/api/pulls").await;
    assert_eq!(list, json!([]), "{list}");

    // Cancelling again (or any other unknown pull) is a 404, not a repeat 202.
    let req = Request::delete("/api/pulls/slow")
        .header(HOST, HOSTPORT)
        .body(Body::empty())
        .unwrap();
    let (status, _, body) = send(router.clone(), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "pull_not_found");
}

/// naru task 1458: `GET /api/pulls` lists a pull while it is streaming, with
/// its live progress; a second `POST /api/pull` for the same model while
/// one is running is a 409 `pull_in_progress`.
#[tokio::test]
async fn api_pulls_lists_an_in_flight_pull() {
    let (url, bytes) = slow_server(2_000_000, 100_000, Duration::from_millis(30));
    let dir = home(&[model_header("slow2", &[]) + &file_entry("big.bin", &url, &bytes)]);
    let router = app(dir.path());

    let post = {
        let router = router.clone();
        tokio::spawn(async move {
            let req = Request::post("/api/pull")
                .header(HOST, HOSTPORT)
                .body(Body::from(r#"{"model":"slow2"}"#))
                .unwrap();
            send(router, req).await
        })
    };

    let deadline = Instant::now() + Duration::from_secs(10);
    let entry = loop {
        let (status, list) = get_json_router(router.clone(), "/api/pulls").await;
        assert_eq!(status, StatusCode::OK);
        let found = list
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["model"] == "slow2")
            .cloned();
        if let Some(entry) = found {
            break entry;
        }
        assert!(
            Instant::now() < deadline,
            "the pull never showed up in /api/pulls"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_eq!(entry["model"], "slow2");
    assert_eq!(entry["status"], "running");
    assert_eq!(entry["model_total"], 1);
    assert!(entry["started_at"].is_string(), "{entry}");

    // A second pull of the same model while one is running is refused.
    let req = Request::post("/api/pull")
        .header(HOST, HOSTPORT)
        .body(Body::from(r#"{"model":"slow2"}"#))
        .unwrap();
    let (status, _, body) = send(router.clone(), req).await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(error_code(&body), "pull_in_progress");

    let (status, ctype, body) = post.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ctype, "application/x-ndjson");
    let body = String::from_utf8(body).unwrap();
    assert!(body.lines().last().unwrap().contains("success"), "{body}");

    let (_, list) = get_json_router(router.clone(), "/api/pulls").await;
    assert_eq!(list, json!([]), "{list}");
}
