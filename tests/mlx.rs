//! §5.3 the MLX sidecar's supervision, without MLX or a model: a
//! stdlib-only fake `naru_audio_mlx` speaks the real `protocol.py` under
//! the system Python, recorded in `config.toml` as `mlx setup` would.
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

const PYTHON: &str = "/usr/bin/python3";
const BOUNDARY: &str = "naru-audio-test-boundary";
/// What the fake's `stats` adds per loaded model.
const FAKE_BYTES: u64 = 123_456_789;

const FAKE_MAIN: &str = r#"
import argparse, os
from .protocol import serve

loaded = set()

def handle(header, samples):
    op = header["op"]
    if op == "load":
        loaded.add(header["model"])
        return {}
    if op == "unload":
        loaded.discard(header["model"])
        return {}
    if op == "stats":
        return {"active_bytes": FAKE_BYTES * len(loaded)}
    if op == "transcribe":
        n = len(samples) // 4
        return {"segments": [{"start": 0.0, "end": n / 16000, "text": "fake %d" % n}]}
    raise ValueError(op)

parser = argparse.ArgumentParser()
parser.add_argument("--socket", required=True)
socket_path = parser.parse_args().socket
with open("sidecar.pid", "w") as f:
    f.write(str(os.getpid()))
serve(socket_path, handle)
"#;

/// A home with `fake-mlx` (and the VAD model it requires) pulled, and the
/// fake sidecar set up.
fn home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    for (name, kind, backend, requires) in [
        ("fake-mlx", "stt", "mlx", vec!["fake-vad"]),
        ("fake-vad", "vad", "sherpa-onnx", vec![]),
    ] {
        std::fs::create_dir_all(models.join(name)).unwrap();
        let manifest = json!({"model": {
            "name": name, "kind": kind, "backend": backend, "languages": ["en"],
            "requires": requires,
        }});
        std::fs::write(
            models.join(name).join("manifest.json"),
            manifest.to_string(),
        )
        .unwrap();
    }
    // Never loaded: the requests ask for `vad=false`.
    std::fs::write(models.join("fake-vad").join("silero_vad.onnx"), b"").unwrap();

    let module = dir.path().join("mlx").join("naru_audio_mlx");
    std::fs::create_dir_all(&module).unwrap();
    std::fs::write(module.join("__init__.py"), "").unwrap();
    std::fs::write(
        module.join("protocol.py"),
        include_str!("../mlx/naru_audio_mlx/protocol.py"),
    )
    .unwrap();
    std::fs::write(
        module.join("__main__.py"),
        FAKE_MAIN.replace("FAKE_BYTES", &FAKE_BYTES.to_string()),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.toml"),
        format!("[mlx]\npython = \"{PYTHON}\"\n"),
    )
    .unwrap();
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

/// One second of a 440 Hz tone, so the energy gate passes it.
fn wav() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for i in 0..16_000 {
        let t = i as f32 / 16_000.0;
        let s = (t * 440.0 * std::f32::consts::TAU).sin() * 0.3;
        writer.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn transcribe(app: &Router) -> (StatusCode, Value) {
    let mut body = Vec::new();
    for (name, value) in [("model", "fake-mlx"), ("vad", "false")] {
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
    body.extend_from_slice(&wav());
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let req = Request::post("/v1/audio/transcriptions")
        .header(HOST, "127.0.0.1:7870")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap();
    send(app, req).await
}

async fn get(app: &Router, uri: &str) -> Value {
    let req = Request::get(uri)
        .header(HOST, "127.0.0.1:7870")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(app, req).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

fn sidecar_pid(home: &Path) -> i32 {
    std::fs::read_to_string(home.join("mlx").join("sidecar.pid"))
        .unwrap()
        .parse()
        .unwrap()
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Killing the sidecar gives the next request 503 `backend_unavailable`
/// and `/health` the reason; the retry restarts it, reloads the model and
/// succeeds; unloading the last MLX model stops it. The model's resident
/// bytes are what `stats` reported.
#[tokio::test]
async fn a_killed_sidecar_is_503_then_the_retry_succeeds() {
    // /usr/bin/python3 is a shim on every Mac; it runs only with the
    // Command Line Tools installed.
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let app = app(home.path());

    let (status, body) = transcribe(&app).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["text"], "fake 16000");
    let ps = get(&app, "/api/ps").await;
    assert_eq!(ps[0]["name"], "fake-mlx", "{ps}");
    assert_eq!(ps[0]["resident_bytes"], FAKE_BYTES, "{ps}");

    let pid = sidecar_pid(home.path());
    // SAFETY: a plain kill of the sidecar this test started.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (status, body) = transcribe(&app).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "backend_unavailable", "{body}");
    let health = get(&app, "/health").await;
    let mlx = &health["backends"][1];
    assert_eq!(mlx["name"], "mlx");
    assert!(
        mlx["reason"]
            .as_str()
            .is_some_and(|r| r.contains("the sidecar died")),
        "{health}"
    );

    let (status, body) = transcribe(&app).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["text"], "fake 16000");
    let restarted = sidecar_pid(home.path());
    assert_ne!(restarted, pid);
    let health = get(&app, "/health").await;
    assert!(health["backends"][1].get("reason").is_none(), "{health}");

    let req = Request::post("/api/load")
        .header(HOST, "127.0.0.1:7870")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "fake-mlx", "keep_alive": 0}).to_string(),
        ))
        .unwrap();
    let (status, body) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["loaded"], false, "{body}");
    // The model is dropped off the runtime, just after the answer.
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(restarted) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!alive(restarted), "the sidecar outlived its last model");
}
