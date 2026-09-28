//! §5.3 the MLX sidecar's supervision, without MLX or a model: a
//! stdlib-only fake `naru_audio_mlx` speaks the real `protocol.py` under
//! the system Python, recorded in `config.toml` as `mlx setup` would.
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use naru_audio::backend::load_tts;
use naru_audio::log::Logger;
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::mlx::sidecar;
use naru_audio::profile::Profile;
use naru_audio::registry::Registry;
use naru_audio::registry::manifest::Kind;
use naru_audio::server::{AppState, router};
use naru_audio::tts::{SynthOptions, TtsError, TtsModel};
use serde_json::{Value, json};
use tower::ServiceExt;

const PYTHON: &str = "/usr/bin/python3";
const BOUNDARY: &str = "naru-audio-test-boundary";
/// What the fake's `stats` adds per loaded model.
const FAKE_BYTES: u64 = 123_456_789;

/// `synth` of "N" streams N chunks of 3 samples, chunk i all i; "N!" fails
/// after them. Each synth appends the chunks it yielded to `synth.log`.
const FAKE_MAIN: &str = r#"
import argparse, json, os, struct
from .protocol import serve

loaded = set()

def chunks(text):
    n = int(text.rstrip("!"))
    yielded = 0
    try:
        for i in range(n):
            yielded += 1
            yield struct.pack("<3f", i, i, i)
        if text.endswith("!"):
            raise RuntimeError("failed after %d" % n)
    finally:
        with open("synth.log", "a") as f:
            f.write("%d\n" % yielded)

def handle(header, samples):
    op = header["op"]
    if op == "load":
        kind = "tts" if header["model"] == "fake-tts" else "stt"
        if header.get("kind") != kind:
            raise ValueError("%s is %s, not %r" % (header["model"], kind, header.get("kind")))
        loaded.add(header["model"])
        return {"sample_rate": 24000} if kind == "tts" else {}
    if op == "synth":
        int(header["text"].rstrip("!"))
        with open("synth.json", "w") as f:
            json.dump(header, f)
        return chunks(header["text"])
    if op == "unload":
        if os.path.exists("unload.fail"):
            raise RuntimeError("unload fails")
        loaded.discard(header["model"])
        return {}
    if op == "stats":
        return {"active_bytes": FAKE_BYTES * len(loaded)}
    if op == "transcribe" and header["model"] == "fake-whisper":
        # Detects "de" unless told; words only when asked for.
        segment = {"start": 0.0, "end": 0.5, "text": "hallo welt"}
        answer = {"segments": [segment], "language": header.get("language", "de")}
        if header.get("words"):
            segment["words"] = [
                {"start": 0.1, "end": 0.2, "text": "hallo"},
                {"start": 0.3, "end": 0.5, "text": "welt"},
            ]
            answer["words"] = True
            # A segment with text but no words must not fail the call.
            answer["segments"].append({"start": 0.5, "end": 0.6, "text": "...", "words": []})
        return answer
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
        ("fake-whisper", "stt", "mlx", vec!["fake-vad"]),
        ("fake-vad", "vad", "sherpa-onnx", vec![]),
    ] {
        std::fs::create_dir_all(models.join(name)).unwrap();
        let manifest = json!({"model": {
            "name": name, "kind": kind, "backend": backend, "languages": ["en", "fr"],
            "requires": requires,
        }});
        std::fs::write(
            models.join(name).join("manifest.json"),
            manifest.to_string(),
        )
        .unwrap();
    }
    let tts = json!({
        "model": {"name": "fake-tts", "kind": "tts", "backend": "mlx", "languages": ["en"]},
        "backend": {"mlx": {"clone": true}},
        "voice": [{"id": "fake", "sid": 0, "default": true}],
    });
    std::fs::create_dir_all(models.join("fake-tts")).unwrap();
    std::fs::write(
        models.join("fake-tts").join("manifest.json"),
        tts.to_string(),
    )
    .unwrap();
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
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
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
    transcribe_with(app, "fake-mlx", &[]).await
}

async fn transcribe_with(app: &Router, model: &str, extra: &[(&str, &str)]) -> (StatusCode, Value) {
    let mut body = Vec::new();
    let fields = [("model", model), ("vad", "false")];
    for &(name, value) in fields.iter().chain(extra) {
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

/// A reload whose unload of the instance it replaces fails still stops a
/// sidecar that is left with no model.
#[test]
fn a_reload_whose_unload_fails_stops_an_empty_sidecar() {
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let sidecar = sidecar::for_home(home.path());
    let dir = home.path().join("models").join("fake-mlx");
    sidecar.load("fake-mlx", Kind::Stt, &dir).unwrap();
    let pid = sidecar_pid(home.path());
    std::fs::write(home.path().join("mlx").join("unload.fail"), "").unwrap();
    assert!(sidecar.load("fake-mlx", Kind::Stt, &dir).is_err());
    assert!(!alive(pid), "the sidecar outlived its last model");
}

/// A sidecar that dies while idle is on `/health` at once, before any
/// request finds it dead; one stopped with its last model is not.
#[tokio::test]
async fn an_idle_crash_is_on_health_at_once() {
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
    let reason = async || get(&app, "/health").await["backends"][1]["reason"].clone();

    let (status, body) = transcribe(&app).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let pid = sidecar_pid(home.path());
    let req = Request::post("/api/load")
        .header(HOST, "127.0.0.1:7870")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "fake-mlx", "keep_alive": 0}).to_string(),
        ))
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK);
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(pid) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!alive(pid), "the sidecar outlived its last model");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(reason().await, Value::Null, "a stop is not a crash");

    let (status, body) = transcribe(&app).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let pid = sidecar_pid(home.path());
    // SAFETY: a plain kill of the sidecar this test started.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    while reason().await.is_null() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reason = reason().await;
    assert!(
        reason
            .as_str()
            .is_some_and(|r| r.contains("the sidecar died")),
        "{reason}"
    );
    assert!(!alive(pid), "the dead sidecar was not reaped");
}

/// Synthesises `text` with the fake: the result and the chunks the sink
/// got. The sink cancels once it has `keep` chunks, and panics on the first
/// if `keep` is 0.
fn synth(tts: &dyn TtsModel, text: &str, keep: usize) -> (Result<(), TtsError>, Vec<Vec<f32>>) {
    let pieces = Arc::new(Mutex::new(Vec::new()));
    let got = pieces.clone();
    let sink = Box::new(move |samples: &[f32]| {
        assert!(keep > 0, "the sink panics");
        let mut got = got.lock().unwrap();
        got.push(samples.to_vec());
        got.len() < keep
    });
    let result = tts.synth(text, "fake", &SynthOptions::default(), sink);
    let pieces = std::mem::take(&mut *pieces.lock().unwrap());
    (result, pieces)
}

/// How many chunks the last synth yielded, from the fake's `synth.log`.
fn yielded(home: &Path) -> usize {
    let log = std::fs::read_to_string(home.join("mlx").join("synth.log")).unwrap();
    log.lines().last().unwrap().parse().unwrap()
}

/// §5.3 `synth`: the chunks reach the sink in order until the end frame.
/// A sink that cancels stops the sidecar early; an error answer, a
/// cancelled stream and a panicking sink each leave the connection in step,
/// so the next synth on the same sidecar gets exactly its own chunks.
#[test]
fn synth_streams_chunks_and_cancelling_leaves_the_sidecar_in_step() {
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let registry = Registry::open(home.path()).unwrap();
    let manifest = registry.pulled_manifest("fake-tts").unwrap();
    let tts = load_tts(&manifest, &registry.model_dir("fake-tts")).unwrap();
    assert_eq!(tts.sample_rate(), 24_000);
    assert_eq!(tts.resident_bytes(), Some(FAKE_BYTES));
    let pid = sidecar_pid(home.path());
    let five: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();

    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);

    // Cancelled after the first chunk: `Ok`, and the sidecar stopped long
    // before the 100 000th.
    let (result, pieces) = synth(&*tts, "100000", 1);
    result.unwrap();
    assert_eq!(pieces, [vec![0.0; 3]]);
    assert!(yielded(home.path()) < 100_000, "{}", yielded(home.path()));
    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);

    // Cancelled on the last chunk: the cancel may land after the end frame,
    // where the sidecar ignores it.
    let (result, pieces) = synth(&*tts, "1", 1);
    result.unwrap();
    assert_eq!(pieces, [vec![0.0; 3]]);
    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);

    // An error after two chunks is the end frame's.
    let (result, pieces) = synth(&*tts, "2!", usize::MAX);
    let err = result.unwrap_err();
    assert!(
        matches!(&err, TtsError::Sidecar(m) if m.contains("failed after 2")),
        "{err}"
    );
    assert_eq!(pieces.len(), 2);
    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);

    // A panicking sink is re-raised after the stream is read to its end.
    let panicked =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| synth(&*tts, "100000", 0)));
    assert!(panicked.is_err());
    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);

    // Checked before the sidecar is asked.
    let (result, _) = synth(&*tts, "5\0", usize::MAX);
    assert!(matches!(result, Err(TtsError::NulInText)));

    assert_eq!(sidecar_pid(home.path()), pid, "the sidecar restarted");

    // Killed, the next synth is `BackendUnavailable`; the restart reloads
    // the model as a TTS model (the fake refuses any other kind), and
    // synthesis works again.
    // SAFETY: a plain kill of the sidecar this test started.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
    std::thread::sleep(Duration::from_millis(200));
    let (result, _) = synth(&*tts, "5", usize::MAX);
    assert!(
        matches!(result, Err(TtsError::BackendUnavailable { .. })),
        "{result:?}"
    );
    let (result, pieces) = synth(&*tts, "5", usize::MAX);
    result.unwrap();
    assert_eq!(pieces, five);
    let restarted = sidecar_pid(home.path());
    assert_ne!(restarted, pid);

    drop(tts);
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(restarted) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!alive(restarted), "the sidecar outlived its last model");
}

/// A sink that blocks (a client that stops reading) does not hold the
/// sidecar: another synth on it completes meanwhile, and the blocked one
/// still gets all its chunks once the sink is released.
#[test]
fn a_blocked_sink_does_not_hold_the_sidecar() {
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let registry = Registry::open(home.path()).unwrap();
    let manifest = registry.pulled_manifest("fake-tts").unwrap();
    let tts = load_tts(&manifest, &registry.model_dir("fake-tts")).unwrap();
    let five: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();

    let (release, blocked) = std::sync::mpsc::channel::<()>();
    let (entered, in_sink) = std::sync::mpsc::channel::<()>();
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let tts = &*tts;
        let first = scope.spawn(move || {
            let pieces = Arc::new(Mutex::new(Vec::new()));
            let got = pieces.clone();
            let sink = Box::new(move |samples: &[f32]| {
                if got.lock().unwrap().is_empty() {
                    let _ = entered.send(());
                    let _ = blocked.recv();
                }
                got.lock().unwrap().push(samples.to_vec());
                true
            });
            let result = tts.synth("5", "fake", &SynthOptions::default(), sink);
            let pieces = std::mem::take(&mut *pieces.lock().unwrap());
            (result, pieces)
        });
        in_sink.recv().unwrap();
        scope.spawn(move || {
            let _ = done.send(synth(tts, "5", usize::MAX));
        });
        let other = finished.recv_timeout(Duration::from_secs(5));
        release.send(()).unwrap();
        let (result, pieces) = other.expect("the blocked sink held the sidecar");
        result.unwrap();
        assert_eq!(pieces, five);
        let (result, pieces) = first.join().unwrap();
        result.unwrap();
        assert_eq!(pieces, five);
    });
}

/// A cloning model's preset voice goes to the sidecar as `voice`; a cloned
/// voice as `reference` (its clip) and `reference_text` (its transcript),
/// with no `voice`. A name outside `voices/` is an unknown voice.
#[test]
fn a_cloned_voice_is_sent_as_its_reference_and_transcript() {
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let amy = home.path().join("voices").join("amy");
    std::fs::create_dir_all(&amy).unwrap();
    std::fs::write(amy.join("ref.wav"), b"").unwrap();
    std::fs::write(amy.join("ref.txt"), "Hello there.\n").unwrap();
    let registry = Registry::open(home.path()).unwrap();
    let manifest = registry.pulled_manifest("fake-tts").unwrap();
    let tts = load_tts(&manifest, &registry.model_dir("fake-tts")).unwrap();
    let sent = |voice: &str| {
        let result = tts.synth("1", voice, &SynthOptions::default(), Box::new(|_| true));
        result.map(|()| {
            let json = std::fs::read(home.path().join("mlx").join("synth.json")).unwrap();
            let header: Value = serde_json::from_slice(&json).unwrap();
            (
                header["voice"].clone(),
                header["reference"].clone(),
                header["reference_text"].clone(),
            )
        })
    };

    assert_eq!(
        sent("fake").unwrap(),
        (json!("fake"), Value::Null, Value::Null)
    );
    assert_eq!(
        sent("amy").unwrap(),
        (
            Value::Null,
            json!(amy.join("ref.wav")),
            json!("Hello there.")
        )
    );
    for voice in ["nope", "../voices/amy"] {
        let err = sent(voice).unwrap_err();
        assert!(matches!(err, TtsError::UnknownVoice { .. }), "{err}");
    }
}

/// A model that instructs is sent the request's `instructions` as
/// `instruct`; with no voices (VoiceDesign), any voice is taken and none
/// is sent. A model that does not instruct is sent no `instruct`.
#[test]
fn instructions_are_sent_as_instruct_only_to_a_model_that_instructs() {
    let runs = std::process::Command::new(PYTHON)
        .args(["-c", ""])
        .output()
        .is_ok_and(|out| out.status.success());
    if !runs {
        eprintln!("skipped: {PYTHON} does not run");
        return;
    }
    let home = home();
    let registry = Registry::open(home.path()).unwrap();
    let options = SynthOptions {
        instructions: Some("A warm, husky woman. Speak slowly.".to_string()),
        ..Default::default()
    };
    let sent = |voice: &str| {
        let manifest = registry.pulled_manifest("fake-tts").unwrap();
        let tts = load_tts(&manifest, &registry.model_dir("fake-tts")).unwrap();
        tts.synth("1", voice, &options, Box::new(|_| true)).unwrap();
        let json = std::fs::read(home.path().join("mlx").join("synth.json")).unwrap();
        let header: Value = serde_json::from_slice(&json).unwrap();
        header
    };

    let header = sent("fake");
    assert_eq!(header["voice"], "fake");
    assert!(header.get("instruct").is_none(), "{header}");

    let design = json!({
        "model": {"name": "fake-tts", "kind": "tts", "backend": "mlx", "languages": ["en"]},
        "backend": {"mlx": {"instruct": true}},
    });
    std::fs::write(
        registry.model_dir("fake-tts").join("manifest.json"),
        design.to_string(),
    )
    .unwrap();
    let header = sent("alloy");
    assert_eq!(header["instruct"], "A warm, husky woman. Speak slowly.");
    assert!(header.get("voice").is_none(), "{header}");
}

/// Whisper's language and word timestamps round-trip through the
/// protocol: a requested language is passed to the sidecar and reported;
/// without one, the sidecar's detected language is; a model that has no
/// notion of a language keeps reporting the manifest's first.
#[tokio::test]
async fn a_whisper_model_reports_the_language_it_used_and_its_words() {
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
    let verbose = [("response_format", "verbose_json")];

    let (status, body) = transcribe_with(&app, "fake-whisper", &verbose).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["text"], "hallo welt");
    assert_eq!(body["language"], "de", "{body}");

    let fr = [("response_format", "verbose_json"), ("language", "fr")];
    let (status, body) = transcribe_with(&app, "fake-whisper", &fr).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["language"], "fr", "{body}");

    let (status, body) = transcribe_with(&app, "fake-mlx", &verbose).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["language"], "en", "{body}");

    // Words: from the model directly. `fake-mlx` answers text without
    // any, which is unsupported; `fake-whisper` answers them.
    let registry = Registry::open(home.path()).unwrap();
    let load = |name: &str| {
        let manifest = registry.pulled_manifest(name).unwrap();
        naru_audio::backend::load_stt(&manifest, &home.path().join("models").join(name)).unwrap()
    };
    let pcm: Vec<f32> = (0..16_000)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.3)
        .collect();
    let whisper = load("fake-whisper");
    let (words, language) = whisper.decode_words_in(&pcm, None, None).unwrap();
    assert_eq!(language.as_deref(), Some("de"));
    let got: Vec<_> = words
        .iter()
        .map(|w| (w.text.as_str(), w.start, w.end))
        .collect();
    assert_eq!(got, [("hallo", 0.1, 0.2), ("welt", 0.3, 0.5)]);
    let (_, language) = whisper.decode_words_in(&pcm, None, Some("fr")).unwrap();
    assert_eq!(language.as_deref(), Some("fr"));
    assert!(matches!(
        load("fake-mlx").decode_words(&pcm, None),
        Err(naru_audio::stt::SttError::WordTimestampsUnsupported)
    ));
}
