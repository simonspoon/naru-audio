//! `GET /v1/audio/live` (naru task 2029, `docs/live-protocol.md`) against the
//! daemon on an ephemeral port.
//!
//! The handshake checks and the Origin guard need no model: `fake-stt` is
//! "pulled" by writing its `manifest.json` alone. Everything after `ready`
//! needs `parakeet-tdt-0.6b-v2-int8` and its Silero model, and skips as
//! `tests/stream.rs` does when they are not pulled; `speak` also needs
//! `kokoro-v1.0`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use naru_audio::log::Logger;
use naru_audio::manager::{BackendLoader, KeepAlive, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::{self, Registry};
use naru_audio::server::{AmbientHost, AppState, Capture, router};
use naru_audio::speaker::{self, Clip, Library, SpeakerVerifier};
use naru_audio::stt::audio;
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

const MODEL: &str = "parakeet-tdt-0.6b-v2-int8";
const TTS: &str = "kokoro-v1.0";

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// A home with `fake-stt` (no model files) and `fake-vad` pulled.
fn fake_home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, kind) in [("fake-stt", "stt"), ("fake-vad", "vad")] {
        let model_dir = dir.path().join("models").join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        let manifest = json!({"model": {
            "name": name, "kind": kind, "backend": "sherpa-onnx", "languages": ["en"],
        }});
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();
    }
    dir
}

/// A fresh home whose `models/` links to the pulled models, so the daemon's
/// `state/` (and any enrollment) never lands in the real home. `None` (skip)
/// when [`MODEL`] is not pulled.
fn model_home() -> Option<tempfile::TempDir> {
    let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .or_else(registry::default_home)?;
    if !Registry::open(&home).ok()?.is_installed(MODEL) {
        let message = format!(
            "{MODEL} not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
            home.display()
        );
        if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
            panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
        }
        eprintln!("skip: {message}");
        return None;
    }
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    #[cfg(windows)]
    use std::os::windows::fs::symlink_dir as symlink;
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("models")).unwrap();
    for entry in std::fs::read_dir(home.join("models")).unwrap() {
        let entry = entry.unwrap();
        symlink(
            entry.path(),
            dir.path().join("models").join(entry.file_name()),
        )
        .unwrap();
    }
    Some(dir)
}

async fn serve(home: &Path) -> String {
    serve_with(home, None).await
}

async fn serve_with(home: &Path, mic: Option<AmbientHost>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(Logger::stderr());
    let registry = Arc::new(Registry::open(home).unwrap());
    let mut settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
    settings.stt_default = MODEL.to_string();
    settings.tts_default = TTS.to_string();
    settings.keep_alive = KeepAlive::For(Duration::from_secs(300));
    let models = ModelManager::new(
        registry.clone(),
        log.clone(),
        settings,
        Arc::new(BackendLoader),
    );
    let app = router(Arc::new(AppState {
        port,
        allow_remote: false,
        started: Instant::now(),
        log,
        registry,
        models: Arc::new(models),
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
        keep_recent: false,
        mic,
    }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("127.0.0.1:{port}")
}

async fn connect(addr: &str) -> Ws {
    let url = format!("ws://{addr}/v1/audio/live");
    tokio_tungstenite::connect_async(url).await.unwrap().0
}

fn start(fields: Value) -> Message {
    let mut start = json!({"type": "start", "mode": "live", "rate": 16000});
    for (k, v) in fields.as_object().unwrap() {
        start[k] = v.clone();
    }
    Message::text(start.to_string())
}

fn text(v: Value) -> Message {
    Message::text(v.to_string())
}

/// Every event until the close frame, and its code; panics after 60 s.
async fn until_close(ws: &mut Ws) -> (Vec<Value>, Option<u16>) {
    let mut events = Vec::new();
    let read = async {
        while let Some(msg) = ws.next().await {
            match msg {
                Ok(Message::Text(t)) => events.push(serde_json::from_str(&t).unwrap()),
                Ok(Message::Close(frame)) => return frame.map(|f| u16::from(f.code)),
                Ok(_) => {}
                Err(e) => panic!("socket error: {e}"),
            }
        }
        None
    };
    let code = tokio::time::timeout(Duration::from_secs(60), read)
        .await
        .expect("no close within 60 s");
    (events, code)
}

/// The next event satisfying `want`, skipping others; a fatal error or a
/// close is a panic. Panics after 60 s.
async fn next_where(ws: &mut Ws, what: &str, want: impl Fn(&Value) -> bool) -> Value {
    let read = async {
        loop {
            match ws.next().await {
                Some(Ok(Message::Text(t))) => {
                    let event: Value = serde_json::from_str(&t).unwrap();
                    if want(&event) {
                        return event;
                    }
                    assert_ne!(event["fatal"], true, "{event}");
                }
                Some(Ok(Message::Close(frame))) => panic!("closed waiting for {what}: {frame:?}"),
                Some(Ok(_)) => {}
                other => panic!("socket ended waiting for {what}: {other:?}"),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(60), read)
        .await
        .unwrap_or_else(|_| panic!("no {what} within 60 s"))
}

async fn next_of(ws: &mut Ws, kind: &str) -> Value {
    next_where(ws, kind, |e| e["type"] == kind).await
}

fn assert_error_close(events: &[Value], close: Option<u16>, code: &str) {
    let last = events.last().expect("an error event");
    assert_eq!(last["type"], "error", "{events:?}");
    assert_eq!(last["code"], code, "{last}");
    assert_eq!(last["fatal"], true, "{last}");
    assert!(last["message"].as_str().is_some_and(|m| !m.is_empty()));
    assert_eq!(close, Some(1008), "{events:?}");
}

async fn session_error(home: &Path, first: Message) -> (Vec<Value>, Option<u16>) {
    let addr = serve(home).await;
    let mut ws = connect(&addr).await;
    ws.send(first).await.unwrap();
    until_close(&mut ws).await
}

fn s16le(pcm: &[f32]) -> Vec<u8> {
    pcm.iter()
        .flat_map(|s| ((s * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
        .collect()
}

fn samples(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(name);
    audio::decode(std::fs::File::open(path).unwrap()).unwrap()
}

/// Two copies of `plain.wav` separated, and followed, by 1 s of silence.
fn two_plains() -> Vec<f32> {
    let speech = samples("plain.wav");
    let gap = vec![0.0f32; 16_000];
    let mut pcm = Vec::new();
    for _ in 0..2 {
        pcm.extend_from_slice(&speech);
        pcm.extend_from_slice(&gap);
    }
    pcm
}

async fn send_audio(ws: &mut Ws, pcm: &[f32]) {
    for frame in pcm.chunks(640) {
        ws.send(Message::binary(s16le(frame))).await.unwrap();
    }
}

/// A ready session in `mode`, and its first `state`.
async fn ready(addr: &str, fields: Value) -> (Ws, Value) {
    let mut ws = connect(addr).await;
    ws.send(start(fields)).await.unwrap();
    let ready = next_of(&mut ws, "ready").await;
    assert_eq!(ready["model"], MODEL, "{ready}");
    assert!(ready["session"].as_str().is_some_and(|s| !s.is_empty()));
    let state = next_of(&mut ws, "state").await;
    (ws, state)
}

/// Stops the session and returns everything it said until the close.
async fn stop(ws: &mut Ws) -> Vec<Value> {
    ws.send(text(json!({"type": "stop"}))).await.unwrap();
    let (events, close) = until_close(ws).await;
    assert_eq!(close, Some(1000), "{events:?}");
    assert_eq!(events.last().unwrap()["type"], "done");
    events
}

#[tokio::test(flavor = "multi_thread")]
async fn binary_before_start_is_1008() {
    let home = fake_home();
    let (events, close) = session_error(home.path(), Message::binary(vec![0u8; 4])).await;
    assert_error_close(&events, close, "invalid_request");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_first_message_other_than_start_is_1008() {
    let home = fake_home();
    let (events, close) = session_error(home.path(), text(json!({"type": "stop"}))).await;
    assert_error_close(&events, close, "invalid_request");
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_start_fields_are_1008() {
    let home = fake_home();
    for (fields, code) in [
        (json!({"mode": "loud"}), "invalid_request"),
        (json!({"mode": null}), "invalid_request"),
        (json!({"rate": 44100}), "unsupported_value"),
        (json!({"format": "f32le"}), "unsupported_value"),
        (json!({"session": ""}), "invalid_request"),
        (json!({"vad": {"threshold": 2.0}}), "invalid_request"),
    ] {
        let (events, close) = session_error(home.path(), start(fields.clone())).await;
        assert_error_close(&events, close, code);
        eprintln!("ok: {fields}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_foreign_origin_is_refused() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    let mut request = format!("ws://{addr}/v1/audio/live")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", "http://evil.example".parse().unwrap());
    let refused = tokio_tungstenite::connect_async(request).await;
    assert!(refused.is_err(), "a foreign Origin must not upgrade");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_without_upgrade_is_400() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    let status = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://{addr}/v1/audio/live"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap()
            .status()
    })
    .await
    .unwrap();
    assert_eq!(status, 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn start_gives_ready_then_state() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, state) = ready(&addr, json!({})).await;
    assert_eq!(state["mode"], "live", "{state}");
    assert_eq!(state["seq"], 1, "{state}");
    assert_eq!(state["resumed"], false, "{state}");
    assert_eq!(state["agent_speaking"], false, "{state}");
    assert_eq!(state["enrolled"], false, "{state}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn mode_changes_emit_state() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    for (mode, seq) in [("muted", 2), ("ambient", 3), ("idle", 4), ("live", 5)] {
        ws.send(text(json!({"type": "mode", "mode": mode})))
            .await
            .unwrap();
        let state = next_of(&mut ws, "state").await;
        assert_eq!(state["mode"], mode, "{state}");
        assert_eq!(state["seq"], seq, "{state}");
    }
    // The same mode again changes nothing; the next change is seq 6.
    ws.send(text(json!({"type": "mode", "mode": "live"})))
        .await
        .unwrap();
    ws.send(text(json!({"type": "agent_speaking", "value": true})))
        .await
        .unwrap();
    let state = next_of(&mut ws, "state").await;
    assert_eq!(state["seq"], 6, "{state}");
    assert_eq!(state["agent_speaking"], true, "{state}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_live_utterance_has_text_and_ids() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    let utterances: Vec<&Value> = events.iter().filter(|e| e["type"] == "utterance").collect();
    assert_eq!(utterances.len(), 2, "{events:?}");
    for (i, u) in utterances.iter().enumerate() {
        assert_eq!(u["id"], i, "{u}");
        assert!(u["text"].as_str().is_some_and(|t| !t.is_empty()), "{u}");
        assert!(u["start"].as_f64() < u["end"].as_f64(), "{u}");
        // Nobody is enrolled, so there is no speaker verdict.
        assert!(u.get("speaker").is_none(), "{u}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn partials_come_before_their_utterance() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(kinds.contains(&"utterance"), "{kinds:?}");
    let first_utterance = kinds.iter().position(|k| *k == "utterance").unwrap();
    assert!(
        kinds[..first_utterance].contains(&"speech"),
        "speech starts before the utterance: {kinds:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_speaking_suppresses_the_utterance() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    ws.send(text(json!({"type": "agent_speaking", "value": true})))
        .await
        .unwrap();
    next_of(&mut ws, "state").await;
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    assert!(
        !events.iter().any(|e| e["type"] == "utterance"),
        "the gate was shut: {events:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn muted_and_idle_hear_nothing_and_ambient_flags() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    for mode in ["muted", "idle"] {
        let (mut ws, state) = ready(&addr, json!({"mode": mode})).await;
        assert_eq!(state["mode"], mode, "{state}");
        send_audio(&mut ws, &two_plains()).await;
        let events = stop(&mut ws).await;
        let heard = events
            .iter()
            .any(|e| matches!(e["type"].as_str(), Some("utterance" | "ambient_flag")));
        assert!(!heard, "{mode} heard something: {events:?}");
    }
    let (mut ws, state) = ready(&addr, json!({"mode": "ambient"})).await;
    assert_eq!(state["mode"], "ambient", "{state}");
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    let flags = events
        .iter()
        .filter(|e| e["type"] == "ambient_flag")
        .count();
    assert_eq!(flags, 2, "{events:?}");
    assert!(
        !events.iter().any(|e| e["type"] == "utterance"),
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn level_events_follow_the_microphone() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    let speech = samples("plain.wav");
    // 15 frames over ~300 ms: at least one level at 10 Hz.
    for frame in speech.chunks(640).take(15) {
        ws.send(Message::binary(s16le(frame))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let level = next_of(&mut ws, "level").await;
    assert!(level["rms"].as_f64().unwrap() > 0.0, "{level}");
    assert_eq!(level["gate"], true, "{level}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn violations_after_start_are_1008() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    for bad in [
        Message::binary(vec![0u8; 64 * 1024 + 2]),
        Message::binary(vec![0u8; 3]),
        Message::text("not json"),
        text(json!({"type": "start", "mode": "live", "rate": 16000})),
        text(json!({"type": "mode", "mode": "loud"})),
        text(json!({"type": "dance"})),
    ] {
        let (mut ws, _) = ready(&addr, json!({})).await;
        ws.send(bad).await.unwrap();
        let (events, close) = until_close(&mut ws).await;
        assert_error_close(&events, close, "invalid_request");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn enroll_mistakes_are_errors_that_keep_the_session() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    for message in [
        json!({"type": "enroll", "action": "finish"}),
        json!({"type": "enroll", "action": "wiggle"}),
    ] {
        ws.send(text(message)).await.unwrap();
        let error = next_of(&mut ws, "error").await;
        assert_eq!(error["fatal"], false, "{error}");
        assert_eq!(error["code"], "invalid_request", "{error}");
    }
    // A recording under a second is refused, and the recording ends.
    ws.send(text(json!({"type": "enroll", "action": "begin"})))
        .await
        .unwrap();
    let state = next_of(&mut ws, "state").await;
    assert_eq!(state["enrolling"], true, "{state}");
    send_audio(&mut ws, &samples("plain.wav")[..4000]).await;
    ws.send(text(json!({"type": "enroll", "action": "finish"})))
        .await
        .unwrap();
    let error = next_of(&mut ws, "error").await;
    assert_eq!(error["fatal"], false, "{error}");
    let state = next_of(&mut ws, "state").await;
    assert_eq!(state["enrolling"], false, "{state}");
    assert_eq!(state["enrolled"], false, "{state}");
    // Still a working session.
    ws.send(text(json!({"type": "mode", "mode": "muted"})))
        .await
        .unwrap();
    assert_eq!(next_of(&mut ws, "state").await["mode"], "muted");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn speak_streams_chunks_with_the_id() {
    let Some(home) = model_home() else { return };
    if !home.path().join("models").join(TTS).exists() {
        eprintln!("skip: {TTS} not pulled");
        return;
    }
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    ws.send(text(
        json!({"type": "speak", "id": "s1", "text": "Hello there."}),
    ))
    .await
    .unwrap();
    let mut chunks = Vec::new();
    loop {
        let event = next_of(&mut ws, "speak_audio").await;
        assert_eq!(event["id"], "s1", "{event}");
        assert_eq!(event["seq"], chunks.len(), "{event}");
        let final_ = event["final"] == true;
        chunks.push(event);
        if final_ {
            break;
        }
    }
    let audio: usize = chunks
        .iter()
        .map(|c| {
            base64::engine::general_purpose::STANDARD
                .decode(c["data"].as_str().unwrap())
                .unwrap()
                .len()
        })
        .sum();
    assert!(
        audio > 4000 && audio.is_multiple_of(2),
        "{audio} bytes of s16le"
    );
    assert!(chunks[0]["rate"].as_u64().unwrap() > 0);
    assert!(chunks.len() >= 2, "audio then a final marker");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_newer_speak_replaces_the_older() {
    let Some(home) = model_home() else { return };
    if !home.path().join("models").join(TTS).exists() {
        eprintln!("skip: {TTS} not pulled");
        return;
    }
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    let long = "This is a fairly long sentence. ".repeat(12);
    ws.send(text(json!({"type": "speak", "id": "a", "text": long})))
        .await
        .unwrap();
    ws.send(text(json!({"type": "speak", "id": "b", "text": "Short."})))
        .await
        .unwrap();
    let mut a_end: Option<Value> = None;
    let mut b_final = false;
    while !(a_end.is_some() && b_final) {
        let event = next_of(&mut ws, "speak_audio").await;
        match (event["id"].as_str().unwrap(), event["final"] == true) {
            ("a", true) => a_end = Some(event),
            ("b", true) => b_final = true,
            ("b", false) => {}
            // Audio of "a" before it was cut.
            ("a", false) => assert!(a_end.is_none(), "audio of a after its end"),
            other => panic!("{other:?}"),
        }
    }
    let a_end = a_end.unwrap();
    assert_eq!(a_end["cancelled"], true, "{a_end}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_speak_with_a_bad_model_is_an_error_that_keeps_the_session() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    ws.send(text(json!({
        "type": "speak", "id": "x", "text": "hi", "model": "no-such-model",
    })))
    .await
    .unwrap();
    let error = next_of(&mut ws, "error").await;
    assert_eq!(error["fatal"], false, "{error}");
    assert_eq!(error["id"], "x", "{error}");
    ws.send(text(json!({"type": "mode", "mode": "muted"})))
        .await
        .unwrap();
    next_of(&mut ws, "state").await;
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reconnect_resyncs_the_state() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, first) = ready(&addr, json!({"session": "resume-1"})).await;
    assert_eq!(first["resumed"], false, "{first}");
    ws.send(text(json!({"type": "mode", "mode": "muted"})))
        .await
        .unwrap();
    let muted = next_of(&mut ws, "state").await;
    assert_eq!(muted["seq"], 2, "{muted}");
    // The connection drops without a stop.
    drop(ws);
    tokio::time::sleep(Duration::from_millis(300)).await;

    // No mode: the session picks up where it was, and says so.
    let mut ws = connect(&addr).await;
    ws.send(start(json!({"session": "resume-1", "mode": null})))
        .await
        .unwrap();
    // `mode: null` is "not given".
    let ready_event = next_of(&mut ws, "ready").await;
    assert_eq!(ready_event["session"], "resume-1", "{ready_event}");
    assert_eq!(ready_event["resumed"], true, "{ready_event}");
    let state = next_of(&mut ws, "state").await;
    assert_eq!(state["resumed"], true, "{state}");
    assert_eq!(state["mode"], "muted", "{state}");
    assert_eq!(state["previous_mode"], "muted", "{state}");
    assert_eq!(state["seq"], 3, "{state}");
    assert_eq!(state["lost_speaks"], json!([]), "{state}");
    // A clean stop forgets the session.
    stop(&mut ws).await;
    let mut ws = connect(&addr).await;
    ws.send(start(json!({"session": "resume-1"})))
        .await
        .unwrap();
    assert_eq!(next_of(&mut ws, "ready").await["resumed"], false);
    assert_eq!(next_of(&mut ws, "state").await["seq"], 1);
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_resume_may_name_a_new_mode() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (ws, _) = ready(&addr, json!({"session": "resume-2", "mode": "ambient"})).await;
    drop(ws);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let (mut ws, state) = ready(&addr, json!({"session": "resume-2", "mode": "live"})).await;
    assert_eq!(state["resumed"], true, "{state}");
    assert_eq!(state["mode"], "live", "{state}");
    assert_eq!(state["previous_mode"], "ambient", "{state}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_start_without_a_mode_or_a_session_to_resume_is_1008() {
    let home = fake_home();
    let (events, close) = session_error(
        home.path(),
        start(json!({"mode": null, "session": "never"})),
    )
    .await;
    assert_error_close(&events, close, "invalid_request");
}

/// A typed frame: the type byte, then s16le.
fn typed(kind: u8, pcm: &[f32]) -> Message {
    let mut bytes = vec![kind];
    bytes.extend(s16le(pcm));
    Message::binary(bytes)
}

/// Plays `reference` and hears `mic` back, frame for frame, on a typed
/// session.
async fn send_with_reference(ws: &mut Ws, reference: &[f32], mic: &[f32]) {
    for (i, frame) in mic.chunks(640).enumerate() {
        let from = (i * 640).min(reference.len());
        let to = (from + 640).min(reference.len());
        if from < to {
            ws.send(typed(1, &reference[from..to])).await.unwrap();
        }
        ws.send(typed(0, frame)).await.unwrap();
    }
}

fn utterances(events: &[Value]) -> usize {
    events.iter().filter(|e| e["type"] == "utterance").count()
}

#[tokio::test(flavor = "multi_thread")]
async fn typed_frames_are_opt_in_and_echoed_in_ready() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    for (fields, want) in [(json!({}), false), (json!({"typed_frames": true}), true)] {
        let mut ws = connect(&addr).await;
        ws.send(start(fields)).await.unwrap();
        let ready = next_of(&mut ws, "ready").await;
        assert_eq!(ready["typed_frames"], want, "{ready}");
        stop(&mut ws).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_typed_microphone_frame_behaves_like_an_untyped_one() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    for frame in two_plains().chunks(640) {
        ws.send(typed(0, frame)).await.unwrap();
    }
    let events = stop(&mut ws).await;
    assert_eq!(utterances(&events), 2, "{events:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_frame_type_is_1008() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    ws.send(typed(7, &[0.0; 160])).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_error_close(&events, close, "invalid_request");
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    ws.send(Message::binary(vec![])).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_error_close(&events, close, "invalid_request");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_microphone_hearing_the_reference_makes_no_utterance() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let speech = two_plains();

    // Echo: the reference is exactly what the microphone hears.
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    send_with_reference(&mut ws, &speech, &speech).await;
    let events = stop(&mut ws).await;
    assert_eq!(utterances(&events), 0, "echo_leak_utterances: {events:?}");

    // Control: no reference at all, the same speech is heard.
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    send_with_reference(&mut ws, &[], &speech).await;
    let events = stop(&mut ws).await;
    assert_eq!(utterances(&events), 2, "short_reply_drops: {events:?}");

    // Control: a reference that is not what the microphone hears.
    let other = samples("mesa-names.wav");
    let (mut ws, _) = ready(&addr, json!({"typed_frames": true})).await;
    send_with_reference(&mut ws, &other, &speech).await;
    let events = stop(&mut ws).await;
    assert_eq!(utterances(&events), 2, "short_reply_drops: {events:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_untyped_session_ignores_the_echo_machinery() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({})).await;
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    assert_eq!(utterances(&events), 2, "{events:?}");
}

// Conversations (naru task 2178): sockets sharing a `conversation` key share
// one mode and `seq`; a `control` socket carries only the mode.

/// A control socket on `conversation`, and its first `state`.
async fn control(addr: &str, conversation: &str, mode: &str) -> (Ws, Value) {
    let mut ws = connect(addr).await;
    ws.send(text(json!({
        "type": "start", "role": "control", "conversation": conversation, "mode": mode,
    })))
    .await
    .unwrap();
    let ready = next_of(&mut ws, "ready").await;
    assert_eq!(ready["role"], "control", "{ready}");
    assert_eq!(ready["conversation"], conversation, "{ready}");
    let state = next_of(&mut ws, "state").await;
    (ws, state)
}

/// The next event, which must be a `state`; panics after 10 s.
async fn exactly_state(ws: &mut Ws) -> Value {
    let event = tokio::time::timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("no state within 10 s")
        .unwrap()
        .unwrap();
    let event: Value = serde_json::from_str(event.to_text().unwrap()).unwrap();
    assert_eq!(event["type"], "state", "{event}");
    event
}

/// Nothing arrives on `ws` for a while.
async fn quiet(ws: &mut Ws) {
    let got = tokio::time::timeout(Duration::from_millis(300), ws.next()).await;
    assert!(got.is_err(), "unexpected {got:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn control_sockets_share_one_conversation_state() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    let conv = "c1-shared";
    let (mut a, first) = control(&addr, conv, "live").await;
    assert_eq!(first["mode"], "live", "{first}");
    assert_eq!(first["seq"], 1, "{first}");
    // B's own mode is ignored: the conversation has one.
    let (mut b, joined) = control(&addr, conv, "idle").await;
    assert_eq!(joined["mode"], "live", "{joined}");
    assert_eq!(joined["seq"], 1, "{joined}");
    let mut seq = 1;
    for (from, mode) in [(0, "muted"), (1, "live"), (1, "muted")] {
        let sender = if from == 0 { &mut a } else { &mut b };
        sender
            .send(text(json!({"type": "mode", "mode": mode})))
            .await
            .unwrap();
        seq += 1;
        for ws in [&mut a, &mut b] {
            let state = exactly_state(ws).await;
            assert_eq!(state["mode"], mode, "{state}");
            assert_eq!(state["seq"], seq, "{state}");
        }
        quiet(&mut a).await;
        quiet(&mut b).await;
    }
    // The same mode again changes nothing.
    a.send(text(json!({"type": "mode", "mode": "muted"})))
        .await
        .unwrap();
    quiet(&mut a).await;
    quiet(&mut b).await;
    // A later joiner sees the truth first, whatever its own start says.
    let (mut c, current) = control(&addr, conv, "live").await;
    assert_eq!(current["mode"], "muted", "{current}");
    assert_eq!(current["seq"], seq, "{current}");
    for ws in [&mut a, &mut b, &mut c] {
        stop(ws).await;
    }
    // The last member gone, the conversation starts afresh.
    let (mut d, fresh) = control(&addr, conv, "idle").await;
    assert_eq!(fresh["mode"], "idle", "{fresh}");
    assert_eq!(fresh["seq"], 1, "{fresh}");
    stop(&mut d).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_conversation_starts_are_1008() {
    let home = fake_home();
    for first in [
        json!({"type": "start", "role": "control", "mode": "live"}),
        json!({"type": "start", "role": "control", "conversation": "", "mode": "live"}),
        json!({"type": "start", "conversation": "", "mode": "live", "rate": 16000}),
        json!({"type": "start", "role": "speaker", "conversation": "c", "mode": "live"}),
        json!({"type": "start", "role": "control", "conversation": "c-new"}),
        json!({"type": "start", "mode": "live"}),
    ] {
        let (events, close) = session_error(home.path(), text(first.clone())).await;
        assert_error_close(&events, close, "invalid_request");
        eprintln!("ok: {first}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_control_socket_refuses_audio_and_speech() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    for bad in [
        Message::binary(vec![0u8; 640]),
        text(json!({"type": "speak", "id": "t1", "text": "hi"})),
        text(json!({"type": "agent_speaking", "value": true})),
        text(json!({"type": "enroll", "action": "begin"})),
        text(json!({"type": "flush"})),
        text(json!({"type": "start", "role": "control", "conversation": "c-bad", "mode": "live"})),
    ] {
        let (mut ws, _) = control(&addr, "c-bad", "live").await;
        ws.send(bad).await.unwrap();
        let (events, close) = until_close(&mut ws).await;
        assert_error_close(&events, close, "invalid_request");
    }
    // A socket that failed left its conversation: this one begins it afresh.
    let (mut ws, fresh) = control(&addr, "c-bad", "idle").await;
    assert_eq!(fresh["mode"], "idle", "{fresh}");
    assert_eq!(fresh["seq"], 1, "{fresh}");
    stop(&mut ws).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_control_mute_shuts_the_audio_sockets_gate() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let conv = "c-audio";
    let (mut audio, first) = ready(&addr, json!({"conversation": conv})).await;
    assert_eq!(first["mode"], "live", "{first}");
    assert_eq!(first["seq"], 1, "{first}");
    let (mut ctl, joined) = control(&addr, conv, "idle").await;
    assert_eq!(joined["mode"], "live", "{joined}");
    assert_eq!(joined["seq"], 1, "{joined}");

    ctl.send(text(json!({"type": "mode", "mode": "muted"})))
        .await
        .unwrap();
    let on_ctl = exactly_state(&mut ctl).await;
    let on_audio = exactly_state(&mut audio).await;
    for state in [&on_ctl, &on_audio] {
        assert_eq!(state["mode"], "muted", "{state}");
        assert_eq!(state["seq"], 2, "{state}");
    }
    // The audio socket's gate is really shut, not just relabelled.
    let speech = samples("plain.wav");
    for frame in speech.chunks(640).take(15) {
        audio.send(Message::binary(s16le(frame))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let level = next_of(&mut audio, "level").await;
    assert!(level["rms"].as_f64().unwrap() > 0.0, "{level}");
    assert_eq!(level["gate"], false, "{level}");
    eprintln!("ran: muted level {level}");

    // Back to live from the audio socket, mirrored on the control socket.
    audio
        .send(text(json!({"type": "mode", "mode": "live"})))
        .await
        .unwrap();
    let back = next_of(&mut audio, "state").await;
    // The muted frames' `level` events reach the control socket first.
    let mirrored = next_of(&mut ctl, "state").await;
    for state in [&back, &mirrored] {
        assert_eq!(state["mode"], "live", "{state}");
        assert_eq!(state["seq"], 3, "{state}");
    }
    let events = stop(&mut audio).await;
    assert!(
        !events.iter().any(|e| e["type"] == "state"),
        "one state per change: {events:?}"
    );
    quiet(&mut ctl).await;
    stop(&mut ctl).await;
}

/// The `level` events among `events`, as `(rms, gate)`.
fn levels(events: &[Value]) -> Vec<(f64, bool)> {
    events
        .iter()
        .filter(|e| e["type"] == "level")
        .map(|e| (e["rms"].as_f64().unwrap(), e["gate"].as_bool().unwrap()))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn level_events_reach_every_socket_of_the_conversation() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let conv = "c-level";
    let (mut audio, _) = ready(&addr, json!({"conversation": conv})).await;
    let (mut ctl, _) = control(&addr, conv, "live").await;
    let speech = samples("plain.wav");
    for frame in speech.chunks(640).take(15) {
        audio.send(Message::binary(s16le(frame))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let on_audio = levels(&stop(&mut audio).await);
    // The audio socket published each before sending it, so all are queued.
    let mut seen = Vec::new();
    while let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_millis(300), ctl.next()).await
    {
        if let Message::Text(t) = msg {
            seen.push(serde_json::from_str::<Value>(&t).unwrap());
        }
    }
    let on_ctl = levels(&seen);
    let level_divergence = on_audio.len().max(on_ctl.len())
        - on_audio.iter().zip(&on_ctl).filter(|(a, c)| a == c).count();
    eprintln!("ran: level_divergence = {level_divergence}, levels = {on_audio:?}");
    assert!(!on_audio.is_empty(), "{on_audio:?}");
    assert_eq!(level_divergence, 0, "audio {on_audio:?} control {on_ctl:?}");
    stop(&mut ctl).await;

    // Without `conversation` a socket's levels stay its own.
    let (mut other, _) = control(&addr, "c-level-other", "live").await;
    let (mut solo, _) = ready(&addr, json!({})).await;
    for frame in speech.chunks(640).take(15) {
        solo.send(Message::binary(s16le(frame))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let level = next_of(&mut solo, "level").await;
    assert!(level["rms"].as_f64().unwrap() > 0.0, "{level}");
    stop(&mut solo).await;
    quiet(&mut other).await;
    stop(&mut other).await;
}

// Ambient mode's wake word (naru task 2208): `wake.wav` is macOS `say -v
// Samantha "Hey Naru. What time is it?"` at 16 kHz, which Parakeet hears as
// "Hey Neru, what time is it?" (a built-in spelling).

/// `wake.wav` followed by 1 s of silence.
fn wake_line() -> Vec<f32> {
    let mut pcm = samples("wake.wav");
    pcm.extend(vec![0.0f32; 16_000]);
    pcm
}

fn wake_events(events: &[Value]) -> usize {
    events.iter().filter(|e| e["type"] == "wake").count()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wake_word_in_ambient_emits_one_wake() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({"mode": "ambient"})).await;
    send_audio(&mut ws, &wake_line()).await;
    let events = stop(&mut ws).await;
    let wake_events = wake_events(&events);
    eprintln!("ran: wake_events = {wake_events}");
    assert_eq!(wake_events, 1, "{events:?}");
    let wake = events.iter().find(|e| e["type"] == "wake").unwrap();
    assert!(wake["at_ms"].is_u64(), "{wake}");
    assert!(wake["context"].is_array(), "{wake}");
    // The wake line is still flagged, and comes first.
    let flag = events
        .iter()
        .position(|e| e["type"] == "ambient_flag")
        .expect("an ambient_flag");
    let at = events.iter().position(|e| e["type"] == "wake").unwrap();
    assert!(flag < at, "{events:?}");
    assert_eq!(
        wake["text"],
        events[flag]["text"].as_str().unwrap().trim(),
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_non_wake_line_emits_no_wake() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({"mode": "ambient"})).await;
    send_audio(&mut ws, &two_plains()).await;
    let events = stop(&mut ws).await;
    let wake_events = wake_events(&events);
    eprintln!("ran: wake_events = {wake_events}");
    assert_eq!(wake_events, 0, "{events:?}");
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "ambient_flag")
            .count(),
        2,
        "{events:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn every_conversation_socket_gets_the_wake() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let conv = "c-wake";
    let (mut audio, first) = ready(&addr, json!({"conversation": conv, "mode": "ambient"})).await;
    assert_eq!(first["mode"], "ambient", "{first}");
    let (mut ctl, _) = control(&addr, conv, "ambient").await;
    send_audio(&mut audio, &wake_line()).await;
    let on_ctl = next_of(&mut ctl, "wake").await;
    let events = stop(&mut audio).await;
    assert_eq!(wake_events(&events), 1, "{events:?}");
    let on_audio = events.iter().find(|e| e["type"] == "wake").unwrap();
    assert_eq!(&on_ctl, on_audio);
    // Exactly one on the control socket too.
    let (rest, close) = {
        ctl.send(text(json!({"type": "stop"}))).await.unwrap();
        until_close(&mut ctl).await
    };
    assert_eq!(close, Some(1000), "{rest:?}");
    assert_eq!(wake_events(&rest), 0, "{rest:?}");
    eprintln!("ran: control wake {on_ctl}");
}

// ---- the server's own listener (naru task 2372) ---------------------------
// `POST /ambient` with the capture channel fed from wav fixtures, not a mic.

type Frames = Arc<std::sync::Mutex<Option<tokio::sync::mpsc::Sender<Vec<f32>>>>>;

/// A host whose "microphone" is a channel the test writes to.
fn injected() -> (AmbientHost, Frames) {
    let frames: Frames = Default::default();
    let host = AmbientHost::new({
        let frames = frames.clone();
        move || {
            let (tx, rx) = tokio::sync::mpsc::channel(4096);
            *frames.lock().unwrap() = Some(tx);
            Ok(Capture {
                mic: None,
                frames: rx,
            })
        }
    });
    (host, frames)
}

async fn inject(frames: &Frames, pcm: &[f32]) {
    let tx = frames.lock().unwrap().clone().expect("the listener is on");
    for frame in pcm.chunks(640) {
        tx.send(frame.to_vec()).await.unwrap();
    }
}

/// One HTTP/1.1 request; the status and the JSON body.
async fn http(addr: &str, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    let mut tcp = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tcp.write_all(req.as_bytes()).await.unwrap();
    let mut raw = String::new();
    tcp.read_to_string(&mut raw).await.unwrap();
    let status = raw[9..12].parse().unwrap();
    let json = raw
        .split_once("\r\n\r\n")
        .and_then(|(_, b)| serde_json::from_str(b).ok())
        .unwrap_or(Value::Null);
    (status, json)
}

/// `GET /ambient/log`'s events of `kind`.
async fn logged(addr: &str, kind: &str) -> Vec<Value> {
    let (status, log) = http(addr, "GET", "/ambient/log", None).await;
    assert_eq!(status, 200, "{log}");
    log["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["type"] == kind)
        .cloned()
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn ambient_needs_a_microphone_and_a_boolean() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    let (status, body) = http(&addr, "GET", "/ambient", None).await;
    assert_eq!(status, 503, "{body}");
    let (host, _) = injected();
    let addr = serve_with(home.path(), Some(host)).await;
    let (status, body) = http(&addr, "GET", "/ambient", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body,
        json!({"on": false, "conversation": null, "mic": null})
    );
    for bad in [
        json!({}),
        json!({"on": "yes"}),
        json!({"on": true, "conversation": ""}),
    ] {
        let (status, body) = http(&addr, "POST", "/ambient", Some(bad.clone())).await;
        assert_eq!(status, 400, "{bad}: {body}");
        assert_eq!(body["error"]["code"], "invalid_request", "{body}");
    }
    let (status, body) = http(&addr, "POST", "/ambient", Some(json!({"on": false}))).await;
    assert_eq!((status, body["on"].clone()), (200, json!(false)), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_listener_wakes_once_and_logs_lines() {
    let Some(home) = model_home() else { return };
    let (host, frames) = injected();
    let addr = serve_with(home.path(), Some(host)).await;
    let conv = "c-host";
    let (status, on) = http(
        &addr,
        "POST",
        "/ambient",
        Some(json!({"on": true, "conversation": conv})),
    )
    .await;
    assert_eq!(status, 200, "{on}");
    assert_eq!(
        (&on["on"], &on["conversation"]),
        (&json!(true), &json!(conv))
    );
    // A socket joining finds the listener's conversation in ambient.
    let (mut ctl, first) = control(&addr, conv, "live").await;
    assert_eq!(first["mode"], "ambient", "{first}");

    inject(&frames, &wake_line()).await;
    let wake = next_of(&mut ctl, "wake").await;
    inject(&frames, &two_plains()).await;
    // Both plain lines too (the wake line's flag came before its wake), on
    // the socket and in the log.
    let mut flags = 1;
    let mut wakes = 1;
    while flags < 3 {
        let e = next_where(&mut ctl, "a line", |e| {
            e["type"] == "ambient_flag" || e["type"] == "wake"
        })
        .await;
        if e["type"] == "wake" {
            wakes += 1;
        } else {
            flags += 1;
        }
    }
    assert_eq!(wakes, 1, "a second wake");
    let logged_wakes = logged(&addr, "wake").await;
    assert_eq!(logged_wakes, vec![wake.clone()]);
    assert_eq!(logged(&addr, "ambient_flag").await.len(), 3);
    eprintln!("ran: host wake {wake}");

    let (status, off) = http(&addr, "POST", "/ambient", Some(json!({"on": false}))).await;
    assert_eq!((status, off["on"].clone()), (200, json!(false)), "{off}");
    // The capture channel's receiver is gone: capture stopped.
    let tx = frames.lock().unwrap().clone().unwrap();
    tokio::time::timeout(Duration::from_secs(5), tx.closed())
        .await
        .expect("capture still open after off");
    let (_, status) = http(&addr, "GET", "/ambient", None).await;
    assert_eq!(status["on"], false, "{status}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_member_playing_a_reply_shuts_the_server_listener() {
    let Some(home) = model_home() else { return };
    let (host, frames) = injected();
    let addr = serve_with(home.path(), Some(host)).await;
    let conv = "c-host-echo";
    let (status, on) = http(
        &addr,
        "POST",
        "/ambient",
        Some(json!({"on": true, "conversation": conv})),
    )
    .await;
    assert_eq!(status, 200, "{on}");
    let (mut audio, _) = ready(&addr, json!({"conversation": conv})).await;
    audio
        .send(text(json!({"type": "agent_speaking", "value": true})))
        .await
        .unwrap();
    let state = next_of(&mut audio, "state").await;
    assert_eq!(state["agent_speaking"], true, "{state}");
    // Let the listener take the state before the frames.
    tokio::time::sleep(Duration::from_millis(300)).await;
    inject(&frames, &wake_line()).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(logged(&addr, "ambient_flag").await.is_empty());
    assert!(logged(&addr, "wake").await.is_empty());
    // Playback over (and the hold past): it hears again.
    audio
        .send(text(json!({"type": "agent_speaking", "value": false})))
        .await
        .unwrap();
    next_of(&mut audio, "state").await;
    tokio::time::sleep(Duration::from_millis(700)).await;
    inject(&frames, &wake_line()).await;
    let wake = next_of(&mut audio, "wake").await;
    eprintln!("ran: wake after playback {wake}");
    stop(&mut audio).await;
}

// ---- voice library (naru task 2260) --------------------------------------

fn voice(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/speaker")
        .join(name);
    audio::decode(std::fs::File::open(path).unwrap()).unwrap()
}

/// `clip` of speech then 1 s of silence, so the VAD closes the segment.
fn spoken(pcm: Vec<f32>) -> Vec<f32> {
    let mut out = pcm;
    out.extend(vec![0.0f32; 16_000]);
    out
}

/// Writes `people` (`(name, owner, clips)`) into `home`'s library, embedded
/// with the pulled speaker model. `None` (skip) when it is not pulled.
fn write_library(home: &Path, live_guard: bool, people: &[(&str, bool, &[&str])]) -> Option<()> {
    let dir = home.join("models").join(speaker::DEFAULT_MODEL);
    if !dir.exists() {
        eprintln!("skip: {} not pulled", speaker::DEFAULT_MODEL);
        return None;
    }
    let verifier = SpeakerVerifier::load(&dir).unwrap();
    let mut lib = Library::default();
    lib.live_guard = live_guard;
    for (name, owner, clips) in people {
        let id = lib.add_person(name, *owner);
        let person = lib.people.iter_mut().find(|p| p.id == id).unwrap();
        for (i, clip) in clips.iter().enumerate() {
            person.clips.push(Clip {
                id: id * 100 + i as u64,
                created_at: 0,
                seconds: 1.0,
                embedding: verifier.embed(&voice(clip)).unwrap(),
                has_audio: false,
            });
        }
    }
    lib.save(home).unwrap();
    Some(())
}

fn heard_utterances(events: &[Value]) -> Vec<&Value> {
    events.iter().filter(|e| e["type"] == "utterance").collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_guard_off_never_drops_another_voice() {
    let Some(home) = model_home() else { return };
    let Some(()) = write_library(home.path(), false, &[("Me", true, &["a1.wav", "a2.wav"])]) else {
        return;
    };
    let addr = serve(home.path()).await;
    let (mut ws, state) = ready(&addr, json!({})).await;
    assert_eq!(state["enrolled"], true, "{state}");
    assert_eq!(state["live_guard"], false, "{state}");
    send_audio(&mut ws, &spoken(voice("b1.wav"))).await;
    let events = stop(&mut ws).await;
    let heard = heard_utterances(&events);
    assert_eq!(heard.len(), 1, "{events:?}");
    // still scored for the tag: someone else
    assert_eq!(heard[0]["speaker"]["is_him"], false, "{}", heard[0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_guard_on_drops_another_voice_but_not_a_short_reply() {
    let Some(home) = model_home() else { return };
    let Some(()) = write_library(home.path(), true, &[("Me", true, &["a1.wav", "a2.wav"])]) else {
        return;
    };
    let addr = serve(home.path()).await;
    let (mut ws, state) = ready(&addr, json!({})).await;
    assert_eq!(state["live_guard"], true, "{state}");
    // the other voice for 5 s is dropped ...
    send_audio(&mut ws, &spoken(voice("b1.wav"))).await;
    // ... his own voice is admitted ...
    send_audio(&mut ws, &spoken(voice("a3.wav"))).await;
    // ... and a short reply in the other voice is never dropped
    let b = voice("b2.wav");
    send_audio(&mut ws, &spoken(b[..b.len().min(24_000)].to_vec())).await;
    let events = stop(&mut ws).await;
    let heard = heard_utterances(&events);
    let ends: Vec<f64> = heard
        .iter()
        .map(|u| u["end"].as_f64().unwrap() - u["start"].as_f64().unwrap())
        .collect();
    eprintln!("{} utterances, lengths {ends:?}", heard.len());
    assert_eq!(heard.len(), 2, "{events:?}");
    assert_eq!(heard[0]["speaker"]["is_him"], true, "{}", heard[0]);
    assert!(ends[1] < 2.0, "{ends:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_library_written_mid_session_is_picked_up() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let (mut ws, state) = ready(&addr, json!({})).await;
    assert_eq!(state["enrolled"], false, "{state}");
    send_audio(&mut ws, &spoken(voice("b1.wav"))).await;
    next_of(&mut ws, "utterance").await;
    let Some(()) = write_library(home.path(), true, &[("Me", true, &["a1.wav", "a2.wav"])]) else {
        return;
    };
    send_audio(&mut ws, &spoken(voice("b2.wav"))).await;
    send_audio(&mut ws, &spoken(voice("a3.wav"))).await;
    let events = stop(&mut ws).await;
    let heard = heard_utterances(&events);
    // b2 was dropped by the guard that arrived mid-session; only a3 is left
    assert_eq!(heard.len(), 1, "{events:?}");
    assert_eq!(heard[0]["speaker"]["person"]["name"], "Me", "{}", heard[0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn ambient_lines_name_the_person_whatever_the_guard() {
    let Some(home) = model_home() else { return };
    let Some(()) = write_library(
        home.path(),
        false,
        &[
            ("Me", true, &["a1.wav", "a2.wav"]),
            ("Sam", false, &["b1.wav", "b2.wav"]),
        ],
    ) else {
        return;
    };
    let addr = serve(home.path()).await;
    let (mut ws, _) = ready(&addr, json!({"mode": "ambient"})).await;
    send_audio(&mut ws, &spoken(voice("a3.wav"))).await;
    send_audio(&mut ws, &spoken(voice("b3.wav"))).await;
    let events = stop(&mut ws).await;
    let names: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "ambient_flag")
        .map(|e| &e["speaker"]["person"]["name"])
        .collect();
    assert_eq!(names, ["Me", "Sam"], "{events:?}");
}
