//! §2.4 `GET /v1/audio/transcriptions/stream` against the daemon on an
//! ephemeral port, and the `stream` CLI client.
//!
//! The handshake checks and the model errors need no model: `fake-stt` is
//! "pulled" by writing its `manifest.json` alone. Everything after `ready`
//! needs `parakeet-tdt-0.6b-v2-int8` and its Silero model, and skips as
//! `tests/stt.rs` does when they are not pulled.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use naru_audio::log::Logger;
use naru_audio::manager::{
    BackendLoader, KeepAlive, LoadError, LoadedModel, Loader, ModelManager, Resident, Settings,
};
use naru_audio::profile::Profile;
use naru_audio::registry::manifest::Manifest;
use naru_audio::registry::{self, Registry};
use naru_audio::server::{AppState, router};
use naru_audio::stt::{Segment, SttError, SttModel, VadConfig, Vocabulary, audio};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

const MODEL: &str = "parakeet-tdt-0.6b-v2-int8";

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

/// A fresh home whose `models/` links to the pulled models under
/// `$NARU_AUDIO_TEST_HOME`, else `$NARU_AUDIO_HOME` or `~/.naru-audio`, so
/// the daemon's `state/` never lands in the real one. `None` (skip) when
/// [`MODEL`] is not pulled there.
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
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("models")).unwrap();
    for entry in std::fs::read_dir(home.join("models")).unwrap() {
        let entry = entry.unwrap();
        std::os::unix::fs::symlink(
            entry.path(),
            dir.path().join("models").join(entry.file_name()),
        )
        .unwrap();
    }
    Some(dir)
}

/// The daemon on an ephemeral port, on the test's runtime; its address.
async fn serve(home: &Path) -> String {
    serve_with(home, Profile::detect().unwrap(), Arc::new(BackendLoader)).await
}

async fn serve_with(home: &Path, profile: Profile, loader: Arc<dyn Loader>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(Logger::stderr());
    let registry = Arc::new(Registry::open(home).unwrap());
    let mut settings = Settings::from_env(profile).unwrap();
    settings.stt_default = MODEL.to_string();
    settings.keep_alive = KeepAlive::For(Duration::from_secs(300));
    let models = ModelManager::new(registry.clone(), log.clone(), settings, loader);
    let app = router(Arc::new(AppState {
        port,
        allow_remote: false,
        started: Instant::now(),
        log,
        registry,
        models: Arc::new(models),
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
    }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("127.0.0.1:{port}")
}

async fn connect(addr: &str) -> Ws {
    let url = format!("ws://{addr}/v1/audio/transcriptions/stream");
    tokio_tungstenite::connect_async(url).await.unwrap().0
}

fn start(fields: Value) -> Message {
    let mut start = json!({"type": "start", "format": "s16le", "sample_rate": 16000});
    for (k, v) in fields.as_object().unwrap() {
        start[k] = v.clone();
    }
    Message::text(start.to_string())
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

/// The next event of `kind`, skipping others; panics after 60 s.
async fn next_of(ws: &mut Ws, kind: &str) -> Value {
    let read = async {
        loop {
            match ws.next().await {
                Some(Ok(Message::Text(t))) => {
                    let event: Value = serde_json::from_str(&t).unwrap();
                    assert_ne!(event["type"], "error", "{event}");
                    if event["type"] == kind {
                        return event;
                    }
                }
                Some(Ok(Message::Close(frame))) => panic!("closed waiting for {kind}: {frame:?}"),
                Some(Ok(_)) => {}
                other => panic!("socket ended waiting for {kind}: {other:?}"),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(60), read)
        .await
        .unwrap_or_else(|_| panic!("no {kind} within 60 s"))
}

/// The last event is an `error` with `code`, then the close is `close`.
fn assert_error_close(events: &[Value], close: Option<u16>, code: &str, expected: u16) {
    let last = events.last().expect("an error event");
    assert_eq!(last["type"], "error", "{events:?}");
    assert_eq!(last["code"], code, "{last}");
    assert!(last["message"].as_str().is_some_and(|m| !m.is_empty()));
    assert_eq!(close, Some(expected), "{events:?}");
}

async fn session_error(home: &Path, first: Message) -> (Vec<Value>, Option<u16>) {
    let addr = serve(home).await;
    let mut ws = connect(&addr).await;
    ws.send(first).await.unwrap();
    until_close(&mut ws).await
}

#[tokio::test]
async fn binary_before_start_is_1008() {
    let home = fake_home();
    let (events, close) = session_error(home.path(), Message::binary(vec![0u8; 640])).await;
    assert_error_close(&events, close, "invalid_request", 1008);
}

#[tokio::test]
async fn a_first_message_other_than_start_is_1008() {
    let home = fake_home();
    for first in [
        r#"{"type":"flush"}"#,
        r#"{"type":"stop"}"#,
        r#"{"type":"nope"}"#,
        r#"{"model":"fake-stt"}"#,
        "not json",
    ] {
        let (events, close) = session_error(home.path(), Message::text(first)).await;
        assert_error_close(&events, close, "invalid_request", 1008);
    }
}

#[tokio::test]
async fn bad_start_fields_are_1008() {
    let home = fake_home();
    for (fields, code) in [
        (json!({"sample_rate": 44100}), "unsupported_value"),
        (json!({"format": "wav"}), "unsupported_value"),
        (json!({"sample_rate": null}), "invalid_request"),
        (json!({"vad": {"threshold": 1.5}}), "invalid_request"),
        (json!({"vad": {"min_speech": -1}}), "invalid_request"),
        (json!({"vad": {"min_silence": 0}}), "invalid_request"),
        (json!({"vad": {"max_segment": 0.5}}), "invalid_request"),
        (json!({"vad": {"max_segment": 31}}), "invalid_request"),
        (json!({"vad": {"pad": 2}}), "invalid_request"),
        // 28.6 s padded: more than the backlog holds with its headroom.
        (
            json!({"vad": {"max_segment": 28, "pad": 0.3}}),
            "invalid_request",
        ),
        (json!({"hotwords": "naru:3.0"}), "invalid_request"),
        (json!({"keep_alive": "soon"}), "invalid_request"),
    ] {
        let mut fields = fields;
        fields["model"] = json!("fake-stt");
        let (events, close) = session_error(home.path(), start(fields.clone())).await;
        assert_error_close(&events, close, code, 1008);
        assert_eq!(events.len(), 1, "{fields}: {events:?}");
    }
}

#[tokio::test]
async fn model_errors_use_the_http_codes() {
    let home = fake_home();
    let (events, close) = session_error(home.path(), start(json!({"model": "nope"}))).await;
    assert_error_close(&events, close, "model_not_found", 1008);

    let (events, close) = session_error(home.path(), start(json!({"model": "fake-vad"}))).await;
    assert_error_close(&events, close, "invalid_request", 1008);

    // `default` is the catalog's parakeet, not pulled here.
    let (events, close) = session_error(home.path(), start(json!({"model": "default"}))).await;
    assert_error_close(&events, close, "model_not_pulled", 1008);
}

/// `fake-stt` has no model files: `loading`, then the load fails. A
/// padded `max_segment` of exactly 28 s passes validation.
#[tokio::test]
async fn a_failed_load_is_loading_then_1011() {
    let home = fake_home();
    for vad in [json!({}), json!({"max_segment": 27.4, "pad": 0.3})] {
        let (events, close) =
            session_error(home.path(), start(json!({"model": "fake-stt", "vad": vad}))).await;
        assert_eq!(events[0]["type"], "loading", "{events:?}");
        assert_error_close(&events, close, "model_load_failed", 1011);
    }
}

/// A plain GET gets the §2.6 envelope.
#[tokio::test]
async fn a_request_without_upgrade_is_400() {
    let home = fake_home();
    let addr = serve(home.path()).await;
    let (status, body) = tokio::task::spawn_blocking(move || {
        let mut resp = ureq::get(format!("http://{addr}/v1/audio/transcriptions/stream"))
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .unwrap();
        let status = resp.status().as_u16();
        let body = resp.body_mut().read_to_string().unwrap();
        (status, serde_json::from_str::<Value>(&body).unwrap())
    })
    .await
    .unwrap();
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request", "{body}");
}

fn samples(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(name);
    audio::decode(std::fs::File::open(path).unwrap()).unwrap()
}

/// Four copies of `plain.wav` separated, and followed, by 1 s of silence.
fn four_plains() -> Vec<f32> {
    let speech = samples("plain.wav");
    let gap = vec![0.0f32; 16_000];
    let mut pcm = Vec::new();
    for _ in 0..4 {
        pcm.extend_from_slice(&speech);
        pcm.extend_from_slice(&gap);
    }
    pcm
}

fn s16le(pcm: &[f32]) -> Vec<u8> {
    pcm.iter()
        .flat_map(|s| ((s * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
        .collect()
}

/// `pcm` as a 16-bit WAV.
fn wav(pcm: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for s in s16le(pcm).as_chunks::<2>().0 {
        writer.write_sample(i16::from_le_bytes(*s)).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

/// Sends `pcm` in 40 ms frames, as fast as the socket takes them.
async fn send_audio(ws: &mut Ws, pcm: &[f32]) {
    for frame in pcm.chunks(640) {
        ws.send(Message::binary(s16le(frame))).await.unwrap();
    }
}

/// Folds case, strips punctuation, collapses whitespace (auris `normalise`).
fn normalise(s: &str) -> String {
    let folded: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

async fn ready(addr: &str, fields: Value) -> Ws {
    let mut ws = connect(addr).await;
    ws.send(start(fields)).await.unwrap();
    let ready = next_of(&mut ws, "ready").await;
    assert_eq!(ready["model"], MODEL, "{ready}");
    assert!(ready["session"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(ready["load_ms"].is_u64(), "{ready}");
    ws
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_without_audio_is_done_then_1000() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let mut ws = ready(&addr, json!({})).await;
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_eq!(events, [json!({"type": "done"})]);
    assert_eq!(close, Some(1000));
}

#[tokio::test(flavor = "multi_thread")]
async fn violations_after_start_are_1008() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    for bad in [
        Message::binary(vec![0u8; 641]),
        Message::binary(vec![0u8; 64 * 1024 + 2]),
        // Past the socket's 1 MiB message cap: refused before it is read.
        Message::binary(vec![0u8; 2 * 1024 * 1024]),
        Message::text(r#"{"type":"partial"}"#),
        start(json!({})),
        Message::text(r#"{"type":"hotwords","hotwords":"naru:3.0"}"#),
    ] {
        let mut ws = ready(&addr, json!({})).await;
        ws.send(bad).await.unwrap();
        let (events, close) = until_close(&mut ws).await;
        assert_error_close(&events, close, "invalid_request", 1008);
    }
}

/// A client still sending when the session fails gets the `error` and the
/// close all the same: frames it left unread would otherwise turn the close
/// into a TCP reset that takes both with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_violation_mid_burst_still_gets_error_then_1008() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let mut ws = ready(&addr, json!({})).await;
    ws.send(Message::binary(vec![0u8; 641])).await.unwrap();
    for _ in 0..2000 {
        // The server may already have closed: later sends can fail.
        if ws.send(Message::binary(vec![0u8; 1280])).await.is_err() {
            break;
        }
    }
    let (events, close) = until_close(&mut ws).await;
    assert_error_close(&events, close, "invalid_request", 1008);
}

/// The finals, in order, say what the batch endpoint says for the same
/// audio; the speech events bracket each one.
#[tokio::test(flavor = "multi_thread")]
async fn finals_match_the_batch_transcript() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let pcm = four_plains();

    let mut ws = ready(&addr, json!({})).await;
    send_audio(&mut ws, &pcm).await;
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_eq!(close, Some(1000), "{events:?}");
    assert_eq!(events.last().unwrap()["type"], "done");
    let finals: Vec<&Value> = events.iter().filter(|e| e["type"] == "final").collect();
    eprintln!("finals: {finals:?}");
    assert_eq!(finals.len(), 4, "{events:?}");
    for (i, f) in finals.iter().enumerate() {
        assert_eq!(f["segment"], i);
        assert!(f["start"].as_f64() < f["end"].as_f64(), "{f}");
        assert!(f["decode_ms"].is_u64(), "{f}");
    }
    let speech: Vec<bool> = events
        .iter()
        .filter(|e| e["type"] == "speech")
        .map(|e| e["active"].as_bool().unwrap())
        .collect();
    assert_eq!(speech, [true, false].repeat(4));

    const BOUNDARY: &str = "naru-audio-test-boundary";
    let mut form = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"
    )
    .into_bytes();
    form.extend_from_slice(&wav(&pcm));
    form.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let batch = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://{addr}/v1/audio/transcriptions"))
            .content_type(format!("multipart/form-data; boundary={BOUNDARY}"))
            .send(&form[..])
            .unwrap()
            .body_mut()
            .read_to_string()
            .unwrap()
    })
    .await
    .unwrap();
    let batch: Value = serde_json::from_str(&batch).unwrap();
    let streamed: Vec<&str> = finals.iter().map(|f| f["text"].as_str().unwrap()).collect();
    assert_eq!(
        normalise(&streamed.join(" ")),
        normalise(batch["text"].as_str().unwrap())
    );
}

/// `flush` closes the segment with no trailing silence at all.
#[tokio::test(flavor = "multi_thread")]
async fn flush_closes_the_segment_now() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let mut ws = ready(&addr, json!({})).await;
    send_audio(&mut ws, &samples("plain.wav")).await;
    ws.send(Message::text(r#"{"type":"flush"}"#)).await.unwrap();
    let last = next_of(&mut ws, "final").await;
    assert_eq!(
        normalise(last["text"].as_str().unwrap()),
        "open the daily notes and add a line about the meeting"
    );
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert!(!events.iter().any(|e| e["type"] == "final"), "{events:?}");
    // Silero's span was still open at `stop`: `stop` ends it.
    let n = events.len();
    assert!(n >= 2, "{events:?}");
    assert_eq!(events[n - 2]["type"], "speech", "{events:?}");
    assert_eq!(events[n - 2]["active"], false, "{events:?}");
    assert_eq!(events[n - 1]["type"], "done");
    assert_eq!(close, Some(1000));
}

/// Unbroken speech longer than `max_segment` is cut into back-to-back
/// segments no longer than it.
#[tokio::test(flavor = "multi_thread")]
async fn max_segment_cuts_long_speech() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let mut ws = ready(&addr, json!({"vad": {"max_segment": 1.0}})).await;
    send_audio(&mut ws, &samples("plain.wav")).await;
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_eq!(close, Some(1000), "{events:?}");
    let finals: Vec<(f64, f64)> = events
        .iter()
        .filter(|e| e["type"] == "final")
        .map(|e| (e["start"].as_f64().unwrap(), e["end"].as_f64().unwrap()))
        .collect();
    eprintln!("cut finals: {events:?}");
    assert!(finals.len() >= 2, "{events:?}");
    for (start, end) in &finals {
        assert!(end - start <= 1.0 + 1e-9, "{finals:?}");
    }
}

/// The real model, counting its decodes.
struct CountingModel {
    inner: Arc<dyn SttModel>,
    decodes: Arc<AtomicUsize>,
}

impl SttModel for CountingModel {
    fn decode_each(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        self.decodes.fetch_add(1, Ordering::SeqCst);
        self.inner.decode_each(pcm16k, hotwords, vad, on_segment)
    }

    fn vad_model(&self) -> Option<&Path> {
        self.inner.vad_model()
    }
}

struct CountingLoader {
    decodes: Arc<AtomicUsize>,
}

impl Loader for CountingLoader {
    fn load(&self, manifest: &Manifest, dir: &Path) -> Result<LoadedModel, LoadError> {
        let loaded = BackendLoader.load(manifest, dir)?;
        let Resident::Stt(inner) = loaded.model else {
            unreachable!("an STT manifest loads an STT model")
        };
        Ok(LoadedModel {
            model: Resident::Stt(Arc::new(CountingModel {
                inner,
                decodes: self.decodes.clone(),
            })),
            measured_bytes: loaded.measured_bytes,
        })
    }
}

/// The daemon with a [`CountingModel`]; its address and decode count.
async fn serve_counting(home: &Path) -> (String, Arc<AtomicUsize>) {
    let decodes = Arc::new(AtomicUsize::new(0));
    let loader = Arc::new(CountingLoader {
        decodes: decodes.clone(),
    });
    let addr = serve_with(home, Profile::detect().unwrap(), loader).await;
    (addr, decodes)
}

/// Without `partials` the model decodes each segment once, and no more.
#[tokio::test(flavor = "multi_thread")]
async fn partials_off_run_no_extra_decode() {
    let Some(home) = model_home() else { return };
    let (addr, decodes) = serve_counting(home.path()).await;
    let mut ws = ready(&addr, json!({})).await;
    send_audio(&mut ws, &four_plains()).await;
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_eq!(close, Some(1000), "{events:?}");
    assert!(!events.iter().any(|e| e["type"] == "partial"), "{events:?}");
    let finals = events.iter().filter(|e| e["type"] == "final").count();
    assert_eq!(finals, 4, "{events:?}");
    assert_eq!(decodes.load(Ordering::SeqCst), finals);
}

/// Paced at real time, `partials` gives each segment at least one partial,
/// every one of them before that segment's final.
#[tokio::test(flavor = "multi_thread")]
async fn partials_come_before_their_final() {
    let Some(home) = model_home() else { return };
    let (addr, decodes) = serve_counting(home.path()).await;
    let mut ws = ready(&addr, json!({"partials": true})).await;
    let mut pcm = samples("plain.wav");
    pcm.extend(vec![0.0f32; 16_000]);
    pcm.extend(samples("plain.wav"));
    pcm.extend(vec![0.0f32; 16_000]);
    for frame in pcm.chunks(640) {
        ws.send(Message::binary(s16le(frame))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
    let (events, close) = until_close(&mut ws).await;
    assert_eq!(close, Some(1000), "{events:?}");
    eprintln!("partials: {events:?}");
    let finals: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| e["type"] == "final")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(finals.len(), 2, "{events:?}");
    let mut partials = 0;
    for (segment, &at) in finals.iter().enumerate() {
        let before = events[..at]
            .iter()
            .filter(|e| e["type"] == "partial" && e["segment"] == segment)
            .count();
        assert!(before >= 1, "no partial for segment {segment}: {events:?}");
        partials += before;
        assert!(
            !events[at..]
                .iter()
                .any(|e| e["type"] == "partial" && e["segment"] == segment),
            "a partial after final {segment}: {events:?}"
        );
    }
    for e in events.iter().filter(|e| e["type"] == "partial") {
        assert!(
            e["start"].is_f64() && e["text"].as_str().is_some_and(|t| !t.is_empty()),
            "{e}"
        );
    }
    assert!(decodes.load(Ordering::SeqCst) >= finals.len() + partials);
}

/// `partials` on an x86_64 build gets a `warning` right after `ready`; on
/// arm64, or without `partials`, nothing.
#[tokio::test(flavor = "multi_thread")]
async fn partials_on_x86_64_warn() {
    let Some(home) = model_home() else { return };
    let ram = 64 * 1024 * 1024 * 1024;
    for (arch, partials, warned) in [
        ("x86_64", true, true),
        ("x86_64", false, false),
        ("aarch64", true, false),
    ] {
        let profile = Profile::new(ram, arch, false);
        let addr = serve_with(home.path(), profile, Arc::new(BackendLoader)).await;
        let mut ws = ready(&addr, json!({"partials": partials})).await;
        ws.send(Message::text(r#"{"type":"stop"}"#)).await.unwrap();
        let (events, close) = until_close(&mut ws).await;
        assert_eq!(close, Some(1000), "{events:?}");
        assert_eq!(events.last().unwrap()["type"], "done");
        if warned {
            assert_eq!(events.len(), 2, "{arch}: {events:?}");
            assert_eq!(events[0]["type"], "warning", "{events:?}");
            assert_eq!(events[0]["code"], "partials_expensive", "{events:?}");
            assert!(events[0]["message"].as_str().is_some_and(|m| !m.is_empty()));
        } else {
            assert_eq!(events.len(), 1, "{arch} partials={partials}: {events:?}");
        }
    }
}

/// The CLI client paces the file, prints the events and a latency per
/// final, and exits 0 on `done`.
#[tokio::test(flavor = "multi_thread")]
async fn the_cli_client_streams_a_file() {
    let Some(home) = model_home() else { return };
    let addr = serve(home.path()).await;
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("four.wav");
    std::fs::write(&file, wav(&four_plains())).unwrap();
    let out = tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_naru-audio"))
            .args(["stream", file.to_str().unwrap(), "--speed", "4"])
            .env("NARU_AUDIO_URL", format!("http://{addr}"))
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let events: Vec<Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let finals: Vec<&Value> = events.iter().filter(|e| e["type"] == "final").collect();
    assert_eq!(finals.len(), 4, "{stdout}");
    assert!(finals.iter().all(|f| f["latency_ms"].is_u64()), "{stdout}");
    assert_eq!(events.last().unwrap()["type"], "done");
}

/// As `transcribe`: exit 3 with the daemon down.
#[test]
fn the_cli_client_with_the_daemon_down_exits_3() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let out = Command::new(env!("CARGO_BIN_EXE_naru-audio"))
        .args(["stream", "/nonexistent.wav"])
        .env("NARU_AUDIO_URL", &url)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
}
