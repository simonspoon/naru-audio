//! §2.3 `POST /v1/audio/speech`, §2.5 `GET /v1/audio/voices` and the §3.7
//! `say` client, over real HTTP/1.1 on an ephemeral port. A fake TTS model
//! gives fixed samples; one test runs the real Kokoro and skips (eprintln
//! and return) when `kokoro-v1.0` is not pulled, as tests/tts.rs does.

use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use naru_audio::log::Logger;
use naru_audio::manager::{
    BackendLoader, KeepAlive, LoadError, LoadedModel, Loader, ModelManager, Resident, Settings,
};
use naru_audio::profile::Profile;
use naru_audio::registry::manifest::{Manifest, Voice};
use naru_audio::registry::{self, Registry};
use naru_audio::server::{AppState, router};
use naru_audio::tts::{Sink, SynthOptions, TtsError, TtsModel};
use serde_json::{Value, json};

/// Samples per sentence the fake gives, and their value.
const PIECE: usize = 1200;
const LEVEL: f32 = 0.25;
/// `LEVEL` as s16le: round(0.25 * 32767) = 8192.
const LEVEL_S16: [u8; 2] = 8192i16.to_le_bytes();

/// What the fake saw and did, shared with the test.
#[derive(Default)]
struct Record {
    loads: AtomicUsize,
    /// Pieces handed to the sink by `endless`.
    pieces: AtomicUsize,
    /// `endless` returned.
    finished: AtomicBool,
    /// The last request's voice and options.
    last: Mutex<Option<(String, SynthOptions)>>,
}

/// A piece of `LEVEL` per `.`-separated sentence, at 24 kHz. `endless`
/// sends pieces until the sink cancels; `fail` fails after one piece and
/// `fail first` before any.
struct FakeTts {
    voices: Vec<Voice>,
    record: Arc<Record>,
}

impl TtsModel for FakeTts {
    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn sample_rate(&self) -> u32 {
        24_000
    }

    fn synth(
        &self,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        mut sink: Sink,
    ) -> Result<(), TtsError> {
        *self.record.last.lock().unwrap() = Some((voice.to_string(), options.clone()));
        let piece = [LEVEL; PIECE];
        match text {
            "endless" => {
                loop {
                    std::thread::sleep(Duration::from_millis(10));
                    self.record.pieces.fetch_add(1, Ordering::SeqCst);
                    if !sink(&piece) {
                        break;
                    }
                }
                self.record.finished.store(true, Ordering::SeqCst);
                Ok(())
            }
            "fail" => {
                sink(&piece);
                Err(TtsError::GenerateFailed)
            }
            "fail first" => Err(TtsError::GenerateFailed),
            _ => {
                for _ in text.split('.').filter(|s| !s.trim().is_empty()) {
                    if !sink(&piece) {
                        break;
                    }
                }
                Ok(())
            }
        }
    }
}

struct FakeLoader(Arc<Record>);

impl Loader for FakeLoader {
    fn load(&self, manifest: &Manifest, _dir: &Path) -> Result<LoadedModel, LoadError> {
        self.0.loads.fetch_add(1, Ordering::SeqCst);
        Ok(LoadedModel {
            model: Resident::Tts(Arc::new(FakeTts {
                voices: manifest.voices.clone(),
                record: self.0.clone(),
            })),
            measured_bytes: None,
        })
    }
}

/// A home with `fake-tts` (two voices, `af_heart` the default) and
/// `fake-stt` "pulled" (a `manifest.json` alone).
fn fake_home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for manifest in [
        json!({"model": {"name": "fake-tts", "kind": "tts", "backend": "sherpa-onnx"},
        "voice": [
            {"id": "af_heart", "sid": 3, "accent": "us", "gender": "f", "default": true},
            {"id": "bm_george", "sid": 26, "accent": "gb", "gender": "m"},
        ]}),
        json!({"model": {"name": "fake-stt", "kind": "stt", "backend": "sherpa-onnx"}}),
    ] {
        let model_dir = dir
            .path()
            .join("models")
            .join(manifest["model"]["name"].as_str().unwrap());
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();
    }
    dir
}

/// The daemon on an ephemeral port, on its own runtime.
struct Server {
    port: u16,
    record: Arc<Record>,
    _home: tempfile::TempDir,
    rt: Option<tokio::runtime::Runtime>,
}

impl Server {
    fn fake() -> Self {
        let record = Arc::new(Record::default());
        Self::start(
            fake_home(),
            "fake-tts",
            Arc::new(FakeLoader(record.clone())),
            record,
        )
    }

    fn start(
        home: tempfile::TempDir,
        tts_default: &str,
        loader: Arc<dyn Loader>,
        record: Arc<Record>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Logger::stderr());
        let registry = Arc::new(Registry::open(home.path()).unwrap());
        let mut settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
        settings.tts_default = tts_default.to_string();
        settings.keep_alive = KeepAlive::For(Duration::from_secs(300));
        let models = ModelManager::new(registry.clone(), log.clone(), settings, loader);
        let app = router(Arc::new(AppState {
            port,
            allow_remote: false,
            started: Instant::now(),
            log,
            registry,
            models: Arc::new(models),
        }));
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
        Server {
            port,
            record,
            _home: home,
            rt: Some(rt),
        }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Writes a request and returns the open connection.
    fn send(&self, method: &str, path: &str, body: &str) -> TcpStream {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        write!(
            s,
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            self.port,
            body.len()
        )
        .unwrap();
        s
    }

    fn request(&self, method: &str, path: &str, body: &str) -> Reply {
        let mut s = self.send(method, path, body);
        let mut raw = Vec::new();
        let mut buf = [0u8; 16 * 1024];
        loop {
            match s.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => raw.extend_from_slice(&buf[..n]),
                // An aborted body may end in a reset.
                Err(e) if e.kind() == ErrorKind::ConnectionReset => break,
                Err(e) => panic!("{e}"),
            }
        }
        Reply::parse(&raw)
    }

    fn speech(&self, request: Value) -> Reply {
        self.request("POST", "/v1/audio/speech", &request.to_string())
    }

    fn ps(&self) -> Vec<Value> {
        let reply = self.request("GET", "/api/ps", "");
        serde_json::from_slice(&reply.body).unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}

struct Reply {
    status: u16,
    /// Lower-cased names.
    headers: HashMap<String, String>,
    /// De-chunked when chunked.
    body: Vec<u8>,
    /// Chunked and ended by the zero-length chunk.
    terminated: bool,
}

impl Reply {
    fn parse(raw: &[u8]) -> Self {
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("a response head");
        let head = String::from_utf8(raw[..split].to_vec()).unwrap();
        let mut lines = head.split("\r\n");
        let status = lines.next().unwrap().split_whitespace().nth(1).unwrap();
        let headers: HashMap<String, String> = lines
            .map(|l| l.split_once(':').unwrap())
            .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        let mut rest = &raw[split + 4..];
        let (mut body, mut terminated) = (Vec::new(), false);
        if headers.get("transfer-encoding").map(String::as_str) == Some("chunked") {
            while let Some(eol) = rest.windows(2).position(|w| w == b"\r\n") {
                let size = std::str::from_utf8(&rest[..eol]).unwrap();
                let size = usize::from_str_radix(size, 16).unwrap();
                if size == 0 {
                    terminated = true;
                    break;
                }
                let start = eol + 2;
                if rest.len() < start + size + 2 {
                    body.extend_from_slice(&rest[start.min(rest.len())..]);
                    break;
                }
                body.extend_from_slice(&rest[start..start + size]);
                rest = &rest[start + size + 2..];
            }
        } else {
            body = rest.to_vec();
        }
        Reply {
            status: status.parse().unwrap(),
            headers,
            body,
            terminated,
        }
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// `n` fake pieces as s16le.
fn pcm(n: usize) -> Vec<u8> {
    LEVEL_S16.repeat(PIECE * n)
}

/// Acceptance: the streamed header's RIFF size is `0x7FFF0024` and its data
/// size `0x7FFF0000`, and the body is chunked with no `Content-Length`.
#[test]
fn streamed_wav_carries_the_0x7fff_sizes() {
    let server = Server::fake();
    let reply =
        server.speech(json!({"model": "fake-tts", "input": "One. Two.", "voice": "af_heart"}));
    assert_eq!(
        reply.status,
        200,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    assert_eq!(reply.header("content-type"), Some("audio/wav"));
    assert_eq!(reply.header("transfer-encoding"), Some("chunked"));
    assert_eq!(reply.header("content-length"), None);
    assert!(reply.terminated);

    let wav = &reply.body;
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[4..8], &[0x24, 0x00, 0xFF, 0x7F]);
    assert_eq!(u32_at(wav, 4), 0x7FFF_0024);
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    assert_eq!(u32_at(wav, 16), 16);
    // PCM, mono, 24 kHz, 48 000 bytes/s, 2-byte frames, 16 bits.
    assert_eq!(&wav[20..24], &[1, 0, 1, 0]);
    assert_eq!(u32_at(wav, 24), 24_000);
    assert_eq!(u32_at(wav, 28), 48_000);
    assert_eq!(&wav[32..36], &[2, 0, 16, 0]);
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(&wav[40..44], &[0x00, 0x00, 0xFF, 0x7F]);
    assert_eq!(u32_at(wav, 40), 0x7FFF_0000);
    assert_eq!(wav[44..], pcm(2));
}

#[test]
fn pcm_is_headerless_s16le_with_its_format_headers() {
    let server = Server::fake();
    let reply = server.speech(json!({"input": "One. Two. Three.", "response_format": "pcm"}));
    assert_eq!(
        reply.status,
        200,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    assert_eq!(reply.header("transfer-encoding"), Some("chunked"));
    assert_eq!(reply.header("x-audio-sample-rate"), Some("24000"));
    assert_eq!(reply.header("x-audio-channels"), Some("1"));
    assert_eq!(reply.header("x-audio-encoding"), Some("s16le"));
    assert!(reply.terminated);
    assert_eq!(reply.body, pcm(3));
}

#[test]
fn stream_false_has_exact_sizes_and_a_content_length() {
    let server = Server::fake();
    let data = (2 * PIECE * 2) as u32;
    let reply = server.speech(json!({"input": "One. Two.", "stream": false}));
    assert_eq!(
        reply.status,
        200,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    assert_eq!(reply.header("transfer-encoding"), None);
    assert_eq!(
        reply.header("content-length"),
        Some((44 + data).to_string().as_str())
    );
    assert_eq!(u32_at(&reply.body, 4), data + 36);
    assert_eq!(u32_at(&reply.body, 40), data);
    assert_eq!(reply.body[44..], pcm(2));

    let reply =
        server.speech(json!({"input": "One. Two.", "stream": false, "response_format": "pcm"}));
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.header("content-length"),
        Some(data.to_string().as_str())
    );
    assert_eq!(reply.body, pcm(2));
}

/// Every refusal answers before a load, with the §2.6 code and `param`.
#[test]
fn bad_requests_are_refused_before_a_load() {
    let server = Server::fake();
    let long = "a".repeat(16_385);
    for (request, status, code, param) in [
        (json!({"input": ""}), 400, "invalid_request", "input"),
        (
            json!({"voice": "af_heart"}),
            400,
            "invalid_request",
            "input",
        ),
        (json!({"input": 5}), 400, "invalid_request", "input"),
        (json!({"input": long}), 413, "payload_too_large", "input"),
        (
            json!({"input": "Hi.", "voice": "alloy"}),
            400,
            "unknown_voice",
            "voice",
        ),
        (
            json!({"input": "Hi.", "response_format": "mp3"}),
            400,
            "unsupported_value",
            "response_format",
        ),
        (
            json!({"input": "Hi.", "speed": 2.5}),
            400,
            "invalid_request",
            "speed",
        ),
        (
            json!({"input": "Hi.", "speed": 0.4}),
            400,
            "invalid_request",
            "speed",
        ),
        (
            json!({"input": "Hi.", "stream_format": "sse"}),
            400,
            "unsupported_value",
            "stream_format",
        ),
        (
            json!({"input": "Hi.", "gap": 6}),
            400,
            "invalid_request",
            "gap",
        ),
        (
            json!({"input": "Hi.", "level": "yes"}),
            400,
            "invalid_request",
            "level",
        ),
        (
            json!({"input": "Hi.", "stream": "no"}),
            400,
            "invalid_request",
            "stream",
        ),
        (
            json!({"input": "Hi.", "model": "nope"}),
            404,
            "model_not_found",
            "model",
        ),
        (
            json!({"input": "Hi.", "model": "fake-stt"}),
            400,
            "invalid_request",
            "model",
        ),
        (
            json!({"input": "Hi.", "model": "kokoro-v1.0"}),
            409,
            "model_not_pulled",
            "model",
        ),
    ] {
        let reply = server.speech(request.clone());
        assert_eq!(reply.status, status, "{request}");
        let body = reply.json();
        assert_eq!(body["error"]["code"], code, "{request}: {body}");
        assert_eq!(body["error"]["param"], param, "{request}");
    }
    let reply = server.speech(json!({"input": "Hi.", "voice": "alloy"}));
    let message = reply.json()["error"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(message.contains("af_heart, bm_george"), "{message}");
    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);
    assert_eq!(server.ps(), Vec::<Value>::new());
}

/// OpenAI's model names stand for the default model; no voice is the
/// manifest's default; the knobs reach the model.
#[test]
fn aliases_defaults_and_options_reach_the_model() {
    let server = Server::fake();
    let last = || server.record.last.lock().unwrap().clone().unwrap();
    for model in ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "default"] {
        let reply =
            server.speech(json!({"model": model, "input": "Hi.", "instructions": "cheerful"}));
        assert_eq!(reply.status, 200, "{model}");
        assert_eq!(last(), ("af_heart".to_string(), SynthOptions::default()));
    }
    let long = "a".repeat(16_384);
    assert_eq!(server.speech(json!({"input": long})).status, 200);

    let reply = server.speech(json!({
        "input": "Hi.", "voice": "bm_george", "speed": 1.5, "gap": 0, "level": false,
    }));
    assert_eq!(reply.status, 200);
    let expected = SynthOptions {
        speed: 1.5,
        gap: 0.0,
        level: false,
    };
    assert_eq!(last(), ("bm_george".to_string(), expected));
}

/// §2.3: an error after the first byte cannot change the 200; the body
/// ends without its zero-length chunk.
#[test]
fn an_error_mid_stream_aborts_the_chunked_body() {
    let server = Server::fake();
    let reply = server.speech(json!({"input": "fail"}));
    assert_eq!(reply.status, 200);
    assert!(!reply.terminated, "a failed stream must not end cleanly");
    assert_eq!(reply.body[44..], pcm(1));
}

/// Before any audio, a failure is still a status code.
#[test]
fn an_error_before_any_audio_is_a_500() {
    let server = Server::fake();
    for stream in [true, false] {
        let reply = server.speech(json!({"input": "fail first", "stream": stream}));
        assert_eq!(reply.status, 500, "stream={stream}");
        assert_eq!(reply.json()["error"]["code"], "internal");
    }
}

/// A client that hangs up mid-stream stops the synthesis, and the model
/// is released.
#[test]
fn a_client_abort_stops_synthesis() {
    let server = Server::fake();
    let mut s = server.send("POST", "/v1/audio/speech", r#"{"input":"endless"}"#);
    let mut buf = [0u8; 4096];
    let mut got = 0;
    while got < 44 + 2 * PIECE {
        got += s.read(&mut buf).unwrap();
    }
    drop(s);

    let deadline = Instant::now() + Duration::from_secs(5);
    while !server.record.finished.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < deadline,
            "synthesis went on after the client left"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let pieces = server.record.pieces.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(server.record.pieces.load(Ordering::SeqCst), pieces);
    let ps = server.ps();
    assert_eq!(ps.len(), 1);
    assert_eq!(ps[0]["busy"], false, "{ps:?}");
}

/// Acceptance: voices answer from the manifest with the model unloaded,
/// before and after, for a pulled model and a catalog one alike.
#[test]
fn voices_answer_without_loading_the_model() {
    let server = Server::fake();
    assert_eq!(server.ps(), Vec::<Value>::new());

    for path in [
        "/v1/audio/voices",
        "/v1/audio/voices?model=fake-tts",
        "/v1/audio/voices?model=tts-1",
    ] {
        let reply = server.request("GET", path, "");
        assert_eq!(reply.status, 200, "{path}");
        assert_eq!(
            reply.json(),
            json!({"model": "fake-tts", "voices": [
                {"id": "af_heart", "accent": "us", "gender": "f", "default": true},
                {"id": "bm_george", "accent": "gb", "gender": "m", "default": false},
            ]}),
            "{path}"
        );
    }

    // Built in, not pulled here: the catalog's manifest answers.
    let reply = server.request("GET", "/v1/audio/voices?model=kokoro-v1.0", "");
    assert_eq!(reply.status, 200);
    let body = reply.json();
    let voices = body["voices"].as_array().unwrap();
    assert_eq!(voices.len(), 54);
    let defaults: Vec<&Value> = voices.iter().filter(|v| v["default"] == true).collect();
    assert_eq!(defaults.len(), 1);
    assert_eq!(defaults[0]["id"], "af_heart");

    for (path, status, code) in [
        ("/v1/audio/voices?model=nope", 404, "model_not_found"),
        ("/v1/audio/voices?model=fake-stt", 400, "invalid_request"),
        ("/v1/audio/voices?model=silero-vad", 400, "invalid_request"),
    ] {
        let reply = server.request("GET", path, "");
        assert_eq!(reply.status, status, "{path}");
        assert_eq!(reply.json()["error"]["code"], code, "{path}");
    }

    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);
    assert_eq!(server.ps(), Vec::<Value>::new());
}

fn say(url: &str, args: &[&str], stdin: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_naru-audio"))
        .arg("say")
        .args(args)
        .env("NARU_AUDIO_URL", url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdin.take().unwrap();
    if let Some(bytes) = stdin {
        pipe.write_all(bytes).unwrap();
    }
    drop(pipe);
    child.wait_with_output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// `say -o -` writes the endpoint's stream; `-o FILE` its `stream=false`
/// body; `-` reads the text from stdin.
#[test]
fn say_writes_what_the_endpoint_returns() {
    let server = Server::fake();
    let streamed = server.speech(json!({"input": "One. Two."})).body;
    let out = say(&server.url(), &["One. Two.", "-o", "-"], None);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(out.stdout, streamed);
    assert_eq!(u32_at(&out.stdout, 40), 0x7FFF_0000);

    let out = say(&server.url(), &["-", "-o", "-"], Some(b"One. Two."));
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(out.stdout, streamed);

    let exact = server
        .speech(json!({"input": "One. Two.", "stream": false}))
        .body;
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("out.wav");
    let out = say(
        &server.url(),
        &["One. Two.", "-o", file.to_str().unwrap()],
        None,
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(out.stdout.is_empty());
    assert_eq!(std::fs::read(&file).unwrap(), exact);

    let out = say(
        &server.url(),
        &["Hi.", "-v", "bm_george", "-s", "1.5", "-o", "-"],
        None,
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let (voice, options) = server.record.last.lock().unwrap().clone().unwrap();
    assert_eq!((voice.as_str(), options.speed), ("bm_george", 1.5));
}

#[test]
fn say_reports_the_daemons_error_and_a_cut_stream() {
    let server = Server::fake();
    let out = say(&server.url(), &["Hi.", "-v", "alloy", "-o", "-"], None);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).starts_with("naru-audio: the model \"fake-tts\" has no voice \"alloy\""),
        "{}",
        stderr(&out)
    );
    assert!(out.stdout.is_empty());

    let out = say(&server.url(), &["fail", "-o", "-"], None);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
}

#[test]
fn say_with_the_daemon_down_exits_3() {
    let url = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    };
    let out = say(&url, &["Hi.", "-o", "-"], None);
    assert_eq!(out.status.code(), Some(3), "{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        format!(
            "Speech isn't available: naru-audio isn't running at {url}. \
             Start it with `brew services start naru-audio`.\n"
        )
    );
}

/// A fresh home whose `models/` links to the pulled models under
/// `$NARU_AUDIO_TEST_HOME`, else `$NARU_AUDIO_HOME` or `~/.naru-audio`, as
/// tests/stream.rs does. `None` (skip) when `kokoro-v1.0` is not pulled.
fn kokoro_home() -> Option<tempfile::TempDir> {
    let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .or_else(registry::default_home)?;
    if !Registry::open(&home).ok()?.is_installed("kokoro-v1.0") {
        let message = format!(
            "kokoro-v1.0 not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
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

/// The real Kokoro through the endpoint: the stream has the 0x7FFF sizes
/// and audio, and the exact WAV parses with `hound` at the stated length.
#[test]
fn kokoro_speaks_over_http() {
    let Some(home) = kokoro_home() else { return };
    let server = Server::start(home, "kokoro-v1.0", Arc::new(BackendLoader), Arc::default());
    let text = "Hello from naru audio. This is the second sentence.";
    let started = Instant::now();
    let reply = server.speech(json!({"input": text}));
    eprintln!(
        "streamed in {:.3} s (with the load)",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(
        reply.status,
        200,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    assert!(reply.terminated);
    assert_eq!(u32_at(&reply.body, 4), 0x7FFF_0024);
    assert_eq!(u32_at(&reply.body, 40), 0x7FFF_0000);
    let seconds = (reply.body.len() - 44) as f64 / 48_000.0;
    assert!(seconds > 1.5, "{seconds} s of audio");

    let reply = server.speech(json!({"input": text, "stream": false}));
    assert_eq!(reply.status, 200);
    let wav = hound::WavReader::new(&reply.body[..]).unwrap();
    let spec = wav.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 24_000, 16)
    );
    assert_eq!(wav.len() as usize, (reply.body.len() - 44) / 2);
}
