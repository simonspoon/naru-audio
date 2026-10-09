//! §2.3 `POST /v1/audio/speech`, §2.5 `GET /v1/audio/voices` and the §3.7
//! `say` client, over real HTTP/1.1 on an ephemeral port. A fake TTS model
//! gives fixed samples; one test runs the real Kokoro and skips (eprintln
//! and return) when `kokoro-v1.0` is not pulled, as tests/tts.rs does.

use std::collections::{BTreeMap, HashMap};
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
use naru_audio::tts::{DEFAULT_SEED, Sink, SynthOptions, TtsError, TtsModel};
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
/// `fail first` before any; `unavailable` fails as a dead backend.
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
            "unavailable" => Err(TtsError::BackendUnavailable {
                backend: "mlx".to_string(),
                reason: "the sidecar died".to_string(),
            }),
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
            pulls: Arc::new(naru_audio::server::PullTracker::new()),
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
        receive(self.send(method, path, body))
    }

    /// A `multipart/form-data` POST of `fields`, each a name and its bytes.
    fn form(&self, path: &str, fields: &[(&str, &[u8])]) -> Reply {
        let mut body = Vec::new();
        for (name, value) in fields {
            write!(
                body,
                "--XBOUNDARY\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            )
            .unwrap();
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--XBOUNDARY--\r\n");
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        write!(
            s,
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\
             Content-Type: multipart/form-data; boundary=XBOUNDARY\r\n\
             Content-Length: {}\r\n\r\n",
            self.port,
            body.len()
        )
        .unwrap();
        s.write_all(&body).unwrap();
        receive(s)
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

/// Reads the reply to the end.
fn receive(mut s: TcpStream) -> Reply {
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
/// manifest's default; the knobs reach the model, but not `instructions`,
/// which this model does not take.
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
    for instructions in [json!({"style": "calm"}), Value::Null] {
        let reply = server.speech(json!({"input": "Hi.", "instructions": instructions}));
        assert_eq!(reply.status, 200, "{instructions}");
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
        instructions: None,
        exaggeration: None,
        reference: None,
        knobs: BTreeMap::new(),
        seed: DEFAULT_SEED,
    };
    assert_eq!(last(), ("bm_george".to_string(), expected));
}

/// `seed` defaults to the fixed constant, an explicit one reaches the
/// model, and anything but a non-negative integer is a 400 before a load.
#[test]
fn seed_defaults_to_the_constant_and_is_validated() {
    let server = Server::fake();
    let seed = || server.record.last.lock().unwrap().clone().unwrap().1.seed;
    assert_eq!(server.speech(json!({"input": "Hi."})).status, 200);
    assert_eq!(seed(), DEFAULT_SEED);
    assert_eq!(
        server.speech(json!({"input": "Hi.", "seed": 1234})).status,
        200
    );
    assert_eq!(seed(), 1234);
    assert_eq!(
        server.speech(json!({"input": "Hi.", "seed": null})).status,
        200
    );
    assert_eq!(seed(), DEFAULT_SEED);
    let loads = server.record.loads.load(Ordering::SeqCst);
    for bad in [json!(-1), json!(1.5), json!("7"), json!(true)] {
        let reply = server.speech(json!({"input": "Hi.", "seed": bad}));
        assert_eq!(reply.status, 400, "{bad}");
        let error = &reply.json()["error"];
        assert_eq!(error["code"], "invalid_request");
        assert_eq!(error["param"], "seed");
    }
    assert_eq!(server.record.loads.load(Ordering::SeqCst), loads);
}

/// Each speech request leaves its audio as `recent/<id>.wav` under the
/// home (a real wav of what was sent), and only the newest 50 stay.
#[test]
fn recent_wavs_are_written_and_pruned_to_fifty() {
    let server = Server::fake();
    let recent = server._home.path().join("recent");
    std::fs::create_dir_all(&recent).unwrap();
    let old = std::time::SystemTime::now() - Duration::from_secs(3600);
    for i in 0..60 {
        let f = std::fs::File::create(recent.join(format!("old-{i}.wav"))).unwrap();
        f.set_modified(old + Duration::from_secs(i)).unwrap();
    }
    // No audio, no file: a failure leaves the recordings alone, so the one
    // new wav below is the only addition.
    assert_eq!(server.speech(json!({"input": "fail first"})).status, 500);
    let reply = server.speech(json!({"input": "One. Two."}));
    assert_eq!(reply.status, 200);
    let wavs = || -> Vec<PathBuf> {
        std::fs::read_dir(&recent)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect()
    };
    // Written once synthesis ends, which is just after the body does.
    let deadline = Instant::now() + Duration::from_secs(10);
    while wavs().len() != 50 || !wavs().iter().any(|p| !p.to_string_lossy().contains("old-")) {
        assert!(
            Instant::now() < deadline,
            "no pruned recent dir: {:?}",
            wavs()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let new: Vec<_> = wavs()
        .into_iter()
        .filter(|p| !p.to_string_lossy().contains("old-"))
        .collect();
    assert_eq!(new.len(), 1);
    // The oldest eleven went; the newest old ones stayed.
    assert!(!recent.join("old-0.wav").exists());
    assert!(!recent.join("old-10.wav").exists());
    assert!(recent.join("old-11.wav").exists());
    let wav = std::fs::read(&new[0]).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(u32_at(&wav, 24), 24_000);
    assert_eq!(u32_at(&wav, 40) as usize, wav.len() - 44);
    assert_eq!(wav.len() - 44, 2 * 2 * PIECE);
}

/// A model that instructs and has no voices (VoiceDesign) is handed the
/// `instructions`, and takes any voice; without `instructions` it is a
/// 400 before a load.
#[test]
fn a_model_that_instructs_gets_the_instructions() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-design");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-design", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"instruct": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );

    for request in [
        json!({"model": "fake-design", "input": "Hi."}),
        json!({"model": "fake-design", "input": "Hi.", "voice": "alloy", "instructions": " "}),
    ] {
        let reply = server.speech(request.clone());
        assert_eq!(reply.status, 400, "{request}");
        let error = &reply.json()["error"];
        assert_eq!(error["code"], "invalid_request", "{request}");
        assert_eq!(error["param"], "instructions", "{request}");
        assert_eq!(
            error["message"],
            "the model \"fake-design\" needs \"instructions\" describing the voice"
        );
    }
    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);

    let instructions = "A warm, husky woman. Speak slowly.";
    for voice in [json!("alloy"), Value::Null] {
        let reply = server.speech(json!({
            "model": "fake-design", "input": "Hi.", "voice": voice, "instructions": instructions,
        }));
        assert_eq!(reply.status, 200, "{voice}");
        let (_, options) = server.record.last.lock().unwrap().clone().unwrap();
        assert_eq!(options.instructions.as_deref(), Some(instructions));
    }
}

/// naru task 1458: `"knobs"` is checked against the model's declared
/// `prompt_format.knobs` before it reaches `SynthOptions` — an unknown
/// name is 400 `unsupported_value`, an out-of-range value is 400
/// `invalid_request`, and a valid one lands in `options.knobs` (or, for
/// `exaggeration`/`speed`, in their own fields alongside the top-level
/// ones).
#[test]
fn knobs_are_checked_against_the_manifest_before_reaching_synth_options() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-knobs");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-knobs", "kind": "tts", "backend": "sherpa-onnx"},
        "voice": [{"id": "only", "sid": 1, "default": true}],
        "backend": {"sherpa-onnx": {"exaggeration": true, "prompt_format": {"knobs": [
            {"name": "temperature", "default": 0.9, "min": 0.0, "max": 2.0},
            {"name": "exaggeration", "default": 0.1, "min": 0.0, "max": 1.0},
        ]}}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );

    // An unknown name.
    let reply = server.speech(json!({
        "model": "fake-knobs", "input": "Hi.", "knobs": {"nope": 1.0},
    }));
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "unsupported_value");
    assert_eq!(error["param"], "knobs");

    // Out of range.
    let reply = server.speech(json!({
        "model": "fake-knobs", "input": "Hi.", "knobs": {"temperature": 3.0},
    }));
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "invalid_request");
    assert_eq!(error["param"], "knobs");
    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);

    // A valid knob reaches `SynthOptions.knobs`; `exaggeration` reaches
    // its own field, even though it is sent inside `knobs` too.
    let reply = server.speech(json!({
        "model": "fake-knobs", "input": "Hi.",
        "knobs": {"temperature": 0.5, "exaggeration": 0.8},
    }));
    assert_eq!(reply.status, 200);
    let (_, options) = server.record.last.lock().unwrap().clone().unwrap();
    assert_eq!(
        options.knobs,
        BTreeMap::from([("temperature".to_string(), 0.5)])
    );
    assert_eq!(options.exaggeration, Some(0.8));
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

/// A backend that went away before any audio (a crashed MLX sidecar) is
/// 503 `backend_unavailable`, as for transcription.
#[test]
fn an_unavailable_backend_before_any_audio_is_a_503() {
    let server = Server::fake();
    for stream in [true, false] {
        let reply = server.speech(json!({"input": "unavailable", "stream": stream}));
        assert_eq!(reply.status, 503, "stream={stream}");
        assert_eq!(reply.json()["error"]["code"], "backend_unavailable");
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
                {"id": "af_heart", "accent": "us", "gender": "f", "default": true, "cloned": false,
                 "origin": "builtin", "description": null, "duration": null, "has_transcript": false},
                {"id": "bm_george", "accent": "gb", "gender": "m", "default": false, "cloned": false,
                 "origin": "builtin", "description": null, "duration": null, "has_transcript": false},
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

/// A cloning model lists and speaks the cloned voices in the home's
/// `voices/`, the first as its default; a model that does not clone
/// refuses them. A voice added while the daemon runs is usable at once.
#[test]
fn a_cloning_model_speaks_the_cloned_voices() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-clone");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let add = |home: &Path, name: &str| {
        let voice = home.join("voices").join(name);
        std::fs::create_dir_all(&voice).unwrap();
        std::fs::write(voice.join("ref.wav"), b"").unwrap();
        std::fs::write(voice.join("ref.txt"), "Hello.").unwrap();
        std::fs::write(voice.join("model.txt"), "fake-clone").unwrap();
    };
    add(home.path(), "zed");
    let voices_dir = home.path().to_path_buf();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );
    let last = || server.record.last.lock().unwrap().clone().unwrap().0;

    let reply = server.request("GET", "/v1/audio/voices?model=fake-clone", "");
    assert_eq!(
        reply.json(),
        json!({"model": "fake-clone", "voices": [
            {"id": "zed", "accent": null, "gender": null, "default": false, "cloned": true,
             "origin": "cloned", "description": null, "duration": null, "has_transcript": true},
        ]})
    );
    // Kokoro-like models are untouched.
    let reply = server.request("GET", "/v1/audio/voices?model=fake-tts", "");
    assert_eq!(reply.json()["voices"].as_array().unwrap().len(), 2);

    add(&voices_dir, "amy");
    let reply = server.speech(json!({"model": "fake-clone", "input": "Hi.", "voice": "amy"}));
    assert_eq!(reply.status, 200);
    assert_eq!(last(), "amy");
    let reply = server.speech(json!({"model": "fake-clone", "input": "Hi."}));
    assert_eq!(reply.status, 200);
    assert_eq!(last(), "amy");

    for (model, voice) in [("fake-clone", "nope"), ("fake-clone", "../voices/amy")] {
        let reply = server.speech(json!({"model": model, "input": "Hi.", "voice": voice}));
        assert_eq!(reply.status, 400, "{model} {voice}");
        assert_eq!(reply.json()["error"]["code"], "unknown_voice");
    }
    let reply = server.speech(json!({"model": "fake-clone", "input": "Hi.", "voice": "nope"}));
    let message = reply.json()["error"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(message.ends_with("use one of: amy, zed"), "{message}");

    // §5.3: an explicit model that "amy" (made for "fake-clone") was not
    // made for is refused by name, not treated as merely unknown.
    let reply = server.speech(json!({"model": "fake-tts", "input": "Hi.", "voice": "amy"}));
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "voice_model_mismatch");
    assert_eq!(
        error["message"],
        "the voice \"amy\" was made for \"fake-clone\", not \"fake-tts\""
    );

    // With no explicit model, "amy" picks its own ("fake-clone") instead of
    // the daemon's default ("fake-tts").
    let reply = server.speech(json!({"input": "Hi.", "voice": "amy"}));
    assert_eq!(reply.status, 200);
    assert_eq!(last(), "amy");

    // A clone that happens to share a name with a built-in voice of the
    // model that would otherwise be used does not shadow it: "af_heart" is
    // fake-tts's own default voice, even though a clone named "af_heart"
    // exists for fake-clone. Neither the implicit default nor an explicit
    // "fake-tts" is redirected to the clone or refused as a mismatch.
    add(&voices_dir, "af_heart");
    let reply = server.speech(json!({"input": "Hi.", "voice": "af_heart"}));
    assert_eq!(reply.status, 200, "{}", reply.json());
    let reply = server.speech(json!({"model": "fake-tts", "input": "Hi.", "voice": "af_heart"}));
    assert_eq!(reply.status, 200, "{}", reply.json());
}

/// A cloning model with no cloned voices and no voice asked for says
/// how to add one, before a load.
#[test]
fn a_cloning_model_without_voices_says_how_to_add_one() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-clone");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );

    let reply = server.speech(json!({"model": "fake-clone", "input": "Hi."}));
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "unknown_voice");
    assert_eq!(
        error["message"],
        "the model \"fake-clone\" has no cloned voices; add one with naru-audio voice add"
    );
    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);
}

/// A model that both clones and instructs (VoxCPM2): a named cloned voice
/// wins even with `instructions` given, but `instructions` alone still
/// designs a voice even once a cloned voice exists — the design branch is
/// not shadowed by `voices` no longer being empty.
#[test]
fn a_model_that_both_clones_and_instructs_prefers_a_named_voice() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-both");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-both", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true, "instruct": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let voice = home.path().join("voices").join("zed");
    std::fs::create_dir_all(&voice).unwrap();
    std::fs::write(voice.join("ref.wav"), b"").unwrap();
    std::fs::write(voice.join("ref.txt"), "Hello.").unwrap();
    std::fs::write(voice.join("model.txt"), "fake-both").unwrap();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );
    let last = || server.record.last.lock().unwrap().clone().unwrap();

    // A named, existing cloned voice wins, even with `instructions` given.
    let reply = server.speech(json!({
        "model": "fake-both", "input": "Hi.", "voice": "zed", "instructions": "cheerful",
    }));
    assert_eq!(reply.status, 200);
    assert_eq!(last().0, "zed");

    // `instructions` alone still designs a voice, though `voices` is no
    // longer empty (the cloned voice above): the model is not forced into
    // cloning "zed" by default just because instructions did not name it.
    let instructions = "A warm, husky woman. Speak slowly.";
    let reply = server.speech(json!({
        "model": "fake-both", "input": "Hi.", "instructions": instructions,
    }));
    assert_eq!(reply.status, 200);
    let (voice, options) = last();
    assert_eq!(voice, "");
    assert_eq!(options.instructions.as_deref(), Some(instructions));
}

/// `secs` seconds of a 440 Hz tone as a 16 kHz mono WAV.
#[cfg(target_os = "macos")]
fn tone(secs: f64) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut out = std::io::Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
    for i in 0..(secs * 16_000.0) as u32 {
        let s = (f64::from(i) * 440.0 * std::f64::consts::TAU / 16_000.0).sin();
        w.write_sample((s * 8000.0) as i16).unwrap();
    }
    w.finalize().unwrap();
    out.into_inner()
}

/// `POST /v1/audio/voices` adds a cloned voice the way `voice add` does:
/// listed and spoken at once. Each refusal is a 4xx that leaves nothing
/// behind in `voices/` or `tmp/`.
#[cfg(target_os = "macos")]
#[test]
fn a_posted_clip_becomes_a_cloned_voice() {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-clone");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    let home_dir = home.path().to_path_buf();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );
    let ok = tone(6.0);
    let post = |fields: &[(&str, &[u8])]| server.form("/v1/audio/voices", fields);

    let reply = post(&[
        ("name", b"amy"),
        ("text", b"  Hello there.\n"),
        ("file", &ok),
        ("model", b"fake-clone"),
    ]);
    assert_eq!(
        reply.status,
        201,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    let body = reply.json();
    assert!((body["duration"].as_f64().unwrap() - 6.0).abs() < 0.05);
    assert_eq!(
        body,
        json!({"id": "amy", "accent": null, "gender": null, "default": false,
               "cloned": true, "origin": "cloned", "description": null,
               "model": "fake-clone", "duration": body["duration"]})
    );
    let voice = home_dir.join("voices").join("amy");
    assert_eq!(
        std::fs::read_to_string(voice.join("ref.txt")).unwrap(),
        "Hello there.\n"
    );
    let spec = hound::WavReader::open(voice.join("ref.wav"))
        .unwrap()
        .spec();
    assert_eq!((spec.channels, spec.sample_rate), (1, 24_000));
    let reply = server.request("GET", "/v1/audio/voices?model=fake-clone", "");
    assert_eq!(reply.json()["voices"][0]["id"], "amy");
    let reply = server.speech(json!({"model": "fake-clone", "input": "Hi.", "voice": "amy"}));
    assert_eq!(reply.status, 200);
    assert_eq!(server.record.last.lock().unwrap().clone().unwrap().0, "amy");

    let (short, long) = (tone(2.0), tone(31.0));
    type Fields<'a> = &'a [(&'a str, &'a [u8])];
    let cases: [(Fields, u16, &str, &str, &str); 11] = [
        (
            &[("text", b"Hi."), ("file", &ok)],
            400,
            "invalid_request",
            "name",
            "\"name\" field is required",
        ),
        (
            &[("name", b"b"), ("text", b"Hi.")],
            400,
            "invalid_request",
            "file",
            "\"file\" field is required",
        ),
        (
            &[("name", b"b"), ("file", &ok)],
            400,
            "invalid_request",
            "text",
            "\"text\" field is required",
        ),
        (
            &[("name", b"b"), ("text", b" \n"), ("file", &ok)],
            400,
            "invalid_request",
            "text",
            "the transcript is empty",
        ),
        (
            &[("name", b"../x"), ("text", b"Hi."), ("file", &ok)],
            400,
            "invalid_request",
            "name",
            "must be a plain name",
        ),
        (
            &[("name", b".x"), ("text", b"Hi."), ("file", &ok)],
            400,
            "invalid_request",
            "name",
            "must be a plain name",
        ),
        (
            &[("name", &[b'a'; 65]), ("text", b"Hi."), ("file", &ok)],
            400,
            "invalid_request",
            "name",
            "voice name is 65 bytes; the cap is 64",
        ),
        (
            &[("name", b"amy"), ("text", b"Hi."), ("file", &ok)],
            409,
            "voice_exists",
            "name",
            "already exists",
        ),
        (
            &[
                ("name", b"b"),
                ("text", b"Hi."),
                ("file", b"not audio at all"),
            ],
            415,
            "unsupported_media_type",
            "file",
            "WAV or MP3",
        ),
        (
            &[("name", b"b"), ("text", b"Hi."), ("file", &short)],
            400,
            "invalid_request",
            "file",
            "the clip is 2.0 s; it must be 3–30 s",
        ),
        (
            &[("name", b"b"), ("text", b"Hi."), ("file", &long)],
            400,
            "invalid_request",
            "file",
            "the clip is 31.0 s; it must be 3–30 s",
        ),
    ];
    for (fields, status, code, param, message) in cases {
        let reply = post(fields);
        let error = &reply.json()["error"];
        assert_eq!(reply.status, status, "{error}");
        assert_eq!(
            (error["code"].as_str(), error["param"].as_str()),
            (Some(code), Some(param)),
            "{error}"
        );
        assert!(
            error["message"].as_str().unwrap().contains(message),
            "{error}"
        );
    }
    let reply = server.request("POST", "/v1/audio/voices", "{}");
    assert_eq!(reply.status, 400);

    // Nothing half-made is left behind, and the uploads are gone.
    let names = |d: &Path| -> Vec<String> {
        std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect()
    };
    assert_eq!(names(&home_dir.join("voices")), ["amy"]);
    // A model load leaves its registry lock file, which is never unlinked.
    let uploads: Vec<String> = names(&home_dir.join("tmp"))
        .into_iter()
        .filter(|n| !n.ends_with(".lock"))
        .collect();
    assert!(uploads.is_empty(), "{uploads:?}");
}

/// naru task 1458 §2.6 `POST /api/voices/preview`: field validation before
/// any clip conversion or model load — a missing field, a model that does
/// not clone, and an unknown model.
#[test]
fn preview_validates_its_fields_before_converting_or_loading() {
    let server = Server::fake();
    let ok: &[u8] = b"not real audio, but validation never reads it";
    let post = |fields: &[(&str, &[u8])]| server.form("/api/voices/preview", fields);

    let field = |name: &'static str, value: &'static [u8]| (name, value);
    type Fields<'a> = &'a [(&'a str, &'a [u8])];
    let cases: [(Fields, u16, &str, &str); 6] = [
        (
            &[field("file", ok), field("input", b"Hi.")],
            400,
            "invalid_request",
            "model",
        ),
        (
            &[field("model", b"fake-clone"), field("input", b"Hi.")],
            400,
            "invalid_request",
            "file",
        ),
        (
            &[field("model", b"fake-clone"), field("file", ok)],
            400,
            "invalid_request",
            "input",
        ),
        (
            &[
                field("model", b"fake-clone"),
                field("file", ok),
                field("input", b"   "),
            ],
            400,
            "invalid_request",
            "input",
        ),
        (
            // "fake-tts" clones nothing: refused before any conversion.
            &[
                field("model", b"fake-tts"),
                field("file", ok),
                field("input", b"Hi."),
            ],
            400,
            "model_does_not_clone",
            "model",
        ),
        (
            &[
                field("model", b"nope"),
                field("file", ok),
                field("input", b"Hi."),
            ],
            404,
            "model_not_found",
            "model",
        ),
    ];
    for (fields, status, code, param) in cases {
        let reply = post(fields);
        assert_eq!(reply.status, status, "{code}: {}", reply.json());
        let error = &reply.json()["error"];
        assert_eq!(error["code"], code, "{code}");
        assert_eq!(error["param"], param, "{code}");
    }
    assert_eq!(server.record.loads.load(Ordering::SeqCst), 0);
}

/// A cloning model with `[backend.sherpa-onnx] clone = true` for
/// `POST /api/voices/preview`, alongside the models `fake_home` already
/// gives `Server::fake`.
#[cfg(target_os = "macos")]
fn fake_clone_home() -> tempfile::TempDir {
    let home = fake_home();
    let dir = home.path().join("models").join("fake-clone");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = json!({
        "model": {"name": "fake-clone", "kind": "tts", "backend": "sherpa-onnx"},
        "backend": {"sherpa-onnx": {"clone": true}},
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
    home
}

/// `POST /api/voices/preview` synthesises from the uploaded clip directly
/// (`SynthOptions.reference`), saves nothing under `voices/`, and always
/// removes its temp files.
#[cfg(target_os = "macos")]
#[test]
fn preview_synthesises_from_a_reference_clip_without_saving_a_voice() {
    let home = fake_clone_home();
    let home_dir = home.path().to_path_buf();
    let record = Arc::new(Record::default());
    let server = Server::start(
        home,
        "fake-tts",
        Arc::new(FakeLoader(record.clone())),
        record,
    );
    let ok = tone(6.0);

    let reply = server.form(
        "/api/voices/preview",
        &[
            ("model", b"fake-clone"),
            ("file", &ok),
            ("text", b"What the clip says."),
            ("input", b"Preview this."),
        ],
    );
    assert_eq!(
        reply.status,
        200,
        "{}",
        String::from_utf8_lossy(&reply.body)
    );
    assert_eq!(reply.header("content-type"), Some("audio/wav"));
    assert_eq!(&reply.body[0..4], b"RIFF");

    let (voice, options) = server.record.last.lock().unwrap().clone().unwrap();
    assert_eq!(voice, "");
    let (wav, text) = options.reference.expect("a reference was passed");
    assert_eq!(text, "What the clip says.");
    // The temp clip existed for the synthesis but is gone once it answers.
    assert!(!wav.exists(), "{}", wav.display());
    assert!(wav.starts_with(home_dir.join("tmp")), "{}", wav.display());

    // Nothing was saved as a voice.
    assert!(!home_dir.join("voices").exists());
    let leftover: Vec<String> = std::fs::read_dir(home_dir.join("tmp"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| !n.ends_with(".lock"))
        .collect();
    assert!(leftover.is_empty(), "{leftover:?}");
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
