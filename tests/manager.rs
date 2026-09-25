//! §3.3–3.5 model manager over a real TCP server, with a fake loader whose
//! loads and decodes block until a test lets them through.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use naru_audio::log::Logger;
use naru_audio::manager::{
    KeepAlive, LoadError, LoadedModel, Loader, ModelManager, Resident, Settings,
};
use naru_audio::profile::Profile;
use naru_audio::registry::Registry;
use naru_audio::registry::manifest::Manifest;
use naru_audio::server::{AppState, router};
use naru_audio::stt::{Segment, SttError, SttModel, VadConfig, Vocabulary};
use serde_json::{Value, json};

/// Two fake STT models, `a` and `b`, of this many resident bytes each.
const MODEL_BYTES: u64 = 600;
/// Room for one model, not two.
const BUDGET: u64 = 1000;

/// Closed, it holds every caller of `pass` until `open`.
struct Gate {
    state: Mutex<(bool, usize)>,
    cv: Condvar,
}

impl Gate {
    fn new(closed: bool) -> Arc<Self> {
        Arc::new(Gate {
            state: Mutex::new((closed, 0)),
            cv: Condvar::new(),
        })
    }

    fn pass(&self) {
        let mut s = self.state.lock().unwrap();
        s.1 += 1;
        self.cv.notify_all();
        while s.0 {
            s = self.cv.wait(s).unwrap();
        }
    }

    /// Waits until `n` callers have reached the gate.
    fn reached(&self, n: usize) {
        let s = self.state.lock().unwrap();
        let (_s, timeout) = self
            .cv
            .wait_timeout_while(s, Duration::from_secs(10), |s| s.1 < n)
            .unwrap();
        assert!(!timeout.timed_out(), "nothing reached the gate");
    }

    fn count(&self) -> usize {
        self.state.lock().unwrap().1
    }

    fn open(&self) {
        self.state.lock().unwrap().0 = false;
        self.cv.notify_all();
    }

    fn close(&self) {
        self.state.lock().unwrap().0 = true;
    }
}

struct FakeModel {
    decode: Arc<Gate>,
}

impl SttModel for FakeModel {
    fn decode_each(
        &self,
        _pcm16k: &[f32],
        _hotwords: Option<&Vocabulary>,
        _vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        self.decode.pass();
        on_segment(Segment {
            start: 0.0,
            end: 0.1,
            text: "fake".to_string(),
        });
        Ok(())
    }
}

struct FakeLoader {
    load: Arc<Gate>,
    decode: Arc<Gate>,
    /// What each load reports as measured, if anything.
    measured: Option<(&'static str, u64)>,
    loads: AtomicUsize,
}

impl FakeLoader {
    fn new(load: Arc<Gate>, decode: Arc<Gate>) -> Arc<Self> {
        Arc::new(FakeLoader {
            load,
            decode,
            measured: None,
            loads: AtomicUsize::new(0),
        })
    }
}

impl Loader for FakeLoader {
    fn load(&self, manifest: &Manifest, _dir: &Path) -> Result<LoadedModel, LoadError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.load.pass();
        Ok(LoadedModel {
            model: Resident::Stt(Arc::new(FakeModel {
                decode: self.decode.clone(),
            })),
            measured_bytes: self
                .measured
                .filter(|(name, _)| *name == manifest.model.name)
                .map(|(_, bytes)| bytes),
        })
    }
}

/// A home with `a` and `b` "pulled" (a `manifest.json` alone).
fn home() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in ["a", "b"] {
        let model_dir = dir.path().join("models").join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        let manifest = json!({"model": {
            "name": name, "kind": "stt", "backend": "sherpa-onnx", "languages": ["en"],
            "resident_bytes": MODEL_BYTES,
        }});
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();
    }
    dir
}

/// The daemon on an ephemeral port, on its own runtime.
struct Server {
    port: u16,
    rt: Option<tokio::runtime::Runtime>,
}

impl Server {
    fn start(home: &Path, loader: Arc<FakeLoader>) -> Self {
        Self::with_budget(home, loader, BUDGET)
    }

    fn with_budget(home: &Path, loader: Arc<FakeLoader>, budget: u64) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Logger::stderr());
        let registry = Arc::new(Registry::open(home).unwrap());
        let mut settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
        settings.budget_bytes = budget;
        settings.stt_default = "a".to_string();
        settings.tts_default = "t".to_string();
        // Not `NARU_AUDIO_KEEP_ALIVE`, whatever the environment says.
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
        Server { port, rt: Some(rt) }
    }

    /// Status and JSON body of one HTTP/1.1 request on a fresh connection.
    fn request(&self, method: &str, path: &str, ctype: &str, body: &[u8]) -> (u16, Value) {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        write!(
            s,
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\
             Content-Type: {ctype}\r\nContent-Length: {}\r\n\r\n",
            self.port,
            body.len()
        )
        .unwrap();
        s.write_all(body).unwrap();
        let mut resp = Vec::new();
        s.read_to_end(&mut resp).unwrap();
        let resp = String::from_utf8(resp).unwrap();
        let (head, body) = resp.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    fn get(&self, path: &str) -> (u16, Value) {
        self.request("GET", path, "application/json", b"")
    }

    fn load(&self, body: Value) -> (u16, Value) {
        self.request(
            "POST",
            "/api/load",
            "application/json",
            body.to_string().as_bytes(),
        )
    }

    /// `/api/ps` names.
    fn loaded(&self) -> Vec<String> {
        let (status, ps) = self.get("/api/ps");
        assert_eq!(status, 200, "{ps}");
        ps.as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap().to_string())
            .collect()
    }

    fn transcribe(&self, fields: &[(&str, &str)]) -> (u16, Value) {
        const BOUNDARY: &str = "naru-audio-test-boundary";
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
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&wav());
        body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
        self.request(
            "POST",
            "/v1/audio/transcriptions",
            &format!("multipart/form-data; boundary={BOUNDARY}"),
            &body,
        )
    }

    /// Polls `/health` every 50 ms for `span`; the slowest answer.
    fn health_max_latency(&self, span: Duration) -> (Duration, usize) {
        let end = Instant::now() + span;
        let (mut max, mut n) = (Duration::ZERO, 0);
        while Instant::now() < end {
            let started = Instant::now();
            let (status, body) = self.get("/health");
            let took = started.elapsed();
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["status"], "ok");
            max = max.max(took);
            n += 1;
            thread::sleep(Duration::from_millis(50));
        }
        (max, n)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
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

/// Acceptance: with the budget below two models, loading the second evicts
/// the idle first.
#[test]
fn loading_past_the_budget_evicts_the_idle_model() {
    let home = home();
    let loader = FakeLoader::new(Gate::new(false), Gate::new(false));
    let server = Server::start(home.path(), loader.clone());

    let (status, body) = server.load(json!({"model": "default", "kind": "stt"}));
    assert_eq!(
        (status, &body),
        (200, &json!({"model": "a", "loaded": true}))
    );
    let (_, ps) = server.get("/api/ps");
    let a = &ps[0];
    assert_eq!(a["name"], "a");
    assert_eq!(a["kind"], "stt");
    assert_eq!(a["backend"], "sherpa-onnx");
    assert_eq!(a["resident_bytes"], MODEL_BYTES);
    assert_eq!(a["busy"], false);
    assert_eq!(a["loading"], false);
    assert!(a["expires_at"].is_string(), "{a}");

    let (_, models) = server.get("/v1/models");
    let entry = |id: &str| {
        models["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(entry("a")["x_loaded"], true);
    assert_eq!(entry("a")["x_default"], true);
    assert_eq!(entry("b")["x_loaded"], false);
    assert_eq!(entry("b")["x_default"], false);

    let (_, health) = server.get("/health");
    assert_eq!(health["stt"]["default"], "a");
    assert_eq!(health["stt"]["loaded"], true);
    assert_eq!(health["profile"]["budget_bytes"], BUDGET);

    let (status, body) = server.load(json!({"model": "b"}));
    assert_eq!(status, 200, "{body}");
    assert_eq!(server.loaded(), ["b"]);
    assert_eq!(loader.loads.load(Ordering::SeqCst), 2);
}

/// Acceptance: the same, but `a` is mid-decode, so `b` gets 507 and `a`
/// stays. A busy model also cannot be deleted.
#[test]
fn loading_past_the_budget_with_a_busy_model_is_507() {
    let home = home();
    let decode = Gate::new(true);
    let loader = FakeLoader::new(Gate::new(false), decode.clone());
    let server = Arc::new(Server::start(home.path(), loader));

    let busy = {
        let server = server.clone();
        thread::spawn(move || server.transcribe(&[("model", "a")]))
    };
    decode.reached(1);

    let (status, body) = server.load(json!({"model": "b"}));
    assert_eq!(status, 507, "{body}");
    assert_eq!(body["error"]["code"], "insufficient_memory");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("busy: a"),
        "{body}"
    );
    let (_, ps) = server.get("/api/ps");
    assert_eq!(ps.as_array().unwrap().len(), 1, "{ps}");
    assert_eq!(ps[0]["name"], "a");
    assert_eq!(ps[0]["busy"], true);
    assert_eq!(ps[0]["expires_at"], Value::Null);

    let (status, body) = server.request("DELETE", "/api/models/a", "text/plain", b"");
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "model_in_use");

    decode.open();
    let (status, body) = busy.join().unwrap();
    assert_eq!((status, &body), (200, &json!({"text": "fake"})));

    // Idle now: `b` evicts it, and `DELETE` of an idle model unloads it.
    assert_eq!(server.load(json!({"model": "b"})).0, 200);
    assert_eq!(server.loaded(), ["b"]);
    assert_eq!(
        server
            .request("DELETE", "/api/models/b", "text/plain", b"")
            .0,
        204
    );
    assert!(server.loaded().is_empty());
}

/// Acceptance: `keep_alive: 0` unloads, from `/api/load` and per request.
#[test]
fn keep_alive_zero_unloads() {
    let home = home();
    let loader = FakeLoader::new(Gate::new(false), Gate::new(false));
    let server = Server::start(home.path(), loader);

    assert_eq!(
        server.load(json!({"model": "a", "keep_alive": "10m"})).0,
        200
    );
    assert_eq!(server.loaded(), ["a"]);
    let (status, body) = server.load(json!({"model": "a", "keep_alive": 0}));
    assert_eq!(
        (status, &body),
        (200, &json!({"model": "a", "loaded": false}))
    );
    assert!(server.loaded().is_empty());

    let (status, body) = server.transcribe(&[("model", "a"), ("keep_alive", "0")]);
    assert_eq!((status, &body), (200, &json!({"text": "fake"})));
    assert!(server.loaded().is_empty());

    // Negative: never expires. A duration: expires.
    assert_eq!(
        server.transcribe(&[("model", "a"), ("keep_alive", "-1")]).0,
        200
    );
    assert_eq!(server.get("/api/ps").1[0]["expires_at"], Value::Null);
    assert_eq!(
        server.transcribe(&[("model", "a"), ("keep_alive", "5m")]).0,
        200
    );
    assert!(server.get("/api/ps").1[0]["expires_at"].is_string());

    let (status, body) = server.transcribe(&[("model", "a"), ("keep_alive", "soon")]);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["param"], "keep_alive");
}

/// A short keep-alive unloads the model once it has been idle that long.
#[test]
fn keep_alive_expires() {
    let home = home();
    let loader = FakeLoader::new(Gate::new(false), Gate::new(false));
    let server = Server::start(home.path(), loader);
    assert_eq!(
        server.load(json!({"model": "a", "keep_alive": "200ms"})).0,
        200
    );
    assert_eq!(server.loaded(), ["a"]);
    thread::sleep(Duration::from_millis(600));
    assert!(server.loaded().is_empty());
}

/// Acceptance: `/health` answers in < 50 ms during a load and during a
/// decode, both blocked for seconds.
#[test]
fn health_is_fast_during_a_load_and_a_decode() {
    const LIMIT: Duration = Duration::from_millis(50);
    let home = home();
    let (load, decode) = (Gate::new(true), Gate::new(true));
    let loader = FakeLoader::new(load.clone(), decode.clone());
    let server = Arc::new(Server::start(home.path(), loader));

    let busy = {
        let server = server.clone();
        thread::spawn(move || server.transcribe(&[("model", "a")]))
    };
    load.reached(1);
    let (_, health) = server.get("/health");
    assert_eq!(health["stt"]["loading"], true, "{health}");
    assert_eq!(server.get("/api/ps").1[0]["loading"], true);
    let (during_load, n_load) = server.health_max_latency(Duration::from_secs(2));

    load.open();
    decode.reached(1);
    let (during_decode, n_decode) = server.health_max_latency(Duration::from_secs(5));
    decode.open();
    assert_eq!(busy.join().unwrap().0, 200);

    eprintln!(
        "/health max latency: {during_load:?} over {n_load} requests during a load, \
         {during_decode:?} over {n_decode} during a decode"
    );
    assert!(during_load < LIMIT, "{during_load:?} during a load");
    assert!(during_decode < LIMIT, "{during_decode:?} during a decode");
}

/// Concurrent requests for one model load it once.
#[test]
fn concurrent_requests_load_once() {
    let home = home();
    let load = Gate::new(true);
    let loader = FakeLoader::new(load.clone(), Gate::new(false));
    let server = Arc::new(Server::start(home.path(), loader.clone()));

    let requests: Vec<_> = (0..3)
        .map(|_| {
            let server = server.clone();
            thread::spawn(move || server.load(json!({"model": "a"})))
        })
        .collect();
    load.reached(1);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(load.count(), 1);
    load.open();
    for r in requests {
        assert_eq!(r.join().unwrap().0, 200);
    }
    assert_eq!(loader.loads.load(Ordering::SeqCst), 1);
    assert_eq!(server.loaded(), ["a"]);
}

/// §3.5: a measured load is recorded in `state/measured.json` and replaces
/// the manifest's estimate, in this daemon and the next.
#[test]
fn measured_bytes_replace_the_estimate() {
    let home = home();
    let loader = Arc::new(FakeLoader {
        load: Gate::new(false),
        decode: Gate::new(false),
        measured: Some(("a", 300)),
        loads: AtomicUsize::new(0),
    });
    let server = Server::start(home.path(), loader);

    assert_eq!(server.load(json!({"model": "a"})).0, 200);
    assert_eq!(server.get("/api/ps").1[0]["resident_bytes"], 300);
    let measured: Value =
        serde_json::from_slice(&std::fs::read(home.path().join("state/measured.json")).unwrap())
            .unwrap();
    assert_eq!(measured, json!({"a": {"sherpa-onnx": 300}}));
    // 300 + 600 fits in 1000; the estimates, 600 + 600, would not.
    assert_eq!(server.load(json!({"model": "b"})).0, 200);
    assert_eq!(server.loaded(), ["a", "b"]);

    // Unloaded, `a` needs its measured 300 again: `b` stays.
    assert_eq!(server.load(json!({"model": "a", "keep_alive": 0})).0, 200);
    assert_eq!(server.load(json!({"model": "a"})).0, 200);
    assert_eq!(server.loaded(), ["a", "b"]);
    drop(server);

    // A new daemon reads the file (this loader measures nothing).
    let server = Server::start(
        home.path(),
        FakeLoader::new(Gate::new(false), Gate::new(false)),
    );
    assert_eq!(server.load(json!({"model": "b"})).0, 200);
    assert_eq!(server.load(json!({"model": "a"})).0, 200);
    assert_eq!(server.loaded(), ["a", "b"]);
}

/// §3.5: memory freed while a load runs would shrink its RSS delta, so
/// that measurement is not recorded and the estimate stands.
#[test]
fn an_unload_during_a_load_discards_the_measurement() {
    let home = home();
    let load = Gate::new(false);
    let loader = Arc::new(FakeLoader {
        load: load.clone(),
        decode: Gate::new(false),
        measured: Some(("b", 300)),
        loads: AtomicUsize::new(0),
    });
    // Room for both, so loading `b` evicts nothing.
    let server = Arc::new(Server::with_budget(home.path(), loader, 2 * MODEL_BYTES));

    assert_eq!(server.load(json!({"model": "a"})).0, 200);
    load.close();
    let loading = {
        let server = server.clone();
        thread::spawn(move || server.load(json!({"model": "b"})))
    };
    load.reached(2);
    assert_eq!(server.load(json!({"model": "a", "keep_alive": 0})).0, 200);
    load.open();
    assert_eq!(loading.join().unwrap().0, 200);

    let (_, ps) = server.get("/api/ps");
    assert_eq!(ps.as_array().unwrap().len(), 1, "{ps}");
    assert_eq!(ps[0]["name"], "b");
    assert_eq!(ps[0]["resident_bytes"], MODEL_BYTES);
    assert!(!home.path().join("state/measured.json").exists());
}

#[test]
fn api_load_validates_its_body() {
    let home = home();
    let server = Server::start(
        home.path(),
        FakeLoader::new(Gate::new(false), Gate::new(false)),
    );
    for (body, param, code) in [
        (json!({"kind": "stt"}), "model", "invalid_request"),
        (json!({"model": "default"}), "kind", "invalid_request"),
        (
            json!({"model": "default", "kind": "vad"}),
            "kind",
            "unsupported_value",
        ),
        (
            json!({"model": "a", "keep_alive": "soon"}),
            "keep_alive",
            "invalid_request",
        ),
    ] {
        let (status, resp) = server.load(body.clone());
        assert_eq!(status, 400, "{body}: {resp}");
        assert_eq!(resp["error"]["param"], param, "{body}");
        assert_eq!(resp["error"]["code"], code, "{body}");
    }
    // `keep_alive: 0` checks the model too.
    for keep_alive in [json!(null), json!(0)] {
        let (status, resp) = server.load(json!({"model": "nope", "keep_alive": keep_alive}));
        assert_eq!(status, 404, "{resp}");
        assert_eq!(resp["error"]["code"], "model_not_found");
    }
    assert!(server.loaded().is_empty());
}

/// §2.5: the TTS default reports ready, problem, loaded and loading as the
/// STT one does, and `/api/load` resolves `default` for `kind: tts` to it.
#[test]
fn health_and_load_follow_the_tts_default() {
    let home = home();
    let server = Server::start(
        home.path(),
        FakeLoader::new(Gate::new(false), Gate::new(false)),
    );
    let (status, health) = server.get("/health");
    assert_eq!(status, 200, "{health}");
    let tts = &health["tts"];
    assert_eq!(tts["default"], "t", "{health}");
    assert_eq!(tts["ready"], false, "{health}");
    // Not pulled, and not in the catalog either.
    assert_eq!(tts["problem"]["code"], "model_not_found", "{health}");
    assert_eq!(tts["loaded"], false, "{health}");
    assert_eq!(tts["loading"], false, "{health}");

    let model_dir = home.path().join("models/t");
    std::fs::create_dir_all(&model_dir).unwrap();
    let manifest = json!({"model": {
        "name": "t", "kind": "tts", "backend": "sherpa-onnx", "resident_bytes": MODEL_BYTES,
    }});
    std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();
    let (_, health) = server.get("/health");
    assert_eq!(health["tts"]["ready"], true, "{health}");
    assert_eq!(health["tts"]["problem"], Value::Null, "{health}");

    let (status, body) = server.load(json!({"model": "default", "kind": "tts"}));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "t");
    assert_eq!(server.loaded(), ["t"]);
    let (_, health) = server.get("/health");
    assert_eq!(health["tts"]["loaded"], true, "{health}");
    assert_eq!(health["stt"]["loaded"], false, "{health}");
}
