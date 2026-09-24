//! §3.7 thin HTTP clients `transcribe`, `health` and `ps`: the binary run
//! against a daemon on an ephemeral port whose fake model returns fixed
//! segments, against a stub, and against a closed port.

mod common;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::Stub;
use naru_audio::log::Logger;
use naru_audio::manager::{KeepAlive, LoadedModel, Loader, ModelManager, Resident, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::Registry;
use naru_audio::registry::manifest::Manifest;
use naru_audio::server::{AppState, router};
use naru_audio::stt::{Segment, SttError, SttModel, VadConfig, Vocabulary};
use serde_json::{Value, json};

const BOUNDARY: &str = "naru-audio-test-boundary";

struct FakeModel;

impl SttModel for FakeModel {
    fn decode_each(
        &self,
        _pcm16k: &[f32],
        _hotwords: Option<&Vocabulary>,
        _vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        for (start, end, text) in [(0.31, 1.5, "Hello there."), (1.75, 3.1, "Second line.")] {
            on_segment(Segment {
                start,
                end,
                text: text.to_string(),
            });
        }
        Ok(())
    }
}

struct FakeLoader;

impl Loader for FakeLoader {
    fn load(&self, _manifest: &Manifest, _dir: &Path) -> Result<LoadedModel, SttError> {
        Ok(LoadedModel {
            model: Resident::Stt(Arc::new(FakeModel)),
            measured_bytes: None,
        })
    }
}

/// The daemon on an ephemeral port, on its own runtime, with `a` "pulled"
/// (a `manifest.json` alone) and `stt_default` as its default STT model.
struct Server {
    port: u16,
    _home: tempfile::TempDir,
    rt: Option<tokio::runtime::Runtime>,
}

impl Server {
    fn start(stt_default: &str) -> Self {
        let home = tempfile::tempdir().unwrap();
        let model_dir = home.path().join("models/a");
        std::fs::create_dir_all(&model_dir).unwrap();
        let manifest = json!({"model": {
            "name": "a", "kind": "stt", "backend": "sherpa-onnx", "languages": ["en"],
        }});
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Logger::stderr());
        let registry = Arc::new(Registry::open(home.path()).unwrap());
        let mut settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
        settings.stt_default = stt_default.to_string();
        settings.keep_alive = KeepAlive::For(Duration::from_secs(300));
        let models = ModelManager::new(
            registry.clone(),
            log.clone(),
            settings,
            Arc::new(FakeLoader),
        );
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
            _home: home,
            rt: Some(rt),
        }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// The endpoint's own answer: status and body of a transcription of
    /// `file` with `fields`.
    fn transcribe(&self, file: &[u8], fields: &[(&str, &str)]) -> (u16, String) {
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
        body.extend_from_slice(file);
        body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());

        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        write!(
            s,
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\
             Content-Type: multipart/form-data; boundary={BOUNDARY}\r\nContent-Length: {}\r\n\r\n",
            self.port,
            body.len()
        )
        .unwrap();
        s.write_all(&body).unwrap();
        let mut resp = Vec::new();
        s.read_to_end(&mut resp).unwrap();
        let resp = String::from_utf8(resp).unwrap();
        let (head, body) = resp.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, body.to_string())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}

/// A URL nothing listens on.
fn closed_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stt/plain.wav")
}

fn run(url: &str, args: &[&str], stdin: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_naru-audio"))
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

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn daemon_down(url: &str) -> String {
    format!(
        "Speech isn't available: naru-audio isn't running at {url}. Start it with `brew services start naru-audio`.\n"
    )
}

/// Acceptance: with the daemon stopped, `transcribe x.wav` exits 3 with the
/// "isn't running" text, and never starts one.
#[test]
fn transcribe_with_the_daemon_down_exits_3() {
    let url = closed_url();
    let out = run(&url, &["transcribe", fixture().to_str().unwrap()], None);
    assert_eq!(out.status.code(), Some(3), "{}", stderr(&out));
    assert_eq!(stderr(&out), daemon_down(&url));
    assert_eq!(stdout(&out), "");
    assert!(TcpStream::connect(url.trim_start_matches("http://")).is_err());
}

/// The daemon is checked before the input is read: a missing file still
/// exits 3, and `-` exits without waiting for stdin to close.
#[test]
fn transcribe_with_the_daemon_down_exits_3_before_reading_input() {
    let url = closed_url();
    let out = run(&url, &["transcribe", "does-not-exist.wav"], None);
    assert_eq!(out.status.code(), Some(3), "{}", stderr(&out));
    assert_eq!(stderr(&out), daemon_down(&url));

    let mut child = Command::new(env!("CARGO_BIN_EXE_naru-audio"))
        .args(["transcribe", "-"])
        .env("NARU_AUDIO_URL", &url)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Held open: a client that drained stdin would never exit.
    let _stdin = child.stdin.take().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            panic!("transcribe - waited on stdin with the daemon down");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(3));
}

#[test]
fn health_and_ps_with_the_daemon_down_exit_3() {
    let url = closed_url();
    for command in ["health", "ps"] {
        let out = run(&url, &[command], None);
        assert_eq!(out.status.code(), Some(3), "{command}: {}", stderr(&out));
        assert_eq!(stderr(&out), daemon_down(&url), "{command}");
    }
}

/// Acceptance: with the daemon running, the output matches the endpoint.
#[test]
fn transcribe_output_matches_the_endpoint() {
    let server = Server::start("a");
    let wav = std::fs::read(fixture()).unwrap();
    let path = fixture();
    let path = path.to_str().unwrap();

    let (status, text) = server.transcribe(&wav, &[("response_format", "text")]);
    assert_eq!(status, 200, "{text}");
    let out = run(&server.url(), &["transcribe", path], None);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out), format!("{text}\n"));
    assert_eq!(stdout(&out), "Hello there. Second line.\n");

    let (status, verbose) = server.transcribe(&wav, &[("response_format", "verbose_json")]);
    assert_eq!(status, 200, "{verbose}");
    let verbose: Value = serde_json::from_str(&verbose).unwrap();
    for args in [
        &["transcribe", path, "--format", "json"][..],
        &["transcribe", "-", "-m", "a", "--format", "json"][..],
    ] {
        let stdin = (args[1] == "-").then_some(&wav[..]);
        let out = run(&server.url(), args, stdin);
        assert!(out.status.success(), "{args:?}: {}", stderr(&out));
        let jsonl = stdout(&out);
        assert_eq!(
            jsonl,
            "{\"type\":\"segment\",\"index\":0,\"text\":\"Hello there.\",\"start\":0.31,\"end\":1.5}\n\
             {\"type\":\"segment\",\"index\":1,\"text\":\"Second line.\",\"start\":1.75,\"end\":3.1}\n\
             {\"type\":\"transcript\",\"text\":\"Hello there. Second line.\"}\n",
            "{args:?}"
        );
        let lines: Vec<Value> = jsonl
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let segments = verbose["segments"].as_array().unwrap();
        assert_eq!(lines.len(), segments.len() + 1);
        for (line, s) in lines.iter().zip(segments) {
            assert_eq!(
                (&line["index"], &line["text"], &line["start"], &line["end"]),
                (&s["id"], &s["text"], &s["start"], &s["end"])
            );
        }
        assert_eq!(lines.last().unwrap()["text"], verbose["text"]);
    }
}

#[test]
fn transcribe_reports_the_daemons_error() {
    let server = Server::start("a");
    let out = run(
        &server.url(),
        &["transcribe", fixture().to_str().unwrap(), "-m", "nope"],
        None,
    );
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        stderr(&out),
        "naru-audio: the model \"nope\" is not in the catalog; see GET /v1/models\n"
    );
    assert_eq!(stdout(&out), "");
}

#[test]
fn health_exits_0_when_the_default_is_ready() {
    let server = Server::start("a");
    let out = run(&server.url(), &["health"], None);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("is ready (stt: a)"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn health_exits_4_when_the_default_is_not_pulled() {
    let server = Server::start("parakeet-tdt-0.6b-v2-int8");
    let out = run(&server.url(), &["health"], None);
    assert_eq!(out.status.code(), Some(4), "{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        "Speech isn't available: the model parakeet-tdt-0.6b-v2-int8 isn't downloaded. \
         Run `naru-audio pull parakeet-tdt-0.6b-v2-int8`.\n"
    );
}

#[test]
fn health_exits_4_on_any_other_problem() {
    let server = Server::start("nope");
    let out = run(&server.url(), &["health"], None);
    assert_eq!(out.status.code(), Some(4), "{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        "Speech isn't available: naru-audio reported: \
         the model \"nope\" is not in the catalog; see GET /v1/models\n"
    );
}

#[test]
fn health_exits_4_on_another_api() {
    let body = json!({"status": "ok", "version": "9.0.0", "api": 2,
        "stt": {"default": "a", "ready": true, "problem": null}});
    let stub = Stub::start(
        &[("/health", body.to_string().into_bytes())],
        Duration::ZERO,
    );
    let out = run(&stub.url(""), &["health"], None);
    assert_eq!(out.status.code(), Some(4), "{}", stderr(&out));
    assert_eq!(
        stderr(&out),
        "Speech isn't available: naru-audio 9.0.0 speaks API 2; this Naru needs API 1. \
         Upgrade with `brew upgrade naru-audio`.\n"
    );
}

#[test]
fn ps_lists_loaded_models() {
    let server = Server::start("a");
    let empty = run(&server.url(), &["ps"], None);
    assert!(empty.status.success(), "{}", stderr(&empty));
    assert_eq!(stdout(&empty).lines().count(), 1, "{}", stdout(&empty));
    assert!(stdout(&empty).starts_with("NAME "));

    let out = run(
        &server.url(),
        &["transcribe", fixture().to_str().unwrap()],
        None,
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let out = run(&server.url(), &["ps"], None);
    assert!(out.status.success(), "{}", stderr(&out));
    let table = stdout(&out);
    let row = table.lines().nth(1).unwrap_or_default();
    let cols: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(cols[..3], ["a", "stt", "sherpa-onnx"], "{table}");
    assert_eq!(cols[4], "idle", "{table}");
    assert_eq!(table.lines().count(), 2, "{table}");
}
