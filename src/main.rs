use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use futures_util::{SinkExt, StreamExt};
use naru_audio::log::{self, Logger};
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::{self, Progress, Pulled, Registry};
use naru_audio::server::{self, AppState};
use naru_audio::stt::audio;
use naru_audio::voices;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message as WsMessage;

#[derive(Parser)]
#[command(name = "naru-audio", version, about = "Local STT/TTS daemon for Naru")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP daemon.
    Serve {
        /// Address to bind, ADDR:PORT.
        #[arg(long, env = "NARU_AUDIO_LISTEN", default_value = server::DEFAULT_LISTEN)]
        listen: SocketAddr,
        /// Allow a non-loopback --listen and skip the Host check.
        #[arg(long)]
        allow_remote: bool,
        /// Log file, rotated at 10 MiB. Defaults to
        /// $(brew --prefix)/var/log/naru-audio.log under Homebrew, else stderr.
        #[arg(long, env = "NARU_AUDIO_LOG")]
        log_file: Option<PathBuf>,
    },
    /// Download models, and what they require, into $NARU_AUDIO_HOME.
    Pull {
        #[arg(required = true)]
        names: Vec<String>,
        /// Pull even if the model's backend cannot run on this machine.
        #[arg(long)]
        force: bool,
    },
    /// List pulled models and catalog models that can run here.
    List {
        /// Print only pulled model names, one per line.
        #[arg(long)]
        names: bool,
    },
    /// Remove pulled models.
    Rm {
        #[arg(required = true)]
        names: Vec<String>,
        /// Remove even if another pulled model requires it.
        #[arg(long)]
        force: bool,
    },
    /// Re-hash pulled models' files against their manifests (all if none named).
    Verify { names: Vec<String> },
    /// Check the daemon at $NARU_AUDIO_URL: exit 0 ready, 3 not running,
    /// 4 the default speech model is not ready.
    Health,
    /// List the daemon's loaded models.
    Ps,
    /// Transcribe a WAV file through the daemon.
    Transcribe {
        /// WAV file, or `-` for stdin.
        file: String,
        /// Model name; the daemon's default STT model if not given.
        #[arg(short, long)]
        model: Option<String>,
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
    /// Speak text through the daemon as 16-bit WAV.
    Say {
        /// Text, or `-` for stdin.
        text: String,
        /// Voice id; the model's default voice if not given. VoiceDesign
        /// models have no voices and ignore it: use --instructions.
        #[arg(short, long)]
        voice: Option<String>,
        /// Model name; if not given, qwen3-tts-0.6b-base-mlx for a cloned
        /// voice, else the daemon's default TTS model.
        #[arg(short, long)]
        model: Option<String>,
        /// Speed, 0.5 to 2.0.
        #[arg(short, long)]
        speed: Option<f64>,
        /// Voice description, which a VoiceDesign model needs; other models
        /// ignore it.
        #[arg(long)]
        instructions: Option<String>,
        /// WAV file, written with exact sizes once synthesis ends; or `-`
        /// for stdout, streamed as each sentence is synthesised.
        #[arg(short, long)]
        output: String,
    },
    /// Stream a WAV file through the daemon's WebSocket at the pace of real
    /// time, printing its events as JSON Lines. Each `final` gains
    /// `latency_ms`: from sending its last speech sample to its arrival.
    Stream {
        /// WAV file; resampled to 16 kHz mono if need be.
        file: PathBuf,
        /// Model name; the daemon's default STT model if not given.
        #[arg(short, long)]
        model: Option<String>,
        /// Milliseconds of audio per frame.
        #[arg(long, default_value_t = 40, value_parser = clap::value_parser!(u64).range(20..=100))]
        frame_ms: u64,
        /// Times faster than real time; 0 sends as fast as possible.
        #[arg(long, default_value_t = 1.0)]
        speed: f64,
        /// Ask for `partial` events.
        #[arg(long)]
        partials: bool,
    },
    /// The MLX sidecar's Python environment (Apple Silicon only).
    Mlx {
        #[command(subcommand)]
        action: MlxAction,
    },
    /// Cloned voices in $NARU_AUDIO_HOME/voices, spoken by any cloning model
    /// (qwen3-tts-0.6b-base-mlx, qwen3-tts-1.7b-base-mlx).
    Voice {
        #[command(subcommand)]
        action: VoiceAction,
    },
}

#[derive(Subcommand)]
enum VoiceAction {
    /// Add a voice from a 5–15 s clip of one speaker (WAV or MP3; 3–30 s
    /// accepted), converted with macOS afconvert.
    Add {
        /// The voice id: no path separators.
        name: String,
        clip: PathBuf,
        /// Exactly what the clip says.
        #[arg(long)]
        text: String,
    },
}

#[derive(Subcommand)]
enum MlxAction {
    /// Install the sidecar's venv with uv into $NARU_AUDIO_HOME/mlx and
    /// record its interpreter in config.toml.
    Setup,
    /// Show the recorded interpreter and whether it imports the sidecar.
    Status,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    /// The transcript text.
    Text,
    /// auris-style JSON Lines: a `segment` line each, then `transcript`.
    Json,
}

#[tokio::main]
async fn main() -> ExitCode {
    let (listen, allow_remote, log_file) = match Cli::parse().command {
        Command::Serve {
            listen,
            allow_remote,
            log_file,
        } => (listen, allow_remote, log_file),
        Command::Health => return health(&daemon_url()),
        Command::Ps => return ps(&daemon_url()),
        Command::Transcribe {
            file,
            model,
            format,
        } => return transcribe(&daemon_url(), &file, model.as_deref(), format),
        Command::Say {
            text,
            voice,
            model,
            speed,
            instructions,
            output,
        } => {
            return say(
                &daemon_url(),
                &text,
                voice.as_deref(),
                model.as_deref(),
                speed,
                instructions.as_deref(),
                &output,
            );
        }
        Command::Stream {
            file,
            model,
            frame_ms,
            speed,
            partials,
        } => {
            return stream(
                &daemon_url(),
                &file,
                model.as_deref(),
                frame_ms,
                speed,
                partials,
            )
            .await;
        }
        Command::Mlx { action } => return mlx_command(action),
        Command::Voice { action } => return voice_command(action),
        // §3.7: these work without a daemon, directly on $NARU_AUDIO_HOME.
        command => return registry_command(command),
    };

    if let Err(msg) = server::check_listen(listen, allow_remote) {
        eprintln!("naru-audio: {msg}");
        return ExitCode::from(2);
    }

    let registry = match open_registry() {
        Ok(r) => Arc::new(r),
        Err(code) => return code,
    };
    let settings = match Profile::detect().and_then(|p| Settings::load(p, registry.home())) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("naru-audio: {e}");
            return ExitCode::from(2);
        }
    };

    let log_file = log_file.or_else(|| {
        std::env::current_exe()
            .ok()
            // brew services runs the opt/ symlink; the Cellar path is its target.
            .and_then(|exe| exe.canonicalize().ok())
            .and_then(|exe| log::brew_log_path(&exe))
    });
    let log = match log_file {
        Some(path) => match Logger::file(&path, log::MAX_LOG_BYTES) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("naru-audio: cannot open log {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        },
        None => Logger::stderr(),
    };

    let listener = match tokio::net::TcpListener::bind(listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("naru-audio: cannot listen on {listen}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let addr = listener.local_addr().unwrap_or(listen);
    let log = Arc::new(log);
    log.info(None, &format!("listening addr={addr}"));
    // Before any model load can start the sidecar.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    naru_audio::mlx::refresh_scripts(registry.home(), &log);

    let models = ModelManager::new(
        registry.clone(),
        log.clone(),
        settings,
        Arc::new(BackendLoader),
    );
    let app = server::router(Arc::new(AppState {
        port: addr.port(),
        allow_remote,
        started: Instant::now(),
        log,
        registry,
        models: Arc::new(models),
    }));
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("naru-audio: server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn open_registry() -> Result<Registry, ExitCode> {
    let Some(home) = registry::default_home() else {
        eprintln!("naru-audio: set NARU_AUDIO_HOME or HOME");
        return Err(ExitCode::FAILURE);
    };
    Registry::open(home).map_err(|e| {
        eprintln!("naru-audio: {e}");
        ExitCode::FAILURE
    })
}

/// §3.7 `mlx setup|status`, on $NARU_AUDIO_HOME; both exit 1 on failure.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn mlx_command(action: MlxAction) -> ExitCode {
    use naru_audio::mlx;

    let Some(home) = registry::default_home() else {
        eprintln!("naru-audio: set NARU_AUDIO_HOME or HOME");
        return ExitCode::FAILURE;
    };
    if let MlxAction::Setup = action
        && let Err(e) = mlx::setup(&home)
    {
        eprintln!("naru-audio: mlx setup: {e}");
        return ExitCode::FAILURE;
    }
    let (python, result) = mlx::status(&home);
    if let Some(python) = python {
        println!("python   {}", python.display());
        println!("exists   {}", if python.is_file() { "yes" } else { "no" });
    }
    match result {
        Ok(()) => {
            println!("imports  yes");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("naru-audio: mlx: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn mlx_command(_: MlxAction) -> ExitCode {
    eprintln!(
        "naru-audio: mlx requires Apple Silicon (arm64 macOS); this build has no MLX backend"
    );
    ExitCode::FAILURE
}

/// `voice add`, on $NARU_AUDIO_HOME; exits 1 on failure.
fn voice_command(action: VoiceAction) -> ExitCode {
    let Some(home) = registry::default_home() else {
        eprintln!("naru-audio: set NARU_AUDIO_HOME or HOME");
        return ExitCode::FAILURE;
    };
    let VoiceAction::Add { name, clip, text } = action;
    match voices::add(&home, &name, &clip, &text) {
        Ok(secs) => {
            if !(5.0..=15.0).contains(&secs) {
                eprintln!("naru-audio: warning: the clip is {secs:.1} s; 5–15 s clones best");
            }
            println!("added voice {name} ({secs:.1} s)");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("naru-audio: voice add: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `pull`, `list`, `rm` and `verify`. Every named model is attempted; the
/// exit code is 1 if any failed.
fn registry_command(command: Command) -> ExitCode {
    let reg = match open_registry() {
        Ok(r) => r,
        Err(code) => return code,
    };
    let mut ok = true;
    let mut fail = |name: &str, e: registry::RegistryError| {
        eprintln!("naru-audio: {name}: {e}");
        ok = false;
    };
    match command {
        Command::Serve { .. }
        | Command::Health
        | Command::Ps
        | Command::Transcribe { .. }
        | Command::Say { .. }
        | Command::Stream { .. }
        | Command::Mlx { .. }
        | Command::Voice { .. } => {
            unreachable!("handled by main")
        }
        Command::Pull { names, force } => {
            let names = if names.iter().any(|n| n == "default") {
                match Profile::detect().and_then(|p| Settings::load(p, reg.home())) {
                    Ok(s) => expand_pull_names(&names, &s.stt_default, &s.tts_default),
                    Err(e) => {
                        eprintln!("naru-audio: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                names
            };
            for name in &names {
                let mut progress = |p: Progress| {
                    if let Progress::Downloading {
                        file, completed: 0, ..
                    } = p
                    {
                        eprintln!("downloading {file}");
                    }
                };
                let pulled = match reg.pull_with(name, force, &mut progress) {
                    Ok(Pulled::Downloaded) => {
                        println!("pulled {name}");
                        true
                    }
                    Ok(Pulled::AlreadyInstalled) => {
                        println!("{name} is already pulled");
                        true
                    }
                    Err(e) => {
                        fail(name, e);
                        false
                    }
                };
                if pulled
                    && let Some(m) = reg.catalog().models.get(name)
                    && m.model.non_commercial
                {
                    eprintln!(
                        "naru-audio: warning: {name} is {}, non-commercial use only{}",
                        m.model.license.as_deref().unwrap_or("unlicensed"),
                        m.model
                            .license_url
                            .as_deref()
                            .map_or(String::new(), |u| format!(" ({u})"))
                    );
                }
            }
        }
        Command::List { names: true } => match reg.pulled() {
            Ok(names) => names.iter().for_each(|n| println!("{n}")),
            Err(e) => fail("list", e),
        },
        Command::List { names: false } => match reg.list() {
            Ok(entries) => {
                println!(
                    "{:<32} {:<4} {:<12} {:<6} {:>14}  {:<24}LICENSE",
                    "NAME", "KIND", "BACKEND", "PULLED", "SIZE", "AVAILABLE"
                );
                for e in entries {
                    let m = &e.manifest.model;
                    let size = e.size_bytes.map_or("-".to_string(), |n| n.to_string());
                    let available = match &e.available {
                        Ok(()) => "yes".to_string(),
                        Err(reason) => format!("no: {reason}"),
                    };
                    let license = match &m.license {
                        Some(l) if m.non_commercial => format!("{l} (non-commercial)"),
                        Some(l) => l.clone(),
                        None => "-".to_string(),
                    };
                    println!(
                        "{:<32} {:<4} {:<12} {:<6} {:>14}  {:<24}{license}",
                        m.name,
                        m.kind.as_str(),
                        m.backend,
                        if e.pulled { "yes" } else { "no" },
                        size,
                        available
                    );
                }
            }
            Err(e) => fail("list", e),
        },
        Command::Rm { names, force } => {
            for name in &names {
                match reg.remove(name, force, &names) {
                    Ok(()) => println!("removed {name}"),
                    Err(e) => fail(name, e),
                }
            }
        }
        Command::Verify { names } => {
            let names = if names.is_empty() {
                match reg.pulled() {
                    Ok(n) => n,
                    Err(e) => {
                        fail("verify", e);
                        Vec::new()
                    }
                }
            } else {
                names
            };
            for name in &names {
                match reg.verify(name) {
                    Ok(()) => println!("{name} ok"),
                    Err(e) => fail(name, e),
                }
            }
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// §4.1 `pull default`: `default` becomes the STT and TTS models `serve`
/// would load; the VAD model comes with the STT model's `requires`. Each
/// name is kept once, first mention wins.
fn expand_pull_names(names: &[String], stt: &str, tts: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in names {
        let expanded = if name == "default" {
            vec![stt, tts]
        } else {
            vec![name.as_str()]
        };
        for n in expanded {
            if !out.iter().any(|o| o == n) {
                out.push(n.to_string());
            }
        }
    }
    out
}

/// §3.7: `transcribe` and `say` exit 3 when the daemon is down, `health`
/// also exits 4 when it is up but speech is not ready.
const EXIT_DAEMON_DOWN: u8 = 3;
const EXIT_NOT_READY: u8 = 4;
/// `health` and `ps` answer from memory (§4.3); a daemon slower than this
/// is treated as down.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// §4.4: `NARU_AUDIO_URL`, else the address `serve` listens on by default.
fn daemon_url() -> String {
    match std::env::var("NARU_AUDIO_URL") {
        Ok(url) if !url.is_empty() => url.trim_end_matches('/').to_string(),
        _ => format!("http://{}", server::DEFAULT_LISTEN),
    }
}

/// A client that returns non-2xx responses, so their §2.6 envelope can be read.
fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(timeout)
        .build()
        .into()
}

/// The status and body of a daemon response. A daemon that cannot be
/// reached is reported with the §4.4 `daemon_down` text and exit 3.
fn read(
    url: &str,
    result: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<(u16, String), ExitCode> {
    match result.and_then(|mut r| Ok((r.status().as_u16(), r.body_mut().read_to_string()?))) {
        Ok(response) => Ok(response),
        Err(
            ureq::Error::Io(_)
            | ureq::Error::Timeout(_)
            | ureq::Error::ConnectionFailed
            | ureq::Error::HostNotFound,
        ) => {
            eprintln!(
                "Speech isn't available: naru-audio isn't running at {url}. Start it with `brew services start naru-audio`."
            );
            Err(ExitCode::from(EXIT_DAEMON_DOWN))
        }
        Err(e) => {
            eprintln!("naru-audio: {url}: {e}");
            Err(ExitCode::FAILURE)
        }
    }
}

/// The §2.6 envelope's message, else the raw status and body.
fn error_message(status: u16, body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| format!("HTTP {status}: {body}"))
}

/// `GET /health`, printing the §4.4 text for any state but `ready`.
fn health(url: &str) -> ExitCode {
    let (status, body) = match read(
        url,
        agent(Some(PROBE_TIMEOUT))
            .get(format!("{url}/health"))
            .call(),
    ) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let health = match serde_json::from_str::<Value>(&body) {
        Ok(v) if status == 200 => v,
        _ => {
            let message = error_message(status, &body);
            eprintln!("Speech isn't available: naru-audio reported: {message}");
            return ExitCode::from(EXIT_NOT_READY);
        }
    };
    let version = health["version"].as_str().unwrap_or("?");
    if health["api"] != 1 {
        eprintln!(
            "Speech isn't available: naru-audio {version} speaks API {}; this Naru needs API 1. Upgrade with `brew upgrade naru-audio`.",
            health["api"]
        );
        return ExitCode::from(EXIT_NOT_READY);
    }
    let stt = &health["stt"];
    let model = stt["default"].as_str().unwrap_or("?");
    if stt["ready"] == true {
        println!("naru-audio {version} at {url} is ready (stt: {model})");
        return ExitCode::SUCCESS;
    }
    let problem = &stt["problem"];
    if problem["code"] == "model_not_pulled" {
        eprintln!(
            "Speech isn't available: the model {model} isn't downloaded. Run `naru-audio pull {model}`."
        );
    } else {
        eprintln!(
            "Speech isn't available: naru-audio reported: {}",
            problem["message"]
                .as_str()
                .unwrap_or("the default speech model is not ready")
        );
    }
    ExitCode::from(EXIT_NOT_READY)
}

/// `GET /api/ps` as a table.
fn ps(url: &str) -> ExitCode {
    let (status, body) = match read(
        url,
        agent(Some(PROBE_TIMEOUT))
            .get(format!("{url}/api/ps"))
            .call(),
    ) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let models = match serde_json::from_str::<Vec<Value>>(&body) {
        Ok(m) if status == 200 => m,
        _ => {
            eprintln!("naru-audio: {}", error_message(status, &body));
            return ExitCode::FAILURE;
        }
    };
    println!(
        "{:<32} {:<4} {:<12} {:>14}  {:<7}  EXPIRES",
        "NAME", "KIND", "BACKEND", "RESIDENT", "STATE"
    );
    let text = |v: &Value| match v {
        Value::Null => "-".to_string(),
        Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    for m in &models {
        let state = if m["loading"] == true {
            "loading"
        } else if m["busy"] == true {
            "busy"
        } else {
            "idle"
        };
        println!(
            "{:<32} {:<4} {:<12} {:>14}  {:<7}  {}",
            text(&m["name"]),
            text(&m["kind"]),
            text(&m["backend"]),
            text(&m["resident_bytes"]),
            state,
            text(&m["expires_at"])
        );
    }
    ExitCode::SUCCESS
}

#[derive(Deserialize)]
struct Verbose {
    text: String,
    segments: Vec<VerboseSegment>,
}

#[derive(Deserialize)]
struct VerboseSegment {
    id: usize,
    start: f64,
    end: f64,
    text: String,
}

/// An auris `--format json` line (§1.1).
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Line<'a> {
    Segment {
        index: usize,
        text: &'a str,
        start: f64,
        end: f64,
    },
    Transcript {
        text: &'a str,
    },
}

/// `POST /v1/audio/transcriptions` with `verbose_json`, printed as `format`.
/// Silence is an empty transcript and exit 0 (§2.2).
fn transcribe(url: &str, file: &str, model: Option<&str>, format: Format) -> ExitCode {
    // A stopped daemon is reported before the input is read: exit 3 even
    // for a missing file, and stdin is not drained. Readiness is not checked.
    if let Err(code) = read(
        url,
        agent(Some(PROBE_TIMEOUT))
            .get(format!("{url}/health"))
            .call(),
    ) {
        return code;
    }
    let audio = if file == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf).map(|_| buf)
    } else {
        std::fs::read(file)
    };
    let audio = match audio {
        Ok(a) => a,
        Err(e) => {
            eprintln!("naru-audio: {file}: {e}");
            return ExitCode::FAILURE;
        }
    };

    const BOUNDARY: &str = "naru-audio-cli-9f2c4e7a1b3d";
    let mut form = Vec::with_capacity(audio.len() + 512);
    let mut field = |name: &str, value: &str| {
        form.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        )
    };
    field("response_format", "verbose_json");
    if let Some(model) = model {
        field("model", model);
    }
    form.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    form.extend_from_slice(&audio);
    form.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());

    let (status, body) = match read(
        url,
        agent(None)
            .post(format!("{url}/v1/audio/transcriptions"))
            .content_type(format!("multipart/form-data; boundary={BOUNDARY}"))
            .send(&form[..]),
    ) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let verbose = match serde_json::from_str::<Verbose>(&body) {
        Ok(v) if status == 200 => v,
        _ => {
            eprintln!("naru-audio: {}", error_message(status, &body));
            return ExitCode::FAILURE;
        }
    };
    match format {
        Format::Text => println!("{}", verbose.text),
        Format::Json => {
            let line = |l: &Line| println!("{}", serde_json::to_string(l).expect("serializable"));
            for s in &verbose.segments {
                line(&Line::Segment {
                    index: s.id,
                    text: &s.text,
                    start: s.start,
                    end: s.end,
                });
            }
            line(&Line::Transcript {
                text: &verbose.text,
            });
        }
    }
    ExitCode::SUCCESS
}

/// The `/v1/audio/speech` body for `say`: optional fields only when given.
fn speech_request(
    model: &str,
    text: &str,
    voice: Option<&str>,
    speed: Option<f64>,
    instructions: Option<&str>,
    stream: bool,
) -> serde_json::Value {
    let mut request = serde_json::json!({
        "model": model, "input": text, "response_format": "wav", "stream": stream,
    });
    if let Some(voice) = voice {
        request["voice"] = voice.into();
    }
    if let Some(speed) = speed {
        request["speed"] = speed.into();
    }
    if let Some(instructions) = instructions {
        request["instructions"] = instructions.into();
    }
    request
}

/// `POST /v1/audio/speech` as `wav`, to `output`. `-` asks for the chunked
/// stream and copies each piece to stdout as it arrives, so a player can
/// start at once; a file asks for `stream=false`, whose header has the exact
/// sizes. A stream the daemon aborts (§2.3) is exit 1.
fn say(
    url: &str,
    text: &str,
    voice: Option<&str>,
    model: Option<&str>,
    speed: Option<f64>,
    instructions: Option<&str>,
    output: &str,
) -> ExitCode {
    // As `transcribe`: a stopped daemon is exit 3, before stdin is read.
    if let Err(code) = read(
        url,
        agent(Some(PROBE_TIMEOUT))
            .get(format!("{url}/health"))
            .call(),
    ) {
        return code;
    }
    let text = if text == "-" {
        let mut buf = String::new();
        match std::io::stdin().read_to_string(&mut buf) {
            Ok(_) => buf,
            Err(e) => {
                eprintln!("naru-audio: stdin: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        text.to_string()
    };
    let stream = output == "-";
    let model = voices::say_model(registry::default_home().as_deref(), model, voice);
    let request = speech_request(model, &text, voice, speed, instructions, stream);

    let response = agent(None)
        .post(format!("{url}/v1/audio/speech"))
        .content_type("application/json")
        .send(request.to_string());
    let mut response = match response {
        Ok(r) if r.status() == 200 => r,
        // Reports the error, and exits 3 for a daemon that went away.
        result => {
            return match read(url, result) {
                Ok((status, body)) => {
                    eprintln!("naru-audio: {}", error_message(status, &body));
                    ExitCode::FAILURE
                }
                Err(code) => code,
            };
        }
    };
    let mut body = response.body_mut().as_reader();
    let copied = if stream {
        let mut out = std::io::stdout().lock();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match body.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => {
                    if let Err(e) = out.write_all(&buf[..n]).and_then(|()| out.flush()) {
                        break Err(format!("stdout: {e}"));
                    }
                }
                Err(e) => break Err(format!("{url}: {e}")),
            }
        }
    } else {
        std::fs::File::create(output)
            .and_then(|mut file| std::io::copy(&mut body, &mut file))
            .map(|_| ())
            .map_err(|e| {
                // No half-written file is left behind.
                let _ = std::fs::remove_file(output);
                format!("{output}: {e}")
            })
    };
    match copied {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("naru-audio: {e}");
            ExitCode::FAILURE
        }
    }
}

/// §2.4 over `ws://`: `start`, the file's samples as s16le frames paced at
/// `speed` times real time, then `stop`. Exits 0 once `done` arrives, 1 on
/// an `error` event or any other end.
async fn stream(
    url: &str,
    file: &Path,
    model: Option<&str>,
    frame_ms: u64,
    speed: f64,
    partials: bool,
) -> ExitCode {
    if !(speed.is_finite() && speed >= 0.0) {
        eprintln!("naru-audio: --speed must be a non-negative number, got {speed}");
        return ExitCode::from(2);
    }
    // As `transcribe`: a stopped daemon is exit 3, before the file is read.
    if let Err(code) = read(
        url,
        agent(Some(PROBE_TIMEOUT))
            .get(format!("{url}/health"))
            .call(),
    ) {
        return code;
    }
    let pcm = match std::fs::File::open(file)
        .map_err(|e| e.to_string())
        .and_then(|f| audio::decode(f).map_err(|e| e.to_string()))
    {
        Ok(pcm) => pcm,
        Err(e) => {
            eprintln!("naru-audio: {}: {e}", file.display());
            return ExitCode::FAILURE;
        }
    };

    let ws_url = match url.strip_prefix("http") {
        Some(rest) => format!("ws{rest}/v1/audio/transcriptions/stream"),
        None => format!("{url}/v1/audio/transcriptions/stream"),
    };
    let (socket, _) = match tokio_tungstenite::connect_async(&ws_url).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("naru-audio: {ws_url}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (mut tx, mut rx) = socket.split();
    let start = serde_json::json!({
        "type": "start", "model": model.unwrap_or("default"), "format": "s16le",
        "sample_rate": audio::TARGET_SAMPLE_RATE, "partials": partials,
    });
    if let Err(e) = tx.send(WsMessage::text(start.to_string())).await {
        eprintln!("naru-audio: {ws_url}: {e}");
        return ExitCode::FAILURE;
    }

    let frame_len = (audio::TARGET_SAMPLE_RATE as u64 * frame_ms / 1000) as usize;
    let frames: Vec<Vec<u8>> = pcm
        .chunks(frame_len)
        .map(|frame| {
            frame
                .iter()
                // The inverse of the daemon's `/ 32768`: 16-bit WAV samples
                // arrive unchanged.
                .flat_map(|s| ((s * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())
                .collect()
        })
        .collect();
    let pace = (speed > 0.0).then(|| Duration::from_secs_f64(frame_ms as f64 / 1000.0 / speed));
    // When each frame went out, for `latency_ms`.
    let sent: Arc<std::sync::Mutex<Vec<Instant>>> = Arc::default();
    // Sending starts at `ready`, alongside the reading.
    let (mut go, sender) = {
        let sent = sent.clone();
        let (go, ready) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            if ready.await.is_err() {
                return Ok(());
            }
            let t0 = tokio::time::Instant::now();
            for (i, frame) in frames.into_iter().enumerate() {
                if let Some(pace) = pace {
                    tokio::time::sleep_until(t0 + pace * i as u32).await;
                }
                sent.lock().unwrap().push(Instant::now());
                tx.send(WsMessage::binary(frame)).await?;
            }
            tx.send(WsMessage::text(r#"{"type":"stop"}"#)).await
        });
        (Some(go), task)
    };

    let mut done = false;
    while let Some(msg) = rx.next().await {
        let text = match msg {
            Ok(WsMessage::Text(text)) => text,
            Ok(WsMessage::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        let Ok(mut event) = serde_json::from_str::<Value>(&text) else {
            println!("{text}");
            continue;
        };
        match event["type"].as_str() {
            Some("ready") => {
                if let Some(go) = go.take() {
                    let _ = go.send(());
                }
            }
            Some("final") => {
                // The frame that carried the segment's last speech sample.
                let end = event["end"].as_f64().unwrap_or(0.0);
                // `end` is exclusive: the last sample is the one before it.
                let last =
                    ((end * audio::TARGET_SAMPLE_RATE as f64).round() as usize).saturating_sub(1);
                let frame = last / frame_len;
                if let Some(at) = sent.lock().unwrap().get(frame) {
                    event["latency_ms"] = (at.elapsed().as_millis() as u64).into();
                }
            }
            Some("done") => done = true,
            _ => {}
        }
        println!("{event}");
    }
    sender.abort();
    if done {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_pull_names, speech_request};

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn pull_default_expands_to_stt_and_tts() {
        assert_eq!(
            expand_pull_names(&names(&["default"]), "stt-a", "tts-b"),
            names(&["stt-a", "tts-b"])
        );
    }

    #[test]
    fn pull_names_are_deduped_in_order() {
        assert_eq!(
            expand_pull_names(
                &names(&["tts-b", "default", "silero-vad", "default", "tts-b"]),
                "stt-a",
                "tts-b"
            ),
            names(&["tts-b", "stt-a", "silero-vad"])
        );
    }

    #[test]
    fn pull_other_names_pass_through() {
        assert_eq!(
            expand_pull_names(&names(&["kokoro-v1.0", "silero-vad"]), "stt-a", "tts-b"),
            names(&["kokoro-v1.0", "silero-vad"])
        );
    }

    #[test]
    fn say_request_carries_instructions_when_given() {
        let request = speech_request(
            "qwen3-tts-1.7b-voicedesign-mlx",
            "hello",
            None,
            None,
            Some("A warm calm man"),
            false,
        );
        assert_eq!(request["instructions"], "A warm calm man");
        assert_eq!(request["model"], "qwen3-tts-1.7b-voicedesign-mlx");
        assert_eq!(request["input"], "hello");
    }

    #[test]
    fn say_request_omits_instructions_when_not_given() {
        let request = speech_request("default", "hello", Some("af_heart"), Some(1.5), None, true);
        assert!(request.get("instructions").is_none());
        assert_eq!(request["voice"], "af_heart");
        assert_eq!(request["speed"], 1.5);
        assert_eq!(request["stream"], true);
    }
}
