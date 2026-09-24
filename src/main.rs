use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use naru_audio::log::{self, Logger};
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::registry::{self, Progress, Pulled, Registry};
use naru_audio::server::{self, AppState};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    let settings = match Profile::detect().and_then(Settings::from_env) {
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
        Command::Serve { .. } | Command::Health | Command::Ps | Command::Transcribe { .. } => {
            unreachable!("handled by main")
        }
        Command::Pull { names, force } => {
            for name in &names {
                let mut progress = |p: Progress| {
                    if let Progress::Downloading {
                        file, completed: 0, ..
                    } = p
                    {
                        eprintln!("downloading {file}");
                    }
                };
                match reg.pull_with(name, force, &mut progress) {
                    Ok(Pulled::Downloaded) => println!("pulled {name}"),
                    Ok(Pulled::AlreadyInstalled) => println!("{name} is already pulled"),
                    Err(e) => fail(name, e),
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
                    "{:<32} {:<4} {:<12} {:<6} {:>14}  AVAILABLE",
                    "NAME", "KIND", "BACKEND", "PULLED", "SIZE"
                );
                for e in entries {
                    let m = &e.manifest.model;
                    let size = e.size_bytes.map_or("-".to_string(), |n| n.to_string());
                    let available = match &e.available {
                        Ok(()) => "yes".to_string(),
                        Err(reason) => format!("no: {reason}"),
                    };
                    println!(
                        "{:<32} {:<4} {:<12} {:<6} {:>14}  {available}",
                        m.name,
                        m.kind.as_str(),
                        m.backend,
                        if e.pulled { "yes" } else { "no" },
                        size
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
