use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use naru_audio::log::{self, Logger};
use naru_audio::registry::{self, Progress, Pulled, Registry};
use naru_audio::server::{self, AppState};

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
}

#[tokio::main]
async fn main() -> ExitCode {
    let (listen, allow_remote, log_file) = match Cli::parse().command {
        Command::Serve {
            listen,
            allow_remote,
            log_file,
        } => (listen, allow_remote, log_file),
        // §3.7: these work without a daemon, directly on $NARU_AUDIO_HOME.
        command => return registry_command(command),
    };

    if let Err(msg) = server::check_listen(listen, allow_remote) {
        eprintln!("naru-audio: {msg}");
        return ExitCode::from(2);
    }

    let registry = match open_registry() {
        Ok(r) => r,
        Err(code) => return code,
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

    let app = server::router(Arc::new(AppState {
        port: addr.port(),
        allow_remote,
        started: Instant::now(),
        log,
        registry: Arc::new(registry),
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
        Command::Serve { .. } => unreachable!("handled by main"),
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
