use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use naru_audio::log::{self, Logger};
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
}

#[tokio::main]
async fn main() -> ExitCode {
    let Command::Serve {
        listen,
        allow_remote,
        log_file,
    } = Cli::parse().command;

    if let Err(msg) = server::check_listen(listen, allow_remote) {
        eprintln!("naru-audio: {msg}");
        return ExitCode::from(2);
    }

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
    }));
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("naru-audio: server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
