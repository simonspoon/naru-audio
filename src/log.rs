//! §4.2 logging: one line per event, `ts level req_id event key=value…`.
//!
//! With a file sink the daemon rotates its own log (Homebrew does not): when a
//! line would take the file past `max_bytes`, the file is renamed to `<path>.1`
//! (replacing any older `.1`) and a fresh file is started.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// Rotation threshold for the daemon log.
pub const MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;

pub struct Logger {
    sink: Mutex<Sink>,
}

enum Sink {
    Stderr,
    File {
        path: PathBuf,
        max_bytes: u64,
        file: File,
    },
}

impl Logger {
    pub fn stderr() -> Self {
        Self {
            sink: Mutex::new(Sink::Stderr),
        }
    }

    /// Append to `path`, rotating to `<path>.1` past `max_bytes`.
    pub fn file(path: &Path, max_bytes: u64) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        Ok(Self {
            sink: Mutex::new(Sink::File {
                path: path.to_path_buf(),
                max_bytes,
                file: open_append(path)?,
            }),
        })
    }

    pub fn info(&self, req_id: Option<&str>, event: &str) {
        self.line("info", req_id, event);
    }

    /// `event` is the event name followed by its `key=value` fields.
    pub fn line(&self, level: &str, req_id: Option<&str>, event: &str) {
        let ts = humantime::format_rfc3339_millis(SystemTime::now());
        let line = format!("{ts} {level} {} {event}\n", req_id.unwrap_or("-"));
        let mut sink = self.sink.lock().unwrap_or_else(|e| e.into_inner());
        // Logging never takes the daemon down: write errors are dropped.
        match &mut *sink {
            Sink::Stderr => {
                let _ = io::stderr().write_all(line.as_bytes());
            }
            Sink::File {
                path,
                max_bytes,
                file,
            } => {
                let len = file.metadata().map(|m| m.len()).unwrap_or(0);
                if len > 0
                    && len + line.len() as u64 > *max_bytes
                    && let Ok(fresh) = rotate(path)
                {
                    *file = fresh;
                }
                let _ = file.write_all(line.as_bytes());
            }
        }
    }
}

fn open_append(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

fn rotate(path: &Path) -> io::Result<File> {
    let mut old = path.as_os_str().to_owned();
    old.push(".1");
    fs::rename(path, &old)?;
    open_append(path)
}

/// The §4.1/§4.2 log path, `$(brew --prefix)/var/log/naru-audio.log`, derived
/// from a Homebrew-installed executable (`<prefix>/Cellar/naru-audio/<ver>/bin/…`).
/// `None` when the binary is not running from a Homebrew Cellar.
pub fn brew_log_path(exe: &Path) -> Option<PathBuf> {
    let mut prefix = exe;
    while let Some(parent) = prefix.parent() {
        if parent.file_name().is_some_and(|n| n == "Cellar") {
            return Some(parent.parent()?.join("var/log/naru-audio.log"));
        }
        prefix = parent;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_count(path: &Path) -> usize {
        fs::read_to_string(path).unwrap().lines().count()
    }

    #[test]
    fn rotates_when_log_crosses_10_mib() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naru-audio.log");
        // A sparse file 10 bytes short of the threshold.
        File::create(&path)
            .unwrap()
            .set_len(MAX_LOG_BYTES - 10)
            .unwrap();

        let log = Logger::file(&path, MAX_LOG_BYTES).unwrap();
        log.info(Some("abc"), "request route=/health status=200 ms=0");

        let rotated = dir.path().join("naru-audio.log.1");
        assert_eq!(fs::metadata(&rotated).unwrap().len(), MAX_LOG_BYTES - 10);
        let fresh = fs::read_to_string(&path).unwrap();
        assert_eq!(fresh.lines().count(), 1);
        assert!(fresh.contains(" info abc request route=/health status=200 ms=0\n"));
    }

    #[test]
    fn does_not_rotate_below_10_mib() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naru-audio.log");
        let log = Logger::file(&path, MAX_LOG_BYTES).unwrap();
        log.info(None, "listening addr=127.0.0.1:7870");
        log.info(None, "listening addr=127.0.0.1:7870");
        assert_eq!(line_count(&path), 2);
        assert!(!dir.path().join("naru-audio.log.1").exists());
    }

    #[test]
    fn rotation_keeps_only_one_old_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naru-audio.log");
        // Tiny threshold: every line after the first rotates.
        let log = Logger::file(&path, 1).unwrap();
        log.info(None, "first");
        log.info(None, "second");
        log.info(None, "third");
        let old = fs::read_to_string(dir.path().join("naru-audio.log.1")).unwrap();
        assert!(old.ends_with(" info - second\n"));
        assert!(
            fs::read_to_string(&path)
                .unwrap()
                .ends_with(" info - third\n")
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn line_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("naru-audio.log");
        let log = Logger::file(&path, MAX_LOG_BYTES).unwrap();
        log.info(None, "listening addr=127.0.0.1:7870");
        let text = fs::read_to_string(&path).unwrap();
        let fields: Vec<&str> = text.trim_end().split(' ').collect();
        assert_eq!(
            fields[1..],
            ["info", "-", "listening", "addr=127.0.0.1:7870"]
        );
        assert!(humantime::parse_rfc3339(fields[0]).is_ok(), "{}", fields[0]);
    }

    #[test]
    fn brew_log_path_from_cellar() {
        assert_eq!(
            brew_log_path(Path::new(
                "/opt/homebrew/Cellar/naru-audio/0.1.0/bin/naru-audio"
            )),
            Some(PathBuf::from("/opt/homebrew/var/log/naru-audio.log"))
        );
        assert_eq!(
            brew_log_path(Path::new("/home/x/naru-audio/target/debug/naru-audio")),
            None
        );
    }
}
