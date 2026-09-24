//! §3 model registry: find a model in the catalog and pull it, `requires`
//! first, into `$NARU_AUDIO_HOME/models/<name>/` (§3.1, §3.2).
//!
//! A pull stages everything in `tmp/<name>/`. Each download streams to
//! `<file>.part`, is checked for size and then sha256, and is renamed into
//! place. Archives are extracted and their `[[file]]`s checked, derived files
//! are generated, and `manifest.json` is written last. Only then is the
//! staging directory renamed to `models/<name>`, so `models/` never holds a
//! partial model; any failure removes the staging directory instead.
//!
//! A per-model `tmp/<name>.lock` (an advisory `flock`, so it holds across
//! threads and processes) serialises pulls: whoever waited finds the model
//! installed and does nothing.

pub mod manifest;

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use manifest::{BPE_VOCAB, Catalog, Manifest};

/// Written last into a staged model; its presence under `models/` means installed.
pub const MANIFEST_JSON: &str = "manifest.json";
const CHUNK_SIZE: usize = 64 * 1024;

#[derive(Debug)]
pub enum RegistryError {
    /// A manifest failed to parse or validate. `origin` names its file.
    Manifest {
        origin: String,
        message: String,
    },
    UnknownModel(String),
    /// `requires` loops back on itself; the chain of names.
    RequiresCycle(Vec<String>),
    Download {
        url: String,
        message: String,
    },
    SizeMismatch {
        file: String,
        expected: u64,
        actual: u64,
    },
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    /// A url-less `[[file]]` that no `[[archive]]` produced.
    MissingFile(String),
    Io {
        context: String,
        error: std::io::Error,
    },
}

impl RegistryError {
    fn io(context: impl Into<String>) -> impl Fn(std::io::Error) -> Self {
        let context = context.into();
        move |error| RegistryError::Io {
            context: context.clone(),
            error,
        }
    }
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::Manifest { origin, message } => {
                write!(f, "invalid manifest {origin}: {message}")
            }
            RegistryError::UnknownModel(name) => write!(f, "unknown model `{name}`"),
            RegistryError::RequiresCycle(chain) => {
                write!(f, "`requires` cycle: {}", chain.join(" -> "))
            }
            RegistryError::Download { url, message } => {
                write!(f, "download of {url} failed: {message}")
            }
            RegistryError::SizeMismatch {
                file,
                expected,
                actual,
            } => write!(f, "{file}: expected {expected} bytes, got {actual}"),
            RegistryError::HashMismatch {
                file,
                expected,
                actual,
            } => write!(f, "{file}: sha256 is {actual}, expected {expected}"),
            RegistryError::MissingFile(file) => {
                write!(f, "{file}: not found after extracting the archives")
            }
            RegistryError::Io { context, error } => write!(f, "{context}: {error}"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pulled {
    Downloaded,
    AlreadyInstalled,
}

pub struct Registry {
    home: PathBuf,
    catalog: Catalog,
}

impl Registry {
    /// Loads the catalog for `home` (`$NARU_AUDIO_HOME`).
    pub fn open(home: impl Into<PathBuf>) -> Result<Self, RegistryError> {
        let home = home.into();
        let catalog = Catalog::load(&home.join("catalog.d"))?;
        Ok(Self { home, catalog })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.home.join("models").join(name)
    }

    pub fn is_installed(&self, name: &str) -> bool {
        self.model_dir(name).join(MANIFEST_JSON).is_file()
    }

    /// Pulls `name` after everything it `requires`. Reports what happened to
    /// `name` itself.
    pub fn pull(&self, name: &str) -> Result<Pulled, RegistryError> {
        let mut order = Vec::new();
        self.resolve(name, &mut Vec::new(), &mut order)?;
        let mut pulled = Pulled::AlreadyInstalled;
        for m in order {
            pulled = self.install(m)?;
        }
        Ok(pulled)
    }

    /// Depth-first, so each model lands after its `requires`.
    fn resolve<'a>(
        &'a self,
        name: &str,
        stack: &mut Vec<String>,
        order: &mut Vec<&'a Manifest>,
    ) -> Result<(), RegistryError> {
        if order.iter().any(|m| m.model.name == name) {
            return Ok(());
        }
        if stack.iter().any(|n| n == name) {
            let mut chain = stack.clone();
            chain.push(name.to_string());
            return Err(RegistryError::RequiresCycle(chain));
        }
        let m = self
            .catalog
            .models
            .get(name)
            .ok_or_else(|| RegistryError::UnknownModel(name.to_string()))?;
        stack.push(name.to_string());
        for r in &m.model.requires {
            self.resolve(r, stack, order)?;
        }
        stack.pop();
        order.push(m);
        Ok(())
    }

    fn install(&self, m: &Manifest) -> Result<Pulled, RegistryError> {
        let name = &m.model.name;
        if self.is_installed(name) {
            return Ok(Pulled::AlreadyInstalled);
        }

        let tmp = self.home.join("tmp");
        std::fs::create_dir_all(&tmp)
            .map_err(RegistryError::io(format!("create {}", tmp.display())))?;
        // Never unlinked: removing a lock file races with a waiter that has
        // it open. Released when `lock` is dropped.
        let lock_path = tmp.join(format!("{name}.lock"));
        let lock = File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(RegistryError::io(format!("open {}", lock_path.display())))?;
        lock.lock()
            .map_err(RegistryError::io(format!("lock {}", lock_path.display())))?;

        // Whoever held the lock before us may have just finished this model.
        if self.is_installed(name) {
            return Ok(Pulled::AlreadyInstalled);
        }

        // Anything here is left from a crashed pull: we hold the lock.
        let staging = tmp.join(name);
        if staging.exists() {
            std::fs::remove_dir_all(&staging).map_err(RegistryError::io(format!(
                "remove stale {}",
                staging.display()
            )))?;
        }
        std::fs::create_dir_all(&staging)
            .map_err(RegistryError::io(format!("create {}", staging.display())))?;

        let result = stage(m, &staging).and_then(|()| promote(&staging, &self.model_dir(name)));
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result.map(|()| Pulled::Downloaded)
    }
}

/// Fills `staging` with the verified model: archives, then files, then
/// derived files, then `manifest.json`.
fn stage(m: &Manifest, staging: &Path) -> Result<(), RegistryError> {
    for a in &m.archives {
        let base = a.url.rsplit('/').next().unwrap_or("archive");
        let part = staging.join(format!("{base}.part"));
        download(&a.url, &part, a.size, sha(&a.sha256))?;
        extract(&part, &a.url, staging, a.strip)?;
        std::fs::remove_file(&part)
            .map_err(RegistryError::io(format!("remove {}", part.display())))?;
    }

    for f in &m.files {
        let dest = staging.join(&f.path);
        match &f.url {
            Some(url) => {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(RegistryError::io(format!("create {}", parent.display())))?;
                }
                let part = staging.join(format!("{}.part", f.path));
                download(url, &part, f.size, sha(&f.sha256))?;
                std::fs::rename(&part, &dest).map_err(RegistryError::io(format!(
                    "rename {} into place",
                    part.display()
                )))?;
            }
            None => {
                let file = match File::open(&dest) {
                    Ok(file) => file,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        return Err(RegistryError::MissingFile(f.path.clone()));
                    }
                    Err(e) => return Err(RegistryError::io(format!("open {}", dest.display()))(e)),
                };
                let got = hash_copy(file, std::io::sink())
                    .map_err(RegistryError::io(format!("read {}", dest.display())))?;
                check(&f.path, f.size, sha(&f.sha256), got)?;
            }
        }
    }

    for d in m.derive() {
        // Validation admits only `bpe.vocab`.
        debug_assert_eq!(d, BPE_VOCAB);
        let tokens_path = staging.join("tokens.txt");
        let tokens = std::fs::read_to_string(&tokens_path)
            .map_err(RegistryError::io(format!("read {}", tokens_path.display())))?;
        let vocab_path = staging.join(BPE_VOCAB);
        std::fs::write(&vocab_path, tokens_to_bpe_vocab(&tokens))
            .map_err(RegistryError::io(format!("write {}", vocab_path.display())))?;
    }

    // §3.1: a copy of the resolved manifest plus `pulled_at`.
    let mut doc = serde_json::to_value(&m.raw).map_err(|e| RegistryError::Manifest {
        origin: m.model.name.clone(),
        message: e.to_string(),
    })?;
    doc["pulled_at"] = humantime::format_rfc3339_seconds(SystemTime::now())
        .to_string()
        .into();
    let json_path = staging.join(MANIFEST_JSON);
    let json = serde_json::to_vec_pretty(&doc).expect("a JSON value always serialises");
    std::fs::write(&json_path, json)
        .map_err(RegistryError::io(format!("write {}", json_path.display())))
}

/// Validation normalised every sha256 to `Some(lowercase hex)`.
fn sha(s: &Option<String>) -> &str {
    s.as_deref().expect("validated manifests pin every sha256")
}

/// Streams `url` into `part`, then checks size and sha256. On any failure
/// `part` is removed.
fn download(url: &str, part: &Path, size: Option<u64>, sha256: &str) -> Result<(), RegistryError> {
    let result = (|| {
        let reader = open_url(url)?;
        // One byte past the pinned size is enough to fail the size check, so
        // an endless body cannot fill the disk.
        let reader: Box<dyn Read> = match size {
            Some(n) => Box::new(reader.take(n.saturating_add(1))),
            None => reader,
        };
        let out =
            File::create(part).map_err(RegistryError::io(format!("create {}", part.display())))?;
        let got = hash_copy(reader, out).map_err(|e| RegistryError::Download {
            url: url.to_string(),
            message: e.to_string(),
        })?;
        check(url, size, sha256, got)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(part);
    }
    result
}

/// §3.2 allows `file:///` for local experiments.
fn open_url(url: &str) -> Result<Box<dyn Read>, RegistryError> {
    let err = |message: String| RegistryError::Download {
        url: url.to_string(),
        message,
    };
    if let Some(path) = url.strip_prefix("file://") {
        return Ok(Box::new(File::open(path).map_err(|e| err(e.to_string()))?));
    }
    let resp = ureq::get(url).call().map_err(|e| err(e.to_string()))?;
    Ok(Box::new(resp.into_body().into_reader()))
}

/// Copies `r` into `w`, returning the byte count and the lowercase hex sha256.
fn hash_copy(mut r: impl Read, mut w: impl Write) -> std::io::Result<(u64, String)> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_SIZE];
    let mut n_total = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        w.write_all(&buf[..n])?;
        n_total += n as u64;
    }
    w.flush()?;
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((n_total, hex))
}

/// Size first (cheap and the likelier failure), then sha256.
fn check(
    file: &str,
    size: Option<u64>,
    sha256: &str,
    (actual_size, actual_sha): (u64, String),
) -> Result<(), RegistryError> {
    if let Some(expected) = size
        && expected != actual_size
    {
        return Err(RegistryError::SizeMismatch {
            file: file.to_string(),
            expected,
            actual: actual_size,
        });
    }
    if actual_sha != sha256 {
        return Err(RegistryError::HashMismatch {
            file: file.to_string(),
            expected: sha256.to_string(),
            actual: actual_sha,
        });
    }
    Ok(())
}

/// Unpacks a `.tar.bz2` or `.tar` into `dest`, dropping `strip` leading path
/// components; entries left empty by the strip are skipped.
fn extract(archive: &Path, url: &str, dest: &Path, strip: usize) -> Result<(), RegistryError> {
    let ioe = RegistryError::io(format!("extract {url}"));
    let file = File::open(archive).map_err(&ioe)?;
    let reader: Box<dyn Read> = if url.ends_with(".tar.bz2") {
        Box::new(bzip2::read::MultiBzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(file)
    };
    let mut tar = tar::Archive::new(reader);
    for entry in tar.entries().map_err(&ioe)? {
        let mut entry = entry.map_err(&ioe)?;
        let path = entry.path().map_err(&ioe)?.into_owned();
        // A link could point later entries outside the model directory.
        let kind = entry.header().entry_type();
        if kind.is_symlink() || kind.is_hard_link() {
            return Err(RegistryError::Download {
                url: url.to_string(),
                message: format!("archive entry `{}` is a link", path.display()),
            });
        }
        let rel: PathBuf = path
            .components()
            .filter(|c| !matches!(c, Component::CurDir))
            .skip(strip)
            .collect();
        if rel.as_os_str().is_empty() {
            continue;
        }
        if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(RegistryError::Download {
                url: url.to_string(),
                message: format!(
                    "archive entry `{}` escapes the model directory",
                    path.display()
                ),
            });
        }
        let out = dest.join(&rel);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(&ioe)?;
        }
        entry.unpack(&out).map_err(&ioe)?;
    }
    Ok(())
}

/// Line n (1-based) of `tokens.txt` becomes `<piece> <-(n-1)>`, where
/// `<piece>` is its first field: auris's `bpe.vocab` (auris/src/model.rs:258).
fn tokens_to_bpe_vocab(tokens_txt: &str) -> String {
    let mut out = String::new();
    for (i, line) in tokens_txt.lines().enumerate() {
        let Some(piece) = line.split_whitespace().next() else {
            continue;
        };
        out.push_str(piece);
        out.push(' ');
        out.push_str(&(-(i as i64)).to_string());
        out.push('\n');
    }
    out
}

/// The single moment `models/<name>` starts to exist. Called under the lock
/// once `is_installed` is false, so anything already at `dest` lacks
/// `manifest.json` and is an incomplete leftover: it is removed.
fn promote(staging: &Path, dest: &Path) -> Result<(), RegistryError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(RegistryError::io(format!("create {}", parent.display())))?;
    }
    if dest.exists() {
        std::fs::remove_dir_all(dest).map_err(RegistryError::io(format!(
            "remove incomplete {}",
            dest.display()
        )))?;
    }
    std::fs::rename(staging, dest).map_err(RegistryError::io(format!(
        "install {} to {}",
        staging.display(),
        dest.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_vocab_numbers_lines_like_auris() {
        assert_eq!(
            tokens_to_bpe_vocab("<unk> 0\n\u{2581}the 1\n\ns 3\n"),
            "<unk> 0\n\u{2581}the -1\ns -3\n"
        );
    }
}
