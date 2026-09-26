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
//! installed and does nothing. `remove` and `verify` take the same lock.
//! Before promoting, a pull also takes the lock of each model its manifest
//! `requires` and checks it is still installed, holding those locks through
//! the rename, so a concurrent `rm` of a requirement either fails the pull
//! or sees the new model and refuses. Locks are only ever taken from a model
//! to its `requires` (never the reverse), and `requires` has no cycles, so
//! this cannot deadlock.

pub mod manifest;

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// In the catalog, but not under `models/`.
    NotPulled(String),
    /// §3.6: `pull` refuses a model whose backend cannot run here.
    BackendUnavailable {
        name: String,
        backend: String,
        reason: String,
    },
    /// §3.2: `rm` refuses a model that other pulled models `require`.
    RequiredBy {
        name: String,
        by: Vec<String>,
    },
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
            RegistryError::NotPulled(name) => write!(f, "model `{name}` is not pulled"),
            RegistryError::BackendUnavailable {
                name,
                backend,
                reason,
            } => write!(
                f,
                "model `{name}` needs backend {backend}, which is unavailable: {reason}"
            ),
            RegistryError::RequiredBy { name, by } => write!(
                f,
                "model `{name}` is required by {}; remove those first or use --force",
                by.join(", ")
            ),
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

/// Pull progress, the §2.5 `/api/pull` NDJSON lines before `success`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress<'a> {
    /// `file` is a `[[file]]` path or an archive's base name; `total` is the
    /// pinned size, if any.
    Downloading {
        file: &'a str,
        completed: u64,
        total: Option<u64>,
    },
    /// The download is complete; its size and sha256 are being checked.
    Verifying,
}

/// Progress is reported at most once per this many bytes, plus at each end.
const PROGRESS_STEP: u64 = 1 << 20;

/// One `list` / `GET /v1/models` row.
#[derive(Debug)]
pub struct Entry {
    pub manifest: Manifest,
    pub available: Result<(), String>,
    pub pulled: bool,
    /// Pulled: `pulled_at` as Unix seconds; otherwise 0.
    pub created: u64,
    /// Pulled: bytes on disk. Otherwise the download size, when every
    /// download pins one.
    pub size_bytes: Option<u64>,
}

/// `$NARU_AUDIO_HOME`, else `~/.naru-audio` (§3.1).
pub fn default_home() -> Option<PathBuf> {
    match std::env::var_os("NARU_AUDIO_HOME") {
        Some(h) if !h.is_empty() => Some(h.into()),
        _ => std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".naru-audio")),
    }
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

    /// `$NARU_AUDIO_HOME` (§3.1).
    pub fn home(&self) -> &Path {
        &self.home
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
        self.pull_with(name, false, &mut |_| {})
    }

    /// `pull`, reporting progress. `force` pulls models whose backend is
    /// unavailable here (§3.6).
    pub fn pull_with(
        &self,
        name: &str,
        force: bool,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Pulled, RegistryError> {
        let mut pulled = Pulled::AlreadyInstalled;
        for m in self.plan(name, force)? {
            pulled = self.install(m, progress)?;
        }
        Ok(pulled)
    }

    /// The errors `pull_with` would raise before downloading anything.
    pub fn check_pull(&self, name: &str, force: bool) -> Result<(), RegistryError> {
        self.plan(name, force).map(|_| ())
    }

    /// `name` and its `requires`, in install order, each with an available
    /// backend unless `force`.
    fn plan(&self, name: &str, force: bool) -> Result<Vec<&Manifest>, RegistryError> {
        let mut order = Vec::new();
        self.resolve(name, &mut Vec::new(), &mut order)?;
        if !force {
            for m in &order {
                crate::backend::available(&m.model.backend).map_err(|reason| {
                    RegistryError::BackendUnavailable {
                        name: m.model.name.clone(),
                        backend: m.model.backend.clone(),
                        reason,
                    }
                })?;
            }
        }
        Ok(order)
    }

    /// Pulled model names, sorted.
    pub fn pulled(&self) -> Result<Vec<String>, RegistryError> {
        let models = self.home.join("models");
        let entries = match std::fs::read_dir(&models) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(RegistryError::io(format!("read {}", models.display()))(e)),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(RegistryError::io(format!("read {}", models.display())))?;
            if let Some(name) = entry.file_name().to_str()
                && self.is_installed(name)
            {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    /// The manifest a pulled model was installed with, from its `manifest.json`.
    pub fn pulled_manifest(&self, name: &str) -> Result<Manifest, RegistryError> {
        let path = self.model_dir(name).join(MANIFEST_JSON);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(self.not_pulled(name));
            }
            Err(e) => return Err(RegistryError::io(format!("read {}", path.display()))(e)),
        };
        let origin = path.display().to_string();
        let raw: toml::Table =
            serde_json::from_slice(&bytes).map_err(|e| RegistryError::Manifest {
                origin: origin.clone(),
                message: e.to_string(),
            })?;
        Manifest::from_table(raw, &origin)
    }

    /// `NotPulled` for a catalog model, `UnknownModel` otherwise.
    fn not_pulled(&self, name: &str) -> RegistryError {
        if self.catalog.models.contains_key(name) {
            RegistryError::NotPulled(name.to_string())
        } else {
            RegistryError::UnknownModel(name.to_string())
        }
    }

    /// Pulled models plus catalog models that can run on this machine
    /// (§2.5), sorted by name. A pulled model is described by its own
    /// `manifest.json`, even if the catalog has since changed. A pulled model
    /// whose `manifest.json` cannot be read is left out with a warning.
    pub fn list(&self) -> Result<Vec<Entry>, RegistryError> {
        let mut entries = std::collections::BTreeMap::new();
        let mut unreadable = std::collections::BTreeSet::new();
        for name in self.pulled()? {
            let manifest = match self.pulled_manifest(&name) {
                Ok(m) => m,
                // Removed since `pulled()` looked.
                Err(RegistryError::NotPulled(_) | RegistryError::UnknownModel(_)) => continue,
                Err(e) => {
                    warn_unreadable(&name, &e);
                    unreadable.insert(name);
                    continue;
                }
            };
            let created = manifest
                .raw
                .get("pulled_at")
                .and_then(|v| v.as_str())
                .and_then(|s| humantime::parse_rfc3339(s).ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            let dir = self.model_dir(&name);
            let size =
                dir_size(&dir).map_err(RegistryError::io(format!("read {}", dir.display())))?;
            entries.insert(
                name,
                Entry {
                    available: crate::backend::available(&manifest.model.backend),
                    manifest,
                    pulled: true,
                    created,
                    size_bytes: Some(size),
                },
            );
        }
        for (name, m) in &self.catalog.models {
            // An unreadable pulled model is still pulled: not listed as pullable.
            if entries.contains_key(name) || unreadable.contains(name) || !m.runs_here() {
                continue;
            }
            let downloads = m.archives.iter().map(|a| a.size);
            let downloads =
                downloads.chain(m.files.iter().filter(|f| f.url.is_some()).map(|f| f.size));
            entries.insert(
                name.clone(),
                Entry {
                    available: crate::backend::available(&m.model.backend),
                    manifest: m.clone(),
                    pulled: false,
                    created: 0,
                    size_bytes: downloads.sum(),
                },
            );
        }
        Ok(entries.into_values().collect())
    }

    /// Removes a pulled model. Unless `force`, refuses while another pulled
    /// model `requires` it (§3.2); models named in `also_removing` do not
    /// count, and neither do pulled models whose `manifest.json` cannot be
    /// read (with a warning).
    pub fn remove(
        &self,
        name: &str,
        force: bool,
        also_removing: &[String],
    ) -> Result<(), RegistryError> {
        let _lock = self.lock_pulled(name)?;
        if !force {
            let mut by = Vec::new();
            for other in self.pulled()? {
                if other == name || also_removing.contains(&other) {
                    continue;
                }
                match self.pulled_manifest(&other) {
                    Ok(m) if m.model.requires.iter().any(|r| r == name) => by.push(other),
                    Ok(_) | Err(RegistryError::NotPulled(_) | RegistryError::UnknownModel(_)) => {}
                    Err(e) => warn_unreadable(&other, &e),
                }
            }
            if !by.is_empty() {
                return Err(RegistryError::RequiredBy {
                    name: name.to_string(),
                    by,
                });
            }
        }
        // Dropping `manifest.json` first uninstalls the model in one step;
        // a leftover directory is what `promote` already replaces.
        let dir = self.model_dir(name);
        let json = dir.join(MANIFEST_JSON);
        std::fs::remove_file(&json)
            .map_err(RegistryError::io(format!("remove {}", json.display())))?;
        std::fs::remove_dir_all(&dir)
            .map_err(RegistryError::io(format!("remove {}", dir.display())))
    }

    /// Re-hashes every `[[file]]` of a pulled model against the manifest it
    /// was installed with (§3.2 `naru-audio verify`).
    pub fn verify(&self, name: &str) -> Result<(), RegistryError> {
        let _lock = self.lock_pulled(name)?;
        let m = self.pulled_manifest(name)?;
        let dir = self.model_dir(name);
        for f in &m.files {
            let path = dir.join(&f.path);
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(RegistryError::MissingFile(f.path.clone()));
                }
                Err(e) => return Err(RegistryError::io(format!("open {}", path.display()))(e)),
            };
            let got = hash_copy(file, std::io::sink(), &mut |_| {})
                .map_err(RegistryError::io(format!("read {}", path.display())))?;
            check(&f.path, f.size, sha(&f.sha256), got)?;
        }
        Ok(())
    }

    /// `lock` for a model that is pulled, checked before and after locking.
    /// Matching `name` against `models/` first keeps a name like `../x`
    /// from reaching the filesystem.
    fn lock_pulled(&self, name: &str) -> Result<File, RegistryError> {
        if !self.pulled()?.iter().any(|n| n == name) {
            return Err(self.not_pulled(name));
        }
        let lock = self.lock(name)?;
        if !self.is_installed(name) {
            return Err(self.not_pulled(name));
        }
        Ok(lock)
    }

    /// The per-model advisory lock, held until the returned file is dropped.
    fn lock(&self, name: &str) -> Result<File, RegistryError> {
        let tmp = self.home.join("tmp");
        std::fs::create_dir_all(&tmp)
            .map_err(RegistryError::io(format!("create {}", tmp.display())))?;
        // Never unlinked: removing a lock file races with a waiter that has
        // it open.
        let lock_path = tmp.join(format!("{name}.lock"));
        let lock = File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(RegistryError::io(format!("open {}", lock_path.display())))?;
        lock.lock()
            .map_err(RegistryError::io(format!("lock {}", lock_path.display())))?;
        Ok(lock)
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

    fn install(
        &self,
        m: &Manifest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Pulled, RegistryError> {
        let name = &m.model.name;
        if self.is_installed(name) {
            return Ok(Pulled::AlreadyInstalled);
        }

        let _lock = self.lock(name)?;
        let tmp = self.home.join("tmp");

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

        let result = stage(m, &staging, progress).and_then(|()| {
            // A `rm` of a requirement may have run while we staged; its locks
            // are held until `promote` is done (see the module doc).
            let _requires = m
                .model
                .requires
                .iter()
                .map(|r| self.lock_pulled(r))
                .collect::<Result<Vec<_>, _>>()?;
            promote(&staging, &self.model_dir(name))
        });
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result.map(|()| Pulled::Downloaded)
    }
}

/// Fills `staging` with the verified model: archives, then files, then
/// derived files, then `manifest.json`.
fn stage(
    m: &Manifest,
    staging: &Path,
    progress: &mut dyn FnMut(Progress),
) -> Result<(), RegistryError> {
    for a in &m.archives {
        let base = a.url.rsplit('/').next().unwrap_or("archive");
        let part = staging.join(format!("{base}.part"));
        download(&a.url, base, &part, a.size, sha(&a.sha256), progress)?;
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
                download(url, &f.path, &part, f.size, sha(&f.sha256), progress)?;
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
                let got = hash_copy(file, std::io::sink(), &mut |_| {})
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
/// `part` is removed. Progress names the download `file`.
fn download(
    url: &str,
    file: &str,
    part: &Path,
    size: Option<u64>,
    sha256: &str,
    progress: &mut dyn FnMut(Progress),
) -> Result<(), RegistryError> {
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
        let report = |completed| Progress::Downloading {
            file,
            completed,
            total: size,
        };
        progress(report(0));
        let mut reported = 0;
        let got = hash_copy(reader, out, &mut |n| {
            if n - reported >= PROGRESS_STEP {
                reported = n;
                progress(report(n));
            }
        })
        .map_err(|e| RegistryError::Download {
            url: url.to_string(),
            message: e.to_string(),
        })?;
        if got.0 != reported {
            progress(report(got.0));
        }
        progress(Progress::Verifying);
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
/// `on_chunk` gets the running byte count after each chunk.
fn hash_copy(
    mut r: impl Read,
    mut w: impl Write,
    on_chunk: &mut dyn FnMut(u64),
) -> std::io::Result<(u64, String)> {
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
        on_chunk(n_total);
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

/// The registry has no logger; stderr reaches the terminal for the CLI and
/// the daemon's log under `brew services`.
fn warn_unreadable(name: &str, e: &RegistryError) {
    eprintln!("naru-audio: warning: skipping pulled model `{name}`: {e}");
}

/// Total size of the regular files under `dir`.
fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        total += if meta.is_dir() {
            dir_size(&entry.path())?
        } else {
            meta.len()
        };
    }
    Ok(total)
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
