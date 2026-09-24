//! Registry pulls against a local HTTP stub (no network).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use naru_audio::registry::manifest::Manifest;
use naru_audio::registry::{Pulled, Registry, RegistryError};
use sha2::{Digest, Sha256};

/// Serves `files` by path over HTTP/1.1 and records every requested path.
struct Stub {
    base: String,
    hits: Arc<Mutex<Vec<String>>>,
}

impl Stub {
    fn start(files: &[(&str, Vec<u8>)], delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let files: Arc<HashMap<String, Vec<u8>>> = Arc::new(
            files
                .iter()
                .map(|(p, b)| (p.to_string(), b.clone()))
                .collect(),
        );
        let hits = Arc::new(Mutex::new(Vec::new()));
        let recorded = hits.clone();
        thread::spawn(move || {
            for conn in listener.incoming() {
                let (files, hits) = (files.clone(), recorded.clone());
                thread::spawn(move || serve(conn.unwrap(), &files, &hits, delay));
            }
        });
        Stub { base, hits }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn hits(&self) -> Vec<String> {
        self.hits.lock().unwrap().clone()
    }
}

fn serve(
    mut s: TcpStream,
    files: &HashMap<String, Vec<u8>>,
    hits: &Mutex<Vec<String>>,
    delay: Duration,
) {
    let mut reader = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap() == 0 || header == "\r\n" {
            break;
        }
    }
    hits.lock().unwrap().push(path.clone());
    thread::sleep(delay);
    match files.get(&path) {
        Some(body) => {
            write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            s.write_all(body).unwrap();
        }
        None => {
            s.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A `$NARU_AUDIO_HOME` whose `catalog.d` holds `manifests`.
fn home(manifests: &[String]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let catalog_d = dir.path().join("catalog.d");
    std::fs::create_dir_all(&catalog_d).unwrap();
    for (i, m) in manifests.iter().enumerate() {
        std::fs::write(catalog_d.join(format!("{i}.toml")), m).unwrap();
    }
    dir
}

fn model_header(name: &str, requires: &[&str]) -> String {
    format!(
        "[model]\nname = \"{name}\"\nkind = \"stt\"\nbackend = \"sherpa-onnx\"\nrequires = {requires:?}\n"
    )
}

fn file_entry(path: &str, url: &str, bytes: &[u8]) -> String {
    format!(
        "[[file]]\npath = \"{path}\"\nurl = \"{url}\"\nsha256 = \"{}\"\nsize = {}\n",
        sha(bytes),
        bytes.len()
    )
}

/// Every path under `dir`, relative, sorted; empty when `dir` is absent.
fn listing(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd {
            let p = e.unwrap().path();
            out.push(p.strip_prefix(root).unwrap().display().to_string());
            if p.is_dir() {
                walk(root, &p, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// Nothing in `models/`, and nothing in `tmp/` but lock files.
fn assert_nothing_left(home: &Path) {
    assert_eq!(listing(&home.join("models")), Vec::<String>::new());
    let tmp = listing(&home.join("tmp"));
    assert!(tmp.iter().all(|p| p.ends_with(".lock")), "{tmp:?}");
}

#[test]
fn mutated_byte_fails_and_leaves_nothing_in_models() {
    let good = b"encoder weights".to_vec();
    let mut served = good.clone();
    served[3] ^= 0x01;
    let stub = Stub::start(&[("/enc.onnx", served)], Duration::ZERO);
    let dir =
        home(&[model_header("m", &[]) + &file_entry("enc.onnx", &stub.url("/enc.onnx"), &good)]);

    let reg = Registry::open(dir.path()).unwrap();
    let err = reg.pull("m").unwrap_err();
    assert!(matches!(err, RegistryError::HashMismatch { .. }), "{err}");
    assert!(!reg.is_installed("m"));
    assert_nothing_left(dir.path());
}

#[test]
fn size_mismatch_fails_and_leaves_nothing_in_models() {
    let good = b"decoder weights".to_vec();
    let stub = Stub::start(
        &[("/dec.onnx", good[..good.len() - 1].to_vec())],
        Duration::ZERO,
    );
    let dir =
        home(&[model_header("m", &[]) + &file_entry("dec.onnx", &stub.url("/dec.onnx"), &good)]);

    let err = Registry::open(dir.path()).unwrap().pull("m").unwrap_err();
    assert!(
        matches!(err, RegistryError::SizeMismatch { expected, actual, .. } if expected == actual + 1),
        "{err}"
    );
    assert_nothing_left(dir.path());
}

#[test]
fn concurrent_pulls_of_one_model_download_once() {
    let a = b"first file".to_vec();
    let b = b"second file".to_vec();
    // The delay keeps the first pull holding the lock while the second arrives.
    let stub = Stub::start(
        &[("/a.onnx", a.clone()), ("/b.txt", b.clone())],
        Duration::from_millis(300),
    );
    let dir = home(&[model_header("m", &[])
        + &file_entry("a.onnx", &stub.url("/a.onnx"), &a)
        + &file_entry("b.txt", &stub.url("/b.txt"), &b)]);

    let reg = Arc::new(Registry::open(dir.path()).unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let pulls: Vec<_> = (0..2)
        .map(|_| {
            let (reg, barrier) = (reg.clone(), barrier.clone());
            thread::spawn(move || {
                barrier.wait();
                reg.pull("m")
            })
        })
        .collect();
    let mut outcomes: Vec<Pulled> = pulls
        .into_iter()
        .map(|t| t.join().unwrap().unwrap())
        .collect();
    outcomes.sort_by_key(|p| *p == Pulled::AlreadyInstalled);
    assert_eq!(outcomes, [Pulled::Downloaded, Pulled::AlreadyInstalled]);

    let mut hits = stub.hits();
    hits.sort();
    assert_eq!(hits, ["/a.onnx", "/b.txt"]);
    let m = reg.model_dir("m");
    assert_eq!(std::fs::read(m.join("a.onnx")).unwrap(), a);
    assert_eq!(std::fs::read(m.join("b.txt")).unwrap(), b);
}

#[test]
fn manifest_missing_sha256_is_rejected() {
    let no_file_sha = model_header("m", &[])
        + "[[file]]\npath = \"enc.onnx\"\nurl = \"http://127.0.0.1:9/enc.onnx\"\nsize = 1\n";
    let err = Registry::open(home(&[no_file_sha]).path())
        .err()
        .expect("catalog.d manifest without sha256 must be rejected");
    assert!(
        matches!(&err, RegistryError::Manifest { message, .. } if message.contains("file `enc.onnx` has no sha256")),
        "{err}"
    );

    let no_archive_sha = model_header("m", &[])
        + "[[archive]]\nurl = \"http://127.0.0.1:9/m.tar.bz2\"\nstrip = 1\n"
        + &format!("[[file]]\npath = \"x\"\nsha256 = \"{}\"\n", sha(b"x"));
    let err = Manifest::parse(&no_archive_sha, "test").unwrap_err();
    assert!(err.to_string().contains("has no sha256"), "{err}");
}

#[test]
fn archive_is_extracted_with_strip_and_files_checked() {
    let model = b"onnx bytes".to_vec();
    let phontab = b"espeak data".to_vec();
    let mut tar = tar::Builder::new(bzip2::write::BzEncoder::new(
        Vec::new(),
        bzip2::Compression::fast(),
    ));
    for (path, data) in [
        ("kokoro/model.onnx", &model),
        ("kokoro/espeak-ng-data/phontab", &phontab),
        ("kokoro/README.md", &b"readme".to_vec()),
    ] {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        tar.append_data(&mut h, path, data.as_slice()).unwrap();
    }
    let archive = tar.into_inner().unwrap().finish().unwrap();

    let stub = Stub::start(&[("/kokoro.tar.bz2", archive.clone())], Duration::ZERO);
    let manifest = model_header("kokoro", &[])
        + &format!(
            "[[archive]]\nurl = \"{}\"\nsha256 = \"{}\"\nsize = {}\nstrip = 1\n",
            stub.url("/kokoro.tar.bz2"),
            sha(&archive),
            archive.len()
        )
        + &format!(
            "[[file]]\npath = \"model.onnx\"\nsha256 = \"{}\"\n",
            sha(&model)
        )
        + &format!(
            "[[file]]\npath = \"espeak-ng-data/phontab\"\nsha256 = \"{}\"\n",
            sha(&phontab)
        );
    let dir = home(&[manifest]);

    let reg = Registry::open(dir.path()).unwrap();
    assert_eq!(reg.pull("kokoro").unwrap(), Pulled::Downloaded);
    assert_eq!(
        listing(&reg.model_dir("kokoro")),
        [
            "README.md",
            "espeak-ng-data",
            "espeak-ng-data/phontab",
            "manifest.json",
            "model.onnx"
        ]
    );
    assert_eq!(
        std::fs::read(reg.model_dir("kokoro").join("model.onnx")).unwrap(),
        model
    );
}

#[test]
fn archive_file_with_wrong_sha_fails_after_extraction() {
    let mut tar = tar::Builder::new(Vec::new());
    let mut h = tar::Header::new_gnu();
    h.set_size(4);
    h.set_mode(0o644);
    h.set_cksum();
    tar.append_data(&mut h, "top/model.onnx", &b"real"[..])
        .unwrap();
    let archive = tar.into_inner().unwrap();

    let stub = Stub::start(&[("/m.tar", archive.clone())], Duration::ZERO);
    let manifest = model_header("m", &[])
        + &format!(
            "[[archive]]\nurl = \"{}\"\nsha256 = \"{}\"\nstrip = 1\n",
            stub.url("/m.tar"),
            sha(&archive)
        )
        + &format!(
            "[[file]]\npath = \"model.onnx\"\nsha256 = \"{}\"\n",
            sha(b"fake")
        );
    let dir = home(&[manifest]);

    let err = Registry::open(dir.path()).unwrap().pull("m").unwrap_err();
    assert!(matches!(err, RegistryError::HashMismatch { .. }), "{err}");
    assert_nothing_left(dir.path());
}

#[test]
fn requires_are_pulled_first() {
    let vad = b"silero".to_vec();
    let enc = b"parakeet".to_vec();
    let stub = Stub::start(
        &[("/vad.onnx", vad.clone()), ("/enc.onnx", enc.clone())],
        Duration::ZERO,
    );
    let dir = home(&[
        model_header("stt", &["vad"]) + &file_entry("enc.onnx", &stub.url("/enc.onnx"), &enc),
        model_header("vad", &[]) + &file_entry("vad.onnx", &stub.url("/vad.onnx"), &vad),
    ]);

    let reg = Registry::open(dir.path()).unwrap();
    assert_eq!(reg.pull("stt").unwrap(), Pulled::Downloaded);
    assert_eq!(stub.hits(), ["/vad.onnx", "/enc.onnx"]);
    assert!(reg.is_installed("vad") && reg.is_installed("stt"));

    assert_eq!(reg.pull("stt").unwrap(), Pulled::AlreadyInstalled);
    assert_eq!(stub.hits().len(), 2);
}

#[test]
fn requires_cycle_is_an_error() {
    let dir = home(&[model_header("a", &["b"]), model_header("b", &["a"])]);
    let err = Registry::open(dir.path()).unwrap().pull("a").unwrap_err();
    assert!(
        matches!(&err, RegistryError::RequiresCycle(chain) if chain == &["a", "b", "a"]),
        "{err}"
    );
}

#[test]
fn bpe_vocab_is_derived_and_manifest_json_written() {
    let tokens = b"<blk> 0\n\xe2\x96\x81the 1\n".to_vec();
    let stub = Stub::start(&[("/tokens.txt", tokens.clone())], Duration::ZERO);
    let dir = home(&[model_header("m", &[])
        + "[backend.sherpa-onnx]\nderive = [\"bpe.vocab\"]\n"
        + &file_entry("tokens.txt", &stub.url("/tokens.txt"), &tokens)]);

    let reg = Registry::open(dir.path()).unwrap();
    reg.pull("m").unwrap();
    let m = reg.model_dir("m");
    assert_eq!(
        std::fs::read_to_string(m.join("bpe.vocab")).unwrap(),
        "<blk> 0\n\u{2581}the -1\n"
    );
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(m.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(json["model"]["name"], "m");
    assert!(json["pulled_at"].is_string(), "{json}");
}

#[test]
fn oversized_body_is_cut_off_one_byte_past_the_pinned_size() {
    let pinned = b"four".to_vec();
    let stub = Stub::start(&[("/big.bin", vec![0u8; 1 << 20])], Duration::ZERO);
    let dir =
        home(&[model_header("m", &[]) + &file_entry("big.bin", &stub.url("/big.bin"), &pinned)]);

    let err = Registry::open(dir.path()).unwrap().pull("m").unwrap_err();
    assert!(
        matches!(
            err,
            RegistryError::SizeMismatch {
                expected: 4,
                actual: 5,
                ..
            }
        ),
        "{err}"
    );
    assert_nothing_left(dir.path());
}

#[test]
fn archive_with_symlink_is_rejected() {
    let mut tar = tar::Builder::new(bzip2::write::BzEncoder::new(
        Vec::new(),
        bzip2::Compression::fast(),
    ));
    let mut h = tar::Header::new_gnu();
    h.set_entry_type(tar::EntryType::Symlink);
    h.set_size(0);
    h.set_mode(0o777);
    tar.append_link(&mut h, "top/escape", "../../..").unwrap();
    let archive = tar.into_inner().unwrap().finish().unwrap();

    let stub = Stub::start(&[("/m.tar.bz2", archive.clone())], Duration::ZERO);
    let manifest = model_header("m", &[])
        + &format!(
            "[[archive]]\nurl = \"{}\"\nsha256 = \"{}\"\nstrip = 1\n",
            stub.url("/m.tar.bz2"),
            sha(&archive)
        )
        + &format!("[[file]]\npath = \"escape\"\nsha256 = \"{}\"\n", sha(b""));
    let dir = home(&[manifest]);

    let err = Registry::open(dir.path()).unwrap().pull("m").unwrap_err();
    assert!(err.to_string().contains("is a link"), "{err}");
    assert_nothing_left(dir.path());
}

#[test]
fn incomplete_model_dir_is_replaced() {
    let enc = b"weights".to_vec();
    let stub = Stub::start(&[("/enc.onnx", enc.clone())], Duration::ZERO);
    let dir =
        home(&[model_header("m", &[]) + &file_entry("enc.onnx", &stub.url("/enc.onnx"), &enc)]);
    let leftover = dir.path().join("models/m");
    std::fs::create_dir_all(&leftover).unwrap();
    std::fs::write(leftover.join("junk"), b"junk").unwrap();

    let reg = Registry::open(dir.path()).unwrap();
    assert_eq!(reg.pull("m").unwrap(), Pulled::Downloaded);
    assert_eq!(listing(&reg.model_dir("m")), ["enc.onnx", "manifest.json"]);
}

#[test]
fn model_name_ending_in_lock_is_rejected() {
    let err = Manifest::parse(&model_header("m.lock", &[]), "test").unwrap_err();
    assert!(err.to_string().contains("must not end in `.lock`"), "{err}");
}
