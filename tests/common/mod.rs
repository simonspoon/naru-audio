//! Shared by the integration tests: a local HTTP stub (no network) and
//! `$NARU_AUDIO_HOME` builders. Each test crate uses a different subset.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};

/// Serves `files` by path over HTTP/1.1 and records every requested path.
pub struct Stub {
    base: String,
    hits: Arc<Mutex<Vec<String>>>,
}

impl Stub {
    pub fn start(files: &[(&str, Vec<u8>)], delay: Duration) -> Self {
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

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub fn hits(&self) -> Vec<String> {
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

pub fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A `$NARU_AUDIO_HOME` whose `catalog.d` holds `manifests`.
pub fn home(manifests: &[String]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let catalog_d = dir.path().join("catalog.d");
    std::fs::create_dir_all(&catalog_d).unwrap();
    for (i, m) in manifests.iter().enumerate() {
        std::fs::write(catalog_d.join(format!("{i}.toml")), m).unwrap();
    }
    dir
}

/// A `file:///` URL for `path`; Windows paths become `file:///C:/dir/x`.
pub fn file_url(path: &std::path::Path) -> String {
    let p = path.to_string_lossy().replace('\\', "/");
    if p.starts_with('/') {
        format!("file://{p}")
    } else {
        format!("file:///{p}")
    }
}

pub fn model_header(name: &str, requires: &[&str]) -> String {
    format!(
        "[model]\nname = \"{name}\"\nkind = \"stt\"\nbackend = \"sherpa-onnx\"\nrequires = {requires:?}\n"
    )
}

pub fn file_entry(path: &str, url: &str, bytes: &[u8]) -> String {
    format!(
        "[[file]]\npath = \"{path}\"\nurl = \"{url}\"\nsha256 = \"{}\"\nsize = {}\n",
        sha(bytes),
        bytes.len()
    )
}
