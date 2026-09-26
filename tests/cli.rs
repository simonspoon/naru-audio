mod common;

use std::path::Path;
use std::process::{Command, Output};

use common::{file_entry, home, model_header};

#[test]
fn non_loopback_listen_without_allow_remote_exits_non_zero() {
    let out = Command::new(env!("CARGO_BIN_EXE_naru-audio"))
        .args(["serve", "--listen", "0.0.0.0:0"])
        .env_remove("NARU_AUDIO_LISTEN")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("non-loopback") && stderr.contains("--allow-remote"),
        "{stderr}"
    );
}

fn run(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_naru-audio"))
        .args(args)
        .env("NARU_AUDIO_HOME", home)
        .output()
        .unwrap()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A home whose catalog has `stt` (requiring `vad`) served from `file:///`
/// sources in `src`, and `remote`, which points at a closed port.
fn stt_and_vad(src: &Path) -> tempfile::TempDir {
    let mut manifests = Vec::new();
    for (name, requires, file, bytes) in [
        ("vad", &[][..], "vad.onnx", &b"silero"[..]),
        ("stt", &["vad"][..], "enc.onnx", &b"parakeet"[..]),
    ] {
        let path = src.join(file);
        std::fs::write(&path, bytes).unwrap();
        let url = format!("file://{}", path.display());
        manifests.push(model_header(name, requires) + &file_entry(file, &url, bytes));
    }
    manifests.push(
        model_header("remote", &[]) + &file_entry("r.onnx", "http://127.0.0.1:9/r.onnx", b"r"),
    );
    home(&manifests)
}

#[test]
fn list_names_prints_bare_pulled_names_offline() {
    let src = tempfile::tempdir().unwrap();
    let home = stt_and_vad(src.path());
    let empty = run(home.path(), &["list", "--names"]);
    assert!(empty.status.success(), "{}", stderr(&empty));
    assert_eq!(stdout(&empty), "");

    let pull = run(home.path(), &["pull", "stt"]);
    assert!(pull.status.success(), "{}", stderr(&pull));
    // Nothing left to fetch from: listing must not need any source.
    drop(src);

    let out = run(home.path(), &["list", "--names"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out), "stt\nvad\n");

    let table = run(home.path(), &["list"]);
    assert!(table.status.success(), "{}", stderr(&table));
    assert!(stdout(&table).contains("remote"), "{}", stdout(&table));
}

/// Naru 1441: pulling a non-commercial model warns on stderr, since `pull`'s
/// stdout ("pulled <name>") is meant to be scripted; `list` tags it too.
#[test]
fn non_commercial_model_warns_on_pull_and_is_tagged_in_list() {
    let src = tempfile::tempdir().unwrap();
    let path = src.path().join("m.onnx");
    std::fs::write(&path, b"weights").unwrap();
    let manifest = "[model]\nname = \"nc\"\nkind = \"tts\"\nbackend = \"sherpa-onnx\"\n\
        license = \"CC-BY-NC-4.0\"\n\
        license_url = \"https://creativecommons.org/licenses/by-nc/4.0/\"\n\
        non_commercial = true\n"
        .to_string()
        + &file_entry("m.onnx", &format!("file://{}", path.display()), b"weights");
    let home = home(&[manifest]);

    let pull = run(home.path(), &["pull", "nc"]);
    assert!(pull.status.success(), "{}", stderr(&pull));
    assert_eq!(stdout(&pull), "pulled nc\n");
    let warning = stderr(&pull);
    assert!(warning.contains("non-commercial"), "{warning}");
    assert!(warning.contains("CC-BY-NC-4.0"), "{warning}");

    let table = stdout(&run(home.path(), &["list"]));
    assert!(
        table
            .lines()
            .any(|l| l.starts_with("nc ") && l.contains("CC-BY-NC-4.0 (non-commercial)")),
        "{table}"
    );
}

/// A pull that fails (here, an unavailable backend, as in
/// `pull_refuses_an_unavailable_backend_without_force`) must not print the
/// non-commercial warning: there is nothing pulled to warn about.
#[test]
fn failed_pull_of_a_non_commercial_model_prints_no_warning() {
    let src = tempfile::tempdir().unwrap();
    let path = src.path().join("m.onnx");
    std::fs::write(&path, b"weights").unwrap();
    let manifest = "[model]\nname = \"nc\"\nkind = \"tts\"\nbackend = \"nope\"\n\
        license = \"CC-BY-NC-4.0\"\n\
        license_url = \"https://creativecommons.org/licenses/by-nc/4.0/\"\n\
        non_commercial = true\n"
        .to_string()
        + &file_entry("m.onnx", &format!("file://{}", path.display()), b"weights");
    let home = home(&[manifest]);

    let refused = run(home.path(), &["pull", "nc"]);
    assert!(!refused.status.success());
    let stderr = stderr(&refused);
    assert!(stderr.contains("unknown backend `nope`"), "{stderr}");
    assert!(!stderr.contains("non-commercial"), "{stderr}");
}

#[test]
fn rm_of_required_model_needs_force() {
    let src = tempfile::tempdir().unwrap();
    let home = stt_and_vad(src.path());
    assert!(run(home.path(), &["pull", "stt"]).status.success());

    let refused = run(home.path(), &["rm", "vad"]);
    assert!(!refused.status.success());
    assert!(
        stderr(&refused).contains("required by stt"),
        "{}",
        stderr(&refused)
    );
    assert_eq!(
        stdout(&run(home.path(), &["list", "--names"])),
        "stt\nvad\n"
    );

    let forced = run(home.path(), &["rm", "vad", "--force"]);
    assert!(forced.status.success(), "{}", stderr(&forced));
    assert_eq!(stdout(&run(home.path(), &["list", "--names"])), "stt\n");

    // Nothing requires stt; an unknown or unpulled name fails.
    assert!(run(home.path(), &["rm", "stt"]).status.success());
    assert!(!run(home.path(), &["rm", "stt"]).status.success());
    assert!(!run(home.path(), &["rm", "../models"]).status.success());
    assert!(home.path().join("models").is_dir());
}

#[test]
fn verify_fails_on_a_tampered_file() {
    let src = tempfile::tempdir().unwrap();
    let home = stt_and_vad(src.path());
    assert!(run(home.path(), &["pull", "stt"]).status.success());

    let ok = run(home.path(), &["verify"]);
    assert!(ok.status.success(), "{}", stderr(&ok));
    assert_eq!(stdout(&ok), "stt ok\nvad ok\n");

    std::fs::write(home.path().join("models/vad/vad.onnx"), b"silerO").unwrap();
    let bad = run(home.path(), &["verify", "vad"]);
    assert!(!bad.status.success());
    assert!(
        stderr(&bad).contains("vad.onnx: sha256 is"),
        "{}",
        stderr(&bad)
    );
    std::fs::write(home.path().join("models/vad/vad.onnx"), b"short").unwrap();
    let short = run(home.path(), &["verify"]);
    assert!(!short.status.success());
    assert!(
        stderr(&short).contains("expected 6 bytes, got 5"),
        "{}",
        stderr(&short)
    );
    assert!(stdout(&short).contains("stt ok"));
}

#[test]
fn pull_refuses_an_unavailable_backend_without_force() {
    let src = tempfile::tempdir().unwrap();
    let path = src.path().join("m.onnx");
    std::fs::write(&path, b"weights").unwrap();
    let manifest = model_header("m", &[]).replace("sherpa-onnx", "nope")
        + &file_entry("m.onnx", &format!("file://{}", path.display()), b"weights");
    let home = home(&[manifest]);

    let refused = run(home.path(), &["pull", "m"]);
    assert!(!refused.status.success());
    assert!(
        stderr(&refused).contains("unknown backend `nope`"),
        "{}",
        stderr(&refused)
    );
    assert_eq!(stdout(&run(home.path(), &["list", "--names"])), "");

    let forced = run(home.path(), &["pull", "m", "--force"]);
    assert!(forced.status.success(), "{}", stderr(&forced));
    assert_eq!(stdout(&run(home.path(), &["list", "--names"])), "m\n");
}

#[test]
fn rm_of_a_model_and_its_dependent_in_one_command_succeeds() {
    let src = tempfile::tempdir().unwrap();
    let home = stt_and_vad(src.path());
    assert!(run(home.path(), &["pull", "stt"]).status.success());

    let out = run(home.path(), &["rm", "vad", "stt"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out), "removed vad\nremoved stt\n");
    assert_eq!(stdout(&run(home.path(), &["list", "--names"])), "");
}

#[test]
fn list_skips_an_unreadable_manifest_json() {
    let src = tempfile::tempdir().unwrap();
    let home = stt_and_vad(src.path());
    assert!(run(home.path(), &["pull", "stt"]).status.success());
    std::fs::write(home.path().join("models/stt/manifest.json"), b"{not json").unwrap();

    let out = run(home.path(), &["list"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("skipping pulled model `stt`"),
        "{}",
        stderr(&out)
    );
    let table = stdout(&out);
    assert!(table.lines().any(|l| l.starts_with("vad ")), "{table}");
    assert!(!table.lines().any(|l| l.starts_with("stt ")), "{table}");
}
