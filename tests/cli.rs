use std::process::Command;

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
