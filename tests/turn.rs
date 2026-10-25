//! naru task 2355: Smart Turn end-of-turn detection on recorded fixtures
//! (`tests/fixtures/stt/turn-*.wav`: a finished sentence, a sentence cut
//! mid-way, and a bare "yes"). Skips (eprintln and return) when
//! `smart-turn-v3.2` is not pulled under `$NARU_AUDIO_TEST_HOME`, else
//! `$NARU_AUDIO_HOME` or `~/.naru-audio`; `NARU_AUDIO_REQUIRE_MODELS=1` turns
//! the skip into a failure. macOS only: elsewhere the detector is always absent.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use naru_audio::registry::{self, Registry};
use naru_audio::stt::audio;
use naru_audio::turn::{self, EndOfTurn};

fn samples(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(name);
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
    audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
}

fn detector() -> Option<Arc<dyn EndOfTurn>> {
    let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .or_else(registry::default_home)?;
    let registry = Registry::open(&home).expect("open registry");
    let found = turn::shared(&registry.model_dir(turn::DEFAULT_MODEL));
    if found.is_none() {
        let message = format!(
            "{} not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
            turn::DEFAULT_MODEL,
            home.display()
        );
        if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
            panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
        }
        eprintln!("skip: {message}");
    }
    found
}

#[test]
fn smart_turn_tells_finished_turns_from_unfinished_ones() {
    let Some(d) = detector() else { return };
    for (name, finished) in [
        ("turn-complete.wav", true),
        ("turn-midsentence.wav", false),
        ("turn-yes.wav", true),
    ] {
        let began = Instant::now();
        let p = d.complete(&samples(name)).expect("an answer");
        eprintln!("{name}: p={p:.4} in {} ms", began.elapsed().as_millis());
        assert_eq!(p >= turn::THRESHOLD, finished, "{name}: p={p}");
    }
}

#[test]
fn audio_longer_than_8_s_is_read_from_its_end() {
    let Some(d) = detector() else { return };
    let finished = samples("turn-complete.wav");
    let mut long = samples("turn-midsentence.wav");
    long.extend(&finished);
    long.extend(&finished);
    long.extend(&finished);
    long.extend(&finished);
    assert!(long.len() > turn::WINDOW);
    assert!(d.complete(&long).expect("an answer") >= turn::THRESHOLD);
}
