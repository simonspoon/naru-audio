//! naru task 2028: speaker verification against recorded fixtures
//! (`tests/fixtures/speaker/README.md`). Model-dependent tests skip (eprintln
//! and return) when `wespeaker-sv-en` is not pulled under
//! `$NARU_AUDIO_TEST_HOME`, else `$NARU_AUDIO_HOME` or `~/.naru-audio`;
//! `NARU_AUDIO_REQUIRE_MODELS=1` turns the skip into a failure.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use naru_audio::log::Logger;
use naru_audio::manager::{BackendLoader, ModelManager, Settings};
use naru_audio::profile::Profile;
use naru_audio::server::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

use naru_audio::registry::{self, Registry};
use naru_audio::speaker::{
    self, DEFAULT_MODEL, DEFAULT_THRESHOLD, Enrollment, MIN_SAMPLES, Reason, SpeakerVerifier,
};
use naru_audio::stt::audio;

fn samples(name: &str) -> Vec<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/speaker")
        .join(name);
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
    audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
}

/// Loaded once and shared; `None` (skip) when the model is not pulled.
fn verifier() -> Option<&'static SpeakerVerifier> {
    static CELL: OnceLock<Option<SpeakerVerifier>> = OnceLock::new();
    CELL.get_or_init(|| {
        let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
            .map(PathBuf::from)
            .or_else(registry::default_home)?;
        let registry = Registry::open(&home).expect("open registry");
        if !registry.is_installed(DEFAULT_MODEL) {
            let message = format!(
                "{DEFAULT_MODEL} not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
                home.display()
            );
            if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
                panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
            }
            eprintln!("skip: {message}");
            return None;
        }
        Some(SpeakerVerifier::load(&registry.model_dir(DEFAULT_MODEL)).expect("load"))
    })
    .as_ref()
}

fn score(v: &SpeakerVerifier, enrolled: &Enrollment, clip: &str) -> f32 {
    let pcm = samples(clip);
    let e = v.embed(&pcm).expect("embed");
    speaker::cosine(&e, &enrolled.embedding)
}

/// Fixture scores (WeSpeaker, TTS voices Daniel = A, Samantha = B): same
/// voice 0.94-0.99, different voice 0.59-0.62. `DEFAULT_THRESHOLD` sits in
/// that gap, nearer the low end so a real voice's noise rejects him less.
#[test]
fn same_voice_scores_above_a_different_voice() {
    let Some(v) = verifier() else { return };
    let enrolled = v.enroll(&[samples("a1.wav")]).unwrap();
    let same = score(v, &enrolled, "a2.wav").min(score(v, &enrolled, "a3.wav"));
    let different = score(v, &enrolled, "b1.wav").max(score(v, &enrolled, "b2.wav"));
    eprintln!("same {same:.3} different {different:.3}");
    assert!(same - different > 0.0);
    assert!(different < DEFAULT_THRESHOLD && DEFAULT_THRESHOLD < same);
}

#[test]
fn enrolled_voice_verifies_and_the_other_does_not() {
    let Some(v) = verifier() else { return };
    let enrolled = v.enroll(&[samples("a1.wav"), samples("a2.wav")]).unwrap();
    assert_eq!(enrolled.count, 2);
    for (clip, him) in [("a3.wav", true), ("b1.wav", false), ("b3.wav", false)] {
        let verdict = speaker::verify(Some(v), Some(&enrolled), &samples(clip), DEFAULT_THRESHOLD);
        assert_eq!(verdict.is_him, him, "{clip}: {verdict:?}");
        assert_eq!(verdict.reason, Reason::Scored);
    }
}

#[test]
fn several_clips_beat_one() {
    let Some(v) = verifier() else { return };
    let one = v.enroll(&[samples("a1.wav")]).unwrap();
    let two = v.enroll(&[samples("a1.wav"), samples("a2.wav")]).unwrap();
    // a3 is held out of both enrollments
    let (s1, s2) = (score(v, &one, "a3.wav"), score(v, &two, "a3.wav"));
    eprintln!("held-out a3: one clip {s1:.3}, two clips {s2:.3}");
    assert!(s2 > s1);
}

#[test]
fn short_reply_and_missing_enrollment_fail_open() {
    let Some(v) = verifier() else { return };
    let enrolled = v.enroll(&[samples("a1.wav")]).unwrap();
    // "Yes." in the other voice would be rejected if it were scored
    let mut short = samples("b1.wav");
    short.truncate(MIN_SAMPLES - 1);
    let verdict = speaker::verify(Some(v), Some(&enrolled), &short, DEFAULT_THRESHOLD);
    assert!(verdict.is_him && verdict.reason == Reason::TooShort);
    let verdict = speaker::verify(Some(v), None, &samples("b1.wav"), DEFAULT_THRESHOLD);
    assert!(verdict.is_him && verdict.reason == Reason::NoEnrollment);
    let yes = samples("a_short.wav");
    assert!(speaker::verify(Some(v), Some(&enrolled), &yes, DEFAULT_THRESHOLD).is_him);
}

#[test]
fn enrollment_from_unusable_clips_is_refused() {
    let Some(v) = verifier() else { return };
    let mut short = samples("a1.wav");
    short.truncate(MIN_SAMPLES - 1);
    assert!(v.enroll(&[short]).is_err());
    assert!(v.enroll(&[]).is_err());
}

// ---- HTTP routes --------------------------------------------------------

const BOUNDARY: &str = "naru-audio-speaker-test-boundary";

/// A fresh router over `home`, as a restarted daemon would build.
fn app(home: &Path) -> Router {
    let log = Arc::new(Logger::stderr());
    let registry = Arc::new(Registry::open(home).unwrap());
    let settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
    let models = ModelManager::new(
        registry.clone(),
        log.clone(),
        settings,
        Arc::new(BackendLoader),
    );
    router(Arc::new(AppState {
        port: 7870,
        allow_remote: false,
        started: Instant::now(),
        log,
        registry,
        models: Arc::new(models),
        pulls: Arc::new(naru_audio::server::PullTracker::new()),
    }))
}

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/speaker")
            .join(name),
    )
    .unwrap()
}

async fn call(home: &Path, method: &str, path: &str, files: &[Vec<u8>]) -> (StatusCode, Value) {
    let mut body = Vec::new();
    for f in files {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
                 Content-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(f);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(HOST, "127.0.0.1:7870")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap();
    let resp = app(home).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn routes_without_the_model_use_the_documented_errors() {
    let home = tempfile::tempdir().unwrap();
    // not pulled: 409 model_not_pulled
    let (status, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/enroll",
        &[fixture_bytes("a1.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    // no enrollment: fails open without needing the model
    let (status, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("b1.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["is_him"], true);
    assert_eq!(body["reason"], "no_enrollment");
    assert_eq!(body["score"], Value::Null);
    // an enrollment but no model: a clip that needs scoring is 409 ...
    Enrollment::from_embeddings(DEFAULT_MODEL, &[vec![1.0, 0.0]])
        .unwrap()
        .save(home.path())
        .unwrap();
    let (status, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("b1.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    // ... while a short reply is still him
    let (status, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("a_short.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (body["is_him"].clone(), body["reason"].clone()),
        (json!(true), json!("too_short"))
    );
    // missing file and bad threshold are 400
    let (status, body) = call(home.path(), "POST", "/v1/audio/speaker/verify", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "file");
}

#[tokio::test]
async fn enrollment_survives_a_restart_and_deletes() {
    let Some(_) = verifier() else { return };
    let real = registry::default_home().unwrap();
    let real = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .unwrap_or(real);
    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(home.path().join("models")).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        real.join("models").join(DEFAULT_MODEL),
        home.path().join("models").join(DEFAULT_MODEL),
    )
    .unwrap();
    #[cfg(not(unix))]
    return;

    let (status, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/enroll",
        &[fixture_bytes("a1.wav"), fixture_bytes("a2.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["count"], 2);
    // every call builds a fresh router: the enrollment is read off disk
    let (_, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("a3.wav")],
    )
    .await;
    assert_eq!(
        (body["is_him"].clone(), body["reason"].clone()),
        (json!(true), json!("scored"))
    );
    let (_, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("b3.wav")],
    )
    .await;
    assert_eq!(body["is_him"], false, "{body}");
    // the score is reported beside the threshold it was judged against
    assert!(body["score"].as_f64().unwrap() < body["threshold"].as_f64().unwrap());

    let (status, body) = call(home.path(), "DELETE", "/v1/audio/speaker/enrollment", &[]).await;
    assert_eq!(
        (status, body["deleted"].clone()),
        (StatusCode::OK, json!(true))
    );
    let (_, body) = call(
        home.path(),
        "POST",
        "/v1/audio/speaker/verify",
        &[fixture_bytes("b3.wav")],
    )
    .await;
    assert_eq!(body["reason"], "no_enrollment");
}
