//! naru task 2028: speaker verification against recorded fixtures
//! (`tests/fixtures/speaker/README.md`). Model-dependent tests skip (eprintln
//! and return) when `campplus-sv-en` is not pulled under
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
    self, Clip, DEFAULT_MODEL, DEFAULT_THRESHOLD, Identity, Library, MIN_SAMPLES, Reason,
    SpeakerVerifier,
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

fn embed(v: &SpeakerVerifier, clip: &str) -> Vec<f32> {
    v.embed(&samples(clip)).expect("embed")
}

/// A library of `(name, owner, clip fixtures)` people, embedded with `v`.
fn library(v: &SpeakerVerifier, people: &[(&str, bool, &[&str])]) -> Library {
    let mut lib = Library::default();
    for (name, owner, clips) in people {
        let id = lib.add_person(name, *owner);
        let person = lib.people.iter_mut().find(|p| p.id == id).unwrap();
        for (i, clip) in clips.iter().enumerate() {
            person.clips.push(Clip {
                id: (id * 100) + i as u64,
                created_at: 0,
                seconds: 1.0,
                embedding: embed(v, clip),
                has_audio: false,
            });
        }
    }
    lib
}

fn verify(v: &SpeakerVerifier, lib: &Library, only_owner: bool, clip: &str) -> Identity {
    speaker::identify(
        Some(v),
        Some(lib),
        only_owner,
        &samples(clip),
        DEFAULT_THRESHOLD,
    )
}

type Method = fn(&[&Vec<f32>], &[f32]) -> f32;

fn max_cosine(clips: &[&Vec<f32>], e: &[f32]) -> f32 {
    clips
        .iter()
        .map(|c| speaker::cosine(e, c))
        .fold(f32::MIN, f32::max)
}

fn mean_cosine(clips: &[&Vec<f32>], e: &[f32]) -> f32 {
    let mut mean = vec![0f32; e.len()];
    for c in clips {
        mean.iter_mut().zip(c.iter()).for_each(|(m, x)| *m += x);
    }
    speaker::cosine(e, &mean)
}

/// Task 2260: how a person's clips combine into a score. Leave-one-out over
/// the fixtures: enrol a voice from all but one of its clips, score the
/// held-out clip (same voice) and the other voice's clips (different), and
/// compare the best-clip (max) score with the cosine to the mean embedding.
/// `person_score` in `src/speaker.rs` uses the winner, the mean.
#[test]
fn clips_combine_by_mean_embedding() {
    let Some(v) = verifier() else { return };
    let a: Vec<Vec<f32>> = ["a1.wav", "a2.wav", "a3.wav"]
        .iter()
        .map(|c| embed(v, c))
        .collect();
    let b: Vec<Vec<f32>> = ["b1.wav", "b2.wav", "b3.wav"]
        .iter()
        .map(|c| embed(v, c))
        .collect();
    let methods: [(&str, Method); 2] = [("max", max_cosine), ("mean", mean_cosine)];
    let mut margins = Vec::new();
    for (name, method) in methods {
        let (mut same, mut different) = (Vec::new(), Vec::new());
        for (voice, other) in [(&a, &b), (&b, &a)] {
            for held in 0..voice.len() {
                let rest: Vec<&Vec<f32>> = (0..voice.len())
                    .filter(|i| *i != held)
                    .map(|i| &voice[i])
                    .collect();
                same.push(method(&rest, &voice[held]));
                different.extend(other.iter().map(|e| method(&rest, e)));
            }
        }
        let lo = same.iter().cloned().fold(f32::MAX, f32::min);
        let hi = different.iter().cloned().fold(f32::MIN, f32::max);
        eprintln!(
            "{name}: same-voice min {lo:.3}, different-voice max {hi:.3}, margin {:.3}",
            lo - hi
        );
        margins.push(lo - hi);
        if name == "mean" {
            assert!(hi < DEFAULT_THRESHOLD && DEFAULT_THRESHOLD < lo);
        }
    }
    eprintln!("margin max {:.3} vs mean {:.3}", margins[0], margins[1]);
    assert!(margins[1] >= margins[0]);
}

/// Task 2354: a clip just past a multiple of 2 s once embedded far from the
/// same clip cut a little shorter (untiled, `a1` scored 0.55 at 2.006 s
/// against 1.994 s, and `a1` against `b1` 0.67); `SpeakerVerifier::embed` tiles.
#[test]
fn a_clip_just_past_two_seconds_embeds_like_one_just_under() {
    let Some(v) = verifier() else { return };
    let (a, b) = (samples("a1.wav"), samples("b1.wav"));
    for (short, long) in [(31_900, 32_100), (63_900, 64_400)] {
        let same = speaker::cosine(
            &v.embed(&a[..short]).expect("embed"),
            &v.embed(&a[..long]).expect("embed"),
        );
        let other = speaker::cosine(
            &v.embed(&a[..long]).expect("embed"),
            &v.embed(&b[..long]).expect("embed"),
        );
        assert!(same > 0.9 && other < DEFAULT_THRESHOLD, "{long}: {same} {other}");
    }
}

#[test]
fn owner_verifies_and_the_other_voice_does_not() {
    let Some(v) = verifier() else { return };
    let lib = library(v, &[("Me", true, &["a1.wav", "a2.wav"])]);
    for (clip, him) in [("a3.wav", true), ("b1.wav", false), ("b3.wav", false)] {
        let id = verify(v, &lib, true, clip);
        assert_eq!(id.verdict.is_him, him, "{clip}: {id:?}");
        assert_eq!(id.verdict.reason, Reason::Scored);
    }
}

#[test]
fn ambient_tagging_names_the_right_person() {
    let Some(v) = verifier() else { return };
    let lib = library(
        v,
        &[
            ("Me", true, &["a1.wav", "a2.wav"]),
            ("Sam", false, &["b1.wav", "b2.wav"]),
        ],
    );
    let name = |clip| {
        let id = verify(v, &lib, false, clip);
        (id.verdict.is_him, id.person.map(|p| p.name))
    };
    assert_eq!(name("a3.wav"), (true, Some("Me".to_string())));
    assert_eq!(name("b3.wav"), (false, Some("Sam".to_string())));
    // a voice nobody in the library matches is an unnamed other
    let only_me = library(v, &[("Me", true, &["a1.wav", "a2.wav"])]);
    let id = verify(v, &only_me, false, "b3.wav");
    assert!(!id.verdict.is_him && id.person.is_none() && id.verdict.reason == Reason::Scored);
}

#[test]
fn short_reply_and_missing_library_fail_open() {
    let Some(v) = verifier() else { return };
    let lib = library(v, &[("Me", true, &["a1.wav"])]);
    // "Yes." in the other voice would be rejected if it were scored
    let mut short = samples("b1.wav");
    short.truncate(MIN_SAMPLES - 1);
    let id = speaker::identify(Some(v), Some(&lib), true, &short, DEFAULT_THRESHOLD);
    assert!(id.verdict.is_him && id.verdict.reason == Reason::TooShort);
    let id = speaker::identify(Some(v), None, true, &samples("b1.wav"), DEFAULT_THRESHOLD);
    assert!(id.verdict.is_him && id.verdict.reason == Reason::NoEnrollment);
    assert!(verify(v, &lib, true, "a_short.wav").verdict.is_him);
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
        keep_recent: false,
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
    Library::update(home.path(), |l| {
        l.ensure_owner();
        l.people[0].clips.push(Clip {
            id: 1,
            created_at: 0,
            seconds: 1.0,
            embedding: vec![1.0, 0.0],
            has_audio: false,
        });
        Ok(())
    })
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
    let lib = Library::load(home.path()).unwrap();
    let me = lib.owner().unwrap();
    assert_eq!((me.name.as_str(), me.clips.len()), ("Me", 2));
    assert!(me.clips.iter().all(|c| c.has_audio));
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

async fn json_call(
    home: &Path,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let (status, bytes) = raw_call(home, method, path, body).await;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn raw_call(
    home: &Path,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(HOST, "127.0.0.1:7870")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.map(|b| b.to_string()).unwrap_or_default()))
        .unwrap();
    let resp = app(home).oneshot(req).await.unwrap();
    let status = resp.status();
    (
        status,
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
}

#[tokio::test]
async fn people_crud_and_the_guard_switch_without_the_model() {
    let home = tempfile::tempdir().unwrap();
    let h = home.path();
    let people = "/v1/audio/speaker/people";
    // empty library, guard off by default
    let (status, body) = json_call(h, "GET", people, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        (body["live_guard"].clone(), body["people"].clone()),
        (json!(false), json!([]))
    );
    let (_, body) = json_call(h, "GET", "/v1/audio/speaker/guard", None).await;
    assert_eq!(body, json!({"live": false}));
    let (status, body) = json_call(
        h,
        "PUT",
        "/v1/audio/speaker/guard",
        Some(json!({"live": true})),
    )
    .await;
    assert_eq!((status, body), (StatusCode::OK, json!({"live": true})));
    let (_, body) = json_call(h, "GET", "/v1/audio/speaker/guard", None).await;
    assert_eq!(body["live"], true);

    // create, rename, move ownership
    let (status, me) = json_call(
        h,
        "POST",
        people,
        Some(json!({"name": " Me ", "owner": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{me}");
    assert_eq!(
        (me["name"].clone(), me["owner"].clone()),
        (json!("Me"), json!(true))
    );
    let (_, sam) = json_call(h, "POST", people, Some(json!({"name": "Sam"}))).await;
    assert_eq!(sam["owner"], false);
    let (status, _) = json_call(h, "POST", people, Some(json!({"name": "  "}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let sam_path = format!("{people}/{}", sam["id"]);
    let (status, body) = json_call(
        h,
        "PATCH",
        &sam_path,
        Some(json!({"name": "Samuel", "owner": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = json_call(h, "GET", people, None).await;
    let owners: Vec<_> = body["people"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["owner"] == true)
        .map(|p| p["name"].clone())
        .collect();
    assert_eq!(owners, [json!("Samuel")], "{body}");
    assert_eq!(body["live_guard"], true);
    // no embeddings leave the daemon
    assert!(!body.to_string().contains("embedding"));
    let (status, _) = json_call(
        h,
        "PATCH",
        &format!("{people}/99"),
        Some(json!({"name": "x"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // clips: no model pulled is 409, an unknown person 404, a clip with no audio 404
    let (status, body) = call(
        h,
        "POST",
        &format!("{sam_path}/clips"),
        &[fixture_bytes("a1.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "model_not_pulled");
    let (status, _) = call(
        h,
        "POST",
        &format!("{people}/99/clips"),
        &[fixture_bytes("a1.wav")],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = raw_call(h, "GET", &format!("{sam_path}/clips/1/audio"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = json_call(h, "DELETE", &sam_path, None).await;
    assert_eq!(
        (status, body["deleted"].clone()),
        (StatusCode::OK, json!(true))
    );
    let (status, _) = json_call(h, "DELETE", &sam_path, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn clips_round_trip_with_audio() {
    let Some(_) = verifier() else { return };
    let real = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .or_else(registry::default_home)
        .unwrap();
    let home = tempfile::tempdir().unwrap();
    let h = home.path();
    std::fs::create_dir_all(h.join("models")).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        real.join("models").join(DEFAULT_MODEL),
        h.join("models").join(DEFAULT_MODEL),
    )
    .unwrap();
    #[cfg(not(unix))]
    return;

    let people = "/v1/audio/speaker/people";
    let (_, sam) = json_call(h, "POST", people, Some(json!({"name": "Sam"}))).await;
    let clips = format!("{people}/{}/clips", sam["id"]);
    // under a second is refused
    let (status, body) = call(h, "POST", &clips, &[fixture_bytes("a_short.wav")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, clip) = call(h, "POST", &clips, &[fixture_bytes("b1.wav")]).await;
    assert_eq!(status, StatusCode::OK, "{clip}");
    assert_eq!(clip["has_audio"], true);
    assert!(clip["seconds"].as_f64().unwrap() > 1.0);
    // listed (no embedding), and the audio comes back as a WAV
    let (_, body) = json_call(h, "GET", people, None).await;
    assert_eq!(body["people"][0]["clips"][0]["id"], clip["id"], "{body}");
    let audio = format!("{clips}/{}/audio", clip["id"]);
    let (status, wav) = raw_call(h, "GET", &audio, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(&wav[..4], b"RIFF");
    assert!(audio::decode(&wav[..]).unwrap().len() > MIN_SAMPLES);
    // a restart sees it; the clip then the person delete with their audio
    let path = Library::clip_path(h, clip["id"].as_u64().unwrap());
    assert!(path.is_file());
    let (status, _) = json_call(h, "DELETE", &format!("{clips}/{}", clip["id"]), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!path.exists());
    let (status, _) = raw_call(h, "GET", &audio, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, clip) = call(h, "POST", &clips, &[fixture_bytes("b2.wav")]).await;
    let path = Library::clip_path(h, clip["id"].as_u64().unwrap());
    let (status, _) = json_call(h, "DELETE", &format!("{people}/{}", sam["id"]), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!path.exists());
}
