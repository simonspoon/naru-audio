//! auris's `tests/fixtures.rs` (d8644d6) through the library. The fixtures
//! and `hotwords.txt` are copied from auris `tests/fixtures/` and
//! `bench/fixtures/`; the expected transcripts are what auris prints for
//! them at d8644d6.
//!
//! Model-dependent tests skip (eprintln and return) when
//! `parakeet-tdt-0.6b-v2-int8` is not pulled under `$NARU_AUDIO_TEST_HOME`,
//! else `$NARU_AUDIO_HOME` or `~/.naru-audio`, as auris skips without its
//! model. `NARU_AUDIO_REQUIRE_MODELS=1` turns the skip into a failure.

use std::path::PathBuf;
use std::sync::OnceLock;

use naru_audio::backend::load_stt;
use naru_audio::registry::{self, Registry};
use naru_audio::stt::audio::{self, AudioError};
use naru_audio::stt::gate::{self, SpeechGate};
use naru_audio::stt::{Segment, SttModel, VadConfig, Vocabulary};

const MODEL: &str = "parakeet-tdt-0.6b-v2-int8";

/// auris's output for `plain.wav` and `stereo-44100.wav`.
const PLAIN: &str = "Open the daily notes and add a line about the meeting.";
/// auris `tests/fixtures/manifest.tsv`.
const PLAIN_REFERENCE: &str = "open the daily notes and add a line about the meeting";
const MESA_NAMES_UNBIASED: &str = "Hey Corvex Hey Halios.";
const MESA_NAMES_BIASED: &str = "Hey qorvex hey helios.";

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(name)
}

fn samples(name: &str) -> Vec<f32> {
    let file = std::fs::File::open(fixture(name)).unwrap_or_else(|e| panic!("open {name}: {e}"));
    audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
}

fn hotwords() -> Vocabulary {
    Vocabulary::parse(&std::fs::read_to_string(fixture("hotwords.txt")).unwrap()).unwrap()
}

/// Folds case, strips punctuation, collapses whitespace (auris `normalise`).
fn normalise(s: &str) -> String {
    let folded: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Loaded once and shared: each load holds ~1.5 GB.
fn model() -> Option<&'static dyn SttModel> {
    static MODEL_CELL: OnceLock<Option<Box<dyn SttModel>>> = OnceLock::new();
    MODEL_CELL
        .get_or_init(|| {
            let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
                .map(PathBuf::from)
                .or_else(registry::default_home)?;
            let registry = Registry::open(&home).expect("open registry");
            if !registry.is_installed(MODEL) {
                let message = format!(
                    "{MODEL} not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
                    home.display()
                );
                if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
                    panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
                }
                eprintln!("skip: {message}");
                return None;
            }
            let manifest = registry.pulled_manifest(MODEL).expect("pulled manifest");
            Some(load_stt(&manifest, &registry.model_dir(MODEL)).expect("load"))
        })
        .as_deref()
}

/// The default request: VAD on at its defaults.
fn transcribe(model: &dyn SttModel, pcm: &[f32], hotwords: Option<&Vocabulary>) -> Vec<Segment> {
    model
        .decode(pcm, hotwords, Some(&VadConfig::default()))
        .expect("decode")
}

fn texts(segments: &[Segment]) -> Vec<&str> {
    segments.iter().map(|s| s.text.as_str()).collect()
}

#[test]
fn plain_transcribes_to_reference_text() {
    let Some(model) = model() else { return };
    let segments = transcribe(model, &samples("plain.wav"), None);
    eprintln!("plain.wav: {segments:?}");
    assert_eq!(texts(&segments), [PLAIN]);
    assert_eq!(normalise(&segments[0].text), PLAIN_REFERENCE);
}

/// The resample and downmix path reaches the 16 kHz mono transcript.
#[test]
fn stereo_44100_resamples_to_the_same_transcript_as_plain() {
    let Some(model) = model() else { return };
    let segments = transcribe(model, &samples("stereo-44100.wav"), None);
    eprintln!("stereo-44100.wav: {segments:?}");
    assert_eq!(texts(&segments), [PLAIN]);
}

/// With VAD off, one utterance decodes exactly as with it on: the gate
/// decides, it never trims.
#[test]
fn vad_off_decodes_the_same_single_utterance() {
    let Some(model) = model() else { return };
    let segments = model
        .decode(&samples("plain.wav"), None, None)
        .expect("decode");
    assert_eq!(texts(&segments), [PLAIN]);
}

/// Silence is an empty result, not an error: `silence.wav`, all-zero PCM,
/// and noise below −60 dBFS, with and without VAD and hotwords.
#[test]
fn silence_yields_no_segments() {
    let Some(model) = model() else { return };
    let vocabulary = hotwords();
    let mut state: u32 = 0x2545F491;
    let quiet_noise: Vec<f32> = (0..16_000 * 2)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * 1e-4
        })
        .collect();
    let cases = [
        ("silence.wav", samples("silence.wav")),
        ("all-zero", vec![0.0; 16_000 * 2]),
        ("quiet noise", quiet_noise),
        ("empty", Vec::new()),
    ];
    for (name, pcm) in &cases {
        for hotwords in [None, Some(&vocabulary)] {
            for vad in [None, Some(&VadConfig::default())] {
                let segments = model
                    .decode(pcm, hotwords, vad)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(segments, [], "{name}");
            }
        }
    }
}

/// Loud enough to pass the energy gate; the Silero gate turns them away.
#[test]
fn nonspeech_noise_yields_no_segments() {
    let Some(model) = model() else { return };
    let vocabulary = hotwords();
    for name in ["nonspeech-white.wav", "nonspeech-rumble.wav"] {
        let pcm = samples(name);
        assert_eq!(transcribe(model, &pcm, None), [], "{name}");
        assert_eq!(transcribe(model, &pcm, Some(&vocabulary)), [], "{name}");
    }
}

/// Model-free: the decode rejects it before any model is involved.
#[test]
fn not_audio_is_a_clean_error() {
    let bytes = std::fs::read(fixture("not-audio.bin")).unwrap();
    let err = audio::decode(&bytes[..]).unwrap_err();
    assert!(matches!(err, AudioError::NotWav(_)), "{err:?}");
    let message = err.to_string();
    assert!(!message.contains('\n'), "{message:?}");
    assert!(message.contains("not a wav file"), "{message:?}");
    assert!(message.contains("expected a wav stream"), "{message:?}");
}

/// Neither name lands unbiased; both land with the vocabulary. The biased
/// run also covers auris's
/// `manufactured_vocabulary_guard_does_not_affect_real_speech`, which is
/// the same call here.
#[test]
fn mesa_names_lands_only_when_biased() {
    let Some(model) = model() else { return };
    let pcm = samples("mesa-names.wav");

    let unbiased = transcribe(model, &pcm, None);
    eprintln!("mesa-names.wav (unbiased): {unbiased:?}");
    assert_eq!(texts(&unbiased), [MESA_NAMES_UNBIASED]);
    let unbiased = normalise(&unbiased[0].text);
    assert!(!unbiased.contains("qorvex") && !unbiased.contains("helios"));

    let biased = transcribe(model, &pcm, Some(&hotwords()));
    eprintln!("mesa-names.wav (biased): {biased:?}");
    assert_eq!(texts(&biased), [MESA_NAMES_BIASED]);
    let biased = normalise(&biased[0].text);
    assert!(biased.contains("qorvex") && biased.contains("helios"));
}

/// auris task 970: Silero false-positives on this burst and biasing
/// invents a transcript of boosted terms; the guard's unbiased decode is
/// empty, so nothing is returned.
#[test]
fn manufactured_vocabulary_transcript_yields_no_segments() {
    let Some(model) = model() else { return };
    let pcm = samples("nonspeech-transient.wav");
    assert_eq!(transcribe(model, &pcm, Some(&hotwords())), []);
    assert_eq!(transcribe(model, &pcm, None), []);
}

/// Four copies of `plain.wav` separated by 1 s of silence: one segment per
/// utterance, in order (auris's multi-utterance stream test). The spans and
/// texts are what `auris --no-daemon --format json` prints for the same
/// audio at d8644d6; each slice keeps its neighbours' context, hence the
/// lowercase and the third one's comma.
#[test]
fn multi_utterance_buffer_yields_a_segment_per_utterance() {
    let Some(model) = model() else { return };
    let speech = samples("plain.wav");
    let gap = vec![0.0f32; 16_000];
    let mut pcm = Vec::new();
    for i in 0..4 {
        if i > 0 {
            pcm.extend_from_slice(&gap);
        }
        pcm.extend_from_slice(&speech);
    }
    let segments = transcribe(model, &pcm, None);
    eprintln!("4 x plain.wav: {segments:?}");
    let expected = [
        (
            0.166,
            2.764,
            "open the daily notes and add a line about the meeting.",
        ),
        (
            3.814,
            6.572,
            "open the daily notes and add a line about the meeting.",
        ),
        (
            7.622,
            10.412,
            "open the daily notes, and add a line about the meeting.",
        ),
        (
            11.462,
            14.24,
            "open the daily notes and add a line about the meeting.",
        ),
    ];
    let actual: Vec<(f64, f64, &str)> = segments
        .iter()
        .map(|s| (s.start, s.end, s.text.as_str()))
        .collect();
    assert_eq!(actual, expected);
}

/// The speech gate's tagger, shared; skips like `model()` when
/// `audio-tagging-ced-tiny` is not pulled.
fn speech_gate() -> Option<&'static SpeechGate> {
    static GATE: OnceLock<Option<SpeechGate>> = OnceLock::new();
    GATE.get_or_init(|| {
        let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
            .map(PathBuf::from)
            .or_else(registry::default_home)?;
        let gate = SpeechGate::load(&home.join("models"));
        if gate.is_none() {
            let message = format!("{} not pulled under {}", gate::MODEL_NAME, home.display());
            if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
                panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
            }
            eprintln!("skip: {message}");
        }
        gate
    })
    .as_ref()
}

/// Speech passes the gate, including a short word.
#[test]
fn speech_gate_allows_speech() {
    let Some(gate) = speech_gate() else { return };
    for name in ["plain.wav", "stereo-44100.wav", "mesa-names.wav"] {
        assert!(gate.allows(&samples(name)), "{name} was refused");
    }
    let word = &samples("plain.wav")[..16_000 * 7 / 10];
    assert!(gate.allows(word), "a 0.7 s word was refused");
}

/// The tagger hears the silence fixture as Silence 0.68 and refuses it; an
/// empty or over-long buffer is never gated.
#[test]
fn speech_gate_refuses_a_clear_non_speech_sound() {
    let Some(gate) = speech_gate() else { return };
    assert!(!gate.allows(&samples("silence.wav")));
    assert!(gate.allows(&[]));
    assert!(gate.allows(&[0.1; 100]));
    assert!(gate.allows(&vec![0.0; 16_000 * 11]));
}
