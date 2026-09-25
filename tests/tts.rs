//! Kokoro v1.0 and Pocket TTS through the library (`backend::load_tts`).
//!
//! Model-dependent tests skip (eprintln and return) when their model is
//! not pulled under `$NARU_AUDIO_TEST_HOME`, else `$NARU_AUDIO_HOME` or
//! `~/.naru-audio`, as tests/stt.rs does. `NARU_AUDIO_REQUIRE_MODELS=1`
//! turns the skip into a failure. Timings are printed; run with
//! `--test-threads=1` to read them undisturbed.

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};
use std::time::Instant;

use naru_audio::backend::load_tts;
use naru_audio::registry::{self, Registry};
use naru_audio::tts::{Sink, SynthOptions, TtsError, TtsModel};

const MODEL: &str = "kokoro-v1.0";
const VOICE: &str = "af_heart";
const POCKET: &str = "pocket-tts-int8";

/// The first `n` sentences of the spike corpus (docs/tts-spike-corpus.txt).
fn corpus(n: usize) -> String {
    include_str!("../docs/tts-spike-corpus.txt")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(n)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Loaded once and shared: a load holds ~0.95 GB.
fn model() -> Option<&'static dyn TtsModel> {
    static MODEL_CELL: OnceLock<Option<Box<dyn TtsModel>>> = OnceLock::new();
    MODEL_CELL.get_or_init(|| load(MODEL)).as_deref()
}

fn pocket() -> Option<&'static dyn TtsModel> {
    static POCKET_CELL: OnceLock<Option<Box<dyn TtsModel>>> = OnceLock::new();
    POCKET_CELL.get_or_init(|| load(POCKET)).as_deref()
}

fn load(name: &str) -> Option<Box<dyn TtsModel>> {
    let home = std::env::var_os("NARU_AUDIO_TEST_HOME")
        .map(PathBuf::from)
        .or_else(registry::default_home)?;
    let registry = Registry::open(&home).expect("open registry");
    if !registry.is_installed(name) {
        let message = format!(
            "{name} not pulled under {} (set NARU_AUDIO_TEST_HOME to override)",
            home.display()
        );
        if std::env::var_os("NARU_AUDIO_REQUIRE_MODELS").is_some_and(|v| v == "1") {
            panic!("{message}; NARU_AUDIO_REQUIRE_MODELS=1 forbids skipping");
        }
        eprintln!("skip: {message}");
        return None;
    }
    let manifest = registry.pulled_manifest(name).expect("pulled manifest");
    Some(load_tts(&manifest, &registry.model_dir(name)).expect("load"))
}

/// One sink call: when it started and returned, on which thread, and the
/// samples it got.
struct Delivery {
    entered: Instant,
    left: Instant,
    thread: ThreadId,
    samples: Vec<f32>,
}

/// Synthesises `text`, recording every delivery; the sink goes on while
/// `keep` says so.
fn synth(
    model: &dyn TtsModel,
    text: &str,
    options: &SynthOptions,
    keep: fn(usize) -> bool,
) -> (Instant, Result<(), TtsError>, Vec<Delivery>) {
    synth_as(model, VOICE, text, options, keep)
}

fn synth_as(
    model: &dyn TtsModel,
    voice: &str,
    text: &str,
    options: &SynthOptions,
    keep: fn(usize) -> bool,
) -> (Instant, Result<(), TtsError>, Vec<Delivery>) {
    let deliveries: Arc<Mutex<Vec<Delivery>>> = Arc::default();
    let log = Arc::clone(&deliveries);
    let sink: Sink = Box::new(move |chunk: &[f32]| {
        let entered = Instant::now();
        let mut log = log.lock().unwrap();
        log.push(Delivery {
            entered,
            left: entered,
            thread: thread::current().id(),
            samples: chunk.to_vec(),
        });
        let n = log.len();
        log.last_mut().unwrap().left = Instant::now();
        keep(n)
    });
    let start = Instant::now();
    let result = model.synth(text, voice, options, sink);
    let deliveries = std::mem::take(&mut *deliveries.lock().unwrap());
    (start, result, deliveries)
}

/// Seconds of near-silence (below -60 dB of the chunk's peak sample) at the
/// start and end of `samples`.
fn silence(samples: &[f32], rate: u32) -> (f64, f64) {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let loud = |s: &f32| s.abs() > peak * 1e-3;
    let lead = samples.iter().position(loud).unwrap_or(samples.len());
    let trail = samples.len() - samples.iter().rposition(loud).map_or(0, |i| i + 1);
    (lead as f64 / rate as f64, trail as f64 / rate as f64)
}

/// The acceptance: each sentence's samples reach the sink on the thread
/// doing the synthesis, from inside sherpa's callback, so the next sentence
/// cannot start synthesising until the sink has returned.
#[test]
fn each_sentence_reaches_the_sink_before_the_next_is_synthesised() {
    let Some(model) = model() else { return };
    let (start, result, deliveries) = synth(model, &corpus(3), &SynthOptions::default(), |_| true);
    let done = Instant::now();
    result.expect("synth");

    assert_eq!(deliveries.len(), 3, "one chunk per sentence");
    for d in &deliveries {
        assert_eq!(
            d.thread,
            thread::current().id(),
            "sink off the synth thread"
        );
    }
    let rate = model.sample_rate() as f64;
    let secs = |t: Instant| t.duration_since(start).as_secs_f64();
    eprintln!(
        "first chunk at {:.3} s; synth wall {:.3} s",
        secs(deliveries[0].entered),
        secs(done)
    );
    for pair in deliveries.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        // The time between one sink return and the next sink call is the
        // next sentence's synthesis; it only starts after the sink returns.
        let synthesis = next.entered.duration_since(prev.left).as_secs_f64();
        eprintln!(
            "sink returned at {:.3} s; next chunk ({:.2} s of audio) arrived at {:.3} s, {synthesis:.3} s later",
            secs(prev.left),
            next.samples.len() as f64 / rate,
            secs(next.entered),
        );
        assert!(next.entered > prev.left);
        // Real synthesis, not a queued piece handed over late: at least 2%
        // of the piece's audio duration (the spike's RTF is ~0.13).
        assert!(synthesis > 0.02 * next.samples.len() as f64 / rate);
    }
}

/// The no-trim decision (docs/tts-spike.md (b)): sherpa's pieces already
/// come trimmed, against Kokoro's ~2 s of raw padding. Also prints the
/// silence each boundary carries before the gap is put back.
#[test]
fn sherpa_pieces_come_trimmed() {
    let Some(model) = model() else { return };
    let options = SynthOptions {
        gap: 0.0,
        level: false,
        ..Default::default()
    };
    let (_, result, deliveries) = synth(model, &corpus(3), &options, |_| true);
    result.expect("synth");
    let rate = model.sample_rate();
    let ends: Vec<(f64, f64)> = deliveries
        .iter()
        .map(|d| silence(&d.samples, rate))
        .collect();
    for (i, (lead, trail)) in ends.iter().enumerate() {
        eprintln!("piece {i}: lead {lead:.3} s, trail {trail:.3} s");
        assert!(*lead < 0.2 && *trail < 0.2, "piece {i} is padded");
    }
    for (i, pair) in ends.windows(2).enumerate() {
        eprintln!(
            "boundary {i}|{}: {:.3} s of silence without a gap",
            i + 1,
            pair[0].1 + pair[1].0
        );
    }
}

/// Cancelling after the first sentence stops sherpa itself: the call
/// returns in a fraction of a full run of the same 10 sentences.
#[test]
fn a_sink_returning_false_stops_sherpa() {
    let Some(model) = model() else { return };
    let text = corpus(10);
    let options = SynthOptions::default();
    let (start, result, full) = synth(model, &text, &options, |_| true);
    let full_s = start.elapsed().as_secs_f64();
    result.expect("synth");
    let (start, result, deliveries) = synth(model, &text, &options, |n| n < 1);
    let cancelled_s = start.elapsed().as_secs_f64();
    result.expect("a cancelled synthesis is Ok");
    eprintln!(
        "full run {full_s:.3} s ({} chunks); cancelled after 1 chunk {cancelled_s:.3} s",
        full.len()
    );
    assert_eq!(full.len(), 10);
    assert_eq!(deliveries.len(), 1);
    assert!(cancelled_s < 0.5 * full_s);
}

/// sherpa splits at ". ", but not after an abbreviation: no gap lands
/// between "Dr." and "Smith".
#[test]
fn an_abbreviation_does_not_split_a_sentence() {
    let Some(model) = model() else { return };
    let options = SynthOptions::default();
    let (_, result, control) = synth(model, "Stop. Go to Washington.", &options, |_| true);
    result.expect("synth");
    assert_eq!(control.len(), 2, "sherpa splits at a sentence's dot");
    let (_, result, deliveries) = synth(model, "Dr. Smith went to Washington.", &options, |_| true);
    result.expect("synth");
    assert_eq!(deliveries.len(), 1);
}

/// The sink runs inside sherpa's `extern "C"` callback; its panic must
/// come back out of `synth` as a panic, not abort the process.
#[test]
fn a_panicking_sink_panics_out_of_synth() {
    let Some(model) = model() else { return };
    let sink: Sink = Box::new(|_| panic!("sink failed"));
    let payload = std::panic::catch_unwind(AssertUnwindSafe(|| {
        model.synth(&corpus(3), VOICE, &SynthOptions::default(), sink)
    }))
    .expect_err("the sink's panic");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"sink failed"));
}

#[test]
fn a_nul_or_a_bad_gap_is_an_error_not_a_panic() {
    let Some(model) = model() else { return };
    let never = || -> Sink { Box::new(|_| panic!("no audio for a refused request")) };
    let err = model
        .synth("Hello\0.", VOICE, &SynthOptions::default(), never())
        .unwrap_err();
    assert!(matches!(err, TtsError::NulInText), "{err}");
    let options = SynthOptions {
        gap: f32::INFINITY,
        ..Default::default()
    };
    let err = model
        .synth("Hello. There.", VOICE, &options, never())
        .unwrap_err();
    assert!(matches!(err, TtsError::InvalidGap(_)), "{err}");
}

#[test]
fn an_unknown_voice_is_an_error() {
    let Some(model) = model() else { return };
    assert!(model.voices().iter().any(|v| v.id == VOICE && v.default));
    let sink: Sink = Box::new(|_| panic!("no audio for an unknown voice"));
    let err = model
        .synth("Hello.", "alloy", &SynthOptions::default(), sink)
        .unwrap_err();
    assert!(matches!(err, TtsError::UnknownVoice { .. }), "{err}");
}

/// The longest run of exact zeros in `samples`, in samples: the sentence gap
/// is digital silence, which a model's own pauses never are.
fn longest_zero_run(samples: &[f32]) -> usize {
    let (mut longest, mut run) = (0, 0);
    for &s in samples {
        run = if s == 0.0 { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    longest
}

/// Voiced RMS (|s| above 300/32768, as the leveller counts it) of `samples`.
fn voiced_rms(samples: &[f32]) -> f32 {
    let voiced: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|s| s.abs() > 300.0 / 32768.0)
        .collect();
    (voiced.iter().map(|s| s * s).sum::<f32>() / voiced.len().max(1) as f32).sqrt()
}

/// Pocket's callback pieces are decoder chunks, several per sentence, so
/// none of them gets the sentence gap: with the default options no chunk
/// carries a gap's run of digital silence. Prints what each boundary looks
/// like: the step between the samples either side of it, against the
/// typical step inside a chunk, and the voiced RMS either side.
#[test]
fn pocket_chunks_reach_the_sink_without_a_gap() {
    let Some(model) = pocket() else { return };
    assert!(model.voices().iter().any(|v| v.id == "bria" && v.default));
    let (start, result, deliveries) =
        synth_as(model, "bria", &corpus(3), &SynthOptions::default(), |_| {
            true
        });
    let done = Instant::now();
    result.expect("synth");
    let rate = model.sample_rate() as f64;
    let audio: usize = deliveries.iter().map(|d| d.samples.len()).sum();
    eprintln!(
        "{} chunks; first at {:.3} s; synth wall {:.3} s for {:.2} s of audio",
        deliveries.len(),
        deliveries[0].entered.duration_since(start).as_secs_f64(),
        done.duration_since(start).as_secs_f64(),
        audio as f64 / rate
    );
    assert!(deliveries.len() > 3, "more chunks than sentences");
    let gap = (SynthOptions::default().gap as f64 * rate) as usize;
    for (i, d) in deliveries.iter().enumerate() {
        let zeros = longest_zero_run(&d.samples);
        eprintln!(
            "chunk {i}: {:.2} s, voiced rms {:.4}, longest zero run {zeros}",
            d.samples.len() as f64 / rate,
            voiced_rms(&d.samples)
        );
        assert!(zeros < gap, "chunk {i} carries a gap");
    }
    let all: Vec<f32> = deliveries.iter().flat_map(|d| d.samples.clone()).collect();
    let mut steps: Vec<f32> = all.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    steps.sort_by(f32::total_cmp);
    let p99 = steps[steps.len() * 99 / 100];
    let mut at = 0;
    for pair in deliveries.windows(2) {
        at += pair[0].samples.len();
        let step = (all[at] - all[at - 1]).abs();
        eprintln!(
            "boundary at {:.2} s: step {step:.4} (p99 step {p99:.4})",
            at as f64 / rate
        );
    }
}

#[test]
fn pocket_refuses_an_unknown_voice() {
    let Some(model) = pocket() else { return };
    let sink: Sink = Box::new(|_| panic!("no audio for an unknown voice"));
    let err = model
        .synth("Hello.", VOICE, &SynthOptions::default(), sink)
        .unwrap_err();
    assert!(matches!(err, TtsError::UnknownVoice { .. }), "{err}");
}
