//! The `sherpa-onnx` [`TtsModel`]: Kokoro or Kyutai Pocket TTS through
//! `OfflineTts`, picked by `[backend.sherpa-onnx] family` (`kokoro` when
//! absent). Kokoro runs as the task 9 spike ran it (§5.1,
//! examples/tts_spike.rs `whole`).
//!
//! Kokoro: `max_num_sentences = 1`, so sherpa's `generate` callback fires
//! once per sentence with that sentence's samples only. Each piece is
//! levelled, gets the sentence gap in front of it (not the first), and goes
//! to the sink from inside the callback, so a sentence reaches the sink
//! before the next one is synthesised.
//!
//! Pocket: one speaker, whose voice is cloned from each `[[voice]]`'s
//! `reference` recording; `sid` and `speed` are ignored. It splits
//! sentences itself and the callback fires once per decoder chunk (1.2 s),
//! mid-sentence, so pieces go to the sink as they come: no gap, which would
//! be a pause mid-word (Pocket leaves its own 0.1–0.3 s between sentences),
//! and no leveller, which measured 2026-09-25 on three sentences swung
//! between -3 and +3 dB from one chunk to the next. The chunks join without
//! a step (the largest boundary step 0.01, against a p99 in-chunk step of
//! 0.13).
//!
//! sherpa splits at a `.` before a space and a capital or a digit ("Dr.
//! Brown", "Fig. 3"; not "Inc. hired"), so the abbreviations kokoro-rs
//! keeps whole lose that dot first ([`keep_abbreviations_whole`]): a gap
//! after "Dr." would be a pause mid-phrase.
//!
//! Not done, per the spike: no character chunker, no trim, no clause split
//! of a long first sentence (its residual risk has not shown up).

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::rc::Rc;

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig, OfflineTtsPocketModelConfig, Wave,
};

use super::level::Leveller;
use super::{Sink, SynthOptions, TtsError, TtsModel};
use crate::registry::manifest::{Manifest, Voice};

/// The longest `gap` accepted, in seconds.
const MAX_GAP: f32 = 5.0;

/// kokoro-rs text.rs:25 `ABBREVIATIONS` (3f722cf): titles and abbreviations
/// whose trailing dot does not end a sentence.
///
/// Rule for membership: an entry stays only if dropping its dot is no worse
/// for pronunciation than sherpa's split plus the gap, since a mispronounced
/// word is a bigger defect than an extra 0.12 s pause. Measured 2026-09-24
/// (af_heart, one sentence per entry, both versions transcribed by
/// parakeet-tdt-0.6b-v2-int8): every entry passed. Where the two
/// transcripts differ, the dotless one reads the same (Dr/Doctor, JR/J.R.,
/// 9 a.m./9am) or better (Mrs, St, Gen, Capt, Sec, e.g); Hon and Col are
/// misread either way.
const ABBREVIATIONS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Mx", "Dr", "Prof", "Rev", "Hon", "Sr", "Jr", "St", "Mt", "Gen", "Col",
    "Sgt", "Capt", "Lt", "Cmdr", "Inc", "Ltd", "Co", "Corp", "Dept", "Est", "Fig", "No", "Vol",
    "Ch", "Sec", "Ref", "Approx", "vs", "etc", "al", "ca", "cf", "viz", "i.e", "e.g", "a.m", "p.m",
    "U.S", "U.K",
];

/// Which sherpa model a `[backend.sherpa-onnx]` table configures: its
/// `family` key, `kokoro` when absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Kokoro,
    Pocket,
}

pub struct SherpaTts {
    tts: OfflineTts,
    name: String,
    family: Family,
    voices: Vec<Voice>,
    /// Pocket: each voice's reference samples and their rate, in `voices`
    /// order, read once here rather than per request. Empty for Kokoro.
    references: Vec<(Vec<f32>, i32)>,
}

impl SherpaTts {
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, TtsError> {
        let name = &manifest.model.name;
        let table = manifest.backend.get(&manifest.model.backend);
        let setting = |key: &str| {
            table
                .and_then(|t| t.get(key))
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
                .ok_or_else(|| TtsError::MissingConfig {
                    model: name.clone(),
                    key: key.to_string(),
                })
        };
        let file = |key: &str| {
            let path = dir.join(setting(key)?);
            if path.exists() {
                Ok(path.to_string_lossy().into_owned())
            } else {
                Err(TtsError::MissingModelFile(path))
            }
        };

        let family = match table.and_then(|t| t.get("family")) {
            None => Family::Kokoro,
            Some(v) => match v.as_str() {
                Some("kokoro") => Family::Kokoro,
                Some("pocket") => Family::Pocket,
                _ => {
                    return Err(TtsError::UnknownFamily {
                        model: name.clone(),
                        family: v.to_string(),
                    });
                }
            },
        };
        let mut model = OfflineTtsModelConfig {
            // The spike measured with every core.
            num_threads: std::thread::available_parallelism().map_or(4, |n| n.get() as i32),
            provider: Some("cpu".to_string()),
            ..Default::default()
        };
        let mut references = Vec::new();
        match family {
            Family::Kokoro => {
                model.kokoro = OfflineTtsKokoroModelConfig {
                    model: Some(file("model")?),
                    voices: Some(file("voices")?),
                    tokens: Some(file("tokens")?),
                    data_dir: Some(file("data_dir")?),
                    // Required here, not left to sherpa: with neither `lang`
                    // nor a lexicon it calls exit() (docs/tts-spike.md).
                    lang: Some(setting("lang")?.to_string()),
                    ..Default::default()
                };
            }
            Family::Pocket => {
                model.pocket = OfflineTtsPocketModelConfig {
                    lm_flow: Some(file("lm_flow")?),
                    lm_main: Some(file("lm_main")?),
                    encoder: Some(file("encoder")?),
                    decoder: Some(file("decoder")?),
                    text_conditioner: Some(file("text_conditioner")?),
                    vocab_json: Some(file("vocab_json")?),
                    token_scores_json: Some(file("token_scores_json")?),
                    // Upstream's example value; one embedding per voice.
                    voice_embedding_cache_capacity: 50,
                };
                for v in &manifest.voices {
                    let path = dir.join(v.reference.as_deref().ok_or_else(|| {
                        TtsError::MissingConfig {
                            model: name.clone(),
                            key: format!("voice.{}.reference", v.id),
                        }
                    })?);
                    let wave = Wave::read(&path.to_string_lossy())
                        .ok_or(TtsError::MissingModelFile(path))?;
                    references.push((wave.samples().to_vec(), wave.sample_rate()));
                }
            }
        }
        let config = OfflineTtsConfig {
            model,
            // Kokoro only; Pocket splits sentences itself.
            max_num_sentences: 1,
            ..Default::default()
        };
        let tts = OfflineTts::create(&config).ok_or(TtsError::CreateFailed)?;
        Ok(SherpaTts {
            tts,
            name: name.clone(),
            family,
            voices: manifest.voices.clone(),
            references,
        })
    }
}

impl TtsModel for SherpaTts {
    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate() as u32
    }

    fn synth(
        &self,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        sink: Sink,
    ) -> Result<(), TtsError> {
        check(text, options)?;
        let index = self
            .voices
            .iter()
            .position(|v| v.id == voice)
            .ok_or_else(|| TtsError::UnknownVoice {
                model: self.name.clone(),
                voice: voice.to_string(),
            })?;
        let mut config = GenerationConfig {
            sid: self.voices[index].sid,
            speed: options.speed,
            ..Default::default()
        };
        if let Some((samples, rate)) = self.references.get(index) {
            config.reference_audio = Some(samples.clone());
            config.reference_sample_rate = *rate;
        }
        let mut stream = Stream::new(options, self.family, self.sample_rate(), sink);
        let outcome = Rc::clone(&stream.outcome);
        let audio = self.tts.generate_with_config(
            &keep_abbreviations_whole(text),
            &config,
            Some(move |piece: &[f32], _progress: f32| stream.piece(piece)),
        );
        outcome.finish(audio.is_some())
    }
}

/// What neither engine is handed: a NUL panics in sherpa's `CString::new`,
/// and a gap that is not a small non-negative number would allocate without
/// bound inside sherpa's callback. The MLX engine (`super::mlx`) checks the
/// same, so a request means the same whatever model serves it.
pub(super) fn check(text: &str, options: &SynthOptions) -> Result<(), TtsError> {
    if text.contains('\0') {
        return Err(TtsError::NulInText);
    }
    // Also false for NaN.
    if !(0.0..=MAX_GAP).contains(&options.gap) {
        return Err(TtsError::InvalidGap(options.gap));
    }
    Ok(())
}

/// Drops the dot of each [`ABBREVIATIONS`] entry before whitespace, so that
/// sherpa's sentence split does not cut there. Matched case-insensitively
/// as kokoro-rs does, but as a whole word, not kokoro-rs's plain suffix: a
/// sentence ending in "normal." must keep its dot.
fn keep_abbreviations_whole(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, c) in text.char_indices() {
        let before_space = text[i + c.len_utf8()..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace);
        if c == '.' && before_space && ends_in_abbreviation(&text[..i]) {
            continue;
        }
        out.push(c);
    }
    out
}

fn ends_in_abbreviation(head: &str) -> bool {
    ABBREVIATIONS.iter().any(|abbr| {
        let Some(start) = head.len().checked_sub(abbr.len()) else {
            return false;
        };
        head.is_char_boundary(start)
            && head[start..].eq_ignore_ascii_case(abbr)
            && !head[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric)
    })
}

/// What the callback leaves for `synth` once `generate` returns.
#[derive(Default)]
struct Outcome {
    cancelled: Cell<bool>,
    /// A panic caught in the callback. sherpa calls it through an
    /// `extern "C"` trampoline, where an unwinding panic aborts the process.
    panic: Cell<Option<Box<dyn Any + Send>>>,
}

impl Outcome {
    /// Re-raises a caught panic here, on the Rust side; else a cancelled
    /// synthesis is `Ok` whatever `generate` returned.
    fn finish(&self, generated: bool) -> Result<(), TtsError> {
        if let Some(payload) = self.panic.take() {
            panic::resume_unwind(payload);
        }
        if generated || self.cancelled.get() {
            Ok(())
        } else {
            Err(TtsError::GenerateFailed)
        }
    }
}

/// What the sink receives, one callback piece at a time.
struct Stream {
    leveller: Option<Leveller>,
    /// Samples of silence before every piece but the first.
    gap: usize,
    started: bool,
    sink: Sink,
    outcome: Rc<Outcome>,
}

impl Stream {
    /// Pocket's pieces are decoder chunks, not sentences, so it gets
    /// neither the sentence gap nor the leveller (module doc).
    fn new(options: &SynthOptions, family: Family, sample_rate: u32, sink: Sink) -> Self {
        let sentences = family == Family::Kokoro;
        Stream {
            leveller: (sentences && options.level).then(Leveller::new),
            gap: if sentences {
                (options.gap * sample_rate as f32) as usize
            } else {
                0
            },
            started: false,
            sink,
            outcome: Rc::default(),
        }
    }

    /// sherpa's callback: one sentence's samples (Kokoro) or one decoder
    /// chunk's (Pocket). Returns whether to go on;
    /// never unwinds.
    fn piece(&mut self, samples: &[f32]) -> bool {
        if self.outcome.cancelled.get() {
            return false;
        }
        let more = match panic::catch_unwind(AssertUnwindSafe(|| self.deliver(samples))) {
            Ok(more) => more,
            Err(payload) => {
                self.outcome.panic.set(Some(payload));
                false
            }
        };
        self.outcome.cancelled.set(!more);
        more
    }

    fn deliver(&mut self, samples: &[f32]) -> bool {
        // Nothing survived phonemisation; it earns no gap either.
        if samples.is_empty() {
            return true;
        }
        // sherpa's pieces come trimmed (≤76 ms either end, the spike), so
        // the gap is put back in full as kokoro-rs does after its trim.
        let gap = if self.started { self.gap } else { 0 };
        let mut chunk = vec![0.0; gap];
        chunk.extend_from_slice(samples);
        if let Some(leveller) = &mut self.leveller {
            leveller.apply(&mut chunk[gap..]);
        }
        self.started = true;
        (self.sink)(&chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::manifest::Catalog;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    const RATE: u32 = 24_000;

    #[derive(Debug, PartialEq)]
    enum Event {
        /// The fake synthesiser starts on piece `n`.
        Synth(usize),
        /// The sink received a chunk of this many samples.
        Sink(usize),
    }

    /// Calls the callback as sherpa does, across an `extern "C"` frame
    /// (sherpa-onnx tts.rs:638), so a panic unwinding out of it aborts the
    /// test binary just as it would the daemon.
    extern "C" fn trampoline(callback: *mut c_void, samples: *const f32, n: usize) -> bool {
        // SAFETY: `generate` passes its live `&mut dyn FnMut` and a whole
        // piece's pointer and length.
        let callback = unsafe { &mut *(callback as *mut &mut dyn FnMut(&[f32]) -> bool) };
        callback(unsafe { std::slice::from_raw_parts(samples, n) })
    }

    /// Stands in for sherpa's `generate`: synthesises each piece in turn
    /// and hands it to the callback, stopping when it returns `false`.
    fn generate(
        pieces: &[Vec<f32>],
        log: &Arc<Mutex<Vec<Event>>>,
        mut callback: impl FnMut(&[f32]) -> bool,
    ) {
        let mut callback: &mut dyn FnMut(&[f32]) -> bool = &mut callback;
        let arg = &mut callback as *mut &mut dyn FnMut(&[f32]) -> bool as *mut c_void;
        for (n, piece) in pieces.iter().enumerate() {
            log.lock().unwrap().push(Event::Synth(n));
            if !trampoline(arg, piece.as_ptr(), piece.len()) {
                return;
            }
        }
    }

    /// Runs `pieces` through a [`Stream`]; returns the event log, the
    /// chunks the sink received and whether it cancelled. The sink stops
    /// after `keep` chunks.
    fn run(
        pieces: &[Vec<f32>],
        options: &SynthOptions,
        keep: usize,
    ) -> (Vec<Event>, Vec<Vec<f32>>, bool) {
        run_as(Family::Kokoro, pieces, options, keep)
    }

    fn run_as(
        family: Family,
        pieces: &[Vec<f32>],
        options: &SynthOptions,
        keep: usize,
    ) -> (Vec<Event>, Vec<Vec<f32>>, bool) {
        let log: Arc<Mutex<Vec<Event>>> = Arc::default();
        let chunks: Arc<Mutex<Vec<Vec<f32>>>> = Arc::default();
        let (sink_log, sink_chunks) = (Arc::clone(&log), Arc::clone(&chunks));
        let sink: Sink = Box::new(move |chunk: &[f32]| {
            sink_log.lock().unwrap().push(Event::Sink(chunk.len()));
            let mut chunks = sink_chunks.lock().unwrap();
            chunks.push(chunk.to_vec());
            chunks.len() < keep
        });
        let mut stream = Stream::new(options, family, RATE, sink);
        let outcome = Rc::clone(&stream.outcome);
        generate(pieces, &log, move |p| stream.piece(p));
        let log = std::mem::take(&mut *log.lock().unwrap());
        let chunks = std::mem::take(&mut *chunks.lock().unwrap());
        (log, chunks, outcome.cancelled.get())
    }

    fn raw(gap: f32) -> SynthOptions {
        SynthOptions {
            gap,
            level: false,
            ..Default::default()
        }
    }

    /// A piece with `pad` zeros at either end around a tone.
    fn piece(pad: usize, voiced: usize) -> Vec<f32> {
        let mut p = vec![0.0; pad];
        p.extend((0..voiced).map(|i| 0.2 * (i as f32 * 0.07).sin()));
        p.extend(vec![0.0; pad]);
        p
    }

    #[test]
    fn first_piece_reaches_the_sink_before_the_second_is_synthesised() {
        let pieces = [piece(0, 1_000), piece(0, 2_000), piece(0, 3_000)];
        let (log, _, _) = run(&pieces, &SynthOptions::default(), usize::MAX);
        let gap = (0.12 * RATE as f32) as usize;
        assert_eq!(
            log,
            [
                Event::Synth(0),
                Event::Sink(1_000),
                Event::Synth(1),
                Event::Sink(gap + 2_000),
                Event::Synth(2),
                Event::Sink(gap + 3_000),
            ]
        );
    }

    #[test]
    fn the_gap_goes_between_pieces_never_before_the_first() {
        let pieces = [piece(0, 500), piece(0, 700)];
        let (_, chunks, _) = run(&pieces, &raw(0.12), usize::MAX);
        let gap = (0.12 * RATE as f32) as usize;
        assert_eq!(gap, 2_880);
        assert_eq!(chunks[0], pieces[0]);
        assert_eq!(chunks[1].len(), gap + 700);
        assert!(chunks[1][..gap].iter().all(|&s| s == 0.0));
        assert_eq!(chunks[1][gap..], pieces[1][..]);
    }

    #[test]
    fn a_zero_gap_joins_pieces_directly() {
        let pieces = [piece(0, 500), piece(0, 700)];
        let (_, chunks, _) = run(&pieces, &raw(0.0), usize::MAX);
        assert_eq!(chunks, pieces);
    }

    #[test]
    fn an_empty_piece_sends_nothing_and_earns_no_gap() {
        let pieces = [vec![], piece(0, 500), vec![], piece(0, 700)];
        let (log, chunks, _) = run(&pieces, &raw(0.12), usize::MAX);
        let gap = (0.12 * RATE as f32) as usize;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], pieces[1]);
        assert_eq!(chunks[1].len(), gap + 700);
        assert_eq!(
            log.iter().filter(|e| matches!(e, Event::Synth(_))).count(),
            4
        );
    }

    /// The no-trim decision (docs/tts-spike.md (b)): a piece's own leading
    /// and trailing silence reaches the sink sample for sample.
    #[test]
    fn pieces_pass_through_untrimmed() {
        let pieces = [piece(1_500, 500), piece(900, 700)];
        let (_, chunks, _) = run(&pieces, &raw(0.0), usize::MAX);
        assert_eq!(chunks, pieces);
    }

    /// With the leveller on, every sample survives: it scales, never cuts,
    /// and the gap stays digital silence.
    #[test]
    fn levelling_keeps_every_sample_and_leaves_the_gap_silent() {
        let pieces = [piece(1_500, 5_000), piece(900, 5_000)];
        let options = SynthOptions {
            gap: 0.12,
            ..Default::default()
        };
        let (_, chunks, _) = run(&pieces, &options, usize::MAX);
        let gap = (0.12 * RATE as f32) as usize;
        assert_eq!(chunks[0].len(), pieces[0].len());
        assert_eq!(chunks[1].len(), gap + pieces[1].len());
        assert!(chunks[1][..gap].iter().all(|&s| s == 0.0));
        // Scaled: the tone's voiced RMS (~0.14) is above the 0.094 target,
        // so the levelled piece differs from the raw one.
        assert_ne!(chunks[0], pieces[0]);
        assert!(chunks[0][..1_500].iter().all(|&s| s == 0.0));
    }

    /// Pocket's pieces are chunks of a sentence: with the default gap and
    /// levelling they still reach the sink sample for sample.
    #[test]
    fn pocket_pieces_get_neither_a_gap_nor_levelling() {
        let pieces = [piece(1_500, 5_000), piece(0, 28_800), piece(900, 5_000)];
        let (_, chunks, _) = run_as(
            Family::Pocket,
            &pieces,
            &SynthOptions::default(),
            usize::MAX,
        );
        assert_eq!(chunks, pieces);
    }

    #[test]
    fn a_sink_returning_false_stops_synthesis() {
        let pieces = [piece(0, 500), piece(0, 700), piece(0, 900)];
        let (log, chunks, cancelled) = run(&pieces, &raw(0.12), 1);
        assert_eq!(chunks.len(), 1);
        assert!(cancelled);
        assert_eq!(log, [Event::Synth(0), Event::Sink(500)]);
    }

    /// Without the catch in `Stream::piece` this aborts the test binary.
    #[test]
    fn a_panicking_sink_resurfaces_as_an_ordinary_panic_after_generate() {
        let sink: Sink = Box::new(|_| panic!("sink failed"));
        let mut stream = Stream::new(&raw(0.12), Family::Kokoro, RATE, sink);
        let outcome = Rc::clone(&stream.outcome);
        let log: Arc<Mutex<Vec<Event>>> = Arc::default();
        generate(&[piece(0, 500), piece(0, 700)], &log, move |p| {
            stream.piece(p)
        });
        // The panic stopped synthesis like a cancel.
        assert_eq!(*log.lock().unwrap(), [Event::Synth(0)]);
        let payload = panic::catch_unwind(AssertUnwindSafe(|| outcome.finish(true)))
            .expect_err("the sink's panic is re-raised");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"sink failed"));
    }

    #[test]
    fn finish_reports_a_failed_generate_unless_cancelled() {
        let outcome = Outcome::default();
        assert!(outcome.finish(true).is_ok());
        assert!(matches!(
            outcome.finish(false),
            Err(TtsError::GenerateFailed)
        ));
        outcome.cancelled.set(true);
        assert!(outcome.finish(false).is_ok());
    }

    #[test]
    fn text_with_a_nul_is_refused_before_sherpa() {
        let err = check("Hello\0 there.", &SynthOptions::default()).unwrap_err();
        assert!(matches!(err, TtsError::NulInText), "{err}");
    }

    #[test]
    fn a_gap_outside_zero_to_five_seconds_is_refused() {
        for gap in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 5.01, 1e9] {
            let options = SynthOptions {
                gap,
                ..Default::default()
            };
            let err = check("Hello.", &options).unwrap_err();
            assert!(matches!(err, TtsError::InvalidGap(_)), "{gap}: {err}");
        }
        for gap in [0.0, 0.12, 5.0] {
            let options = SynthOptions {
                gap,
                ..Default::default()
            };
            assert!(check("Hello.", &options).is_ok(), "{gap}");
        }
    }

    #[test]
    fn abbreviations_lose_the_dot_that_would_split_them() {
        for (text, want) in [
            (
                "Dr. Smith went to Washington.",
                "Dr Smith went to Washington.",
            ),
            ("Ask mrs. Jones, e.g. today.", "Ask mrs Jones, e.g today."),
            ("Acme Inc.\nNext line.", "Acme Inc\nNext line."),
            ("Born in the U.S. in 1990.", "Born in the U.S in 1990."),
            // At the end of the text the dot splits nothing.
            ("He works for Acme Inc.", "He works for Acme Inc."),
        ] {
            assert_eq!(keep_abbreviations_whole(text), want, "{text}");
        }
    }

    #[test]
    fn sentence_ends_that_merely_end_like_an_abbreviation_keep_their_dot() {
        for text in [
            "That is normal. Then we left.",
            "It was a Dr.Who episode.",
            "Pi is 3.14 today. Yes.",
            "Stop. Go.",
        ] {
            assert_eq!(keep_abbreviations_whole(text), text);
        }
    }

    fn kokoro() -> Manifest {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        cat.models["kokoro-v1.0"].clone()
    }

    /// Placeholder files where the kokoro manifest looks for them.
    fn fake_model_dir() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for f in ["model.onnx", "voices.bin", "tokens.txt"] {
            std::fs::write(tmp.path().join(f), b"").unwrap();
        }
        std::fs::create_dir(tmp.path().join("espeak-ng-data")).unwrap();
        tmp
    }

    /// sherpa exits the process without `lang`, so it must never be called
    /// without one.
    #[test]
    fn a_manifest_without_lang_fails_before_sherpa() {
        let tmp = fake_model_dir();
        for lang in [None, Some("")] {
            let mut m = kokoro();
            let table = m.backend.get_mut("sherpa-onnx").unwrap();
            match lang {
                None => drop(table.remove("lang")),
                Some(v) => drop(table.insert("lang".into(), v.into())),
            }
            let err = SherpaTts::load(&m, tmp.path()).err().expect("an error");
            assert!(
                matches!(&err, TtsError::MissingConfig { key, .. } if key == "lang"),
                "{err}"
            );
        }
    }

    #[test]
    fn a_missing_model_file_is_an_error() {
        let tmp = fake_model_dir();
        std::fs::remove_file(tmp.path().join("voices.bin")).unwrap();
        let err = SherpaTts::load(&kokoro(), tmp.path())
            .err()
            .expect("an error");
        assert!(
            matches!(&err, TtsError::MissingModelFile(p) if p.ends_with("voices.bin")),
            "{err}"
        );
    }

    fn pocket() -> Manifest {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        cat.models["pocket-tts-int8"].clone()
    }

    /// An absent `family` and `family = "kokoro"` both load Kokoro (its
    /// voices.bin is looked for); `"pocket"` loads Pocket (its lm_flow is).
    #[test]
    fn family_picks_the_model_config() {
        let tmp = fake_model_dir();
        std::fs::remove_file(tmp.path().join("voices.bin")).unwrap();
        let mut explicit = kokoro();
        let table = explicit.backend.get_mut("sherpa-onnx").unwrap();
        table.insert("family".into(), "kokoro".into());
        for m in [kokoro(), explicit] {
            let err = SherpaTts::load(&m, tmp.path()).err().expect("an error");
            assert!(
                matches!(&err, TtsError::MissingModelFile(p) if p.ends_with("voices.bin")),
                "{err}"
            );
        }
        let err = SherpaTts::load(&pocket(), tmp.path())
            .err()
            .expect("an error");
        assert!(
            matches!(&err, TtsError::MissingModelFile(p) if p.ends_with("lm_flow.int8.onnx")),
            "{err}"
        );
    }

    #[test]
    fn an_unknown_family_is_an_error() {
        let tmp = fake_model_dir();
        for family in [toml::Value::from("vits"), toml::Value::from(1)] {
            let mut m = kokoro();
            let table = m.backend.get_mut("sherpa-onnx").unwrap();
            table.insert("family".into(), family);
            let err = SherpaTts::load(&m, tmp.path()).err().expect("an error");
            assert!(matches!(&err, TtsError::UnknownFamily { .. }), "{err}");
        }
    }

    /// Placeholder files where the pocket manifest looks for them, then the
    /// references: a voice without one, or whose file is missing, fails
    /// before sherpa.
    #[test]
    fn a_pocket_voice_needs_its_reference() {
        let tmp = tempfile::tempdir().unwrap();
        let m = pocket();
        for (key, value) in &m.backend["sherpa-onnx"] {
            // `prompt_format` (naru_1457) is a table, not a model-file
            // name, same reason `family` is excluded here.
            if key != "family" && key != "prompt_format" {
                std::fs::write(tmp.path().join(value.as_str().unwrap()), b"").unwrap();
            }
        }
        let err = SherpaTts::load(&m, tmp.path()).err().expect("an error");
        assert!(
            matches!(&err, TtsError::MissingModelFile(p) if p.ends_with("test_wavs/bria.wav")),
            "{err}"
        );
        let mut m = pocket();
        m.voices[0].reference = None;
        let err = SherpaTts::load(&m, tmp.path()).err().expect("an error");
        assert!(
            matches!(&err, TtsError::MissingConfig { key, .. } if key == "voice.bria.reference"),
            "{err}"
        );
    }

    #[test]
    fn sherpa_tts_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SherpaTts>();
    }
}
