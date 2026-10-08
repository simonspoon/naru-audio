//! The speech gate: a small AudioSet tagger (CED-tiny) run on each utterance
//! right before the recognizer.
//!
//! Silero (and the clients' RMS VADs) pass loud non-speech such as a dog
//! bark, and Parakeet then invents words for it ("That's easy for you.").
//! The tagger names what the sound is, so a bark or a clap can be refused
//! like the Silero gate refuses: the utterance is dropped and nothing is
//! transcribed.
//!
//! **The rule vetoes, it does not demand speech.** Short real speech ("yes",
//! "stop") tags as Speech with only 0.3–0.6 and overlaps the speech score of
//! a bark (measured, see [`decide`]), so no speech threshold separates them.
//! What does separate them is a *clear other sound* with no confident
//! speech beside it: an utterance is refused only when its strongest
//! non-speech label reaches [`VETO_PROB`], outweighs its strongest speech
//! label by [`DOMINANCE`], and that speech label is below [`SPEECH_PROB`].
//! Speech over music (music 0.8, speech 0.6) passes.
//!
//! **Fails open.** No model, a model that will not load, or an utterance too
//! long to be one stray noise all transcribe as before.

use std::path::Path;

use sherpa_onnx::{AudioEvent, AudioTagging, AudioTaggingConfig, AudioTaggingModelConfig};

use super::audio::TARGET_SAMPLE_RATE;

/// The catalog model providing the tagger (`models/<name>/`).
pub const MODEL_NAME: &str = "audio-tagging-ced-tiny";

const MODEL_FILE: &str = "model.int8.onnx";
const LABELS_FILE: &str = "class_labels_indices.csv";

/// AudioSet classes 0..=15 are Speech, its sub-kinds, Narration,
/// Babbling, Speech synthesizer, and the raised or hushed voice (Shout ..
/// Whispering). Laughter, crying and singing are not speech.
const SPEECH_CLASSES: std::ops::RangeInclusive<i32> = 0..=15;

/// A non-speech label at or above this probability is a clear other sound.
const VETO_PROB: f32 = 0.4;

/// The non-speech label must also be this many times the speech label, so a
/// toss-up (a cut-off word tagged Mantra 0.44 against Speech 0.42) passes.
const DOMINANCE: f32 = 1.5;

/// A speech label at or above this probability is confident speech, which
/// outranks any other sound heard with it.
const SPEECH_PROB: f32 = 0.5;

/// Longer audio is never gated: one buffer that long is not one stray noise
/// (a whole recording sent with `vad=false`), and the tagger's cost grows
/// with length.
const MAX_GATED_SECONDS: usize = 10;

/// Shorter audio is never gated: too little to tag, and the model needs
/// some frames to hear anything. Set at 0.6 s because a one-word reply
/// ("no", "hi", "sure", 0.4–0.6 s) tags as Music, Synthesizer or Buzz at
/// 0.4–0.7 with speech under 0.4, and the veto then drops a real answer
/// (naru task 1685). A bark or clap shorter than this passes too.
const MIN_GATED_SECONDS: f32 = 0.6;

/// All 527 AudioSet classes, so the strongest of each side is always seen.
const TOP_K: i32 = 527;

pub struct SpeechGate {
    tagger: AudioTagging,
}

impl SpeechGate {
    /// Loads the tagger from `models/<MODEL_NAME>/` under `models_dir`.
    /// `None` (logged unless the model is simply not pulled) means no gate.
    pub fn load(models_dir: &Path) -> Option<SpeechGate> {
        let dir = models_dir.join(MODEL_NAME);
        let (model, labels) = (dir.join(MODEL_FILE), dir.join(LABELS_FILE));
        if !model.is_file() || !labels.is_file() {
            eprintln!(
                "naru-audio: speech gate off: {MODEL_NAME} not pulled (run `naru-audio pull {MODEL_NAME}`)"
            );
            return None;
        }
        let config = AudioTaggingConfig {
            model: AudioTaggingModelConfig {
                ced: Some(model.to_string_lossy().into_owned()),
                num_threads: 1,
                ..Default::default()
            },
            labels: Some(labels.to_string_lossy().into_owned()),
            top_k: TOP_K,
        };
        let tagger = AudioTagging::create(&config);
        if tagger.is_none() {
            eprintln!(
                "naru-audio: speech gate off: failed to load {}",
                model.display()
            );
        }
        tagger.map(|tagger| SpeechGate { tagger })
    }

    /// Whether `samples` (16 kHz mono) may go to the recognizer.
    pub fn allows(&self, samples: &[f32]) -> bool {
        if !is_gated_length(samples.len()) {
            return true;
        }
        let stream = self.tagger.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        let events = self.tagger.compute(&stream, TOP_K);
        let allowed = decide(&events);
        if !allowed {
            let top: Vec<String> = events
                .iter()
                .take(3)
                .map(|e| format!("{}={:.2}", e.name.split(',').next().unwrap_or(""), e.prob))
                .collect();
            eprintln!(
                "naru-audio: speech gate refused {:.2}s utterance: {}",
                samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
                top.join(" ")
            );
        }
        allowed
    }
}

/// Whether a clip of `len` samples is long enough, and short enough, to gate.
fn is_gated_length(len: usize) -> bool {
    let rate = TARGET_SAMPLE_RATE as usize;
    len >= (MIN_GATED_SECONDS * rate as f32) as usize && len <= MAX_GATED_SECONDS * rate
}

/// The rule over a tagger's scored labels; see the module doc. Measured with
/// CED-tiny on 64 real ESC-50 clips (dog barks, claps, knocks, glass,
/// coughs, laughs, clicks) and on real and `say` speech down to 0.6 s: the
/// barks and claps score their top label at 0.45–0.95 with speech at most
/// 0.42 (a bark at 0.78 carried speech 0.42, so a speech threshold alone
/// would pass it), while no real speech clip, whole or cut to 0.7 s, had a
/// non-speech label that both reached 0.4 and outweighed its speech label
/// 1.5 to 1.
fn decide(events: &[AudioEvent]) -> bool {
    let strongest = |speech: bool| {
        events
            .iter()
            .filter(|e| SPEECH_CLASSES.contains(&e.index) == speech)
            .map(|e| e.prob)
            .fold(0.0f32, f32::max)
    };
    let (other, speech) = (strongest(false), strongest(true));
    !(other >= VETO_PROB && other >= DOMINANCE * speech && speech < SPEECH_PROB)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(index: i32, prob: f32) -> AudioEvent {
        AudioEvent {
            name: format!("class {index}"),
            index,
            prob,
        }
    }

    #[test]
    fn a_confident_bark_with_weak_speech_is_refused() {
        // Dog 74, Bark 75; Speech 0.42 (measured on a real dog clip).
        assert!(!decide(&[ev(74, 0.78), ev(0, 0.42), ev(7, 0.1)]));
    }

    #[test]
    fn a_plain_bark_at_the_veto_edge_is_refused() {
        assert!(!decide(&[ev(74, 0.47), ev(0, 0.018)]));
    }

    #[test]
    fn dominance_boundary_is_refused() {
        assert!(!decide(&[ev(74, 0.6), ev(0, 0.4)]));
    }

    #[test]
    fn a_toss_up_between_speech_and_another_sound_passes() {
        // A real word cut short: Mantra 0.44 against Speech 0.42.
        assert!(decide(&[ev(300, 0.44), ev(7, 0.42)]));
    }

    #[test]
    fn short_speech_with_no_other_sound_passes() {
        assert!(decide(&[ev(7, 0.52), ev(0, 0.43), ev(500, 0.07)]));
    }

    #[test]
    fn faint_speech_over_a_faint_sound_passes() {
        assert!(decide(&[ev(500, 0.35), ev(0, 0.2)]));
    }

    #[test]
    fn confident_speech_outranks_music() {
        assert!(decide(&[ev(137, 0.8), ev(0, 0.6)]));
    }

    #[test]
    fn whispering_counts_as_speech() {
        assert!(decide(&[ev(500, 0.55), ev(15, 0.6)]));
    }

    #[test]
    fn laughter_alone_is_refused() {
        assert!(!decide(&[ev(21, 0.6), ev(0, 0.1)]));
    }

    #[test]
    fn only_clips_between_the_bounds_are_gated() {
        let rate = TARGET_SAMPLE_RATE as usize;
        assert!(!is_gated_length(rate / 2));
        assert!(is_gated_length(rate));
        assert!(!is_gated_length(MAX_GATED_SECONDS * rate + 1));
    }

    #[test]
    fn no_labels_passes() {
        assert!(decide(&[]));
    }

    #[test]
    fn missing_model_is_no_gate_not_a_panic() {
        assert!(SpeechGate::load(Path::new("/nonexistent/naru-audio-gate-test")).is_none());
    }
}
