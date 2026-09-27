//! §3.6 text-to-speech. A backend's `load(manifest, dir)`
//! (`crate::backend::load_tts`) returns a [`TtsModel`]; `synth` streams
//! mono f32 samples at [`TtsModel::sample_rate`] to a sink as each sentence
//! is synthesised.
//!
//! What §5.1 planned to port from kokoro-rs, after the task 9 spike
//! (docs/tts-spike.md): the character chunker is dropped for sherpa's own
//! sentence split plus the `generate` callback, trim.rs is not ported
//! (sherpa leaves at most 76 ms at either end of a piece), and the
//! [`level::Leveller`] and the gap between sentences are kept.

pub mod level;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod mlx;
pub(crate) mod sherpa;

use std::path::PathBuf;

use crate::registry::manifest::Voice;

/// Receives each piece of audio in order and returns `false` to cancel the
/// rest. It is called on the synthesising thread, before the next piece is
/// synthesised.
pub type Sink = Box<dyn FnMut(&[f32]) -> bool + Send>;

/// The per-request knobs (§2.3 `speed` and `instructions`, and the
/// `gap`/`level`/`exaggeration` extensions).
#[derive(Debug, Clone, PartialEq)]
pub struct SynthOptions {
    /// 0.5–2.0; the API checks the range.
    pub speed: f32,
    /// Seconds of silence put back between sentences (kokoro-rs `--gap`),
    /// 0–5; `synth` rejects anything else.
    pub gap: f32,
    /// Whether the [`level::Leveller`] runs.
    pub level: bool,
    /// What the voice should be and how it should speak, for a model that
    /// takes it (`Manifest::instructs`); `None` for any other.
    pub instructions: Option<String>,
    /// Emotion exaggeration, 0-1, for a model that takes it
    /// (`Manifest::exaggerates`, mlx-audio's Chatterbox); `None` for any
    /// other.
    pub exaggeration: Option<f32>,
    /// A reference clip and its transcript to clone from directly, without
    /// a saved voice (`POST /api/voices/preview`, naru task 1458): wins
    /// over any named voice in `MlxTts::synth`. The `sherpa-onnx` backend
    /// never clones, so it ignores this.
    pub reference: Option<(PathBuf, String)>,
}

impl Default for SynthOptions {
    fn default() -> Self {
        SynthOptions {
            speed: 1.0,
            gap: 0.12,
            level: true,
            instructions: None,
            exaggeration: None,
            reference: None,
        }
    }
}

pub trait TtsModel: Send + Sync {
    /// The manifest's `[[voice]]` table.
    fn voices(&self) -> &[Voice];

    /// Of every sample `synth` produces.
    fn sample_rate(&self) -> u32;

    /// Synthesises `text` in `voice` (a [`Voice::id`]) and hands the audio
    /// to `sink` a sentence at a time. A cancelled synthesis is `Ok`.
    fn synth(
        &self,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        sink: Sink,
    ) -> Result<(), TtsError>;

    /// The bytes the model holds outside the daemon's own RSS (§3.5: an
    /// MLX model lives in the sidecar), when it reports them.
    fn resident_bytes(&self) -> Option<u64> {
        None
    }
}

#[derive(Debug)]
pub enum TtsError {
    /// §3.6: the manifest's backend cannot run here, or cannot load TTS.
    BackendUnavailable {
        backend: String,
        reason: String,
    },
    /// The manifest is not a TTS model.
    NotTts(String),
    /// The manifest's `[backend.<name>]` table lacks a key the backend needs.
    MissingConfig {
        model: String,
        key: String,
    },
    /// `[backend.sherpa-onnx] family` names no model family sherpa is
    /// wired for here.
    UnknownFamily {
        model: String,
        family: String,
    },
    MissingModelFile(std::path::PathBuf),
    /// `OfflineTts::create` returned `None`.
    CreateFailed,
    UnknownVoice {
        model: String,
        voice: String,
    },
    /// The text holds a NUL, which sherpa cannot take.
    NulInText,
    /// `gap` is not in 0–5 s.
    InvalidGap(f32),
    /// `generate` returned no audio without being cancelled.
    GenerateFailed,
    /// The MLX sidecar answered with an error.
    Sidecar(String),
}

impl std::fmt::Display for TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsError::BackendUnavailable { backend, reason } => {
                write!(f, "backend `{backend}` unavailable: {reason}")
            }
            TtsError::NotTts(name) => write!(f, "`{name}` is not a text-to-speech model"),
            TtsError::MissingConfig { model, key } => {
                write!(f, "`{model}` has no backend `{key}` setting")
            }
            TtsError::UnknownFamily { model, family } => write!(
                f,
                "`{model}` has unknown family {family}; expected \"kokoro\" or \"pocket\""
            ),
            TtsError::MissingModelFile(path) => {
                write!(f, "model file missing: {}", path.display())
            }
            TtsError::CreateFailed => write!(f, "failed to create the offline TTS"),
            TtsError::UnknownVoice { model, voice } => {
                write!(f, "`{model}` has no voice `{voice}`")
            }
            TtsError::NulInText => write!(f, "the text contains a NUL character"),
            TtsError::InvalidGap(gap) => write!(f, "gap {gap} is not between 0 and 5 seconds"),
            TtsError::GenerateFailed => write!(f, "synthesis produced no audio"),
            TtsError::Sidecar(message) => write!(f, "the MLX sidecar failed: {message}"),
        }
    }
}

impl std::error::Error for TtsError {}
