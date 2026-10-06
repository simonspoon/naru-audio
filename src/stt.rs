//! §3.6 speech-to-text: the auris engine as a library. A backend's
//! `load(manifest, dir)` (`crate::backend::load_stt`) returns an
//! [`SttModel`]; `decode` turns 16 kHz mono samples into [`Segment`]s.
//!
//! Nothing transcribed (silence, or every gate rejected the audio) is
//! `Ok(vec![])`, never an error (§2.2 "Silence contract").

pub mod audio;
pub mod engine;
pub mod gate;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod mlx;
pub(crate) mod sherpa;
pub mod spoken;
pub mod vad;
pub mod vocabulary;

pub use vad::VadConfig;
pub use vocabulary::Vocabulary;

use std::path::Path;

/// One utterance's transcript. `start`/`end` are seconds from the start of
/// the audio: the speech Silero detected (the whole buffer with VAD off),
/// not the slice that was decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// One word within a [`Segment`], for voice-prep's crop-by-word-span
/// (naru task 1461 §8). `start`/`end` are seconds from the start of the
/// audio, like `Segment`'s.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The engine's confidence in the word, 0..=1; `None` where the engine
    /// reports none (only MLX Whisper does).
    pub confidence: Option<f32>,
}

pub trait SttModel: Send + Sync {
    /// Transcribes `pcm16k`, biased towards `hotwords` when given and
    /// non-empty. `vad: None` skips the Silero gate and decodes the whole
    /// buffer as one utterance (§2.2 `vad=false`). Each segment goes to
    /// `on_segment` as soon as it is decoded (§2.2 `stream`).
    fn decode_each(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError>;

    /// [`SttModel::decode_each`], collected.
    fn decode(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
    ) -> Result<Vec<Segment>, SttError> {
        let mut segments = Vec::new();
        self.decode_each(pcm16k, hotwords, vad, &mut |s| segments.push(s))?;
        Ok(segments)
    }

    /// [`Segment`]s split into [`Word`]s with their own timestamps
    /// (naru task 1461 §8: voice-prep's transcript step). `Err(SttError::
    /// WordTimestampsUnsupported)` by default; only a backend that reports
    /// per-token timestamps (`sherpa-onnx`'s Parakeet, the MLX Whisper via
    /// [`SttModel::decode_words_in`]) overrides it.
    fn decode_words(
        &self,
        _pcm16k: &[f32],
        _vad: Option<&VadConfig>,
    ) -> Result<Vec<Word>, SttError> {
        Err(SttError::WordTimestampsUnsupported)
    }

    /// [`SttModel::decode_each`] for a multilingual model (naru task 1466):
    /// `language` is the caller's requested language, `None` to let the
    /// model detect it. Returns the language the decode used (the request's,
    /// else the one detected), `None` when the backend has no notion of one;
    /// the default ignores `language` and returns `None`.
    fn decode_each_in(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        _language: Option<&str>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<Option<String>, SttError> {
        self.decode_each(pcm16k, hotwords, vad, on_segment)
            .map(|()| None)
    }

    /// [`SttModel::decode_words`] with [`SttModel::decode_each_in`]'s
    /// `language` in and language used out.
    fn decode_words_in(
        &self,
        pcm16k: &[f32],
        vad: Option<&VadConfig>,
        _language: Option<&str>,
    ) -> Result<(Vec<Word>, Option<String>), SttError> {
        self.decode_words(pcm16k, vad).map(|words| (words, None))
    }

    /// [`SttModel::decode_words_in`] that keeps fillers, false starts and
    /// repetitions when `verbatim` and the backend can (the MLX Whisper);
    /// the `bool` answered is whether the decode was verbatim. The default
    /// is a normal decode, answered `false`.
    fn decode_words_verbatim(
        &self,
        pcm16k: &[f32],
        vad: Option<&VadConfig>,
        language: Option<&str>,
        _verbatim: bool,
    ) -> Result<(Vec<Word>, Option<String>, bool), SttError> {
        self.decode_words_in(pcm16k, vad, language)
            .map(|(words, language)| (words, language, false))
    }

    /// The Silero model streaming sessions segment with (§2.4); `None`
    /// when the model has none, and cannot stream.
    fn vad_model(&self) -> Option<&Path> {
        None
    }

    /// The bytes the model holds outside the daemon's own RSS (§3.5: an
    /// MLX model lives in the sidecar), when it reports them.
    fn resident_bytes(&self) -> Option<u64> {
        None
    }
}

#[derive(Debug)]
pub enum SttError {
    /// §3.6: the manifest's backend cannot run here, or cannot load STT.
    BackendUnavailable {
        backend: String,
        reason: String,
    },
    /// The manifest is not an STT model.
    NotStt(String),
    /// No `requires`d model provides `silero_vad.onnx`.
    NoVadModel(String),
    Engine(engine::EngineError),
    Vad(vad::VadError),
    /// The MLX sidecar answered a request with an error (§5.3).
    Sidecar(String),
    /// [`SttModel::decode_words`]'s default: this backend reports no
    /// per-token timestamps.
    WordTimestampsUnsupported,
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttError::BackendUnavailable { backend, reason } => {
                write!(f, "backend `{backend}` unavailable: {reason}")
            }
            SttError::NotStt(name) => write!(f, "`{name}` is not a speech-to-text model"),
            SttError::NoVadModel(name) => {
                write!(
                    f,
                    "`{name}` requires no model with a {}",
                    sherpa::VAD_FILENAME
                )
            }
            SttError::Engine(e) => e.fmt(f),
            SttError::Vad(e) => e.fmt(f),
            SttError::Sidecar(message) => write!(f, "the MLX sidecar failed: {message}"),
            SttError::WordTimestampsUnsupported => {
                write!(f, "this model does not report word timestamps")
            }
        }
    }
}

impl std::error::Error for SttError {}

impl From<engine::EngineError> for SttError {
    fn from(e: engine::EngineError) -> Self {
        SttError::Engine(e)
    }
}

impl From<vad::VadError> for SttError {
    fn from(e: vad::VadError) -> Self {
        SttError::Vad(e)
    }
}
