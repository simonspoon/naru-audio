//! §3.6 speech-to-text: the auris engine as a library. A backend's
//! `load(manifest, dir)` (`crate::backend::load_stt`) returns an
//! [`SttModel`]; `decode` turns 16 kHz mono samples into [`Segment`]s.
//!
//! Nothing transcribed (silence, or every gate rejected the audio) is
//! `Ok(vec![])`, never an error (§2.2 "Silence contract").

pub mod audio;
pub mod engine;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) mod mlx;
pub(crate) mod sherpa;
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
