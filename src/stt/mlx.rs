//! The `mlx` [`SttModel`] (§5.3): the sherpa-onnx model's energy and
//! Silero gates in the daemon, and a decode per utterance by Parakeet
//! in-process (`crate::mlx::parakeet`). It has no hotword biasing, so
//! `hotwords` are ignored, and reports no language or word timestamps.

use std::path::{Path, PathBuf};

use super::audio::TARGET_SAMPLE_RATE;
use super::gate::SpeechGate;
use super::sherpa::{load_gate, utterances, vad_model};
use super::vad::VadConfig;
use super::vocabulary::Vocabulary;
use super::{Segment, SttError, SttModel};
use crate::mlx::parakeet::Parakeet;
use crate::registry::manifest::Manifest;

pub struct MlxStt {
    model: Parakeet,
    vad_model: PathBuf,
    gate: Option<SpeechGate>,
}

impl MlxStt {
    /// `dir` is `models/<name>/` (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, SttError> {
        let name = &manifest.model.name;
        let unavailable = |reason: String| SttError::BackendUnavailable {
            backend: manifest.model.backend.clone(),
            reason,
        };
        if !name.starts_with("parakeet") {
            return Err(unavailable(format!("no native MLX engine for {name}")));
        }
        let vad_model = vad_model(manifest, dir)?;
        Ok(MlxStt {
            model: Parakeet::load(dir).map_err(unavailable)?,
            vad_model,
            gate: load_gate(dir),
        })
    }
}

impl SttModel for MlxStt {
    /// One decode per utterance.
    fn decode_each(
        &self,
        pcm16k: &[f32],
        _hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        let rate = TARGET_SAMPLE_RATE as f64;
        for (span, from, to) in utterances(pcm16k, &self.vad_model, self.gate.as_ref(), vad)? {
            let text = self
                .model
                .transcribe(&pcm16k[from..to])
                .map_err(SttError::Native)?;
            if !text.is_empty() {
                on_segment(Segment {
                    start: span.start as f64 / rate,
                    end: (span.start + span.len) as f64 / rate,
                    text,
                });
            }
        }
        Ok(())
    }

    fn vad_model(&self) -> Option<&Path> {
        Some(&self.vad_model)
    }

    fn resident_bytes(&self) -> Option<u64> {
        Some(self.model.resident_bytes())
    }

    fn engine(&self) -> Option<(&'static str, Option<&str>)> {
        Some(("native", None))
    }
}
