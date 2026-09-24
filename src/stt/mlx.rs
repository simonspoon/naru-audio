//! The `mlx` [`SttModel`] (§5.3): the sherpa-onnx model's energy and
//! Silero gates in the daemon, and a decode per utterance in the sidecar
//! (`crate::mlx::sidecar`). parakeet-mlx has no hotword biasing, so
//! `hotwords` are ignored.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::audio::TARGET_SAMPLE_RATE;
use super::sherpa::{utterances, vad_model};
use super::vad::VadConfig;
use super::vocabulary::Vocabulary;
use super::{Segment, SttError, SttModel};
use crate::mlx::sidecar::{self, Sidecar};
use crate::registry::manifest::Manifest;

pub struct MlxStt {
    sidecar: Arc<Sidecar>,
    name: String,
    /// Which load of `name` this is, for [`Sidecar::unload`].
    instance: u64,
    vad_model: PathBuf,
    /// What the load added to the sidecar's MLX active memory.
    resident_bytes: u64,
}

impl MlxStt {
    /// `dir` is `models/<name>/`, in the home whose sidecar loads it (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, SttError> {
        let vad_model = vad_model(manifest, dir)?;
        let home = dir.parent().and_then(Path::parent).unwrap_or(dir);
        let sidecar = sidecar::for_home(home);
        let name = manifest.model.name.clone();
        let (instance, resident_bytes) = sidecar.load(&name, dir)?;
        Ok(MlxStt {
            sidecar,
            name,
            instance,
            vad_model,
            resident_bytes,
        })
    }
}

impl SttModel for MlxStt {
    fn decode_each(
        &self,
        pcm16k: &[f32],
        _hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        let rate = TARGET_SAMPLE_RATE as f64;
        for (span, from, to) in utterances(pcm16k, &self.vad_model, vad)? {
            let text = self
                .sidecar
                .transcribe(&self.name, &pcm16k[from..to])?
                .join(" ");
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
        Some(self.resident_bytes)
    }
}

/// Unloading the model unloads it from the sidecar, and the last MLX model
/// out stops the sidecar.
impl Drop for MlxStt {
    fn drop(&mut self) {
        self.sidecar.unload(&self.name, self.instance);
    }
}
