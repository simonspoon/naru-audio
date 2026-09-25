//! The `mlx` [`TtsModel`] (§5.3): mlx-audio in the sidecar
//! (`crate::mlx::sidecar`), which streams each chunk as it is generated.
//! The model is whatever mlx-audio's `load_model` makes of the pulled
//! directory; a voice is its `[[voice]]` id, handed to mlx-audio's
//! `generate` as `voice`, and its `reference` recording, if any, as
//! `ref_audio`. `sid` is not used. `speed` is passed on as mlx-audio's
//! `speed`, which Qwen3-TTS ignores.
//!
//! The sidecar is held for a whole synthesis, so `synth` never lets the
//! sink hold it: chunks go through an unbounded queue to a thread that
//! feeds the sink, and the sidecar is free once generation ends, however
//! slowly the client reads.
//!
//! No gap and no leveller, as for Pocket (`super::sherpa`): the chunks are
//! fixed lengths of audio cut mid-sentence (0.32 s, `STREAMING_INTERVAL` in
//! the sidecar), so a gap would be a pause mid-word, and a gain per chunk
//! would step in level from one to the next. `gap` is still checked, so a
//! request means the same whatever model serves it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use serde_json::json;

use super::sherpa::check;
use super::{Sink, SynthOptions, TtsError, TtsModel};
use crate::mlx::sidecar::{self, Sidecar};
use crate::registry::manifest::{Kind, Manifest, Voice};
use crate::stt::SttError;

pub struct MlxTts {
    sidecar: Arc<Sidecar>,
    name: String,
    /// Which load of `name` this is, for [`Sidecar::unload`].
    instance: u64,
    dir: PathBuf,
    voices: Vec<Voice>,
    /// What the sidecar's `load` reported.
    sample_rate: u32,
    /// What the load added to the sidecar's MLX active memory.
    resident_bytes: u64,
}

impl MlxTts {
    /// `dir` is `models/<name>/`, in the home whose sidecar loads it (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, TtsError> {
        let home = dir.parent().and_then(Path::parent).unwrap_or(dir);
        let sidecar = sidecar::for_home(home);
        let name = manifest.model.name.clone();
        let (instance, resident_bytes, answer) =
            sidecar.load(&name, Kind::Tts, dir).map_err(tts_error)?;
        let mut tts = MlxTts {
            sidecar,
            name,
            instance,
            dir: dir.to_path_buf(),
            voices: manifest.voices.clone(),
            sample_rate: 0,
            resident_bytes,
        };
        // An error here drops `tts`, which unloads the model.
        tts.sample_rate = answer["sample_rate"]
            .as_u64()
            .and_then(|r| u32::try_from(r).ok())
            .filter(|&r| r > 0)
            .ok_or_else(|| TtsError::Sidecar("`load` has no sample_rate".to_string()))?;
        Ok(tts)
    }
}

impl TtsModel for MlxTts {
    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn synth(
        &self,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        mut sink: Sink,
    ) -> Result<(), TtsError> {
        check(text, options)?;
        let voice =
            self.voices
                .iter()
                .find(|v| v.id == voice)
                .ok_or_else(|| TtsError::UnknownVoice {
                    model: self.name.clone(),
                    voice: voice.to_string(),
                })?;
        let mut request = json!({"text": text, "voice": voice.id, "speed": options.speed});
        if let Some(reference) = &voice.reference {
            request["reference"] = json!(self.dir.join(reference));
        }
        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        std::thread::scope(|scope| {
            // A cancel drops `rx`, so the next send fails and the sidecar
            // is told to stop. A panic in the sink does the same, and
            // resumes when the scope joins.
            scope.spawn(move || {
                for piece in rx {
                    if !sink(&piece) {
                        break;
                    }
                }
            });
            let result = self
                .sidecar
                .synth(&self.name, request, &mut |samples: &[f32]| {
                    tx.send(samples.to_vec()).is_ok()
                })
                .map_err(tts_error);
            // Ends the feeding thread once the queue is empty.
            drop(tx);
            result
        })
    }

    fn resident_bytes(&self) -> Option<u64> {
        Some(self.resident_bytes)
    }
}

/// Unloading the model unloads it from the sidecar, and the last MLX model
/// out stops the sidecar.
impl Drop for MlxTts {
    fn drop(&mut self) {
        self.sidecar.unload(&self.name, self.instance);
    }
}

/// The sidecar's errors are [`SttError`]s, which it was written for: a dead
/// sidecar stays `BackendUnavailable`, which `/v1/audio/speech` answers
/// with 503 before any audio, and an error answer is [`TtsError::Sidecar`].
fn tts_error(e: SttError) -> TtsError {
    match e {
        SttError::BackendUnavailable { backend, reason } => {
            TtsError::BackendUnavailable { backend, reason }
        }
        SttError::Sidecar(message) => TtsError::Sidecar(message),
        other => TtsError::Sidecar(other.to_string()),
    }
}
