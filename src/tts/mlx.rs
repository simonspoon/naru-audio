//! The `mlx` [`TtsModel`] (§5.3): mlx-audio in the sidecar
//! (`crate::mlx::sidecar`), which streams each chunk as it is generated.
//! The model is whatever mlx-audio's `load_model` makes of the pulled
//! directory; a voice is its `[[voice]]` id, handed to mlx-audio's
//! `generate` as `voice`, and its `reference` recording, if any, as
//! `ref_audio`. A model that clones (`crate::voices`) also takes a cloned
//! voice's name: its clip goes as `ref_audio` and its transcript as
//! `ref_text`, with no `voice`. A model that instructs
//! (`Manifest::instructs`) is sent the request's `instructions` as
//! mlx-audio's `instruct`; if it has no voices (VoiceDesign), that is its
//! voice, and the voice asked for is ignored. VoxCPM2 both clones and
//! instructs: a named cloned voice wins (`crate::server::speech::voice`),
//! else `instructions` designs one. `sid` is not used. `speed` is passed
//! on as mlx-audio's `speed`, which Qwen3-TTS ignores; Chatterbox and
//! VoxCPM2 ignore it too (VoxCPM2's `generate` has no `speed` parameter at
//! all), but the sidecar shim rejects a non-1.0 `speed` for either
//! outright, so a caller gets a clear error instead of normal-speed audio
//! (`naru_audio_mlx/__main__.py`). A model that exaggerates
//! (`Manifest::exaggerates`, Chatterbox) is sent the request's
//! `exaggeration` as mlx-audio's `exaggeration`.
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
use crate::voices;

pub struct MlxTts {
    sidecar: Arc<Sidecar>,
    name: String,
    /// Which load of `name` this is, for [`Sidecar::unload`].
    instance: u64,
    dir: PathBuf,
    voices: Vec<Voice>,
    /// The home whose cloned voices the model speaks in, if it clones.
    cloned: Option<PathBuf>,
    /// Whether the model takes `instructions`.
    instructs: bool,
    /// Whether the model takes `exaggeration`.
    exaggerates: bool,
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
            cloned: manifest.clones().then(|| home.to_path_buf()),
            instructs: manifest.instructs(),
            exaggerates: manifest.exaggerates(),
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
        let mut request = json!({"text": text, "speed": options.speed});
        if let Some((wav, ref_text)) = &options.reference {
            // §2.6 `POST /api/voices/preview` (naru task 1458): an explicit
            // reference clip wins over any named voice, including an empty
            // one the preview endpoint passes since it has no saved voice.
            request["reference"] = json!(wav);
            request["reference_text"] = json!(ref_text);
        } else if let Some(v) = self.voices.iter().find(|v| v.id == voice) {
            request["voice"] = json!(v.id);
            if let Some(reference) = &v.reference {
                request["reference"] = json!(self.dir.join(reference));
            }
        } else if let Some(c) = self
            .cloned
            .as_deref()
            .and_then(|home| voices::find(home, voice))
        {
            request["reference"] = json!(c.wav);
            request["reference_text"] = json!(c.text);
        } else if !(self.instructs && self.voices.is_empty()) {
            return Err(TtsError::UnknownVoice {
                model: self.name.clone(),
                voice: voice.to_string(),
            });
        }
        if self.instructs
            && let Some(instructions) = &options.instructions
        {
            request["instruct"] = json!(instructions);
        }
        if self.exaggerates
            && let Some(exaggeration) = options.exaggeration
        {
            request["exaggeration"] = json!(exaggeration);
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
