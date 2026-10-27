//! The `mlx` [`SttModel`] (§5.3): the sherpa-onnx model's energy and
//! Silero gates in the daemon, and a decode per utterance in the sidecar
//! (`crate::mlx::sidecar`): parakeet-mlx or mlx-whisper, by model name.
//! Neither has hotword biasing, so `hotwords` are ignored; only Whisper
//! takes a language and reports word timestamps. With `[mlx] native_stt`,
//! Parakeet decodes in-process instead (`crate::mlx::parakeet`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::audio::TARGET_SAMPLE_RATE;
use super::gate::SpeechGate;
use super::sherpa::{load_gate, utterances, vad_model};
use super::vad::VadConfig;
use super::vocabulary::Vocabulary;
use super::{Segment, SttError, SttModel, Word};
use crate::mlx::parakeet::Parakeet;
use crate::mlx::sidecar::{self, Sidecar};
use crate::registry::manifest::{Kind, Manifest};

pub struct MlxStt {
    engine: Engine,
    name: String,
    vad_model: PathBuf,
    gate: Option<SpeechGate>,
}

enum Engine {
    Sidecar {
        sidecar: Arc<Sidecar>,
        /// Which load of `name` this is, for [`Sidecar::unload`].
        instance: u64,
        /// What the load added to the sidecar's MLX active memory.
        resident_bytes: u64,
    },
    Native(Parakeet),
}

impl MlxStt {
    /// `dir` is `models/<name>/`, in the home whose sidecar loads it (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, SttError> {
        let vad_model = vad_model(manifest, dir)?;
        let home = dir.parent().and_then(Path::parent).unwrap_or(dir);
        let name = manifest.model.name.clone();
        let unavailable = |reason: String| SttError::BackendUnavailable {
            backend: manifest.model.backend.clone(),
            reason,
        };
        let engine = if crate::mlx::native_stt(home, &name).map_err(unavailable)? {
            Engine::Native(Parakeet::load(dir).map_err(unavailable)?)
        } else {
            let sidecar = sidecar::for_home(home);
            let (instance, resident_bytes, _) = sidecar.load(&name, Kind::Stt, dir)?;
            Engine::Sidecar {
                sidecar,
                instance,
                resident_bytes,
            }
        };
        Ok(MlxStt {
            engine,
            name,
            vad_model,
            gate: load_gate(dir),
        })
    }
}

impl SttModel for MlxStt {
    fn decode_each(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        self.decode_each_in(pcm16k, hotwords, vad, None, on_segment)
            .map(|_| ())
    }

    /// One sidecar decode per utterance. With no `language`, the first
    /// utterance's detected language is the hint for the rest, so one
    /// request is transcribed in one language.
    fn decode_each_in(
        &self,
        pcm16k: &[f32],
        _hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        language: Option<&str>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<Option<String>, SttError> {
        let rate = TARGET_SAMPLE_RATE as f64;
        let mut language = language.map(str::to_string);
        for (span, from, to) in utterances(pcm16k, &self.vad_model, self.gate.as_ref(), vad)? {
            let text = match &self.engine {
                Engine::Native(model) => model
                    .transcribe(&pcm16k[from..to])
                    .map_err(SttError::Native)?,
                Engine::Sidecar { sidecar, .. } => {
                    let done = sidecar.transcribe(
                        &self.name,
                        &pcm16k[from..to],
                        language.as_deref(),
                        false,
                        false,
                    )?;
                    language = language.or(done.language);
                    done.segments
                        .into_iter()
                        .map(|s| s.text)
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            };
            if !text.is_empty() {
                on_segment(Segment {
                    start: span.start as f64 / rate,
                    end: (span.start + span.len) as f64 / rate,
                    text,
                });
            }
        }
        Ok(language)
    }

    /// Only a model whose sidecar answers `"words": true` (Whisper)
    /// supports this, whatever else it answers; any other is
    /// [`SttError::WordTimestampsUnsupported`]. Each word is offset by its
    /// utterance's slice start, the sample the sidecar's times count from.
    fn decode_words_in(
        &self,
        pcm16k: &[f32],
        vad: Option<&VadConfig>,
        language: Option<&str>,
    ) -> Result<(Vec<Word>, Option<String>), SttError> {
        self.decode_words_verbatim(pcm16k, vad, language, false)
            .map(|(words, language, _)| (words, language))
    }

    /// [`SttModel::decode_words_in`] with the sidecar's `"verbatim"` set;
    /// a model that answers words answers verbatim too.
    fn decode_words_verbatim(
        &self,
        pcm16k: &[f32],
        vad: Option<&VadConfig>,
        language: Option<&str>,
        verbatim: bool,
    ) -> Result<(Vec<Word>, Option<String>, bool), SttError> {
        let rate = TARGET_SAMPLE_RATE as f64;
        let mut language = language.map(str::to_string);
        let mut words = Vec::new();
        for (_, from, to) in utterances(pcm16k, &self.vad_model, self.gate.as_ref(), vad)? {
            // Native Parakeet reports no word times, as the sidecar's does not.
            let Engine::Sidecar { sidecar, .. } = &self.engine else {
                return Err(SttError::WordTimestampsUnsupported);
            };
            let done = sidecar.transcribe(
                &self.name,
                &pcm16k[from..to],
                language.as_deref(),
                true,
                verbatim,
            )?;
            language = language.or(done.language);
            if !done.words {
                return Err(SttError::WordTimestampsUnsupported);
            }
            // Whisper's word times can precede the VAD span's start; they
            // count from the slice start `from`, which is what this adds.
            let offset = from as f64 / rate;
            for segment in done.segments {
                words.extend(segment.words.into_iter().map(|w| Word {
                    start: offset + w.start,
                    end: offset + w.end,
                    text: w.text,
                    confidence: w.confidence,
                }));
            }
        }
        Ok((words, language, verbatim))
    }

    fn decode_words(&self, pcm16k: &[f32], vad: Option<&VadConfig>) -> Result<Vec<Word>, SttError> {
        self.decode_words_in(pcm16k, vad, None).map(|(w, _)| w)
    }

    fn vad_model(&self) -> Option<&Path> {
        Some(&self.vad_model)
    }

    fn resident_bytes(&self) -> Option<u64> {
        Some(match &self.engine {
            Engine::Sidecar { resident_bytes, .. } => *resident_bytes,
            Engine::Native(model) => model.resident_bytes(),
        })
    }
}

/// Unloading the model unloads it from the sidecar, and the last MLX model
/// out stops the sidecar.
impl Drop for MlxStt {
    fn drop(&mut self) {
        if let Engine::Sidecar {
            sidecar, instance, ..
        } = &self.engine
        {
            sidecar.unload(&self.name, *instance);
        }
    }
}
