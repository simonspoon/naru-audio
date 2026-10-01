//! The `mlx` [`SttModel`] (§5.3): the sherpa-onnx model's energy and
//! Silero gates in the daemon, and a decode per utterance in the sidecar
//! (`crate::mlx::sidecar`): parakeet-mlx or mlx-whisper, by model name.
//! Neither has hotword biasing, so `hotwords` are ignored; only Whisper
//! takes a language and reports word timestamps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::audio::TARGET_SAMPLE_RATE;
use super::sherpa::{utterances, vad_model};
use super::vad::VadConfig;
use super::vocabulary::Vocabulary;
use super::{Segment, SttError, SttModel, Word};
use crate::mlx::sidecar::{self, Sidecar};
use crate::registry::manifest::{Kind, Manifest};

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
        let (instance, resident_bytes, _) = sidecar.load(&name, Kind::Stt, dir)?;
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
        for (span, from, to) in utterances(pcm16k, &self.vad_model, vad)? {
            let done = self.sidecar.transcribe(
                &self.name,
                &pcm16k[from..to],
                language.as_deref(),
                false,
                false,
            )?;
            language = language.or(done.language);
            let text = done
                .segments
                .into_iter()
                .map(|s| s.text)
                .collect::<Vec<_>>()
                .join(" ");
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
        for (_, from, to) in utterances(pcm16k, &self.vad_model, vad)? {
            let done = self.sidecar.transcribe(
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
