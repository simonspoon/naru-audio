//! The `sherpa-onnx` [`SttModel`]: the transcribe path of auris
//! `src/cli.rs` (d8644d6) over a whole buffer — the energy gate, the Silero
//! gate, a decode per utterance, and the manufactured-vocabulary guard.

use std::path::{Path, PathBuf};

use super::audio::{self, READ_CHUNK_FRAMES, TARGET_SAMPLE_RATE};
use super::engine::{EngineConfig, EngineError, Recognizer};
use super::vad::{Span, Vad, VadConfig};
use super::vocabulary::{Vocabulary, looks_manufactured};
use super::{Segment, SttError, SttModel};
use crate::registry::manifest::Manifest;

/// The file a `requires`d VAD model provides.
pub const VAD_FILENAME: &str = "silero_vad.onnx";

pub struct SherpaStt {
    recognizer: Recognizer,
    vad_model: PathBuf,
}

impl SherpaStt {
    /// `dir` is `models/<name>/`; a `requires`d model is its sibling
    /// `models/<required>/` (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, SttError> {
        let vad_model = manifest
            .model
            .requires
            .iter()
            .filter_map(|name| Some(dir.parent()?.join(name).join(VAD_FILENAME)))
            .find(|path| path.is_file())
            .ok_or_else(|| SttError::NoVadModel(manifest.model.name.clone()))?;
        let recognizer = Recognizer::load(&EngineConfig {
            model_dir: dir.to_path_buf(),
            ..Default::default()
        })?;
        Ok(SherpaStt {
            recognizer,
            vad_model,
        })
    }

    /// One utterance; `None` when it decodes to nothing or the guard
    /// discards it.
    fn decode_utterance(
        &self,
        samples: &[f32],
        vocabulary: Option<&Vocabulary>,
        hotwords: Option<&str>,
    ) -> Result<Option<String>, EngineError> {
        let text = match hotwords {
            Some(h) => self.recognizer.decode_with_hotwords(samples, h)?,
            None => self.recognizer.decode(samples)?,
        };
        let text = text.trim();
        if text.is_empty() {
            return Ok(None);
        }

        // The manufactured-vocabulary guard (auris task 970): on non-speech
        // audio, biasing can invent a transcript out of boosted terms. When
        // the transcript looks like that, decode the same audio unbiased;
        // only if that is empty too is the biased transcript discarded. A
        // failed confirming decode is not evidence, so it keeps the text.
        if let Some(v) = vocabulary
            && looks_manufactured(text, &v.terms)
            && let Ok(unbiased) = self.recognizer.decode(samples)
            && unbiased.trim().is_empty()
        {
            return Ok(None);
        }
        Ok(Some(text.to_string()))
    }
}

impl SttModel for SherpaStt {
    fn decode_each(
        &self,
        pcm16k: &[f32],
        hotwords: Option<&Vocabulary>,
        vad: Option<&VadConfig>,
        on_segment: &mut dyn FnMut(Segment),
    ) -> Result<(), SttError> {
        // An empty vocabulary means no vocabulary: auris never hands
        // sherpa-onnx a wholly empty hotwords string.
        let vocabulary = hotwords.filter(|v| !v.terms.is_empty());
        let hotwords = vocabulary.map(Vocabulary::hotwords_string);

        // Parakeet hallucinates on digital silence, so the recognizer (and
        // the VAD) never sees it.
        if audio::is_silent(pcm16k) {
            return Ok(());
        }

        // (utterance, slice start, slice end). Slices partition the buffer
        // at utterance starts, so one utterance decodes the whole buffer
        // (`super::vad`).
        let mut utterances: Vec<(Span, usize, usize)> = Vec::new();
        match vad {
            None => utterances.push((
                Span {
                    start: 0,
                    len: pcm16k.len(),
                },
                0,
                pcm16k.len(),
            )),
            Some(cfg) => {
                let vad = Vad::load(&self.vad_model, cfg)?;
                let mut segmenter = vad.segmenter();
                let mut spans = Vec::new();
                // Fed and drained per read chunk, as auris's streaming loop
                // does; the detector's queue is finite.
                for chunk in pcm16k.chunks(READ_CHUNK_FRAMES) {
                    segmenter.accept(chunk);
                    while let Some(span) = segmenter.next_span() {
                        spans.push(span);
                    }
                }
                segmenter.finish();
                while let Some(span) = segmenter.next_span() {
                    spans.push(span);
                }

                let mut slice_start = 0;
                for (i, span) in spans.iter().enumerate() {
                    let cut = match spans.get(i + 1) {
                        Some(next) => next.start.max(slice_start),
                        None => pcm16k.len(),
                    };
                    utterances.push((*span, slice_start, cut));
                    slice_start = cut;
                }
            }
        }

        let rate = TARGET_SAMPLE_RATE as f64;
        for (span, from, to) in utterances {
            if let Some(text) =
                self.decode_utterance(&pcm16k[from..to], vocabulary, hotwords.as_deref())?
            {
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
}
