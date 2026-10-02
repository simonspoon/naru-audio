//! The `sherpa-onnx` [`SttModel`]: the transcribe path of auris
//! `src/cli.rs` (d8644d6) over a whole buffer — the energy gate, the Silero
//! gate, a decode per utterance, and the manufactured-vocabulary guard.

use std::path::{Path, PathBuf};

use super::audio::{self, READ_CHUNK_FRAMES, TARGET_SAMPLE_RATE};
use super::engine::{EngineConfig, EngineError, Recognizer};
use super::vad::{Span, Vad, VadConfig};
use super::vocabulary::{Vocabulary, looks_manufactured};
use super::{Segment, SttError, SttModel, Word};
use crate::registry::manifest::Manifest;

/// The file a `requires`d VAD model provides.
pub const VAD_FILENAME: &str = "silero_vad.onnx";

pub struct SherpaStt {
    recognizer: Recognizer,
    vad_model: PathBuf,
}

impl SherpaStt {
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, SttError> {
        let vad_model = vad_model(manifest, dir)?;
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

        let rate = TARGET_SAMPLE_RATE as f64;
        for (span, from, to) in utterances(pcm16k, &self.vad_model, vad)? {
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

    /// naru task 1461 §8: the same utterance gate as `decode_each`, but
    /// each utterance's tokens are merged into words at a leading-space
    /// boundary (checked against a real pulled `parakeet-tdt-0.6b-v2-int8`
    /// decode: sherpa-onnx's NeMo/BPE `OfflineRecognizerResult.tokens`
    /// reports pieces like `" He"`, `"ll"`, `"o"`, `" there"` — the
    /// SentencePiece `\u{2581}` marker `tokens.txt` uses is already
    /// rendered back to an ordinary space at this layer, not passed
    /// through raw), each word's timestamp offset by its utterance's
    /// start.
    fn decode_words(&self, pcm16k: &[f32], vad: Option<&VadConfig>) -> Result<Vec<Word>, SttError> {
        let rate = TARGET_SAMPLE_RATE as f64;
        let mut words = Vec::new();
        for (span, from, to) in utterances(pcm16k, &self.vad_model, vad)? {
            let recognized = self.recognizer.decode_with_tokens(&pcm16k[from..to])?;
            let utt_start = span.start as f64 / rate;
            let utt_end = (span.start + span.len) as f64 / rate;
            words.extend(merge_words(
                &recognized.tokens,
                &recognized.timestamps,
                utt_start,
                utt_end,
            ));
        }
        Ok(words)
    }

    fn vad_model(&self) -> Option<&Path> {
        Some(&self.vad_model)
    }
}

/// The word-start marker in a decoded `OfflineRecognizerResult.tokens`
/// piece — an ordinary leading space, not the raw SentencePiece `\u{2581}`
/// `tokens.txt` uses (see `decode_words`'s doc comment).
const WORD_BOUNDARY: char = ' ';

/// Merges one utterance's tokens and per-token start times (seconds from
/// the utterance's own start) into words, offset by `utt_start`; the last
/// word closes at `utt_end` since there is no timestamp past the last
/// token. A token with no `WORD_BOUNDARY` prefix and no word open yet
/// (should not happen; guarded rather than panicking) starts one anyway,
/// so nothing is silently dropped.
fn merge_words(tokens: &[String], timestamps: &[f32], utt_start: f64, utt_end: f64) -> Vec<Word> {
    if tokens.is_empty() || tokens.len() != timestamps.len() {
        return Vec::new();
    }
    struct Building {
        start: f64,
        text: String,
    }
    let mut words: Vec<Word> = Vec::new();
    let mut building: Option<Building> = None;
    for (tok, &ts) in tokens.iter().zip(timestamps) {
        let start = utt_start + ts as f64;
        match tok.strip_prefix(WORD_BOUNDARY) {
            Some(piece) => {
                if let Some(b) = building.take() {
                    words.push(Word {
                        start: b.start,
                        end: start,
                        text: b.text,
                        confidence: None,
                    });
                }
                building = Some(Building {
                    start,
                    text: piece.to_string(),
                });
            }
            None => match building.as_mut() {
                Some(b) => b.text.push_str(tok),
                None => {
                    building = Some(Building {
                        start,
                        text: tok.clone(),
                    })
                }
            },
        }
    }
    if let Some(b) = building {
        words.push(Word {
            start: b.start,
            end: utt_end,
            text: b.text,
            confidence: None,
        });
    }
    words
        .into_iter()
        .filter(|w| !w.text.trim().is_empty())
        .collect()
}

/// The `requires`d model's Silero file. `dir` is `models/<name>/`; a
/// `requires`d model is its sibling `models/<required>/` (§3.1).
pub(super) fn vad_model(manifest: &Manifest, dir: &Path) -> Result<PathBuf, SttError> {
    manifest
        .model
        .requires
        .iter()
        .filter_map(|name| Some(dir.parent()?.join(name).join(VAD_FILENAME)))
        .find(|path| path.is_file())
        .ok_or_else(|| SttError::NoVadModel(manifest.model.name.clone()))
}

/// The gates in front of the recognizer, shared with the MLX model: the
/// utterances of `pcm16k` to decode, as (utterance, slice start, slice end).
pub(super) fn utterances(
    pcm16k: &[f32],
    vad_model: &Path,
    vad: Option<&VadConfig>,
) -> Result<Vec<(Span, usize, usize)>, SttError> {
    // Parakeet hallucinates on digital silence, so the recognizer (and
    // the VAD) never sees it.
    if audio::is_silent(pcm16k) {
        return Ok(Vec::new());
    }

    // Slices partition the buffer at utterance starts, so one utterance
    // decodes the whole buffer (`super::vad`).
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
            let vad = Vad::load(vad_model, cfg)?;
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
    Ok(utterances)
}

#[cfg(test)]
mod merge_words_tests {
    use super::*;

    fn tok(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn tokens_split_at_the_word_boundary_marker() {
        // " hel" "lo" " there" -> "hello" (0.0-0.6) "there" (0.6-utt_end),
        // the shape a real decode gives (checked against a real pulled
        // parakeet-tdt-0.6b-v2-int8 decode, naru task 1461 §8).
        let tokens = vec![tok(" hel"), tok("lo"), tok(" there")];
        let timestamps = vec![0.0, 0.2, 0.6];
        let words = merge_words(&tokens, &timestamps, 10.0, 11.0);
        assert_eq!(words.len(), 2);
        assert!((words[0].start - 10.0).abs() < 1e-6);
        assert!((words[0].end - 10.6).abs() < 1e-6);
        assert_eq!(words[0].text, "hello");
        assert!((words[1].start - 10.6).abs() < 1e-6);
        assert!((words[1].end - 11.0).abs() < 1e-6);
        assert_eq!(words[1].text, "there");
    }

    #[test]
    fn mismatched_lengths_yield_no_words_rather_than_a_panic() {
        assert_eq!(merge_words(&[tok("a")], &[], 0.0, 1.0), Vec::new());
    }

    #[test]
    fn empty_tokens_yield_no_words() {
        assert_eq!(merge_words(&[], &[], 0.0, 1.0), Vec::new());
    }
}
