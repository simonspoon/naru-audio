//! sherpa-onnx's `OfflineRecognizer` bound to a Parakeet TDT model, ported
//! from auris `src/engine.rs` (d8644d6). `load` pays the ~4 s cold start;
//! `decode` takes `&self`, so one recognizer serves every request.

use std::path::{Path, PathBuf};

use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
};

use super::audio::TARGET_SAMPLE_RATE;
use crate::registry::manifest::BPE_VOCAB;

/// The transducer graphs and token table. `bpe.vocab` is checked separately
/// but is just as required: per-request hotwords need the tokenizer
/// configured at construction.
const REQUIRED_MODEL_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// Only what varies per load; what the model type fixes (`nemo_transducer`,
/// `modified_beam_search`) is not a choice.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Holds [`REQUIRED_MODEL_FILES`] and `bpe.vocab`.
    pub model_dir: PathBuf,
    /// ONNX Runtime intra-op threads.
    pub threads: i32,
    /// The global hotword boost, fixed at construction; a term's `:boost`
    /// overrides it. 3.0, not sherpa-onnx's 1.5: auris measured 1.5 as too
    /// low for a per-term boost to reach the beam at all
    /// (auris docs/vocabulary.md).
    pub hotwords_score: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            model_dir: PathBuf::new(),
            threads: 8,
            hotwords_score: 3.0,
        }
    }
}

#[derive(Debug)]
pub enum EngineError {
    MissingModelDir(PathBuf),
    MissingModelFile(PathBuf),
    MissingBpeVocab(PathBuf),
    /// `OfflineRecognizer::create` returned `None` despite a valid-looking
    /// config.
    CreateFailed,
    /// A stream produced no result.
    DecodeFailed,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::MissingModelDir(dir) => {
                write!(f, "model directory not found: {}", dir.display())
            }
            EngineError::MissingModelFile(path) => {
                write!(f, "model file missing: {}", path.display())
            }
            EngineError::MissingBpeVocab(path) => write!(
                f,
                "hotwords require a bpe vocab at {} but it is missing",
                path.display()
            ),
            EngineError::CreateFailed => write!(f, "failed to create the offline recognizer"),
            EngineError::DecodeFailed => write!(f, "decode produced no result"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Fails fast on a bad directory before any ONNX Runtime session exists.
fn validate_model_dir(cfg: &EngineConfig) -> Result<(), EngineError> {
    if !cfg.model_dir.is_dir() {
        return Err(EngineError::MissingModelDir(cfg.model_dir.clone()));
    }
    for file in REQUIRED_MODEL_FILES {
        let path = cfg.model_dir.join(file);
        if !path.is_file() {
            return Err(EngineError::MissingModelFile(path));
        }
    }
    let bpe_vocab = cfg.model_dir.join(BPE_VOCAB);
    if !bpe_vocab.is_file() {
        return Err(EngineError::MissingBpeVocab(bpe_vocab));
    }
    Ok(())
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// A loaded Parakeet recognizer; `Send + Sync` via `OfflineRecognizer`.
pub struct Recognizer {
    inner: OfflineRecognizer,
}

impl Recognizer {
    pub fn load(cfg: &EngineConfig) -> Result<Self, EngineError> {
        validate_model_dir(cfg)?;

        let model_config = OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: Some(path_str(&cfg.model_dir.join("encoder.int8.onnx"))),
                decoder: Some(path_str(&cfg.model_dir.join("decoder.int8.onnx"))),
                joiner: Some(path_str(&cfg.model_dir.join("joiner.int8.onnx"))),
            },
            tokens: Some(path_str(&cfg.model_dir.join("tokens.txt"))),
            num_threads: cfg.threads,
            debug: false,
            model_type: Some("nemo_transducer".to_string()),
            modeling_unit: Some("bpe".to_string()),
            bpe_vocab: Some(path_str(&cfg.model_dir.join(BPE_VOCAB))),
            ..Default::default()
        };

        let config = OfflineRecognizerConfig {
            model_config,
            decoding_method: Some("modified_beam_search".to_string()),
            hotwords_score: cfg.hotwords_score,
            ..Default::default()
        };

        let inner = OfflineRecognizer::create(&config).ok_or(EngineError::CreateFailed)?;
        Ok(Recognizer { inner })
    }

    /// Decodes one utterance of 16 kHz mono samples, unbiased.
    pub fn decode(&self, samples: &[f32]) -> Result<String, EngineError> {
        let stream = self.inner.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .ok_or(EngineError::DecodeFailed)
    }

    /// Decodes one utterance against a per-request hotwords string: `term
    /// :boost` entries joined with `/` ([`super::vocabulary::Vocabulary::hotwords_string`]).
    /// Changing the terms never rebuilds the recognizer.
    pub fn decode_with_hotwords(
        &self,
        samples: &[f32],
        hotwords: &str,
    ) -> Result<String, EngineError> {
        let stream = self.inner.create_stream_with_hotwords(hotwords);
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .ok_or(EngineError::DecodeFailed)
    }

    /// [`Self::decode`], keeping the per-token text and timestamps
    /// (naru task 1461 §8): NeMo transducer models report both without
    /// needing an `enable_token_timestamps` flag (that config field exists
    /// only on `OfflineWhisperModelConfig`), so this is a plain decode that
    /// reads more of the same result.
    pub fn decode_with_tokens(&self, samples: &[f32]) -> Result<Recognized, EngineError> {
        let stream = self.inner.create_stream();
        stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        let result = stream.get_result().ok_or(EngineError::DecodeFailed)?;
        Ok(Recognized {
            text: result.text,
            tokens: result.tokens,
            timestamps: result.timestamps.unwrap_or_default(),
        })
    }
}

/// One utterance's text with its tokens and their start times, from
/// [`Recognizer::decode_with_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct Recognized {
    pub text: String,
    pub tokens: Vec<String>,
    /// Seconds from the start of the utterance, one per `tokens` entry;
    /// empty when the model reports none.
    pub timestamps: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_dir_is_an_error_not_a_panic() {
        let cfg = EngineConfig {
            model_dir: PathBuf::from("/nonexistent/naru-audio-engine-test-model"),
            ..Default::default()
        };
        let err = Recognizer::load(&cfg).err().expect("expected an error");
        assert!(matches!(err, EngineError::MissingModelDir(_)));
    }

    #[test]
    fn incomplete_model_dir_is_an_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("encoder.int8.onnx"), b"").unwrap();
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let err = Recognizer::load(&cfg).err().expect("expected an error");
        assert!(matches!(err, EngineError::MissingModelFile(_)));
    }

    #[test]
    fn missing_bpe_vocab_is_an_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for file in REQUIRED_MODEL_FILES {
            std::fs::write(tmp.path().join(file), b"").unwrap();
        }
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let err = Recognizer::load(&cfg).err().expect("expected an error");
        assert!(matches!(err, EngineError::MissingBpeVocab(_)));
    }

    #[test]
    fn global_hotword_boost_is_three() {
        assert_eq!(EngineConfig::default().hotwords_score, 3.0);
    }

    #[test]
    fn recognizer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Recognizer>();
    }
}
