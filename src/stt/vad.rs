//! The Silero VAD gate, ported from auris `src/vad.rs` (d8644d6).
//!
//! **A decision, not a filter.** A [`Span`] is offsets only; the recognizer
//! is never handed Silero's cut. auris measured trimming to the detected
//! spans (even padded) changing 35 of 40 real-room transcripts, and turning
//! a non-speech false positive into a hallucinated "Uh.". So a buffer with
//! one utterance is decoded whole, and one with several is partitioned at
//! utterance starts (see `super::sherpa`). The gate can only refuse to run
//! the recognizer on audio with no speech in it.

use std::path::{Path, PathBuf};

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use super::audio::TARGET_SAMPLE_RATE;

/// Silero's frame size.
const WINDOW_SIZE: i32 = 512;

/// The detector's ring buffer, in seconds: headroom over one whole
/// [`MAX_SPEECH_SECONDS`] segment queued while more audio arrives.
const BUFFER_SECONDS: f32 = 30.0;

/// Longest span before the detector force-closes it. From auris
/// docs/latency.md's decode budget; it does not change accept/reject.
const MAX_SPEECH_SECONDS: f32 = 8.0;

/// Silero's stock trailing silence; where one utterance ends and the next
/// begins.
pub const MIN_SILENCE_SECONDS: f32 = 0.5;

/// The per-request knobs (§2.2 `vad_threshold`, `vad_min_speech`,
/// `vad_min_silence`).
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// Speech-probability threshold.
    pub threshold: f32,
    /// Spans shorter than this many seconds are dropped.
    pub min_speech: f32,
    /// Seconds of trailing silence that close a span.
    pub min_silence: f32,
}

impl Default for VadConfig {
    /// `threshold = 0.2`, not Silero's 0.5: auris's sweep (task 968) found
    /// 0.5 and 0.4 miss a genuinely quiet real recording, while 0.10–0.38
    /// accepted all 43 real clips and rejected white noise, rumble and
    /// silence; 0.2 is the middle of that band. No threshold separates the
    /// transient-noise fixture from speech.
    fn default() -> Self {
        VadConfig {
            threshold: 0.2,
            min_speech: 0.25,
            min_silence: MIN_SILENCE_SECONDS,
        }
    }
}

#[derive(Debug)]
pub enum VadError {
    MissingModel(PathBuf),
    /// `VoiceActivityDetector::create` returned `None`.
    CreateFailed,
}

impl std::fmt::Display for VadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VadError::MissingModel(path) => {
                write!(f, "VAD model not found: {}", path.display())
            }
            VadError::CreateFailed => write!(f, "failed to create the voice activity detector"),
        }
    }
}

impl std::error::Error for VadError {}

/// A loaded Silero detector. Its settings are fixed at construction, so a
/// request with its own [`VadConfig`] loads its own (the model is ~2 MB).
pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    /// `model` is `silero_vad.onnx`.
    pub fn load(model: &Path, cfg: &VadConfig) -> Result<Vad, VadError> {
        if !model.is_file() {
            return Err(VadError::MissingModel(model.to_path_buf()));
        }

        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_string_lossy().into_owned()),
                threshold: cfg.threshold,
                min_silence_duration: cfg.min_silence,
                min_speech_duration: cfg.min_speech,
                window_size: WINDOW_SIZE,
                max_speech_duration: MAX_SPEECH_SECONDS,
            },
            sample_rate: TARGET_SAMPLE_RATE as i32,
            num_threads: 1,
            ..Default::default()
        };

        let inner =
            VoiceActivityDetector::create(&config, BUFFER_SECONDS).ok_or(VadError::CreateFailed)?;
        Ok(Vad { inner })
    }

    /// Starts a fresh pass over one stream; resets the detector so nothing
    /// carries over from the last one.
    pub fn segmenter(&self) -> Segmenter<'_> {
        self.inner.reset();
        Segmenter {
            vad: self,
            pending: Vec::new(),
        }
    }
}

/// Where one closed utterance sits, in samples from the start of the pass.
/// (auris calls this `Segment`; here that name is the transcript segment.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub len: usize,
}

/// One pass of the detector, fed incrementally. Holds back a partial
/// window between calls so a short chunk never shifts window alignment.
pub struct Segmenter<'a> {
    vad: &'a Vad,
    pending: Vec<f32>,
}

impl Segmenter<'_> {
    pub fn accept(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        let window = WINDOW_SIZE as usize;
        let whole = self.pending.len() - self.pending.len() % window;
        for chunk in self.pending[..whole].chunks(window) {
            self.vad.inner.accept_waveform(chunk);
        }
        self.pending.drain(..whole);
    }

    /// End of stream: feeds the remainder and closes any open span.
    pub fn finish(&mut self) {
        if !self.pending.is_empty() {
            self.vad.inner.accept_waveform(&self.pending);
            self.pending.clear();
        }
        self.vad.inner.flush();
    }

    /// The next closed utterance, if any. Drain it: the queue is finite.
    pub fn next_span(&mut self) -> Option<Span> {
        let segment = self.vad.inner.front()?;
        let found = Span {
            start: segment.start().max(0) as usize,
            len: segment.n().max(0) as usize,
        };
        drop(segment);
        self.vad.inner.pop();
        Some(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_the_contract() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.threshold, 0.2);
        assert_eq!(cfg.min_speech, 0.25);
        assert_eq!(cfg.min_silence, 0.5);
    }

    #[test]
    fn missing_model_is_an_error_not_a_panic() {
        let err = Vad::load(
            Path::new("/nonexistent/naru-audio-vad-test-model.onnx"),
            &VadConfig::default(),
        )
        .err()
        .expect("expected an error");
        assert!(matches!(err, VadError::MissingModel(_)));
    }
}
