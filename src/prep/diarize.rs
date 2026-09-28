//! Offline speaker diarization (naru task 1461 §8), the `speaker-diarization-en`
//! catalog model: pyannote segmentation finds single-speaker spans, the 3D-Speaker
//! embedding vectorises each, and sherpa-onnx clusters them into speakers.
//! Loaded fresh per request, like `stt::vad::Vad` (the model is ~35 MB, cheap
//! next to a whole clip decode) — never held resident via `ModelManager`.

use std::path::{Path, PathBuf};

use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};

/// The segmentation model's file in the pulled model directory.
pub const SEGMENTATION_FILENAME: &str = "model.onnx";
/// The speaker embedding model's file.
pub const EMBEDDING_FILENAME: &str = "embedding.onnx";

#[derive(Debug)]
pub enum DiarizeError {
    MissingModel(PathBuf),
    /// `OfflineSpeakerDiarization::create` returned `None`.
    CreateFailed,
}

impl std::fmt::Display for DiarizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiarizeError::MissingModel(path) => {
                write!(f, "diarization model not found: {}", path.display())
            }
            DiarizeError::CreateFailed => {
                write!(f, "failed to create the speaker diarizer")
            }
        }
    }
}

impl std::error::Error for DiarizeError {}

/// One speaker-labelled span, seconds from the start of the audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarizedSpan {
    pub start: f64,
    pub end: f64,
    /// A clustering index, stable within one `diarize` call only — not an
    /// identity across clips.
    pub speaker: i32,
}

/// Clustering knobs for [`Diarizer::load`], threaded from the transcribe
/// request body (`num_speakers`/`cluster_threshold`) so a caller who knows
/// a clip's speaker count, or wants a different merge threshold, can avoid
/// two real speakers being clustered into one. Both fields must be set at
/// load: the upstream C API has no post-creation setter for clustering.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterOptions {
    /// Exact speaker count, overriding `threshold` when this is set; same
    /// as `FastClusteringConfig::num_clusters >= 0`.
    pub num_speakers: Option<u32>,
    /// Cosine-distance merge threshold; ignored when `num_speakers` is
    /// set. Sherpa-onnx's own default, used when this is `None`, is 0.5.
    pub threshold: Option<f32>,
}

pub struct Diarizer {
    inner: OfflineSpeakerDiarization,
}

impl Diarizer {
    /// `dir` is the pulled `speaker-diarization-en` model directory.
    pub fn load(dir: &Path, options: ClusterOptions) -> Result<Self, DiarizeError> {
        let segmentation = dir.join(SEGMENTATION_FILENAME);
        if !segmentation.is_file() {
            return Err(DiarizeError::MissingModel(segmentation));
        }
        let embedding = dir.join(EMBEDDING_FILENAME);
        if !embedding.is_file() {
            return Err(DiarizeError::MissingModel(embedding));
        }
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation.to_string_lossy().into_owned()),
                    ..Default::default()
                },
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding.to_string_lossy().into_owned()),
                ..Default::default()
            },
            // Cluster by similarity, not a fixed speaker count, unless the
            // caller told us the count: a clip's speaker count is normally
            // what this is for.
            clustering: FastClusteringConfig {
                num_clusters: options.num_speakers.map_or(-1, |n| n as i32),
                threshold: options.threshold.unwrap_or(0.5),
            },
            ..Default::default()
        };
        let inner = OfflineSpeakerDiarization::create(&config).ok_or(DiarizeError::CreateFailed)?;
        Ok(Diarizer { inner })
    }

    /// The sample rate `pcm` must already be at.
    pub fn sample_rate(&self) -> i32 {
        self.inner.sample_rate()
    }

    /// Diarizes one whole clip, sorted by start time.
    pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<DiarizedSpan>, DiarizeError> {
        let result = self.inner.process(pcm).ok_or(DiarizeError::CreateFailed)?;
        Ok(result
            .sort_by_start_time()
            .into_iter()
            .map(|s| DiarizedSpan {
                start: s.start as f64,
                end: s.end as f64,
                speaker: s.speaker,
            })
            .collect())
    }
}
