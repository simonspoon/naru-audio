//! Speaker verification and enrollment (naru task 2028), the `wespeaker-sv-en`
//! catalog model: a WeSpeaker ResNet34 embedding (via sherpa-onnx) turns a clip
//! into a vector, and cosine similarity against the enrolled voice says whether
//! it is "him" — the speaker guard `docs/live-mode-spec.md` leaves to the
//! server. Loaded fresh per request like `prep::diarize::Diarizer`.
//!
//! **Fails open** everywhere (memory: he often answers with just "yes"): no
//! enrollment, no pulled model, an enrollment made with another model, or a
//! clip shorter than [`MIN_SAMPLES`] all say `is_him = true`, so a guard built
//! on this never drops a short reply or goes silent on a missing model.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

/// The catalog model name; also tagged into every enrollment.
pub const DEFAULT_MODEL: &str = "wespeaker-sv-en";
/// The embedding model's file in the pulled model directory.
pub const EMBEDDING_FILENAME: &str = "embedding.onnx";
/// Sample rate the embedding is computed at.
pub const SAMPLE_RATE: i32 = 16_000;
/// Clips shorter than this (1 s) are not scored: a "yes" or a number is too
/// short for a stable embedding and must never be dropped.
pub const MIN_SAMPLES: usize = 16_000;
/// Cosine score at or above which a clip is "him". Calibrated from the
/// fixtures in `tests/speaker.rs`: same voice scored 0.94-0.99, a different
/// voice 0.59-0.62. Chosen nearer the low end, since wrongly dropping him is
/// worse than admitting someone else; re-measure against a real enrollment.
pub const DEFAULT_THRESHOLD: f32 = 0.7;

#[derive(Debug)]
pub enum SpeakerError {
    MissingModel(PathBuf),
    /// `SpeakerEmbeddingExtractor::create` returned `None`.
    CreateFailed,
    /// The extractor produced no embedding for the clip.
    EmbedFailed,
    /// Enrolling needs at least one usable clip.
    NoClips,
    /// An enrollment's file could not be read or written.
    Store(String),
}

impl std::fmt::Display for SpeakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeakerError::MissingModel(path) => {
                write!(f, "speaker model not found: {}", path.display())
            }
            SpeakerError::CreateFailed => write!(f, "failed to create the speaker extractor"),
            SpeakerError::EmbedFailed => write!(f, "the speaker model produced no embedding"),
            SpeakerError::NoClips => write!(f, "enrolling needs at least one clip"),
            SpeakerError::Store(m) => write!(f, "enrollment store: {m}"),
        }
    }
}

impl std::error::Error for SpeakerError {}

pub struct SpeakerVerifier {
    inner: SpeakerEmbeddingExtractor,
}

impl SpeakerVerifier {
    /// `dir` is the pulled `wespeaker-sv-en` model directory.
    pub fn load(dir: &Path) -> Result<Self, SpeakerError> {
        let model = dir.join(EMBEDDING_FILENAME);
        if !model.is_file() {
            return Err(SpeakerError::MissingModel(model));
        }
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let inner = SpeakerEmbeddingExtractor::create(&config).ok_or(SpeakerError::CreateFailed)?;
        Ok(SpeakerVerifier { inner })
    }

    /// The L2-normalised embedding of 16 kHz mono `pcm`.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>, SpeakerError> {
        let stream = self
            .inner
            .create_stream()
            .ok_or(SpeakerError::EmbedFailed)?;
        stream.accept_waveform(SAMPLE_RATE, pcm);
        stream.input_finished();
        if !self.inner.is_ready(&stream) {
            return Err(SpeakerError::EmbedFailed);
        }
        let mut v = self
            .inner
            .compute(&stream)
            .ok_or(SpeakerError::EmbedFailed)?;
        normalize(&mut v);
        Ok(v)
    }

    /// Enrolls from `clips`, each embedded then averaged. Clips under
    /// [`MIN_SAMPLES`] are skipped; none usable is [`SpeakerError::NoClips`].
    pub fn enroll(&self, clips: &[Vec<f32>]) -> Result<Enrollment, SpeakerError> {
        let mut embeddings = Vec::new();
        for clip in clips.iter().filter(|c| c.len() >= MIN_SAMPLES) {
            embeddings.push(self.embed(clip)?);
        }
        Enrollment::from_embeddings(DEFAULT_MODEL, &embeddings)
    }
}

/// The enrolled voice: the mean of several clips' embeddings. Stored as JSON
/// tagged with the model, since embeddings of different models are not
/// comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Enrollment {
    pub model: String,
    /// How many clips went into `embedding`.
    pub count: usize,
    /// L2-normalised mean of the clips' normalised embeddings.
    pub embedding: Vec<f32>,
}

impl Enrollment {
    /// Averages already-normalised `embeddings`; they must share one length.
    pub fn from_embeddings(model: &str, embeddings: &[Vec<f32>]) -> Result<Self, SpeakerError> {
        let first = embeddings.first().ok_or(SpeakerError::NoClips)?;
        let mut mean = vec![0f32; first.len()];
        for e in embeddings {
            if e.len() != mean.len() {
                return Err(SpeakerError::EmbedFailed);
            }
            for (m, x) in mean.iter_mut().zip(e) {
                *m += x;
            }
        }
        normalize(&mut mean);
        Ok(Enrollment {
            model: model.to_string(),
            count: embeddings.len(),
            embedding: mean,
        })
    }

    pub fn path(home: &Path) -> PathBuf {
        home.join("speaker").join("enrollment.json")
    }

    /// `Ok(None)` when nothing is enrolled.
    pub fn load(home: &Path) -> Result<Option<Self>, SpeakerError> {
        match std::fs::read(Self::path(home)) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| SpeakerError::Store(e.to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(SpeakerError::Store(e.to_string())),
        }
    }

    /// Atomic: written beside the file, then renamed over it.
    pub fn save(&self, home: &Path) -> Result<(), SpeakerError> {
        let path = Self::path(home);
        let store = |e: std::io::Error| SpeakerError::Store(e.to_string());
        std::fs::create_dir_all(path.parent().expect("has a parent")).map_err(store)?;
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec(self).map_err(|e| SpeakerError::Store(e.to_string()))?;
        std::fs::write(&tmp, json).map_err(store)?;
        std::fs::rename(&tmp, &path).map_err(store)
    }

    /// Removes the enrollment; `true` if there was one.
    pub fn delete(home: &Path) -> Result<bool, SpeakerError> {
        match std::fs::remove_file(Self::path(home)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(SpeakerError::Store(e.to_string())),
        }
    }
}

/// Why a verdict came out as it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Scored against the enrollment.
    Scored,
    NoEnrollment,
    /// The enrollment was made with a different model.
    ModelMismatch,
    /// Under [`MIN_SAMPLES`].
    TooShort,
    /// The model is not pulled or failed to embed.
    ModelUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Verdict {
    pub score: Option<f32>,
    pub is_him: bool,
    pub reason: Reason,
}

impl Verdict {
    fn open(reason: Reason) -> Self {
        Verdict {
            score: None,
            is_him: true,
            reason,
        }
    }
}

/// The pure decision. `embedding` is the clip's embedding, `None` when the
/// model could not produce one. Every case but a scored one fails open.
pub fn judge(
    enrollment: Option<&Enrollment>,
    model: &str,
    samples: usize,
    embedding: Option<&[f32]>,
    threshold: f32,
) -> Verdict {
    let Some(enrolled) = enrollment else {
        return Verdict::open(Reason::NoEnrollment);
    };
    if enrolled.model != model {
        return Verdict::open(Reason::ModelMismatch);
    }
    if samples < MIN_SAMPLES {
        return Verdict::open(Reason::TooShort);
    }
    match embedding {
        Some(e) if e.len() == enrolled.embedding.len() => {
            let score = cosine(e, &enrolled.embedding);
            Verdict {
                score: Some(score),
                is_him: score >= threshold,
                reason: Reason::Scored,
            }
        }
        _ => Verdict::open(Reason::ModelUnavailable),
    }
}

/// Verdict for `pcm` against `enrollment`, embedding only when it can matter
/// (an enrollment exists, the clip is long enough, the model is present).
pub fn verify(
    verifier: Option<&SpeakerVerifier>,
    enrollment: Option<&Enrollment>,
    pcm: &[f32],
    threshold: f32,
) -> Verdict {
    let embedding = match (verifier, enrollment) {
        (Some(v), Some(_)) if pcm.len() >= MIN_SAMPLES => v.embed(pcm).ok(),
        _ => None,
    };
    judge(
        enrollment,
        DEFAULT_MODEL,
        pcm.len(),
        embedding.as_deref(),
        threshold,
    )
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Cosine similarity; 0 when either vector is zero or lengths differ.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (
        a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        b.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enrolled(v: &[f32]) -> Enrollment {
        Enrollment::from_embeddings(DEFAULT_MODEL, &[v.to_vec()]).unwrap()
    }

    #[test]
    fn cosine_math() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn enrollment_is_the_normalised_mean() {
        let e = Enrollment::from_embeddings("m", &[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert_eq!(e.count, 2);
        let r = std::f32::consts::FRAC_1_SQRT_2;
        assert!((e.embedding[0] - r).abs() < 1e-6 && (e.embedding[1] - r).abs() < 1e-6);
        assert!(matches!(
            Enrollment::from_embeddings("m", &[]),
            Err(SpeakerError::NoClips)
        ));
        assert!(Enrollment::from_embeddings("m", &[vec![1.0], vec![1.0, 0.0]]).is_err());
    }

    #[test]
    fn threshold_decides_a_scored_clip() {
        let e = enrolled(&[1.0, 0.0]);
        let long = MIN_SAMPLES;
        let same = judge(Some(&e), DEFAULT_MODEL, long, Some(&[1.0, 0.0]), 0.5);
        assert!(same.is_him && same.reason == Reason::Scored);
        let other = judge(Some(&e), DEFAULT_MODEL, long, Some(&[0.0, 1.0]), 0.5);
        assert!(!other.is_him && other.score == Some(0.0));
        // exactly at the threshold admits
        let edge = judge(Some(&e), DEFAULT_MODEL, long, Some(&[0.6, 0.8]), 0.6);
        assert!(edge.is_him);
        let strict = judge(Some(&e), DEFAULT_MODEL, long, Some(&[0.6, 0.8]), 0.7);
        assert!(!strict.is_him);
    }

    #[test]
    fn every_fail_open_path_is_him() {
        let e = enrolled(&[1.0, 0.0]);
        let orthogonal: &[f32] = &[0.0, 1.0];
        let v = judge(None, DEFAULT_MODEL, MIN_SAMPLES, Some(orthogonal), 0.5);
        assert!(v.is_him && v.reason == Reason::NoEnrollment);
        let v = judge(Some(&e), "other-model", MIN_SAMPLES, Some(orthogonal), 0.5);
        assert!(v.is_him && v.reason == Reason::ModelMismatch);
        let v = judge(
            Some(&e),
            DEFAULT_MODEL,
            MIN_SAMPLES - 1,
            Some(orthogonal),
            0.5,
        );
        assert!(v.is_him && v.reason == Reason::TooShort && v.score.is_none());
        let v = judge(Some(&e), DEFAULT_MODEL, MIN_SAMPLES, None, 0.5);
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
        let v = judge(
            Some(&e),
            DEFAULT_MODEL,
            MIN_SAMPLES,
            Some(&[1.0, 0.0, 0.0]),
            0.5,
        );
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
        // no verifier at all (model not pulled): open, even with a long clip
        let v = verify(None, Some(&e), &vec![0.0; MIN_SAMPLES * 2], 0.5);
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
    }

    #[test]
    fn enrollment_round_trips_and_deletes() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(Enrollment::load(home.path()).unwrap(), None);
        let e = enrolled(&[0.6, 0.8]);
        e.save(home.path()).unwrap();
        assert_eq!(Enrollment::load(home.path()).unwrap(), Some(e));
        assert!(Enrollment::delete(home.path()).unwrap());
        assert!(!Enrollment::delete(home.path()).unwrap());
        assert_eq!(Enrollment::load(home.path()).unwrap(), None);
    }

    #[test]
    fn missing_model_dir_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            SpeakerVerifier::load(dir.path()),
            Err(SpeakerError::MissingModel(_))
        ));
    }
}
