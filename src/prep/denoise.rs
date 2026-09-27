//! Offline speech denoising (naru task 1461 §8), the `speech-denoiser-gtcrn`
//! catalog model (GTCRN, MIT). Loaded fresh per request, like [`super::diarize::Diarizer`].

use std::path::{Path, PathBuf};

use sherpa_onnx::{
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserGtcrnModelConfig,
    OfflineSpeechDenoiserModelConfig,
};

pub const MODEL_FILENAME: &str = "gtcrn_simple.onnx";

#[derive(Debug)]
pub enum DenoiseError {
    MissingModel(PathBuf),
    /// `OfflineSpeechDenoiser::create` returned `None`.
    CreateFailed,
}

impl std::fmt::Display for DenoiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenoiseError::MissingModel(path) => {
                write!(f, "denoiser model not found: {}", path.display())
            }
            DenoiseError::CreateFailed => write!(f, "failed to create the speech denoiser"),
        }
    }
}

impl std::error::Error for DenoiseError {}

pub struct Denoiser {
    inner: OfflineSpeechDenoiser,
}

impl Denoiser {
    /// `dir` is the pulled `speech-denoiser-gtcrn` model directory.
    pub fn load(dir: &Path) -> Result<Self, DenoiseError> {
        let model = dir.join(MODEL_FILENAME);
        if !model.is_file() {
            return Err(DenoiseError::MissingModel(model));
        }
        let config = OfflineSpeechDenoiserConfig {
            model: OfflineSpeechDenoiserModelConfig {
                gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                    model: Some(model.to_string_lossy().into_owned()),
                },
                ..Default::default()
            },
        };
        let inner = OfflineSpeechDenoiser::create(&config).ok_or(DenoiseError::CreateFailed)?;
        Ok(Denoiser { inner })
    }

    /// The sample rate `run` expects `pcm` at.
    pub fn sample_rate(&self) -> i32 {
        self.inner.sample_rate()
    }

    /// Denoises one whole clip; the result may be at a different sample
    /// rate than `sample_rate` requires as input (GTCRN keeps it).
    pub fn run(&self, pcm: &[f32], input_sample_rate: i32) -> Vec<f32> {
        self.inner.run(pcm, input_sample_rate).samples
    }
}
