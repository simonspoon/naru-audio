//! §3.6 backends: `available()`, used by `pull` and `/v1/models`, and
//! loading. `sherpa-onnx` and `mlx` (§5.3) STT and TTS load so far; voice-prep's
//! diarization and denoise models (naru task 1461 §8) are `sherpa-onnx` only,
//! so they need no per-backend dispatch, just the `Kind` check.

use std::path::Path;

use crate::prep::denoise::{DenoiseError, Denoiser};
use crate::prep::diarize::{DiarizeError, Diarizer};
use crate::prep::isolate::{IsolateError, Isolator};
use crate::registry::manifest::{Kind, Manifest};
use crate::stt::{SttError, SttModel, sherpa::SherpaStt};
use crate::tts::{TtsError, TtsModel, sherpa::SherpaTts};

/// Whether `backend` can run on this machine, or why not.
pub fn available(backend: &str) -> Result<(), String> {
    available_on(
        backend,
        cfg!(all(target_arch = "aarch64", target_os = "macos")),
    )
}

/// [`available`] on a machine that is Apple Silicon or not.
fn available_on(backend: &str, apple_silicon: bool) -> Result<(), String> {
    match backend {
        // Compiled in on every platform.
        "sherpa-onnx" => Ok(()),
        // Compiled in on aarch64-apple-darwin only.
        "mlx" if apple_silicon => Ok(()),
        "mlx" => Err("requires Apple Silicon".to_string()),
        other => Err(format!("unknown backend `{other}`")),
    }
}

/// Loads the STT model described by `manifest` from its pulled `dir`
/// (`models/<name>/`).
pub fn load_stt(manifest: &Manifest, dir: &Path) -> Result<Box<dyn SttModel>, SttError> {
    let backend = &manifest.model.backend;
    let unavailable = |reason: String| SttError::BackendUnavailable {
        backend: backend.clone(),
        reason,
    };
    available(backend).map_err(unavailable)?;
    if manifest.model.kind != Kind::Stt {
        return Err(SttError::NotStt(manifest.model.name.clone()));
    }
    match backend.as_str() {
        "sherpa-onnx" => Ok(Box::new(SherpaStt::load(manifest, dir)?)),
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        "mlx" => Ok(Box::new(crate::stt::mlx::MlxStt::load(manifest, dir)?)),
        _ => Err(unavailable(
            "cannot load speech-to-text models yet".to_string(),
        )),
    }
}

/// Loads the diarization model described by `manifest` from its pulled
/// `dir`. Every `[model] kind = "diarization"` manifest is `sherpa-onnx`
/// (the only backend that implements it), so unlike [`load_stt`]/[`load_tts`]
/// there is nothing to dispatch on.
pub fn load_diarizer(manifest: &Manifest, dir: &Path) -> Result<Diarizer, DiarizeError> {
    debug_assert_eq!(manifest.model.kind, Kind::Diarization);
    Diarizer::load(dir)
}

/// Loads the denoise model described by `manifest` from its pulled `dir`,
/// as [`load_diarizer`].
pub fn load_denoiser(manifest: &Manifest, dir: &Path) -> Result<Denoiser, DenoiseError> {
    debug_assert_eq!(manifest.model.kind, Kind::Denoise);
    Denoiser::load(dir)
}

/// Loads the source-separation model described by `manifest` from its
/// pulled `dir`, as [`load_diarizer`].
pub fn load_isolator(manifest: &Manifest, dir: &Path) -> Result<Isolator, IsolateError> {
    debug_assert_eq!(manifest.model.kind, Kind::Separation);
    Isolator::load(dir)
}

/// Loads the TTS model described by `manifest` from its pulled `dir`.
pub fn load_tts(manifest: &Manifest, dir: &Path) -> Result<Box<dyn TtsModel>, TtsError> {
    let backend = &manifest.model.backend;
    let unavailable = |reason: String| TtsError::BackendUnavailable {
        backend: backend.clone(),
        reason,
    };
    available(backend).map_err(unavailable)?;
    if manifest.model.kind != Kind::Tts {
        return Err(TtsError::NotTts(manifest.model.name.clone()));
    }
    match backend.as_str() {
        "sherpa-onnx" => Ok(Box::new(SherpaTts::load(manifest, dir)?)),
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        "mlx" => Ok(Box::new(crate::tts::mlx::MlxTts::load(manifest, dir)?)),
        _ => Err(unavailable(
            "cannot load text-to-speech models yet".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::manifest::Catalog;

    /// §3.6 on Intel: the MLX parakeet is listed (its platforms name
    /// macos-x86_64, so `Registry::list` keeps it unpulled), with
    /// `x_available:false` and the reason.
    #[test]
    fn intel_lists_the_mlx_model_as_unavailable() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let mlx = &cat.models["parakeet-tdt-0.6b-v2-mlx"];
        assert!(mlx.runs_on("macos", "x86_64"));
        assert_eq!(
            available_on(&mlx.model.backend, false),
            Err("requires Apple Silicon".to_string())
        );
        assert!(mlx.runs_on("macos", "aarch64"));
        assert_eq!(available_on(&mlx.model.backend, true), Ok(()));
        assert!(!mlx.runs_on("linux", "x86_64"));
    }

    #[test]
    fn load_tts_refuses_an_stt_model() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let stt = &cat.models["parakeet-tdt-0.6b-v2-int8"];
        let err = load_tts(stt, Path::new("/nonexistent")).err().unwrap();
        assert!(matches!(err, TtsError::NotTts(_)), "{err}");
    }
}
