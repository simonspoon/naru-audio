//! §3.6 backends: `available()`, used by `pull` and `/v1/models`, and
//! loading. Only `sherpa-onnx` STT loads so far.

use std::path::Path;

use crate::registry::manifest::{Kind, Manifest};
use crate::stt::{SttError, SttModel, sherpa::SherpaStt};

/// Whether `backend` can run on this machine, or why not.
pub fn available(backend: &str) -> Result<(), String> {
    match backend {
        // Compiled in on every platform.
        "sherpa-onnx" => Ok(()),
        // Compiled in on aarch64-apple-darwin only.
        "mlx" if cfg!(all(target_arch = "aarch64", target_os = "macos")) => Ok(()),
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
        _ => Err(unavailable(
            "cannot load speech-to-text models yet".to_string(),
        )),
    }
}
