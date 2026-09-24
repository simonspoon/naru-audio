//! §3.6 backends: `available()`, used by `pull` and `/v1/models`, and
//! loading. `sherpa-onnx` and `mlx` (§5.3) STT load so far.

use std::path::Path;

use crate::registry::manifest::{Kind, Manifest};
use crate::stt::{SttError, SttModel, sherpa::SherpaStt};

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
}
