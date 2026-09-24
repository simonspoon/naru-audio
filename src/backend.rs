//! §3.6 backend availability. Loading lands with the runtime tasks; for now
//! only `available()` exists, so `pull` and `/v1/models` can use it.

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
