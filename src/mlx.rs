//! §5.3 MLX on Apple Silicon, in-process through mlx-rs ([`native`]): the
//! Parakeet STT, Breeze and Qwen3-TTS engines, and the `[mlx]` keys of
//! `config.toml` (§3.1) they read.

pub mod breeze;
pub mod qwen3_tts;
pub mod native;
pub mod parakeet;

use std::path::Path;

use crate::registry::read_config;

/// Whether the native Breeze engine caches its depth decoder
/// ([`breeze::Params::depth_cache`]): `[mlx] breeze_depth_cache = true`
/// (naru task 2383; off by default, as it samples differently from
/// mlx-audio).
pub fn breeze_depth_cache(home: &Path) -> Result<bool, String> {
    let config = read_config(home)?;
    let on = match config.get("mlx").and_then(|t| t.get("breeze_depth_cache")) {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or("config.toml: `[mlx] breeze_depth_cache` is not true or false")?,
    };
    Ok(on)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::config_path;

    #[test]
    fn breeze_depth_cache_is_off_by_default() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(false));
        std::fs::write(config_path(home.path()), "[mlx]\nother = 1\n").unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(false));
        std::fs::write(
            config_path(home.path()),
            "[mlx]\nbreeze_depth_cache = true\n",
        )
        .unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nbreeze_depth_cache = 1\n").unwrap();
        assert!(breeze_depth_cache(home.path()).is_err());
    }
}
