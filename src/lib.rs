//! naru-audio: local STT/TTS daemon for Naru (docs/design.md).

pub mod backend;
pub mod error;
pub mod log;
pub mod manager;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod mlx;
pub mod prep;
pub mod profile;
pub mod registry;
pub mod server;
pub mod stt;
pub mod tts;
pub mod voices;
