## Local dependency paths

These answer "where does X live". Check here first instead of running a filesystem-wide search (root-rooted `find /`, grep, or a python import-and-locate); the root-search guard hook denies those.

- MLX python venv: `~/.naru-audio/mlx/.venv` (`/Users/simonspoon/.naru-audio/mlx/.venv`)
  - site-packages: `~/.naru-audio/mlx/.venv/lib/python3.12/site-packages` (glob: `python3*`)
  - `mlx_audio/` and `mlx_whisper/` sit directly in that site-packages dir
- sherpa-onnx (Cargo.lock pins 1.13.6 for both crates):
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/sherpa-onnx-1.13.6`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/sherpa-onnx-sys-1.13.6` (glob: `index.crates.io-*`; a stray `-sys-1.13.8` also exists, unused)
  - prebuilt static libs: `target/sherpa-onnx-prebuilt/` in the repo
  - Rust usage: `src/tts/sherpa.rs`, `src/stt/sherpa.rs`
- Cargo crate docs: `cargo doc --no-deps --open`, or read source under `~/.cargo/registry/src/index.crates.io-*/<crate>-<ver>/` (`target/doc` does not exist until `cargo doc` runs)
- Build docs: `README.md` at the repo root
