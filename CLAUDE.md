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
- Native MLX (mlx-rs 0.32, naru task 2378; Apple Silicon only): `src/mlx/native.rs`, check with `naru-audio mlx check`
  - mlx-rs/mlx-sys source: `~/.cargo/registry/src/index.crates.io-*/mlx-rs-0.32.0`, `mlx-sys-0.6.0` (docs.rs build is broken; read source)
  - build needs CMake + Xcode Metal toolchain (`xcrun -f metal`); `mlx.metallib` (~184 MB) builds into `target/mlx/` (`MLX_RS_METAL_PATH` in `.cargo/config.toml`); build.rs symlinks it to `target/<profile>/mlx.metallib`. Installs must copy it next to the binary.
- Cargo crate docs: `cargo doc --no-deps --open`, or read source under `~/.cargo/registry/src/index.crates.io-*/<crate>-<ver>/` (`target/doc` does not exist until `cargo doc` runs)
- Build docs: `README.md` at the repo root
- launchd (opt-in, macOS): `scripts/launchd.sh install|uninstall`, gate `scripts/launchd-check.sh`; see `docs/launchd.md` (never alongside the brew service)
- Ambient gate + state machine (inert, naru task 2031): `src/ambient/`, `docs/ambient.md`; llama.cpp behind the off-by-default `llama` feature

## Dev environment

- Dev work uses the dev daemon on :7871 (naru task 2092), started with `scripts/dev-audio.sh` (state in `~/.naru-audio-dev`).
- Never install to, restart or overwrite the real install: launchd `com.naru.audio`, port :7870, `~/.naru-audio`.
- Full details live in the naru repo's `docs/dev-env.md` (`/Users/simonspoon/naru-projects/projects/tools/naru/docs/dev-env.md`), which exists once naru task 2094 lands.
