# Deploying native Qwen3-TTS (naru task 2389)

From this build on, `qwen3-tts-0.6b-base-mlx` and `qwen3-tts-1.7b-base-mlx`
run in-process (mlx-rs, no Python) by default, as 8-bit Breeze and Parakeet
already do. The audio is byte-identical to the sidecar's for the same seed,
cloned voices included; first audio arrives in ~0.07-0.09 s instead of
~0.25-0.28 s (task 2388). The Python sidecar stays as the fallback for one
release: a failed native load logs `native_load_failed` and loads the model
in the sidecar instead.

## The real install today

`~/.naru-audio/config.toml`:

```toml
[mlx]
python = "/Users/simonspoon/.naru-audio/mlx/.venv/bin/python"

[defaults]
stt = "whisper-large-v3-turbo-mlx"
tts = "qwen3-tts-1.7b-base-mlx"
```

No config change is needed: `qwen3-tts-1.7b-base-mlx` goes native once the
new binary runs. Keep `[mlx] python`: Whisper STT still runs in the sidecar.

The LaunchAgent `com.naru.audio` runs `~/.cargo/bin/naru-audio serve`
(logs `~/.naru-audio/serve.log`). The binary installed there (2026-10-25)
predates native MLX, so this is also the first install that needs
`mlx.metallib` beside it.

## Install

Run from the repo (`tools/naru-audio`); needs CMake and the Xcode Metal
toolchain (`xcrun -f metal`).

```sh
cargo build --release
cp ~/.cargo/bin/naru-audio ~/.cargo/bin/naru-audio.pre-2389   # for rollback
rm ~/.cargo/bin/naru-audio      # cp over a running binary trips OS_REASON_CODESIGNING
cp target/release/naru-audio ~/.cargo/bin/naru-audio
cp target/release/mlx.metallib ~/.cargo/bin/mlx.metallib       # follows the symlink, ~184 MB
~/.cargo/bin/naru-audio mlx check                              # metallib found, GPU matmul ok
launchctl kickstart -k gui/$(id -u)/com.naru.audio
```

## Check

```sh
naru-audio say "Native check, one two three." -m qwen3-tts-1.7b-base-mlx -v Jarvis -o /tmp/native-check.wav
grep 'load model=qwen3-tts-1.7b-base-mlx' ~/.naru-audio/serve.log | tail -1   # ends engine=native
curl -s http://127.0.0.1:7870/api/ps                                       # "engine":"native"
```

`engine=sidecar` with a `native_load_failed` line before it means the native
load failed (most often `mlx.metallib` missing beside the binary); speech
still works through the sidecar.

## Roll back

Back to the sidecar, same binary: add the key under `[mlx]` and restart (the
engine is picked when the model loads):

```toml
[mlx]
python = "/Users/simonspoon/.naru-audio/mlx/.venv/bin/python"
native_qwen3_tts = false
```

```sh
launchctl kickstart -k gui/$(id -u)/com.naru.audio
```

Back to the previous binary:

```sh
rm ~/.cargo/bin/naru-audio
cp ~/.cargo/bin/naru-audio.pre-2389 ~/.cargo/bin/naru-audio
launchctl kickstart -k gui/$(id -u)/com.naru.audio
```

`native_tts = false` (8-bit Breeze) and `native_stt = false` (Parakeet) are
the matching switches for the other native engines; neither is in the
real config's defaults today.

## Verified (dev, ~/naru-scratch/2389)

- Fresh home, no `config.toml`, no `mlx/` venv: both sizes speak a cloned
  voice (Her3 on 0.6B, Jarvis on 1.7B), `/api/ps` reports `engine=native`, the
  daemon has no child process, and the PCM is byte-identical to task 2388's
  sidecar output for the same text and seed.
- `native_qwen3_tts = false` with the venv: both sizes load in the sidecar
  (`engine=sidecar`, python child running), same bytes.
- `cargo test --lib`: 339 passed.
