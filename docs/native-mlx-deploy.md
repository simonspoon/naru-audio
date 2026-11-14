# Deploying native Qwen3-TTS (naru task 2389)

> History, superseded by task 2391 below: the Python sidecar no longer
> exists, so there is no sidecar fallback or `[mlx]` switch to roll back to.
> The Install and Check sections stay current.

From this build on, `qwen3-tts-0.6b-base-mlx` and `qwen3-tts-1.7b-base-mlx`
run in-process (mlx-rs, no Python), as 8-bit Breeze and Parakeet do. The
audio is byte-identical to the old sidecar's for the same seed, cloned voices
included; first audio arrives in ~0.07-0.09 s instead of ~0.25-0.28 s
(task 2388).

## The real install today

`~/.naru-audio/config.toml`:

```toml
[defaults]
stt = "whisper-large-v3-turbo-mlx"
tts = "qwen3-tts-1.7b-base-mlx"
```

No config change is needed for Qwen3-TTS: `qwen3-tts-1.7b-base-mlx` runs
in-process once the new binary runs. (The `stt` default is dropped; see task
2390 below. Any `[mlx] python` key is ignored.)

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

If the native load fails (most often `mlx.metallib` missing beside the
binary), the request gets 503 `backend_unavailable` with the reason; there is
no other engine to fall back to.

## Roll back

Back to the previous binary:

```sh
rm ~/.cargo/bin/naru-audio
cp ~/.cargo/bin/naru-audio.pre-2389 ~/.cargo/bin/naru-audio
launchctl kickstart -k gui/$(id -u)/com.naru.audio
```

(The `[mlx] native_qwen3_tts`, `native_tts` and `native_stt = false`
switches that once selected the sidecar were removed in task 2391 and are
ignored.)

## Verified (dev, ~/naru-scratch/2389)

- Fresh home, no `config.toml`, no `mlx/` venv: both sizes speak a cloned
  voice (Her3 on 0.6B, Jarvis on 1.7B), `/api/ps` reports `engine=native`, the
  daemon has no child process, and the PCM is byte-identical to task 2388's
  sidecar output for the same text and seed.
- At the time, `native_qwen3_tts = false` loaded both sizes in the sidecar
  with the same bytes (historical: the sidecar and that switch are gone).
- `cargo test --lib`: 339 passed.

# Deploying the dropped-model removal (naru task 2390)

From this build on the catalog no longer has `whisper-large-v3-turbo-mlx`,
the Qwen3-ASR pair, both Chatterbox models, IndexTTS, OmniVoice, VoxCPM2,
`qwen3-tts-0.6b-mlx` or the Qwen3-TTS VoiceDesign model. A default naming one
still starts the daemon: `serve` logs one line per replaced default,

```
… warn - dropped_model stt=whisper-large-v3-turbo-mlx fallback=parakeet-tdt-0.6b-v2-mlx (model removed; set another in config.toml)
```

and uses the fallback (STT: `parakeet-tdt-0.6b-v2-mlx`; `qwen3-tts-0.6b-mlx`:
`qwen3-tts-0.6b-base-mlx`; any other dropped TTS model:
`qwen3-tts-1.7b-base-mlx`). A `[defaults.voices]` entry for a dropped model
carries over to its fallback. A request that names a dropped model itself
gets the usual unknown-model error. Weights already pulled under
`~/.naru-audio/models/` stay on disk until `naru-audio rm <name>`.

## The real install

Its `stt = "whisper-large-v3-turbo-mlx"` is dropped, so STT moves to native
Parakeet MLX; `tts = "qwen3-tts-1.7b-base-mlx"` is kept. The config is
changed only as a deploy step by the live main thread, never by an agent:

1. Pull the replacement first, so the first transcription does not wait on a
   download: `naru-audio pull parakeet-tdt-0.6b-v2-mlx`.
2. Install the binary as above (Install), then set `~/.naru-audio/config.toml`:

   ```toml
   [defaults]
   stt = "parakeet-tdt-0.6b-v2-mlx"
   tts = "qwen3-tts-1.7b-base-mlx"
   ```

   Without the edit the daemon still uses Parakeet, but logs the
   `dropped_model` warning on every start.
3. Restart the LaunchAgent as above, then check that the newest start in
   `~/.naru-audio/serve.log` has no `dropped_model` line and that
   `load model=parakeet-tdt-0.6b-v2-mlx` ends `engine=native`.

Roll back: restore the previous binary as above and put
`stt = "whisper-large-v3-turbo-mlx"` back.

# Deploying the Python sidecar removal (naru task 2391)

From this build on naru-audio has no Python at all: every `mlx` model runs
in-process (Parakeet, 8-bit Breeze, both Qwen3-TTS Base sizes). Gone:
`naru-audio mlx setup` and `mlx status` (`mlx check` stays), the
`[mlx] native_stt`, `native_tts` and `native_qwen3_tts` fallback switches,
`[mlx] python`, and the venv on the LaunchAgent's `PATH`. Left-over keys in
`config.toml` are ignored, so the real config still loads unchanged. A
failed native load is now 503 `backend_unavailable` with the reason; there
is nothing to fall back to.

`breeze-tts-2-mlx` (bf16) and `breeze-tts-2-4bit-mlx` had no native port and
are dropped like the models in task 2390: a default naming one falls back to
`breeze-tts-2-8bit-mlx` with a `dropped_model` warning; pulled copies are
hidden until `naru-audio rm`. The real install has both pulled and one
cloned voice made for the 4-bit model; that voice no longer plays until it
is re-made for `breeze-tts-2-8bit-mlx`.

## Upgrade steps (live main thread only, never an agent)

1. Recommended, not required: do the task 2390 steps first if they have not
   run. The real `stt` default is still `whisper-large-v3-turbo-mlx`; without
   the edit the daemon falls back to Parakeet with a warning.
2. Install the binary and `mlx.metallib` as under Install above. Restart by
   re-rendering the LaunchAgent, so its `PATH` drops
   `~/.naru-audio/mlx/.venv/bin` (`scripts/launchd.sh install` rewrites
   `~/Library/LaunchAgents/com.naru.audio.plist` and restarts it).
3. Check: `naru-audio mlx check`, then the Check section above (every
   `load model=…` ends `engine=native`), and no Python child of the daemon:
   `pgrep -lf naru_audio_mlx` prints nothing.
4. Remove the stale Python environment (~1.3 GB) and its config key:
   `rm -rf ~/.naru-audio/mlx`, then delete the `[mlx]` table
   (`python = …`) from `~/.naru-audio/config.toml`.
5. Optional, to free disk: `naru-audio rm breeze-tts-2-mlx breeze-tts-2-4bit-mlx`
   (and the task 2390 models). `uv` itself is no longer needed by naru-audio.

Roll back: restore the previous binary (Roll back above), skipping step 4
until the new build has run for a while; the previous binary (pre-2391)
still finds the venv and its `[mlx] python`, which this build ignores.
