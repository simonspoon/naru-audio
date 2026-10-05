# naru-audio

A local speech-to-text (STT) and text-to-speech (TTS) daemon. It works like
Ollama: one long-running process, models pulled by name, loaded on first use
and unloaded when idle, behind an HTTP API on `127.0.0.1:7870`.

- `/v1/*` is OpenAI-compatible (`/v1/audio/transcriptions`,
  `/v1/audio/speech`, `/v1/models`), so stock OpenAI clients work.
- `/api/*` is native and Ollama-shaped (pull, remove, load, loaded models).
- A WebSocket at `/v1/audio/transcriptions/stream` transcribes live audio.
- The same binary is a CLI: pull models, check health, transcribe a file,
  speak text, clone a voice.

It was built as the speech backend for Naru, but any HTTP client can use it.
The full design (API, error codes, registry, memory budget, sidecar
protocol) is in [docs/design.md](docs/design.md).

## Contents

For users: [Requirements](#requirements) ·
[Install](#install) · [First run](#first-run) · [Models](#models) ·
[Voices and cloning](#voices-and-cloning) · [Configuration](#configuration) ·
[HTTP API](#http-api) · [Health check](#health-check) ·
[Troubleshooting](#troubleshooting) · [Uninstall](#uninstall)

For developers: [Building from source](#building-from-source) ·
[Repository layout](#repository-layout) · [Running tests](#running-tests) ·
[The MLX sidecar](#the-mlx-sidecar) · [Release process](#release-process) ·
[Contributing](#contributing) · [License](#license)

---

## Requirements

- **macOS or Linux, arm64 or x86_64.** The default models run on the
  built-in sherpa-onnx backend (ONNX Runtime, statically linked), on every
  platform, with no Python and no GPU.
- **Apple Silicon, for the MLX models only.** Models whose backend is `mlx`
  (Parakeet and Whisper on MLX, the Qwen3-TTS family, and so voice cloning) run in a
  Python sidecar on arm64 macOS. They need [`uv`](https://docs.astral.sh/uv/)
  and a one-time `naru-audio mlx setup`. On other machines they are listed
  as unavailable.
- **macOS for `voice add`**: clips are converted with the system `afconvert`.
- **Disk and memory for models.** Models are downloaded on request, not
  bundled. The defaults take about 1 GB on disk (see [Models](#models)).

The daemon picks a hardware profile at startup: `large` for arm64 with at
least 24 GB of RAM, `small` otherwise. The profile sets the memory budget
for loaded models (50% of RAM on `large`; the lower of 40% of RAM and 6 GiB
on `small`). Both profiles default to the same models.

## Install

### Homebrew

The formula is [Formula/naru-audio.rb](Formula/naru-audio.rb). The release
workflow publishes it to the `simonspoon/tap` tap when a version is tagged
(see [Release process](#release-process)); until the first tagged release,
[build from source](#building-from-source).

```sh
brew install simonspoon/tap/naru-audio
naru-audio pull default
brew services start naru-audio
```

The formula installs one binary. Its service runs `naru-audio serve`,
restarts it if it crashes (a `brew services stop` is respected), and logs to
`$(brew --prefix)/var/log/naru-audio.log`, which the daemon rotates at
10 MiB into `naru-audio.log.1`.

### From source

See [Building from source](#building-from-source), then put
`target/release/naru-audio` on your `PATH` and run `naru-audio serve`
yourself (it logs to stderr when it is not installed by Homebrew).

## First run

```sh
# Download the default STT and TTS models (and the VAD model they require)
# into ~/.naru-audio. This works without the daemon running.
naru-audio pull default

# Start the daemon: as a service, or in the foreground.
brew services start naru-audio        # or: naru-audio serve

naru-audio health                     # exit 0 when speech is ready
naru-audio say "Hello from naru-audio." -o hello.wav
naru-audio transcribe hello.wav
```

`say` writes a 16-bit mono WAV; `-o -` streams it to stdout instead, one
sentence at a time. Both `say` and `transcribe` take `-` for stdin.
Other client commands:

```sh
naru-audio transcribe hello.wav --format json    # one JSON line per segment, then the transcript
naru-audio stream hello.wav                      # replay a WAV over the WebSocket in real time
naru-audio ps                                    # models the daemon has loaded
naru-audio list                                  # pulled models and those that can run here
```

Every command has `--help`.

## Models

The catalog is built into the binary (the TOML files in [catalog/](catalog/)).
Every file has a pinned URL and sha256; downloads are checked before they are
used.

| Name | Kind | Backend | Download |
|---|---|---|---|
| `parakeet-tdt-0.6b-v2-int8` | STT (English), **default** | sherpa-onnx | 0.66 GB |
| `kokoro-v1.0` | TTS, **default** | sherpa-onnx | 0.35 GB |
| `silero-vad` | VAD, pulled with the STT models | sherpa-onnx | 2 MB |
| `pocket-tts-int8` | TTS (English) | sherpa-onnx | 0.10 GB |
| `parakeet-tdt-0.6b-v2-mlx` | STT (English) | mlx | 2.5 GB |
| `whisper-large-v3-turbo-mlx` | STT, 99 languages, word timestamps | mlx | 1.6 GB |
| `qwen3-tts-0.6b-mlx` | TTS, preset voices | mlx | 2.0 GB |
| `qwen3-tts-0.6b-base-mlx` | TTS, speaks cloned voices | mlx | 2.0 GB |
| `qwen3-tts-1.7b-base-mlx` | TTS, speaks cloned voices | mlx | 3.1 GB |
| `qwen3-tts-1.7b-voicedesign-mlx` | TTS, voice from a text description | mlx | 3.1 GB |
| `chatterbox-tts-8bit-mlx` | TTS, speaks cloned voices | mlx | 1.3 GB |
| `chatterbox-turbo-8bit-mlx` | TTS, speaks cloned voices, inline `[laugh]`-style tags, English | mlx | 1.0 GB |
| `indextts-1.5-mlx` | TTS, speaks cloned voices, does not stream | mlx | 1.4 GB |
| `voxcpm2-8bit-mlx` | TTS, clones or designs a voice, 48 kHz | mlx | 3.2 GB |
| `omnivoice-bf16-mlx` | TTS, clones or designs a voice, 646 languages, does not stream, **non-commercial** | mlx | 1.6 GB |
| `breeze-tts-2-mlx` | TTS, clones or designs a voice, **non-commercial** | mlx | 7.6 GB |

Each model's license is in its catalog file (`license = …`); check it before
you use a model's output. `naru-audio list` and `GET /v1/models` show it
(`x_license`, `x_license_url`), and pulling a non-commercial model
(`pocket-tts-int8`, per its README; `omnivoice-bf16-mlx`, CC-BY-NC-4.0;
`breeze-tts-2-mlx`, BreezeBlue's own research/non-commercial license)
prints a warning; `x_non_commercial` marks it in the API.

```sh
naru-audio pull pocket-tts-int8       # pull by name; `default` means the configured STT and TTS models
naru-audio verify                     # re-hash every pulled model's files
naru-audio rm pocket-tts-int8         # remove; refuses a model another pulled model requires, unless --force
naru-audio list --names               # pulled model names, one per line
```

`pull` refuses a model whose backend cannot run on this machine unless you
pass `--force`. You can add your own manifests (same format as
[catalog/](catalog/), `file:///` URLs allowed, sha256 still required) in
`~/.naru-audio/catalog.d/*.toml`; they are merged over the built-in catalog.

## Voices and cloning

List a model's voices through the daemon:

```sh
curl 'http://127.0.0.1:7870/v1/audio/voices?model=kokoro-v1.0'
naru-audio say "A different voice." -v am_adam -o adam.wav
```

`kokoro-v1.0`'s default voice is `af_heart`; `-s` sets the speed (0.5 to 2.0).

**Cloning** needs Apple Silicon and the MLX sidecar:

```sh
naru-audio mlx setup                              # once; needs uv on PATH
naru-audio pull qwen3-tts-0.6b-base-mlx
naru-audio voice add myvoice clip.wav --text "Exactly what the clip says."
naru-audio say "This is my cloned voice." -v myvoice -o cloned.wav
```

- The clip is one speaker, 5–15 s for best results (3–30 s accepted), in WAV,
  MP3 or anything else `afconvert` reads. `--text` must be its exact
  transcript.
- Recording a reference that clones well (a Qwen3-TTS Base voice leans on it
  for the whole timbre, and a flaw in it is copied into every sentence):
  - 8–12 s of one speaker in a quiet room without echo, close to the
    microphone at a steady distance; no music, no second voice, no
    clipping, no noise-reduction artefacts. Speak in the register you want
    the voice to have, at an even pace, not read stiffly.
  - Start and end on a beat of silence (a few hundred ms), and end on a
    finished sentence, not mid-word; a clip cut off mid-phrase makes the
    clone trail off or ramble.
  - `--text` is the exact transcript of the clip, word for word, with the
    punctuation as spoken, numbers written the way they were said, and
    nothing added or left out. A wrong transcript misaligns the in-context
    clone; Qwen3-TTS Base reads it, Chatterbox ignores it.
  - Record mono at 24 kHz or more (any rate is converted); the stored
    `ref.wav` is 24 kHz mono 16-bit, so there is nothing to gain from
    more. Prefer WAV or another lossless source over a re-encoded MP3.
- The voice is stored as `~/.naru-audio/voices/<name>/ref.wav` (24 kHz mono),
  `ref.txt` and `model.txt`, the model it was made for. An existing voice is
  never replaced; delete its directory to redo it. Voices are read on each
  request, so a new one works without restarting the daemon.
- `voice add` (the CLI) always records `qwen3-tts-0.6b-base-mlx`; it has no
  `--model` yet. `POST /v1/audio/voices` takes one (§2.5 below), for
  `-m qwen3-tts-1.7b-base-mlx`, `-m chatterbox-tts-8bit-mlx`,
  `-m chatterbox-turbo-8bit-mlx`,
  `-m indextts-1.5-mlx`, `-m voxcpm2-8bit-mlx`, `-m omnivoice-bf16-mlx` or
  `-m breeze-tts-2-mlx` instead. `say` with a cloned voice and no `-m` picks
  the model it was made for; an explicit `-m` that voice was not made for
  is refused rather than silently ignored.
- `GET /v1/models` names each model's cloning and voice-design capability
  (`x_clone`, `x_clone_requires_transcript`, `x_instruct`), so a client
  picking a model knows what it can do before offering it. `GET
  /v1/audio/voices?model=<name>` lists only the clones made for `<name>`,
  alongside its own preset voices; `?model=clones` still lists every
  cloned voice regardless of model, each naming which one it is for.
  Voices can also be added with `POST /v1/audio/voices` and exported,
  model and all, with `GET /v1/audio/voices/{name}` ([docs/design.md
  §2.5](docs/design.md)).
- `GET /v1/models`' `x_prompt_format.knobs` names each model's own
  generation knobs beyond `speed`/`exaggeration` — Qwen3-TTS's
  `temperature`/`top_p`/`top_k`, Chatterbox's `cfg_weight`, VoxCPM2's
  `cfg_value`/`inference_timesteps`, and so on — each with `min`, `max`,
  `default` and, for one that takes an integer, `step`. `POST
  /v1/audio/speech` and `POST /api/voices/preview` take them as
  `"knobs":{"<name>":<number>}`; a name not in `x_prompt_format.knobs` or a
  value outside its range is 400 ([docs/design.md §2.7](docs/design.md)).
- `POST /v1/audio/speech` also takes `"seed":<non-negative integer>`
  (default 0, so a voice sounds the same from one reply to the next); the
  MLX backend seeds its sampling with it before each generate, and speaks a
  cloned voice in chunks of whole sentences (a chunk closes at 15 words, so
  short sentences pair up) with the reference re-applied to each. A cloned
  Qwen3-TTS voice samples at `temperature` 0.55, `top_p` 0.8, `top_k` 20,
  tighter than mlx-audio's own, which lets the timbre drift between
  sentences; `knobs` override them.
  Every request's audio is also saved as `~/.naru-audio/recent/<id>.wav`
  (the `id=` in its `tts_request` log line); the newest 50 are kept.
- `chatterbox-tts-8bit-mlx` (MIT) needs no transcript and ignores `speed`
  entirely — a non-1.0 `--speed` with it fails the request instead of
  quietly synthesising at normal speed. It takes `--exaggeration` (0-1, an
  emotion-exaggeration dial); every other model ignores it. Its first load
  fetches a small shared tokenizer from Hugging Face, so it needs network
  access once even though every other model runs fully offline after
  `pull`.
- `chatterbox-turbo-8bit-mlx` (MIT, English) is Chatterbox Turbo: it clones
  from the reference alone and reads tags in the text itself, passed
  through untouched: `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`,
  `[clear throat]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`. It has no
  `exaggeration` or `cfg_weight`, ignores `speed` as Chatterbox does, and
  needs the same one-time network fetch of the shared tokenizer.
- `voxcpm2-8bit-mlx` (Apache-2.0) also needs no transcript and, like
  Chatterbox, does not support `speed` (mlx-audio has no `speed` parameter
  for it at all) — a non-1.0 `--speed` fails the request the same way. It
  speaks at 48 kHz, not the 24 kHz every other model here uses.
- `indextts-1.5-mlx` (Apache-2.0) also needs no transcript and does not
  support `speed` (no `speed` parameter, only `**kwargs`), same as
  Chatterbox and VoxCPM2. Unlike every other model here, it does not
  stream: mlx-audio generates the whole utterance before yielding
  anything, so `-o -` still works but all the audio arrives in one piece
  at the end rather than as it is produced.
- `omnivoice-bf16-mlx` (**CC-BY-NC-4.0, non-commercial only** — k2-fsa's
  README: "The pre-trained model is licensed under the CC-BY-NC due to
  constraints from its training data") clones or designs a voice like
  VoxCPM2, speaks 646 languages, and does not support `speed` either. Like
  IndexTTS, it does not stream — one chunk at the end, so `-o -` still
  works but nothing plays until synthesis finishes. Its cloned voice's
  transcript (`--text`) is not ignored: mlx-audio prepends it to the
  spoken text, so it should still be the clip's exact words.
- `breeze-tts-2-mlx` (**BreezeBlue Research and Non-Commercial License,
  non-commercial only** — the LICENSE: "This Agreement permits research
  and non-commercial use of the Model Materials free of charge... it does
  not grant any commercial rights") clones or designs a voice like
  VoxCPM2 and OmniVoice, and does not support `speed` either (no `speed`
  parameter). Unlike those two, it does stream at the sidecar's usual
  cadence with no extra handling needed.

**Voice design**: `qwen3-tts-1.7b-voicedesign-mlx` has no voices; it makes
one up from `--instructions`, which it requires:

```sh
naru-audio pull qwen3-tts-1.7b-voicedesign-mlx
naru-audio say "Good evening." -m qwen3-tts-1.7b-voicedesign-mlx \
  --instructions "A warm, husky woman in her thirties. Speak slowly." -o designed.wav
```

`voxcpm2-8bit-mlx`, `omnivoice-bf16-mlx` and `breeze-tts-2-mlx` do both:
`say -v myvoice -m voxcpm2-8bit-mlx` (or `-m omnivoice-bf16-mlx` / `-m
breeze-tts-2-mlx`) clones a voice added the same way as above, and `say -m
voxcpm2-8bit-mlx --instructions "..."` designs one instead, with no voice
named — the two are mutually exclusive per request, and a named cloned
voice always wins if both are given.

## Configuration

All state lives in `$NARU_AUDIO_HOME`, default `~/.naru-audio`:

```
~/.naru-audio/
├── config.toml     optional settings (below)
├── catalog.d/      your own model manifests
├── models/<name>/  pulled models
├── voices/<name>/  cloned voices
├── state/          measured memory use per model
├── tmp/            partial downloads and locks
└── mlx/            the MLX sidecar's venv (Apple Silicon)
```

`config.toml` (every key is optional):

```toml
[defaults]
stt = "parakeet-tdt-0.6b-v2-int8"   # the model used when a request names none, or `default`
tts = "kokoro-v1.0"
keep_alive = "5m"                   # how long an idle model stays loaded; 0 unloads once idle, negative never

[memory]
max_resident = "8GiB"               # budget for loaded models; idle ones are evicted, oldest first, to fit

[mlx]
python = "…"                        # written by `naru-audio mlx setup`; do not edit
```

Environment variables:

| Variable | Effect |
|---|---|
| `NARU_AUDIO_HOME` | State directory (default `~/.naru-audio`). |
| `NARU_AUDIO_LISTEN` | `serve`'s bind address (default `127.0.0.1:7870`); same as `--listen`. |
| `NARU_AUDIO_LOG` | `serve`'s log file; same as `--log-file`. |
| `NARU_AUDIO_STT_MODEL`, `NARU_AUDIO_TTS_MODEL` | Default models; override `config.toml`. |
| `NARU_AUDIO_KEEP_ALIVE` | Default keep-alive; overrides `config.toml`. |
| `NARU_AUDIO_URL` | Where the client commands find the daemon (default `http://127.0.0.1:7870`). |

A request's own `model` and `keep_alive` win over all of these. Under
`brew services` the environment is the formula's, so use `config.toml`
there.

**Network access.** The daemon listens on loopback only. `serve --listen
0.0.0.0:7870 --allow-remote` exposes it to the network and turns off the
`Host` header check; there is no authentication, so only do this on a
network you trust. A request with an `Origin` header (a browser fetch or
WebSocket) is allowed only when it names exactly the `Host` the request came
in on — same-origin — and that `Host` is itself allowed; `--allow-remote`
still requires the Origin/Host match. This is what lets the built-in
`/admin` UI (served by this daemon, same origin) call the API from the
browser; a page on any other site gets 403.

## HTTP API

```sh
curl http://127.0.0.1:7870/health
curl http://127.0.0.1:7870/v1/models
curl http://127.0.0.1:7870/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"kokoro-v1.0","input":"Hello there.","voice":"af_heart"}' -o speech.wav
curl http://127.0.0.1:7870/v1/audio/transcriptions -F file=@speech.wav -F model=default
```

| Route | Purpose |
|---|---|
| `GET /health` | Readiness of the default STT and TTS models and of each backend. |
| `GET /v1/models` | Pulled and runnable models (OpenAI list shape, plus `x_*` fields). |
| `POST /v1/audio/transcriptions` | WAV in, text out (`json`, `text`, `verbose_json`; SSE with `stream=true`). |
| `GET /v1/audio/transcriptions/stream` | WebSocket streaming STT. |
| `POST /v1/audio/speech` | Text in, streamed `wav` or `pcm` out. |
| `GET`/`POST /v1/audio/voices`, `GET`/`PATCH`/`DELETE /v1/audio/voices/{name}` | List, add, export, rename/update and remove cloned and designed voices. |
| `GET`/`POST /api/voices/{model}/{voice}/sample`, `POST /api/voices/preview` | A voice's own or a cached (`GET`) or freshly generated and cached (`POST`) sample clip; a one-off preview from an uploaded clip. |
| `POST /api/pull`, `DELETE /api/models/{name}` | Pull (NDJSON progress) and remove models. |
| `GET`/`DELETE /api/pulls/{name}` | List pulls in progress; cancel one. |
| `GET /api/ps`, `POST /api/load` | Loaded models; warm or unload one. |
| `GET`/`PUT /api/defaults` | The live default STT/TTS model and each TTS model's default voice. |

Errors use OpenAI's envelope, `{"error":{"message","type","code","param"}}`.
Fields, extensions and every error code are in
[docs/design.md §2](docs/design.md).

## Health check

```sh
naru-audio health
```

It calls `GET /health` on `$NARU_AUDIO_URL` and exits:

| Exit | Meaning |
|---|---|
| 0 | The daemon is up and the default STT model is ready. |
| 3 | The daemon is not running (or did not answer within 5 s). |
| 4 | The daemon is up but the default STT model is not ready, for example not pulled. The message says what to run. |

`GET /health` itself answers from memory and never waits behind a decode.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `naru-audio isn't running at http://127.0.0.1:7870` (exit 3) | Start it: `brew services start naru-audio` or `naru-audio serve`. If it runs on another address, set `NARU_AUDIO_URL`. |
| `the model … isn't downloaded. Run naru-audio pull …` (exit 4), or HTTP 409 `model_not_pulled` | Run the command it prints. The daemon never downloads a model by itself. |
| Service won't stay up | Read `$(brew --prefix)/var/log/naru-audio.log`. |
| `cannot listen on 127.0.0.1:7870` | Something else holds the port; use `--listen 127.0.0.1:PORT` (and `NARU_AUDIO_URL` for the clients). |
| `refusing to listen on non-loopback address … without --allow-remote` | Add `--allow-remote`, after reading [Network access](#configuration). |
| HTTP 403 `forbidden_origin` | Browsers are refused by design; call the daemon from a server-side process. |
| HTTP 503 `backend_unavailable` on an MLX model | Not Apple Silicon, or the sidecar is not set up: run `naru-audio mlx setup`, then `naru-audio mlx status`. |
| `` `uv` is not on PATH `` | `brew install uv`, then `naru-audio mlx setup` again. |
| Log line `mlx_env_stale` after an upgrade | The sidecar's dependencies changed: run `naru-audio mlx setup`. |
| `/health` reports running under Rosetta | You installed the x86_64 build on Apple Silicon; install the arm64 one. |
| HTTP 507 `insufficient_memory` | Loaded models are busy and the next one does not fit: raise `[memory] max_resident` or use a smaller model. |
| A model fails to load after files were changed | `naru-audio verify NAME`; if it fails, `naru-audio rm NAME` and pull it again. |

## Uninstall

```sh
brew services stop naru-audio
brew uninstall naru-audio
rm -rf ~/.naru-audio     # models, voices and config (or your $NARU_AUDIO_HOME)
```

The log is left at `$(brew --prefix)/var/log/naru-audio.log`; delete it too
if you like.

---

## Building from source

You need a stable Rust toolchain (the crate uses edition 2024). The first
build downloads sherpa-onnx's prebuilt static libraries from its GitHub
releases, so it needs network access.

```sh
git clone https://github.com/simonspoon/naru-audio
cd naru-audio
cargo build --release
./target/release/naru-audio --help
```

The MLX backend is compiled in only for `aarch64-apple-darwin`. No Python
is needed to build; the sidecar's files are embedded in the binary.

## Repository layout

```
src/
  main.rs            CLI (clap): serve and every subcommand
  server.rs          HTTP router, Host/Origin guards, health, models, registry routes
  server/            transcriptions, speech and voices, WebSocket streaming
  registry.rs        pull, verify, remove; registry/manifest.rs parses catalog TOML
  manager.rs         default models, config.toml, keep-alive, memory budget
  profile.rs         hardware profile (RAM, arch, Rosetta)
  backend.rs         which backends can run here; loads STT/TTS models
  stt/  tts/         sherpa-onnx and MLX engines, VAD, hotwords, leveller
  mlx.rs, mlx/       `mlx setup|status` and the sidecar process
  voices.rs          cloned voices
  log.rs, error.rs   log lines and rotation; the error envelope
catalog/             built-in model manifests (embedded at build time)
mlx/                 the Python sidecar (naru_audio_mlx), its pyproject and uv.lock
tests/               integration tests; tests/fixtures/ holds audio fixtures
examples/            tts_spike.rs, the benchmark behind docs/tts-spike.md
docs/                design.md (the spec), tts-spike.md and its corpus
Formula/             the Homebrew formula
.github/workflows/   ci.yml, release.yml
```

## Running tests

CI ([ci.yml](.github/workflows/ci.yml)) runs these (tests and clippy on macOS
and Ubuntu); run them before you open a pull request:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

The tests need no network. Tests that exercise real models load them from
`$NARU_AUDIO_TEST_HOME`, else `$NARU_AUDIO_HOME` or `~/.naru-audio`, and print
`skip:` when a model is not pulled. To make a missing model a failure instead, pull the models and
set `NARU_AUDIO_REQUIRE_MODELS=1`:

```sh
naru-audio pull default pocket-tts-int8
NARU_AUDIO_REQUIRE_MODELS=1 cargo test --all-targets
```

`tests/mlx.rs` runs only on Apple Silicon. It drives the real sidecar
protocol against a stdlib-only fake, so it needs `/usr/bin/python3` but not
MLX.

## The MLX sidecar

MLX models run in a Python process the daemon supervises, not in Rust:

- **Source:** [mlx/naru_audio_mlx/](mlx/naru_audio_mlx/) (`__main__.py`
  runs parakeet-mlx, mlx-whisper and mlx-audio; `protocol.py` defines the frames).
  [mlx/pyproject.toml](mlx/pyproject.toml) pins Python 3.12 and the
  libraries; [mlx/uv.lock](mlx/uv.lock) locks them.
- **Install:** these files are embedded in the binary. `naru-audio mlx
  setup` writes them to `~/.naru-audio/mlx/`, runs `uv sync --frozen`
  there, and records the venv's interpreter in `config.toml` `[mlx]
  python`. `naru-audio mlx status` checks that it imports.
- **Runtime:** the daemon starts `python -m naru_audio_mlx --socket …` on
  the first MLX request and talks length-prefixed JSON frames (plus raw
  f32 samples) over a Unix socket. On each `serve` start it refreshes the
  deployed `.py` files from the binary; a changed `pyproject.toml` or
  `uv.lock` needs `naru-audio mlx setup` again.
- **Changing dependencies:** edit `mlx/pyproject.toml`, run `uv lock` in
  `mlx/`, commit both, rebuild, then `naru-audio mlx setup`.
- **Python tests** use the installed venv, from `mlx/`:

  ```sh
  cd mlx
  ~/.naru-audio/mlx/.venv/bin/python -m unittest tests.test_synth tests.test_transcribe_whisper
  ```

To add an MLX model, add a catalog entry with `backend = "mlx"`; see the
`qwen3-tts-*-mlx.toml` files and [docs/design.md §5.3](docs/design.md).

## Release process

Releases are cut by pushing a `v*` tag ([release.yml](.github/workflows/release.yml)):

1. Bump `version` in `Cargo.toml` (it is what `naru-audio --version` and
   `/health` report) and commit.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The workflow builds `naru-audio-darwin-arm64`, `-darwin-amd64`,
   `-linux-arm64` and `-linux-amd64`, creates a GitHub release with those
   binaries and `checksums.txt`, and asks the `simonspoon/homebrew-tap`
   repository to update its `naru-audio` formula (this needs the
   `HOMEBREW_TAP_TOKEN` secret).

[Formula/naru-audio.rb](Formula/naru-audio.rb) is the formula's template:
its `sha256` values are placeholders, filled in by the tap from the release.

## Contributing

Issues and pull requests are welcome. There is no separate contributing
guide yet; these are the conventions the code follows:

- **Design first.** [docs/design.md](docs/design.md) is the spec, and code
  comments cite its sections (`§2.3`). A change to behaviour updates the
  design doc in the same pull request.
- **Tests with the change.** Add or update a test under `tests/` (or a unit
  test beside the code); tests must not need the network. Run the three CI
  commands under [Running tests](#running-tests).
- **Models are pinned.** A new catalog entry gives every file a URL at a
  fixed revision, its size and its sha256, and states the model's license.
- **Commit messages** use a type prefix: `feat:`, `fix:`, `docs:`.
- Never log audio or transcript text.

## License

naru-audio is licensed under GPL-3.0-or-later, as declared in
[Cargo.toml](Cargo.toml) and the Homebrew formula. The models it downloads
have their own licenses, listed in each [catalog](catalog/) file.
