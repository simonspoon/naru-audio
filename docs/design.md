# naru-audio — design

Status: design, task 1356. Date: 2026-09-24. Nothing here is built yet.

naru-audio is one local daemon that does both speech-to-text (STT) and
text-to-speech (TTS). It works like Ollama: a long-running process, models
pulled by name, loaded on demand and unloaded when idle, and an HTTP API
that other apps can call. It replaces **auris** and **kokoro-rs**.

Paths below are relative to the repo named in each section. `naru/` means
the naru repo. Claims we could not
verify are marked **[assumption]**, and all of them are collected in §7.

## 1. Inventory

### 1.1 auris (Rust 2024, GPL-3.0)

| Area | What exists today |
|---|---|
| Engine | `sherpa-onnx =1.13.6` with static ONNX Runtime, CPU only. `OfflineRecognizer` (auris/src/engine.rs:215) runs `modified_beam_search` (engine.rs:209) with `nemo_transducer`/`bpe`. |
| Model | `parakeet-tdt-0.6b-v2-int8` from HF `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` pinned at commit `1ab9323565ddb038682214b292f588070a538ce2` (model.rs:36). Four files with pinned size and sha256 (model.rs:46–67), about 661 MB. `bpe.vocab` is generated from `tokens.txt`. |
| VAD | Silero `silero_vad.onnx` from `csukuangfj/vad`, sha256-pinned (model.rs:82, :89). |
| Storage | `$AURIS_HOME` (default `~/.cache/auris`) holds `models/<name>/`, `silero_vad.onnx` and `auris.sock`. Downloads go to `.part`, are size- and sha-checked, then renamed (model.rs:8, :357). |
| Model choice | `-m NAME\|PATH`, then `AURIS_MODEL`, then the built-in default. |
| CLI | `auris [PATH]` with `-m`, `--format text\|json`, `--vocabulary-file` (`term :boost` lines), `--no-download`, `--vad-threshold 0.2`, `--vad-min-speech 0.25`, `--vad-min-silence 0.5`, `--no-vad`, `--no-daemon`, `--socket`, `-q`. Subcommands `serve`, `status`, `stop`. `--list-models` prints bare names one per line, works offline, and exits 0. |
| Exit codes | 0 ok, 1 nothing transcribed, 2 usage error, 130 interrupted. |
| Input | WAV only, any rate from 1 kHz to 384 kHz (audio.rs:107–108), resampled to 16 kHz mono. Caps are 256 MiB (audio.rs:27) and 10 minutes (audio.rs:33). |
| Output | JSON Lines: `{"type":"speech","active","at"}`, `{"type":"segment","index","text","start","end"}`, `{"type":"transcript","text"}`. There are no partials; the decision is recorded in auris/docs/streaming.md. |
| Guards | An energy gate `is_silent` (audio.rs:76) and a Silero gate run before decoding, because Parakeet produces "Okay." on silence. A speech span found by the VAD means the **original, untrimmed** audio is decoded (docs/streaming.md). `looks_manufactured` (vocabulary.rs:240) triggers an unbiased re-decode when hotwords look invented. The global hotword score is 3.0; a term's own `:boost` overrides it (vocabulary.rs:35). |
| Daemon | Unix socket, one request per connection, ops `transcribe\|status\|stop` carrying raw f32le samples. It auto-starts on a client call and exits after 300 s idle (daemon.rs:56). Connections are handled one at a time, and it refuses a request for a different model. `status` blocks behind a running decode. |
| Performance | Warm RSS about 1.5 GB, load about 4 s. On the i9-9880H CPU, RTF is about 0.06 for decode alone (warm) and 0.10 end to end. |
| Designed, not built | Post-ASR correction pass (docs/correction.md; name F1 goes from 69% to 89%). |
| Install | `brew install simonspoon/tap/auris`, a prebuilt binary for darwin and linux, arm64 and amd64 (homebrew-tap/Formula/auris.rb). No `service do` block. |

### 1.2 kokoro-rs (Rust 2024, v0.1.2)

| Area | What exists today |
|---|---|
| Engine | `ort 2.0.0-rc.13` with `load-dynamic`. The ORT 1.23.2 dylib is downloaded at runtime (kokoro-rust/src/models.rs:27); `ORT_DYLIB_PATH` overrides it. `espeak-rs 0.2` builds espeak-ng from source, which needs cmake, clang and libclang (Cargo.toml:17–23). Also cpal and hound. |
| License | GPL-3.0-or-later (kokoro-rust/Cargo.toml:6). |
| Model | Kokoro v1.0 82M. `kokoro-v1.0.onnx` is about 326 MB and `voices-v1.0.bin` is an npz with 54 voices, each `(510,1,256)` f32. Both come from the thewh1teagle/kokoro-onnx release `model-files-v1.0` (models.rs:19). **No checksum.** |
| Storage | `$KOKORO_HOME` (`~/.cache/kokoro-rs`). `KOKORO_MODEL` and `KOKORO_VOICES` override the paths. |
| CLI | `kokoro-rs [TEXT]` (text from args or stdin), `-o FILE\|-`, `-v` (default `af_heart`), `-s 0.5–2.0`, `-l en-us`, `-d device`, `--list-voices` (loads the whole ONNX session, which is a known mistake), `--list-devices`, `--chunk-chars`, `--chunk-phonemes 500` (max 510), `--gap 0.12`, `--no-download`, `-q`. |
| Output | 24 kHz mono PCM16 WAV. When streaming to stdout the RIFF sizes are `0xFFFFFFFF` (audio.rs:384). |
| Latency | The first chunk is 100 phonemes and later chunks double up to 500 (text.rs:42–48), budgeted with espeak phonemes (punctuation.rs:251). Kokoro's silence padding is trimmed from each chunk (trim.rs). Playback is fed by a 4-deep bounded channel. A loudness `Leveller` runs (level.rs:49), and gaps between sentences are reinserted. |
| Process model | No daemon. Every call spawns a process and loads the model. |
| Install | `brew install simonspoon/tap/kokoro-rs`. |

### 1.3 How Naru calls them

Both tools are child processes of `naru serve`, which listens on port 7770
by default (naru/src/cli.rs:158). There is no HTTP between Naru and either tool.

| Naru code | Behaviour |
|---|---|
| src/core/listen.rs:34 `auris_bin()` | Reads `NARU_AURIS_BIN`/`MESA_AURIS_BIN` through `core::env::var("AURIS_BIN")`, and otherwise runs `auris` from **PATH**. |
| listen.rs:68 `models()` | Runs `auris --no-download --list-models` once and caches the result in a `OnceLock` **for the life of the process**. |
| listen.rs:256 `transcribe()` | Runs `auris -q --format json [-m model]` with the WAV on stdin and keeps the last `transcript` line. It passes no `--vocabulary-file`: auris implements it (auris/src/cli.rs:575–579), but the comment at listen.rs:257–259 still says it does not. When auris exits 1 (nothing heard), Naru returns an error carrying auris's stderr (listen.rs:298–314), which becomes a 503. |
| src/core/speech.rs:167 `kokoro_bin()` | Reads `NARU_KOKORO_BIN`, otherwise uses PATH. |
| speech.rs:195 `voices()` | Runs `--list-voices` and caches the result in a `OnceLock`. |
| speech.rs:258 `start()` | Runs `kokoro-rs -q -o - [-v voice]` and streams 16 KiB chunks. |
| speech.rs:464 `fix_wav_sizes` | Patches the sizes to `0x7fff0000` (`STREAM_DATA_LEN`, speech.rs:162) because Safari rejects `0xFFFFFFFF`. |
| src/api.rs:4822 `transcribe_live` | `POST /api/live/transcribe` takes `{audio_base64}` and returns `{text}`. The cap is 25 MiB (`LIVE_AUDIO_MAX`, src/core/store.rs:1506). A failure returns 503. |
| api.rs:4907 `transcribe_available` | `GET /api/live/transcribe` returns `{available}`. |
| api.rs (speak routes) | `GET /api/live/turns/{id}/speak`, `GET /api/inbox/{id}/speak` and `GET /api/config/speech/preview` stream `audio/wav`. |
| config | `~/.naru/config.json` keys `speech.voice` (`af_heart`) and `listen.model` (`parakeet-tdt-0.6b-v2-int8`), validated against the binaries' lists (src/core/config.rs:2194–2208, 2377–2392). An empty list means any well-formed name is accepted. |
| frontend (silence) | `isSilentTranscribe` (liveRecognition.ts:443) matches "no speech"/"nothing transcribed" in that 503's text, and LiveHub.tsx:1769 hides the error when it matches. Silence is detected by string-matching another binary's stderr. |
| frontend | `components/LiveHub.tsx` captures through an AudioWorklet at 16 kHz. `liveVad.ts:40–44` is an RMS VAD: onset 0.02, release 0.012, hangover 700 ms, max segment 20 s. Each utterance goes through `liveAudio.ts:160 wavFromFrames` and is sent as one POST. Barge-in runs a second capture. TTS plays in `<audio>`, falling back to `speechStream.ts`/`wavStream.ts` (2 s prebuffer). |
| naru-ios | Uses the same routes and refuses to fall back: `MesaKit/TranscriptionQueue.swift:147–148` shows "Speech isn't available: auris isn't installed on the mesa machine." |

Dictation latency today is roughly hangover (700 ms) + WAV/base64 upload + a
cold-or-warm decode of the whole utterance.

### 1.4 The flat-meter incident (session 134)

1. After the rename to naru, the Naru service on one machine could not find
   `auris` on its PATH. The likely cause is that launchd/Homebrew gave the
   service a shorter PATH **[assumption: the service's PATH was not captured]**.
2. `GET /api/live/transcribe` returned `available:false`, and that answer stayed
   in the `OnceLock` until the process restarted.
3. `listenPath()` (frontend/src/liveRecognition.ts:144–161, the fallback is at
   line 159) silently chose the browser's Web Speech API.
4. The level meter renders only on the auris path (LiveHub.tsx:3136,
   `path === 'auris' && recognizes`), so it sat flat. The only clue was the hint
   "Listening through this browser".

Design lessons, used in §4:
- **L1**: a probe is never cached for the life of the process.
- **L2**: the daemon is found by URL, never by a PATH lookup.
- **L3**: an unavailable engine is a visible error, never a silent switch to
  another engine.

### 1.5 Capabilities naru-audio must keep

- [ ] Parakeet TDT 0.6B v2 int8 via sherpa-onnx, CPU, on Intel x86_64 and arm64.
- [ ] `modified_beam_search` decoding with hotwords (global score 3.0, per-term `:boost` override), including the `looks_manufactured` guard.
- [ ] Energy gate plus Silero gate before decoding; decode the untrimmed audio.
- [ ] VAD tunables: threshold 0.2, min speech 0.25 s, min silence 0.5 s, and a way to turn VAD off.
- [ ] Segment output with `index/start/end/text`, a final transcript, and `speech active` events.
- [ ] "Nothing transcribed" reported as its own outcome, separate from an error.
- [ ] Input caps: 256 MiB and 10 minutes. Any rate from 1 kHz to 384 kHz resampled to 16 kHz mono.
- [ ] sha256-pinned downloads written to `.part`, then renamed; work offline once pulled (`--no-download`).
- [ ] A cheap, offline listing of model names; a bare-name listing for scripts.
- [ ] Warm process (no 4 s load per request) with an idle unload.
- [ ] Kokoro v1.0 with all 54 voices by name, default `af_heart`, speed 0.5–2.0, English (today `-l en-us`).
- [ ] 24 kHz mono PCM16 output, streamed so that playback starts after the first chunk.
- [ ] Growing chunk sizes (today 100 → 500 phonemes; a character-based equivalent here, §5.1), silence trim of each chunk (trim.rs), sentence gaps (0.12 s), loudness leveller.
- [ ] Safari-safe streaming WAV headers (`0x7fff0000`).
- [ ] A voice listing that does not load the model.
- [ ] Prebuilt binaries through `simonspoon/tap`.
- [ ] Env overrides for home and model paths, for tests and stubs.

## 2. API

### 2.1 Bind and port

- Default is `127.0.0.1:7870`. Override with `--listen ADDR:PORT` or `NARU_AUDIO_LISTEN`.
- Why 7870: it is not Naru's 7770, and it stays clear of the 77xx ports
  Naru's tests and dev servers already use (7771, 7777, 7781, 7790, 7794 all
  appear in naru and naru-ios). It is unassigned in `/etc/services` and avoids
  the well-known 7860 (Gradio), 7880 (LiveKit) and 11434 (Ollama).
- Non-loopback binds need `--listen 0.0.0.0:7870 --allow-remote`. Without the
  flag the daemon refuses to start with a non-loopback address.
- DNS-rebinding and drive-by guards: by default `Host` must be `127.0.0.1:<port>`,
  `localhost:<port>` or `[::1]:<port>`. `--allow-remote` **skips the Host
  check**, since LAN clients send the machine's LAN name or address; this is
  exactly what Naru's `--lan` does (naru/src/cli.rs:160–161). A request carrying
  an `Origin` header (a browser fetch or WebSocket) is allowed only when the
  origin's authority is exactly the `Host` it arrived on — same-origin, case-
  insensitively — and that `Host` is itself allowed; with `--allow-remote` the
  Host check is skipped but the Origin/Host match is still required. Anything
  else, including a cross-site `Origin`, is 403 `forbidden_origin`. This is
  what lets `/admin` (§3) — served same-origin by this daemon — call its own
  API from the browser, while a page on another site still cannot.

Two namespaces: `/v1/*` is OpenAI-compatible; `/api/*` is native and
Ollama-shaped (registry, load state).

### 2.2 `POST /v1/audio/transcriptions`

Request is `multipart/form-data`.

| OpenAI field | Support |
|---|---|
| `file` | **Supported.** WAV (PCM 8/16/24/32-bit int or 32-bit float, any rate from 1 kHz to 384 kHz, any channel count, downmixed and resampled). Anything else returns 415 `unsupported_media_type`. Caps: 256 MiB and 10 minutes, otherwise 413. |
| `model` | **Supported.** Accepts a registry name, or `default` for the machine's default STT model. `whisper-1`, `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` are aliases for `default`, so stock OpenAI clients work. Unknown names return 404 `model_not_found`. |
| `response_format` | **Supported:** `json` (default), `text`, `verbose_json`. `srt`/`vtt` return 400 `unsupported_value`. |
| `language` | **Validated.** Absent or `en` is accepted. Anything the model's manifest does not list returns 400 (Parakeet v2 is English-only). |
| `timestamp_granularities[]` | `segment` is **supported** with `verbose_json`. `word` returns 400 `unsupported_value`. |
| `stream` | **Supported.** `true` returns SSE: one `transcript.text.delta` per VAD segment as it is decoded, then `transcript.text.done` carrying the full text. |
| `prompt` | **Ignored.** Hotwords are not free text; use `hotwords`. |
| `temperature` | **Ignored.** Beam search is deterministic. |
| `chunking_strategy`, `include[]` | **Ignored.** |

Extensions (extra form fields; OpenAI SDKs pass them through `extra_body`):
- `hotwords`: one string of newline-separated `term[ :boost]` lines, the auris `--vocabulary-file` syntax. A line without `:boost` uses the global 3.0. The WebSocket (§2.4) uses the same string.
- `vad`: `true` (default) or `false`.
- `vad_threshold`, `vad_min_speech`, `vad_min_silence`: seconds and probability, the same as the auris flags.
- `keep_alive`: see §3.4.

Responses:
```json
// json
{"text": "Add a task to the naru board."}
// verbose_json
{"task":"transcribe","language":"en","duration":3.42,"text":"…",
 "segments":[{"id":0,"start":0.31,"end":3.10,"text":"…"}],
 "x_model":"parakeet-tdt-0.6b-v2-int8","x_decode_ms":212}
```
Nothing transcribed (silence, or every gate rejected the audio) returns
**200 with `"text":""`** and `segments: []`, as OpenAI does. This takes the
place of auris exit code 1.

**Silence contract, end to end.** An empty transcript is a success everywhere.
The daemon returns 200 `{"text":""}`. Naru's `POST /api/live/transcribe`
returns 200 `{"text":""}` rather than a 503 (for the legacy engine too, by
mapping auris exit 1 to `Ok("")`). LiveHub drops empty text without saying
anything. `isSilentTranscribe` is deleted, so nothing matches on stderr text
any more (tasks 17 and 18).

### 2.3 `POST /v1/audio/speech`

The request body is JSON.

| OpenAI field | Support |
|---|---|
| `model` | **Supported.** Accepts a registry name or `default`. `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` are aliases for `default`. |
| `input` | **Supported.** Must be non-empty. The cap is 16 384 chars (OpenAI's is 4 096; we raise it because Naru reads whole turns). Over the cap returns 413. |
| `voice` | **Supported.** A voice id from `/v1/audio/voices` (for example `af_heart`). OpenAI names (`alloy`, …) return 400 `unknown_voice` with the valid list in `message`. We do not map them, because mapping would silently pick a voice nobody chose. |
| `response_format` | **Supported:** `wav` (default here; OpenAI's default is mp3) and `pcm`. `mp3/opus/aac/flac` return 400 `unsupported_value`. See the open question on mp3. |
| `speed` | **Supported** in the range 0.5–2.0. Outside that range returns 400. Qwen3-TTS ignores it silently; Chatterbox, VoxCPM2, IndexTTS, OmniVoice and Breeze do not support it at all, so a non-1.0 `speed` on any of them fails the synthesis (500 `internal_error`) instead of coming out at normal speed unannounced. |
| `stream_format` | `audio` is **supported**. `sse` returns 400 `unsupported_value`. |
| `instructions` | **Supported** by a model whose manifest sets `instruct = true` (§3.2), passed to mlx-audio as `instruct`; **ignored** by every other model (Kokoro cannot be steered by a prompt). For Qwen3-TTS 1.7B VoiceDesign, which has no preset voices, it is the voice's description plus any style direction in one string (for example "A warm, husky woman in her thirties, a little amused. Speak slowly."); `voice` is accepted and ignored, and a request without `instructions` returns 400 before any load. VoxCPM2, OmniVoice and Breeze also clone (`clone = true` too): a named cloned voice wins over `instructions` if both are given, and only a request with no matching voice needs `instructions` (`voice` in `src/server/speech.rs`). |

Extensions: `gap` (default 0.12 s), `level` (default `true`, the leveller),
`exaggeration` (0–1, **supported** only by a model whose manifest sets
`exaggeration = true`, passed to mlx-audio as `exaggeration`; Chatterbox's
emotion-exaggeration dial. **Ignored** by every other model), `keep_alive`,
`knobs` (§2.7, naru task 1458: every other manifest-declared generation
knob, by name).
There is **no per-request `lang`**. sherpa fixes `lang` in
`OfflineTtsKokoroModelConfig` when the model is created (sherpa-onnx
tts.rs:166), with values such as `en`/`es` rather than `en-us`. Language
therefore belongs to the manifest, with one catalog name per language
(`kokoro-v1.0` is English; a Spanish entry would be `kokoro-v1.0-es`).

**Streaming output.** The response is always `Transfer-Encoding: chunked`,
and the first bytes go out as soon as the first text chunk is synthesised.
Text is split with a character-based version of kokoro-rs's schedule (§5.1):
the first chunk is ≤100 chars at a clause or sentence boundary, and later
chunks double up to 400 chars. Each chunk's leading and trailing silence is
trimmed before it is sent.

- `pcm`: raw s16le, 24 kHz, mono, no header (matches OpenAI's `pcm`), plus
  `X-Audio-Sample-Rate: 24000`, `X-Audio-Channels: 1`, `X-Audio-Encoding: s16le`.
- `wav`: a 44-byte header followed by PCM. **Fix for the RIFF size problem:**
  the length is unknown at header time, so the daemon writes
  `data` size = `0x7FFF0000` and RIFF size = `0x7FFF0000 + 36`. These are the
  same values Naru's `fix_wav_sizes` writes today (speech.rs:162, :464), and
  browsers accept them when the stream is read with `fetch()` and played
  through Web Audio, which stops at the real end of the stream. `<audio src>`
  on the live chunked stream does not work in Safari (it sends Range requests
  and seeks to the declared end), so `<audio>` playback must use `stream=false`.
  `0xFFFFFFFF` is never emitted.
  - When `stream=false` is passed (an extension), the daemon buffers everything
    and writes exact sizes plus `Content-Length`. Use this for file export.
- An error **after** the first byte cannot change the status code. The
  daemon aborts the chunked body without the terminating zero-length chunk,
  so the client sees a truncated transfer rather than a clean end, and it
  logs `tts_stream_aborted` with the cause.

### 2.4 Streaming STT: `GET /v1/audio/transcriptions/stream` (WebSocket)

**Transport: WebSocket.**
- It is full-duplex: audio goes up while results come down on the same connection.
- Binary frames carry PCM without base64.
- Naru's axum server can proxy it (§6.2).
- SSE is one-way, and chunked POST upload is not usable from browsers.

The protocol is our own and is **not** OpenAI Realtime-compatible. Realtime
needs base64 audio inside JSON, 24 kHz defaults, and a session/response
model we do not need.

**Honesty about the model.** Parakeet TDT is an offline model: it decodes
a closed buffer and has no incremental state. The low latency comes from
structure, not from the model:
1. **Server-side VAD segmentation.** Silero, through sherpa-onnx
   `VoiceActivityDetector`, runs on the stream as it arrives. When an
   utterance closes (`min_silence` = 0.5 s of silence), its audio, padded by
   300 ms on each side from the ring buffer, is decoded right away. Every
   utterance except the last is decoded while the person is still talking,
   so the critical path is `min_silence + decode(last segment)`. On the i9,
   a 5 s tail is 500 ms + ~300 ms (RTF 0.06), which is about 0.8 s after the
   person stops talking. Today it is 700 ms + upload + a decode of the whole
   recording.
2. **Speech-activity events** in place of partial text. auris/docs/streaming.md
   shows that what Naru's timers actually use from interim results is a
   "still talking" heartbeat. `speech` events give that heartbeat at zero
   decode cost.
3. **Optional partials** (`"partials": true`, off by default). While a
   segment is open, the daemon re-decodes the growing window every 700 ms,
   but only if the previous partial decode has finished and the window is
   ≤ 8 s. Trade-offs:
   - The cost is quadratic in segment length. At 8 s on the i9 that is about
     0.5 s of CPU per tick, which competes with TTS on the same cores.
   - Partials can revise any word, not just the last one.
   - The final result always replaces them.
   - On x86_64 the daemon sends a `warning` event when partials are requested.

**Handshake.**
1. The client connects to `ws://127.0.0.1:7870/v1/audio/transcriptions/stream`.
2. The first client frame must be text:
```json
{"type":"start","model":"default","format":"s16le","sample_rate":16000,
 "vad":{"threshold":0.2,"min_speech":0.25,"min_silence":0.5,"max_segment":20.0,"pad":0.3},
 "partials":false,"hotwords":"naru :4.0\nkhora","keep_alive":"5m"}
```
   - `format` is `s16le` (preferred, half the bytes) or `f32le`.
   - `sample_rate` must be 16000. Any other value returns an error; the client resamples. LiveHub already captures at 16 kHz.
   - Mono only.
3. The server answers `{"type":"ready","session":"a1b2","model":"parakeet-tdt-0.6b-v2-int8","load_ms":0}`.
   If the model had to load, `load_ms` is non-zero, and the server may send
   `{"type":"loading"}` first.

**Audio framing:** binary frames of raw samples only, up to 64 KiB each;
20–100 ms (640–3 200 bytes of s16le) is recommended. A frame that is not a
whole number of samples is an error. Frames carry no timestamp: time is the
sample count since `start`, in seconds.

**Client control (text frames):** `{"type":"flush"}` closes the current
segment now (push-to-talk release, the "send" button). `{"type":"stop"}`
flushes, delivers every final, sends `done` and closes.
`{"type":"hotwords","hotwords":"…"}` (the same string format) applies from the next segment on.

**Server events (text frames).**
```json
{"type":"speech","active":true,"at":1.28}
{"type":"partial","segment":3,"start":12.40,"text":"add a task to"}
{"type":"final","segment":3,"start":12.40,"end":15.02,"text":"Add a task to the board.","decode_ms":188}
{"type":"warning","code":"partials_expensive","message":"…"}
{"type":"error","code":"model_not_pulled","message":"…"}
{"type":"done"}
```
- `speech` uses the same shape as auris's JSONL line.
- A `final` with an empty `text` is never sent, and a gated segment is simply dropped, unless partials were sent for that segment: then an empty `final` closes them.
- `segment` indexes increase monotonically. `final`s arrive in segment order.

**VAD placement.** VAD runs **in the daemon** and is authoritative. Clients
send continuously while the microphone is open. A client RMS gate
(liveVad.ts) remains allowed, but only to decide what to send, never to
decide segment boundaries. `max_segment` (20 s, the same as liveVad) forces a
cut at the lowest-energy point in the last 1 s. The energy gate and the Silero
gate from auris run on every segment before it is decoded.

**Backpressure:** each session queues at most 30 s of undecoded audio;
overflow sends `error` `backlog` and closes with 1013. All sessions share one
decode FIFO per loaded model. Health and listing never wait on it (§2.5).

**Close codes:** 1000 normal, 1008 protocol violation, 1011 internal,
1013 overloaded or backlog. An `error` event always precedes any non-1000 close.

### 2.5 Health, models, voices, registry

| Route | Response |
|---|---|
| `GET /health` | `200 {"status":"ok","version":"0.1.0","api":1,"pid":123,"uptime_s":4410,"stt":{"default":"parakeet-tdt-0.6b-v2-int8","ready":true,"problem":null},"tts":{"default":"kokoro-v1.0","ready":false,"problem":{"code":"model_not_pulled","message":"run `naru-audio pull kokoro-v1.0`"}},"backends":[{"name":"sherpa-onnx","available":true},{"name":"mlx","available":false,"reason":"requires Apple Silicon"}]}`. It is served from in-memory state and **never blocks behind a decode**, which fixes auris's `status` problem. `ready` means pulled and loadable, not necessarily loaded. `stt`/`tts` `default` reads the live value `PUT /api/defaults` sets (or the env override, naru task 1458), not just what the daemon started with. |
| `GET /v1/models` | OpenAI list: `{"object":"list","data":[{"id","object":"model","created","owned_by":"naru-audio","x_kind":"stt\|tts\|vad","x_backend","x_available","x_unavailable_reason","x_pulled","x_loaded","x_loaded_at","x_size_bytes","x_default","x_license","x_license_url","x_non_commercial","x_clone","x_clone_requires_transcript","x_instruct","x_design","x_design_voice_model","x_languages","x_source","x_voice_count","x_cloned_voice_count","x_prompt_format"}]}`. It lists pulled models plus catalog models that can run on this machine. `?pulled=true` filters the list. `x_clone`/`x_clone_requires_transcript`/`x_instruct` are `Manifest::clones`/`clone_requires_transcript`/`instructs` (§3.2, §5.3): whether the model speaks in a cloned reference recording, whether that clone needs the transcript alongside it, and whether it takes `instructions`. `x_design` is `instructs()` again, named for "can design a voice from a description"; `x_design_voice_model` is the model a voice designed with it becomes usable with — itself for a model that both clones and instructs (VoxCPM2), else `backend.<backend>.design_voice_model` (Qwen3-TTS VoiceDesign saves to its base model), else `null`. `x_languages` is `[model] languages`; `x_source` is `[model] source` or the Hugging Face `owner/repo` derived from the first file URL, else `null`. `x_voice_count`/`x_cloned_voice_count` count `voices::of`'s result and the cloned subset of it. `x_loaded_at` is when the model finished loading (RFC3339), `null` while not loaded. `x_prompt_format` is `Manifest::prompt_format` (§3.2): `null` for a non-TTS model, else `{"style"?,"inline"?,"knobs":[{"name","default","min","max","step"?}],"hint"?}` documenting how the model reads `instructions` and what else it takes (§2.7). All meaningless fields are still sent, always `false`/`null`, for STT/VAD. |
| `GET /v1/audio/voices?model=kokoro-v1.0` | `{"model":"kokoro-v1.0","voices":[{"id":"af_heart","accent":"us","gender":"f","default":true,"cloned":false,"origin":"builtin","description":null,"duration":null,"has_transcript":false},…]}`. It is read **from the manifest** and never loads the model, which fixes the kokoro-rs `--list-voices` mistake. `cloned` is true for a cloned voice (§5.3) from `$NARU_AUDIO_HOME/voices/<name>/` **made for this model**, false for the manifest's own; a clone made for a different model is not listed here at all. `origin` is `"builtin"`, `"designed"` (a cloned voice with a `design.txt`) or `"cloned"` (naru task 1458); `description` is `design.txt`, trimmed, or `null`; `duration` is the clip's length in seconds from `ref.wav` (`hound`), `null` for a built-in voice; `has_transcript` is whether `ref.txt` holds anything after trimming. `?model=clones` is not a real model: it lists every cloned voice regardless of the model it was made for, as every listing did before per-model voices, and each entry also carries its own `model` alongside the same `origin`/`description`/`duration`/`has_transcript`. |
| `GET /v1/audio/voices/{name}` | Exports a cloned voice: `200 {"name":"amy","text":"Hello there.","model":"qwen3-tts-0.6b-base-mlx","description":null,"wav_base64":"UklGR…"}`, where `text` is its `ref.txt`, `model` is its `model.txt` (`CLONE_MODEL` for a voice that predates per-model voices), `description` is its `design.txt` if it has one (naru task 1458), and `wav_base64` is its `ref.wav`, byte for byte, in standard base64. Errors: 400 `invalid_request` for a bad `name` (as `POST`); 404 `voice_not_found` if there is no cloned voice of that name (a built-in voice is not one). |
| `POST /v1/audio/voices` | Adds a cloned or designed voice (§5.3), the same code path as `naru-audio voice add`. `multipart/form-data` fields: `name` (the voice id: no path separators, no leading `.`), `file` (the clip, WAV or MP3 or anything else macOS `afconvert` reads; 3–30 s accepted, 5–15 s best), `text` (exactly what the clip says), `model` (which model it is for, a TTS model whose manifest sets `clone = true`; defaults to `CLONE_MODEL` if not given, same as every voice before per-model voices) and `description` (optional; written as `design.txt` when non-blank, naru task 1458 — its mere presence, not its content, is what makes the voice `"designed"` rather than `"cloned"`). The clip is converted to 24 kHz mono and written to `$NARU_AUDIO_HOME/voices/<name>/` as `ref.wav`, `ref.txt`, `model.txt` and, if given, `design.txt`, all or nothing. Returns `201 {"id":"amy","accent":null,"gender":null,"default":false,"cloned":true,"origin":"cloned","description":null,"model":"qwen3-tts-0.6b-base-mlx","duration":6.2}`: the entry `GET /v1/audio/voices?model=` lists for that model, plus the clip's length in seconds. Recreating an exported voice (`GET /v1/audio/voices/{name}` above) elsewhere is the same request with its `model` carried over. Errors: 400 `invalid_request` for a missing field, a bad `name`, an empty `text` or a clip outside 3–30 s (`param` is the field); 400 `model_does_not_clone` if `model` is a TTS model that never speaks in cloned voices; 404 `model_not_found` if `model` is not in the catalog; 409 `voice_exists` if the name is taken (an existing voice is never replaced); 415 `unsupported_media_type` if `afconvert` cannot read the clip; 413 over the 256 MiB body cap. |
| `PATCH /v1/audio/voices/{name}` | `{"name"?,"text"?,"description"?}` (naru task 1458) renames a cloned voice's directory and/or overwrites its `ref.txt`/`design.txt`, each only where given. A configured default voice (`PUT /api/defaults`) follows the rename; `DELETE` below clears it instead. Returns `200` with the voice as `GET /v1/audio/voices` lists it. Errors: 400 `invalid_request` for a bad new `name`; 404 `voice_not_found` for an unknown cloned voice; 409 `voice_exists` if the new name is taken; 409 `builtin_voice` for a built-in name, as `DELETE` also refuses. |
| `GET /api/voices/{model}/{voice}/sample` | `audio/wav` (naru task 1458): a cloned or designed voice's own `ref.wav`, byte for byte, or a built-in voice's cached preview at `$NARU_AUDIO_HOME/state/previews/<model>/<voice>.wav`. Read-only: it never loads a model or synthesises, even asked to with the old `?generate=true` (naru task 1458 ST2 review — a cross-site `<audio src>` sends no `Origin` header, so the Host/Origin guard, §2.1, would otherwise let a `GET` generate straight through). `DELETE /api/models/{name}` removes a model's whole preview cache. Errors: 400 `invalid_request` for a bad `model`/`voice`; 404 `voice_not_found` for a voice that is none of the model's own or cloned voices; 404 `preview_not_cached` for a built-in voice with nothing cached — `POST` the same path to generate one. |
| `POST /api/voices/{model}/{voice}/sample` | The same cached preview `GET` serves if there is one, else loads the model, synthesises a fixed line, caches it under `state/previews/` and answers with that — a built-in voice is never generated by `GET`, only `POST`, which the Host/Origin guard actually covers. A cloned voice has nothing to generate, so it answers its own `ref.wav`, same as `GET`. Same errors as `GET`, plus whatever loading the model can return (409 `model_not_pulled`, 503 `backend_unavailable`, …). |
| `POST /api/voices/preview` | `multipart/form-data` fields `model` (a TTS model that clones), `file` (the reference clip), `text` (its transcript), `input` (what to say) and `knobs` (optional, a JSON object `{"<name>":<number>}`, §2.7) synthesise a one-off `audio/wav` clip, not streamed, in that voice without saving anything under `voices/` (naru task 1458): the clip is converted the same way `POST /v1/audio/voices` converts an upload and always removed once synthesis ends. Errors: 400 `invalid_request` for a missing field, a clip outside 3–30 s, or `knobs` not a JSON object of finite numbers; 400 `unsupported_value` for a `knobs` name the model does not declare; 400 `model_does_not_clone`; 404 `model_not_found`; 415 `unsupported_media_type` if `afconvert` cannot read the clip. |
| `POST /api/pull` | `{"model":"kokoro-v1.0"}` returns an NDJSON stream of `{"status":"downloading","file","completed","total","model_completed","model_total"}` lines, then `{"status":"verifying",…}`, then `{"status":"success"}`. `model_completed`/`model_total` count through the plan (`name` plus its `requires`). It is idempotent; a second pull of the same model while one is already streaming is 409 `pull_in_progress` (naru task 1458) — see `GET`/`DELETE /api/pulls` below. |
| `GET /api/pulls` | Pulls still streaming: `[{"model","file","completed","total","model_completed","model_total","status":"running","started_at"}]` (naru task 1458). |
| `DELETE /api/pulls/{name}` | Cancels a running pull: 202, and its `POST /api/pull` stream ends with `{"error":{"code":"pull_cancelled"}}` instead of `success`; no staging directory is left under `tmp/`. 404 `pull_not_found` if no pull of `name` is running (naru task 1458). |
| `DELETE /api/models/{name}` | 204. Returns 409 `model_in_use` while the model is loaded and busy; an idle loaded model is unloaded first. |
| `GET /api/ps` | Loaded models: `[{"name","kind","backend","resident_bytes","expires_at","busy","loading","loaded_at"}]`. `loaded_at` (RFC3339) is `null` while `loading`. |
| `POST /api/load` | `{"model":"default"\|name,"kind":"stt","keep_alive":"10m"}` warms a model (`kind` is required only with `default`; for a named model, `kind` is otherwise derived from its own manifest, not assumed — naru task 1458 — and a `kind` that disagrees with it is 400 `kind_mismatch`). `keep_alive:0` unloads it. |
| `GET`/`PUT /api/defaults` | `GET` (and `PUT`'s response) return `{"stt","tts","voices":{"<tts model>":"<voice>"},"env_override":{"stt":bool,"tts":bool}}`: the live default STT/TTS model and each TTS model's default voice. `PUT` body: `{"stt"?,"tts"?,"voices"?:{"<tts model>":"<voice>"\|null}}`; `stt`/`tts` must each be a pulled model of that kind (the usual 404/409/400), and each `voices` key a pulled TTS model whose value is one of its voices, else 404 `voice_not_found`. Applied live at once (`/health`, `/v1/models`' `x_default`, `GET /v1/audio/voices`' `default`, and `POST /v1/audio/speech`'s default voice all read it) and persisted to `config.toml`'s `[defaults]`/`[defaults.voices]`. `NARU_AUDIO_STT_MODEL`/`NARU_AUDIO_TTS_MODEL`, if set, still wins over what this sets for as long as the process runs — `env_override` says which (naru task 1458). |

### 2.6 Error model

Every non-2xx HTTP response uses OpenAI's envelope:
```json
{"error":{"message":"the model \"foo\" is not in the catalog; see GET /v1/models",
          "type":"invalid_request_error","code":"model_not_found","param":"model"}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `invalid_request` / `unsupported_value` / `unknown_voice` | A field is malformed or has an unsupported value (`param` names the field). |
| 403 | `forbidden_origin` / `forbidden_host` | The Host or Origin guard (§2.1) rejected the request. |
| 404 | `model_not_found` | The name is in neither the catalog nor the user catalog. |
| 404 | `not_found` | No route for the path. |
| 400 | `model_does_not_clone` | `POST /v1/audio/voices` with a `model` that is a TTS model but never speaks in cloned voices. |
| 400 | `voice_model_mismatch` | `POST /v1/audio/speech` names both a cloned voice and, explicitly, a `model` it was not made for. |
| 404 | `voice_not_found` | `GET /v1/audio/voices/{name}` with no cloned voice of that name. |
| 405 | `method_not_allowed` | The route exists but not for this method. |
| 409 | `model_not_pulled` | Known but not downloaded. The message gives the exact `naru-audio pull` command. The daemon **never auto-pulls** on an inference request. |
| 409 | `model_in_use` | `rm` of a busy model. |
| 409 | `model_required` | `rm` of a model another pulled model `requires` (§3.2). |
| 409 | `voice_exists` | `POST /v1/audio/voices` with a name already taken, or `PATCH .../{name}` renaming onto one. |
| 409 | `builtin_voice` | `PATCH`/`DELETE /v1/audio/voices/{name}` on a built-in voice, never in `voices/` (naru task 1458). |
| 404 | `preview_not_cached` | `GET /api/voices/{model}/{voice}/sample` for a built-in voice with nothing cached; `POST` the same path to generate one (naru task 1458). |
| 409 | `pull_in_progress` | `POST /api/pull` for a model already streaming (naru task 1458). |
| 400 | `kind_mismatch` | `POST /api/load`'s `kind` disagrees with the named model's own (naru task 1458). |
| 404 | `pull_not_found` | `DELETE /api/pulls/{name}` with no pull of that model running (naru task 1458). |
| 413 | `payload_too_large` | Input over 256 MiB or 10 minutes, or text over 16 384 chars. |
| 415 | `unsupported_media_type` | The audio is not WAV, or a voice clip `afconvert` cannot read. |
| 503 | `backend_unavailable` | For example mlx on Intel, or a crashed sidecar. `message` gives the reason. |
| 503 | `model_load_failed` | The files are present but the backend rejected them (bad sha after tampering, ORT error). |
| 507 | `insufficient_memory` | The model cannot fit in the budget even after evicting idle models (§3.5). |
| 500 | `internal` | A bug. The log has a request id, and the response carries `X-Request-Id`. |
| — | `pull_failed` | In-stream, not an HTTP status: the `{"error":…}` line (`type` `server_error`) that ends a `POST /api/pull` NDJSON stream after a failure. The response itself is already 200. |
| — | `pull_cancelled` | In-stream: the `{"error":…}` line that ends a `POST /api/pull` NDJSON stream after `DELETE /api/pulls/{name}` (naru task 1458). |

Every response carries `X-Request-Id`. WebSocket errors use the same `code`
strings inside `{"type":"error"}`.

### 2.7 Knobs

`Manifest::prompt_format`'s `knobs` (§3.2, exposed as `x_prompt_format.knobs`
in `GET /v1/models`) name a model's generation controls beyond `speed` and
`exaggeration`, each declared with a `min`, `max` and `default` a client can
build a slider from, and a `step` only where the underlying kwarg takes an
integer (mlx-audio's `num_steps`, `inference_timesteps`): Qwen3-TTS's
`temperature`/`top_p`/`top_k`, Chatterbox's `cfg_weight` (alongside `exaggeration`,
which is also a declared knob), Breeze's `cfg_scale`, OmniVoice's
`num_steps`/`guidance_scale`, VoxCPM2's `cfg_value`/`inference_timesteps`,
and Kokoro's `speed` (documented here too, even though it travels as the
request's own top-level `speed` field, not through `knobs`). Every catalog
kwarg name was checked against the installed mlx-audio's `generate` (naru
task 1458): none needed renaming or dropping, though a future model's knob
might be swallowed by `generate`'s own `**kwargs` and would need to be
caught the same way there.

`POST /v1/audio/speech` and `POST /api/voices/preview` both take
`"knobs":{"<name>":<number>}`. A name not in the model's own
`x_prompt_format.knobs` is 400 `unsupported_value` (`param` is `"knobs"`); a
value that is not a finite number, or falls outside its `min`/`max`, is 400
`invalid_request` (`param` is `"knobs"`). `exaggeration` and `speed` inside
`knobs` are equivalent to the top-level `exaggeration`/`speed` fields —
they land in `SynthOptions`' own fields, checked the same way; every other
knob lands in `SynthOptions.knobs`, a name-to-number map the `mlx` backend
sends the sidecar as `request["knobs"]`, cast to a JSON integer first where
the manifest's `step` says the kwarg takes one (`MlxTts::synth`), then
merged straight into mlx-audio's `generate` kwargs
(`naru_audio_mlx/__main__.py`). The `sherpa-onnx` backend has no knob
beyond `speed` (its own field), so it never sees `SynthOptions.knobs`.

## 3. Model registry

### 3.1 Storage layout

```
$NARU_AUDIO_HOME            default ~/.naru-audio
├── config.toml             user config (§3.3)
├── catalog.d/*.toml        user/experimental manifests, merged over the built-in catalog
├── models/
│   ├── parakeet-tdt-0.6b-v2-int8/
│   │   ├── manifest.json   copy of the resolved manifest + pulled_at
│   │   ├── encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt
│   │   └── bpe.vocab       generated after verify (derived, not hashed)
│   ├── silero-vad/silero_vad.onnx
│   └── kokoro-v1.0/        model.onnx, voices.bin, tokens.txt, espeak-ng-data/,
│                           lexicon-*.txt, dict/   (exact set pinned by task 10)
├── state/measured.json     measured resident bytes per model+backend
├── tmp/                    *.part downloads, per-model .lock files
└── mlx/                    sidecar venv (arm64 only, §5.3)
```

We use a per-model directory rather than Ollama's content-addressed blobs,
because sherpa-onnx takes file paths with fixed names and no blob is shared
between our models. Silero VAD is its own model (`kind = "vad"`), and STT
manifests depend on it through `requires`.

### 3.2 Manifest format

The catalog is TOML embedded in the binary (`catalog/*.toml`), and
`catalog.d/` overrides it. Example with the hashes auris already pins
(auris/src/model.rs:36, 46–67):

```toml
[model]
name        = "parakeet-tdt-0.6b-v2-int8"
kind        = "stt"                      # stt | tts | vad
backend     = "sherpa-onnx"              # sherpa-onnx | mlx
platforms   = ["macos-arm64", "macos-x86_64", "linux-x86_64", "linux-arm64"]
languages   = ["en"]
license     = "CC-BY-4.0"
license_url = "https://creativecommons.org/licenses/by/4.0/"
# non_commercial = true                  # set when the license restricts commercial use (§1441)
resident_bytes = 1_600_000_000           # estimate; state/measured.json corrects it
requires    = ["silero-vad"]

[backend.sherpa-onnx]
model_type = "nemo_transducer"
decoding   = "modified_beam_search"
derive     = ["bpe.vocab"]               # generated from tokens.txt after verify

[[file]]
path   = "encoder.int8.onnx"
url    = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/encoder.int8.onnx"
sha256 = "a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab"
size   = 652_184_296                     # auris model.rs pins sizes too
# … decoder, joiner, tokens.txt likewise

# A TTS manifest adds:
# [backend.sherpa-onnx] lang="en" lexicon=["lexicon-us-en.txt", …] dict_dir="dict"   (fixed at load)
# [[voice]] id="af_heart" sid=3 accent="us" gender="f" default=true
# [backend.mlx] clone=true      speaks in the cloned voices (§5.3)
# [backend.mlx] clone_requires_transcript=true  clone needs ref_text too (§5.3)
# [backend.mlx] instruct=true   takes the request's `instructions` (§2.3, §5.3)
# An archive source: [[archive]] url=… sha256=… strip=1, followed by [[file]] entries whose sha256 is checked after extraction.
```

Rules:
- **Every file and archive has a sha256, no exceptions**; the loader rejects
  a manifest missing one. This ends kokoro-rs's unverified downloads.
- Downloads stream to `tmp/<name>/<file>.part`, check size and then sha256, and
  are renamed into place. This is auris's scheme (model.rs:8, :281, :357).
  A per-model `.lock` serialises concurrent pulls.
- `pull` fetches `requires` first. `rm` refuses to remove a model that another
  pulled model requires unless `--force` is passed.
- On load, the daemon re-checks sizes (cheap). A full re-hash is `naru-audio verify [NAME]`.
- A user manifest with a `file:///` URL is allowed for local experiments. It still needs a sha256.
- `manifest.json` is a pinned copy from pull time (§3.1): a model pulled
  before a catalog update never picks it up, even after an upgrade,
  because a re-`pull` of an already-installed model is a no-op. The one
  exception (naru task 1458 ST7) is `prompt_format` and
  `design_voice_model`: `Registry::pulled_manifest` overlays both from the
  live catalog on every read, since they only describe how to talk to a
  model already staged and hashed — never a file, a size, or anything
  `install`/`remove` decide by — so an old install still gets today's
  per-model controls and Design save target without a re-download.
  Everything else (files, archives, `resident_bytes`, `pulled_at`, ...)
  stays exactly as pinned.

### 3.3 Per-machine defaults

Model resolution order, first match wins:
1. The request's `model` field.
2. `NARU_AUDIO_STT_MODEL` / `NARU_AUDIO_TTS_MODEL`.
3. `config.toml` `[defaults] stt = "…" tts = "…"`.
4. The hardware profile.

| Profile | Detected by | Default STT | Default TTS | Budget (§3.5) |
|---|---|---|---|---|
| `small` | x86_64, or RAM < 24 GB | parakeet-tdt-0.6b-v2-int8 | kokoro-v1.0 (sherpa, fp32; int8 if the spike accepts it) | min(40% RAM, 6 GiB) |
| `large` | arm64 and RAM ≥ 24 GB | parakeet-tdt-0.6b-v2-int8 | kokoro-v1.0 | 50% RAM (32 GiB on the M5 Max) |

The `large` profile deliberately uses the same defaults. MLX models only
become defaults after a benchmark task shows they beat the ONNX baseline on
latency or accuracy. Until then they are opt-in by name. The default choice
is portable, so a config file copied between machines keeps working. RAM is
read from `sysctl hw.memsize`, and arch from `std::env::consts::ARCH` plus
`sysctl.proc_translated`, so an x86_64 build under Rosetta is flagged in
`/health` as a problem.

### 3.4 Load and unload (keep-alive)

- A model loads on its first request or on `POST /api/load`; `loading` shows in `/api/ps` and `/health`.
- `keep_alive` is accepted per request, as a duration string (`"5m"`) or as
  seconds. `0` means unload once idle. A negative value means never unload.
  The default comes from `NARU_AUDIO_KEEP_ALIVE`, then `config.toml`
  `[defaults] keep_alive = "5m"` (same forms), then **5m**,
  which is the same as auris's 300 s and Ollama's default.
- A model's timer restarts when its last in-flight request or WS session ends.
  A model with an open WS session is never unloaded.
- The **daemon** never exits when idle. launchd owns its lifetime (§4), unlike
  auris, which had to exit to free memory.
- Naru calls `POST /api/load {"model":"default","kind":"stt"}` when Live opens, so the
  first utterance does not pay the ~4 s load.

### 3.5 Memory budget and eviction

- `max_resident_bytes` comes from the profile table, and `config.toml` `[memory] max_resident = "8GiB"` overrides it.
- Before loading model X, the daemon computes `need = measured(X) ?? manifest.resident_bytes`.
  If `sum(loaded) + need > budget`, it evicts **idle** loaded models in LRU
  order until X fits.
- If the only models left are busy, the request gets 507
  `insufficient_memory`. The daemon never loads past the budget and hopes.
- After each load, the RSS delta (`task_info` on macOS) is written to
  `state/measured.json`, so estimates converge to what this machine actually uses.
- MLX models are counted using the resident number the sidecar reports
  (`mx.get_active_memory()`) **[assumption: API name in current MLX]**.
- On Intel, parakeet (~1.5 GB) plus Kokoro (~0.5–1 GB **[assumption]**) fit
  within the 6 GiB cap together, so the normal Live session (listen plus speak)
  never evicts.

### 3.6 Backend selection

Each manifest names exactly one `backend`. A backend implements
`available() -> Result<(), reason>` and `load(manifest, dir)`, which returns
an `SttModel` (`decode(pcm16k, hotwords) -> Vec<Segment>`) or a `TtsModel`
(`voices()`, `synth(text, voice, speed, sink)` where `sink` receives f32
chunks and can return `false` to cancel).

- `sherpa-onnx` is compiled in on every platform.
- `mlx` is compiled in on `aarch64-apple-darwin` only. It is a client for the sidecar (§5.3).
- A model whose backend is unavailable shows up in `/v1/models` with
  `x_available:false` and a reason. `pull` refuses it with 503
  `backend_unavailable` unless `--force` is passed.
- If two manifests serve the same model on different backends, they are two
  names (`parakeet-tdt-0.6b-v2-int8`, `parakeet-tdt-0.6b-v2-mlx`). The daemon
  never picks one silently.

### 3.7 CLI

```
naru-audio serve [--listen ADDR:PORT] [--allow-remote]
naru-audio pull NAME…            naru-audio rm NAME… [--force]
naru-audio list [--names]        # --names: bare names one per line, pulled only (auris --list-models contract)
naru-audio ps                    naru-audio verify [NAME…]
naru-audio health                # GET /health, exit 0 ready / 3 daemon down / 4 default model not ready
naru-audio transcribe FILE|- [-m M] [--format text|json]   # thin HTTP client
naru-audio say TEXT|- [-v V] [-s S] [--instructions I] [-o FILE|-]   # thin HTTP client
naru-audio mlx setup|status      # arm64 only
```

`pull`, `list`, `rm` and `verify` work **without** a daemon, on
`$NARU_AUDIO_HOME` under the same `.lock`, so a fresh install can pull before
the service starts. `transcribe` and `say` need the daemon and never start
one; if it is down they exit 3 with the §4.4 message.
`say --instructions` sends the request's `instructions`: the voice description
a VoiceDesign model needs (it has no voices and ignores `-v`); other models
ignore it.

## 4. Service lifecycle on macOS

### 4.1 Homebrew service

This will be the first `service do` block in the tap: none of the formulae in
homebrew-tap/Formula have one (checked).

```ruby
class NaruAudio < Formula
  # url/sha256 per arch, install and test: same pattern as Formula/auris.rb
  service do
    run [opt_bin/"naru-audio", "serve"]
    keep_alive crashed: true
    process_type :interactive          # audio latency; avoid background throttling
    log_path var/"log/naru-audio.log"
    error_log_path var/"log/naru-audio.log"
    environment_variables RUST_LOG: "naru_audio=info"
  end
  def caveats = "naru-audio pull default && brew services start naru-audio"
end
```

The daemon needs **no PATH**: sherpa-onnx, ONNX Runtime and espeak-ng are
statically linked (§5.1), and the MLX interpreter path is stored absolute in
`config.toml`. That removes session 134's root cause (L2). `keep_alive
crashed: true` restarts after a crash but respects `brew services stop`.
`pull default` pulls the profile's STT, TTS and VAD models.

### 4.2 Logs

- There is one line per event on stderr, which goes to `$(brew --prefix)/var/log/naru-audio.log`.
- Format: `ts level req_id event key=value…`. Events: `listening addr=…`,
  `load model=… ms=… rss=…`, `unload model=… reason=idle|evict|rm`,
  `request route=… status=… ms=…`, `ws_open`, `ws_close code=…`, `tts_stream_aborted`.
- Audio and transcript text are **never** logged.
- The daemon rotates its own log at 10 MiB, keeping one `.1` file, because
  Homebrew does not rotate logs.

### 4.3 Health check

`GET /health` (§2.5) is the only probe. It answers within 50 ms even during a
decode because it reads an `ArcSwap` snapshot; `naru-audio health` wraps it
for scripts. Startup binds, reads the catalog and pulled state, and is healthy
within milliseconds; models load lazily.

### 4.4 How Naru detects the daemon (fail loudly)

**Finding it (L2).** Naru config `audio.url` (default
`http://127.0.0.1:7870`), overridden by `NARU_AUDIO_URL`. No PATH lookup and
no spawning: launchd starts the daemon, never Naru.

**Probe cache (L1).** `core::audio::probe()` calls `GET /health` with a
500 ms timeout and caches the answer with a **TTL**: 10 s when `ready`,
2 s for any failure.

Any failed real request (transcribe, speak, a WS open) invalidates the cache
immediately. There is no `OnceLock` anywhere on this path. The model and
voice lists (`/v1/models`, `/v1/audio/voices`) are cached with the same TTLs.

**Probe result.** `GET /api/live/transcribe` changes from `{available}` to:
```json
{"available":false,"state":"daemon_down","engine":"naru-audio",
 "url":"http://127.0.0.1:7870",
 "message":"Speech isn't available: naru-audio isn't running at http://127.0.0.1:7870. Start it with `brew services start naru-audio`.",
 "checked_at":"2026-09-24T12:00:03Z"}
```
`available` stays in the response for old clients.

| `state` | Condition | Error text (`message`) |
|---|---|---|
| `ready` | health ok and `stt.ready` | none |
| `daemon_down` | connection refused or timeout | "Speech isn't available: naru-audio isn't running at {url}. Start it with `brew services start naru-audio`." |
| `model_missing` | `stt.problem.code == model_not_pulled` | "Speech isn't available: the model {m} isn't downloaded. Run `naru-audio pull {m}`." |
| `incompatible` | `api` ≠ 1 | "Speech isn't available: naru-audio {v} speaks API {n}; this Naru needs API 1. Upgrade with `brew upgrade naru-audio`." |
| `error` | any other problem | "Speech isn't available: naru-audio reported: {problem.message}" |

TTS uses the same states and texts with "Naru's voice isn't available: …".

**UI state (L3).**
- `listenPath()` gains a fourth result, `'unavailable'`, whenever the server
  says `state !== 'ready'`. The browser Web Speech path is chosen **only** when
  the user set `listen.engine = "browser"`; it is never a fallback.
- LiveHub shows the server's `message` as an error banner where the meter would
  be, with a **Retry** button that re-probes and a copyable command. A flat
  meter now always means "no sound", never "no engine". The mic button reads
  "Speech unavailable".
- If a WS closes mid-session, the banner appears within one probe (≤ 2 s), and
  the open segment's captured audio is kept so Retry can resend it.
- iOS shows the server's `message` in place of the hard-coded
  `voiceUnavailableMessage`, keeping its refuse-to-fall-back posture.

**Server-side loudness.** Naru logs `warn audio state ready -> daemon_down url=…`
on every change of state, so a background service's log shows the moment it
broke.

## 5. Language and runtime

### 5.1 Rust daemon, sherpa-onnx as the portable baseline

- **Rust 2024 with tokio and axum.** Naru already uses axum 0.8 with `ws`.
  Neither predecessor uses tokio or axum (both are synchronous CLIs; auris
  downloads with `ureq`). auris's engine, audio gates, VAD, vocabulary and
  model-download code is ported and moved behind `spawn_blocking`.
- **sherpa-onnx `=1.13.6`, `static` feature, for both STT and TTS.**
  Verified in the vendored crates: `sherpa-onnx-sys` build.rs:23–24 links
  `piper_phonemize` and `espeak-ng` statically, and build.rs:239/242 fetch
  prebuilt `osx-x64`/`osx-arm64` libs, so the Intel i9 is covered.
  `sherpa_onnx::OfflineTts` (tts.rs:520) takes an
  `OfflineTtsKokoroModelConfig { model, voices, tokens, data_dir, lang, … }`
  (tts.rs:158), and `generate_with_config` (tts.rs:556) has a progress
  callback. `VoiceActivityDetector` (vad.rs:191) and `OnlineRecognizer`
  (online_asr.rs:351) are there too. **Verified since:** an independent
  reviewer confirmed that the sherpa Kokoro/espeak-ng path works on both
  osx-x64 and osx-arm64.
- **espeak-ng and Kokoro.** Drop `ort` and `espeak-rs`. That removes the
  cmake/clang/libclang source build of espeak-ng, the runtime ORT 1.23.2 dylib
  download, and the risk of two ONNX Runtimes in one process (sherpa's static
  copy plus ort's dlopened one). Phonemisation still runs espeak-ng, inside
  sherpa's static lib, reading the model's `espeak-ng-data/` (`data_dir`). Its
  build cost is **zero** beyond the prebuilt download auris already pays.
- **What we must port to keep parity.**
  - sherpa addresses voices by `sid`, so the manifest maps `id → sid` and the API keeps names.
  - **The chunker cannot be ported as is.** kokoro-rs text.rs budgets chunks
    in *phonemes*, and it gets them from `punctuation::phonemize_preserving`
    (punctuation.rs:251), which calls espeak through espeak-rs. sherpa's Rust
    API exposes no phonemiser. naru-audio therefore uses a **character-based**
    schedule with the same shape: the first chunk is ≤100 chars at a
    clause/sentence boundary, later chunks double up to 400, and `generate` is
    called once per chunk. Task 9 also tests whether sherpa's own sentence
    split (`max_num_sentences`, tts.rs:391) plus the `generate` callback gives
    equal time-to-first-audio. If it does, the chunker is dropped.
  - **Silence trim.** Kokoro pads every chunk with silence (about 2 s before
    the first). kokoro-rs trims it in trim.rs, a port of `librosa.effects.trim`
    with no dependencies, and naru-audio ports it too. Task 9 checks whether
    sherpa already trims.
  - The `Leveller` (level.rs) is pure Rust and is ported unchanged.
- **Risk.** **[assumption: sherpa's `kokoro-multi-lang-v1_0` package carries
  the same v1.0 82M weights and 54 voices in its own `voices.bin` format.]**
  That package needs `lang` or a lexicon, plus `dict/` (tts.rs:164–166).
  Task 9 is a spike that checks quality, voice
  mapping and latency against kokoro-rs before TTS is built on it. **Plan B**
  if it fails: run kokoro-rs's existing `ort` engine as a supervised worker
  process (the §5.3 framing), which keeps the two ONNX Runtimes in separate
  processes but brings back the espeak-rs cmake build.
- STT speed on this exact i9 is measured (RTF 0.06 warm). Kokoro through
  sherpa on the i9 is **[assumption]** fast enough, as kokoro-rs is today;
  task 9 measures it.

### 5.2 Rejected alternatives for the baseline

- **Python daemon** (FastAPI + onnxruntime): needs Python on every machine and throws away the tested Rust in auris and kokoro-rs.
- **Swift + CoreML**: macOS-only, and loses Linux, which auris supports today.
- **Go + ONNX C API**: no advantage over Rust, and no code reuse.
- **whisper.cpp**: different models. Parakeet stays (auris's accuracy and hotword work depends on it); whisper could become a later backend.

### 5.3 MLX on Apple Silicon

MLX is for "bigger or experimental" models on the M5 Max. The realistic paths are:

| Option | What it takes | Verdict |
|---|---|---|
| **mlx-rs** crate (oxideai) | Safe Rust over mlx-c. You must **reimplement every model architecture in Rust** (FastConformer + TDT decoder, StyleTTS2/iSTFTNet, and whatever comes next), plus the weight loading. The crate is pre-1.0 and follows MLX releases **[assumption: still pre-1.0, with no Parakeet or Kokoro ports available]**. | Rejected for now. A port per experimental model defeats the point of "experimental". |
| **mlx-c FFI** directly | The same porting cost as mlx-rs, plus hand-written unsafe bindings. | Rejected. |
| **Python mlx sidecar** | The daemon supervises a Python process that runs community implementations (`parakeet-mlx`, `mlx-audio`, which covers Kokoro and other TTS/STT models, and `mlx-whisper`) **[assumption: package names and model coverage as of 2026-09]**. New models arrive in Python first. | **Chosen.** |

Sidecar design:
- **Install:** `naru-audio mlx setup` creates `~/.naru-audio/mlx/.venv` with
  `uv` from a lock file embedded in the binary, and writes the absolute
  interpreter path to `config.toml`. If `uv` is missing, setup fails with the
  exact `brew install uv` command. The Homebrew formula has no dependency on
  it, so Intel machines never see Python.
- **Process:** `python -m naru_audio_mlx --socket ~/.naru-audio/mlx/sidecar.sock`
  (the module is embedded and written out at setup). It starts on the first
  `mlx` request and stops when the last MLX model unloads. A crash gives 503
  `backend_unavailable` to requests in flight, restarts with backoff (1, 2,
  4 s, then waits for the next request), and sets `/health` `backends[mlx].reason`.
- **Protocol:** length-prefixed frames over the Unix socket (the idea behind
  auris's daemon framing, docs/daemon.md). Each frame is a JSON header
  `{op: load|unload|transcribe|synth|stats, model, …}`, optionally followed by
  raw f32le samples (the full list is in `mlx/naru_audio_mlx/protocol.py`).
  `load` names the model's kind; a TTS load answers with its `sample_rate`.
  A `synth` answer is a stream of `{samples: S}` PCM frames, one per
  mlx-audio chunk (0.32 s) as it is generated, terminated by an
  `{end: true, ok, error?}` header. To cancel, the daemon sends
  `{op: cancel}` once and reads on to the `end` header: the sidecar checks
  for it after each frame and stops generating, and ignores a cancel that
  arrives after the end, which has no answer. The connection is never
  left mid-stream. `stats` returns resident bytes for §3.5.
- **TTS models:** mlx-audio's `load_model` loads the pulled directory, so a
  new mlx-audio model is a catalog entry (`backend = "mlx"`, `kind =
  "tts"`, its `[[voice]]` ids passed as mlx-audio's `voice`, or a
  `reference` recording as `ref_audio`). Qwen3-TTS 0.6B CustomVoice 8-bit
  is the first. Qwen3-TTS 0.6B and 1.7B Base 8-bit
  (`qwen3-tts-0.6b-base-mlx`, `qwen3-tts-1.7b-base-mlx`) have no preset
  voices; each manifest sets `[backend.mlx] clone = true`, and each speaks in the cloned
  voices under `$NARU_AUDIO_HOME/voices/<name>/` (`ref.wav`, 24 kHz mono,
  and `ref.txt`, its transcript — required here, so both manifests also set
  `clone_requires_transcript = true`: mlx-audio's `qwen3_tts.py` only takes
  the in-context cloning path, `use_icl`, when both `ref_audio` and
  `ref_text` are given), added with `naru-audio voice add <name>
  <clip> --text <transcript>` or `POST /v1/audio/voices` (§2.5) and listed by `/v1/audio/voices?model=` after the
  manifest's. Each goes to mlx-audio as `ref_audio` and `ref_text`. No gap or leveller: the chunks are cut mid-sentence, as
  with Pocket. **Voices are keyed by model** (task 1454): alongside `ref.wav`
  and `ref.txt`, a voice's directory holds `model.txt`, the model it was
  cloned or designed for; `voices::of` (which every listing and `speech`
  goes through) only offers a manifest the clones made for its own model, a
  voice with no `model.txt` — everything added before this — is attributed
  to `CLONE_MODEL` (`qwen3-tts-0.6b-base-mlx`), and `POST /v1/audio/speech`
  with a cloned voice and an explicit `model` it was not made for is
  refused (400 `voice_model_mismatch`) rather than silently using the wrong
  model or answering `unknown_voice` as if the voice did not exist; with no
  explicit `model`, the voice's own is used, generalising the CLI's
  `say_model` (still `CLONE_MODEL` for a legacy voice).
- **Instructions:** a manifest that sets `[backend.mlx] instruct = true`
  gets the request's `instructions` (§2.3) in the `synth` header as
  `instruct`, which the sidecar passes to mlx-audio's `generate`; no other
  model is sent it. Qwen3-TTS 1.7B VoiceDesign 8-bit
  (`qwen3-tts-1.7b-voicedesign-mlx`) is the only one: it has no preset or
  cloned voices and makes its voice up from `instructions`, so it needs
  them, and the `voice` asked for is ignored. The flag is ours, not
  mlx-audio's: its guard that drops `instruct` for the 0.6B models is
  dead code, so a model without the flag must not be sent it.
- **Chatterbox** (`chatterbox-tts-8bit-mlx`, MIT) is a third cloning model:
  like the Qwen3-TTS Base pair it sets `[backend.mlx] clone = true` and has
  no preset voices, but it needs no transcript (mlx-audio's Chatterbox
  ignores `ref_text`, sent anyway for a uniform request; `clone_requires_transcript` is unset, so `false`). It does not
  support `speed` at all; the sidecar shim rejects a non-1.0 `speed` for it
  rather than silently generating at normal speed. It takes the request's
  `exaggeration` (0–1, an emotion-exaggeration dial), gated by
  `[backend.mlx] exaggeration = true` the same way `instruct` gates
  `instructions`. Its loader also fetches a small shared dependency,
  `mlx-community/S3TokenizerV2`, straight from the Hub rather than from the
  manifest's own files; the sidecar shim lifts `HF_HUB_OFFLINE` for that one
  call, so it downloads once and is cached under `~/.cache/huggingface`
  like any other `huggingface_hub` download.
- **VoxCPM2** (`voxcpm2-8bit-mlx`, Apache-2.0) is a fourth cloning model,
  and the first to set both `[backend.mlx] clone = true` and `instruct =
  true`: it clones a reference the same way as the Qwen3-TTS Base pair and
  Chatterbox (no preset voices, no transcript needed — mlx-audio's VoxCPM2
  takes `ref_text` but never reads it in that mode), and separately makes a
  voice up from `instructions` (mlx-audio's `instruct`, prepended to the
  text as `(instruct)text`) when no cloned voice is named. `voice` in
  `src/server/speech.rs` resolves a named, existing cloned voice first,
  even if `instructions` is also given, and only falls back to the design
  path — and only then requires `instructions` — once no voice matched;
  every other model with `instruct = true` has no voices at all, so this
  ordering was never exercised before. VoxCPM2 does not support `speed`
  either — mlx-audio's `generate` for it has no `speed` parameter, so
  it would otherwise be silently swallowed by `**kwargs` — and the sidecar
  shim rejects a non-1.0 `speed` for it the same way it does for
  Chatterbox. It is also the first model above 24 kHz: it reports
  `sample_rate: 48000` at `load`, which `MlxTts::load` reads into
  `TtsModel::sample_rate`, used end to end for the WAV header,
  `x-audio-sample-rate` and `/v1/models` — nothing downstream assumes
  24 kHz.
- **IndexTTS** (`indextts-1.5-mlx`, Apache-2.0) is a fifth cloning model:
  like Chatterbox and VoxCPM2 it sets `[backend.mlx] clone = true`, has no
  preset voices and needs no transcript (mlx-audio's IndexTTS has no
  `ref_text` parameter, swallowed by `**kwargs`), and does not support
  `speed` either, rejected the same way. It is the first model whose
  mlx-audio `generate` never streams: it yields exactly one chunk, after
  the whole utterance has decoded, regardless of `stream` or
  `streaming_interval`; `say -o -` still works (the WAV header's RIFF sizes
  already handle a stream of unknown length, §2.4), but the first and only
  audio is out only once generation finishes, not progressively like every
  other MLX TTS model. mlx-community's `config.json` for it also has no
  `tokenizer_name`, which mlx-audio's `ModelArgs` requires to find
  `tokenizer.model`; since the manifest's `config.json` is hash-pinned and
  re-hashed by `naru-audio verify`, the sidecar never edits it in place.
  On `load` it instead builds a scratch directory with every file
  symlinked in and its own patched `config.json` (`tokenizer_name`
  pointing at the scratch directory itself), loads from there, and
  removes it once `load_model` has read everything it needs
  (`naru_audio_mlx/__main__.py`, `_indextts_load_dir`).
- **OmniVoice** (`omnivoice-bf16-mlx`, CC-BY-NC-4.0, **non-commercial**) is
  a sixth cloning model, and the second (after VoxCPM2) that also designs
  a voice from a description: `[backend.mlx] clone = true` and `instruct =
  true`, no preset voices, 646 languages (k2-fsa/OmniVoice's card).
  Unlike VoxCPM2's cloning path, its cloned voice's transcript is not
  ignored — mlx-audio's OmniVoice `generate` takes `ref_text` and prepends
  it to the spoken text before tokenizing — so `--text` is used, not just
  accepted and dropped. It is still optional, though (`ref_text: Optional[str]
  = None`, cloning gated on `ref_audio` alone), so `clone_requires_transcript`
  is unset, `false`. It does not support `speed` either (no `speed`
  parameter, only `**kwargs`), rejected the same way as Chatterbox,
  VoxCPM2 and IndexTTS, and does not stream: like IndexTTS, mlx-audio's
  `generate` yields exactly one chunk after the whole utterance has
  decoded. The catalog pins mlx-community/OmniVoice-**bf16**, not their
  8-bit conversion: mlx-audio's OmniVoice `sanitize()` splits the
  checkpoint's combined `audio_embeddings.weight` / `audio_heads.weight`
  tensors into 8 per-codebook tensors by exact key match, but does not do
  the same for their quantized `.scales` / `.biases` siblings, so loading
  the 8-bit repo fails with mlx's `load_weights` rejecting those keys as
  "not in model" — confirmed against mlx-audio as pinned in
  `mlx/pyproject.toml` (2026-09-26); bf16 has no `quantization` key in its
  config, so `apply_quantization` never runs and the mismatch does not
  arise.
- **Breeze TTS 2** (`breeze-tts-2-mlx`, BreezeBlue Research and
  Non-Commercial License, **non-commercial**) is a seventh cloning model,
  and the third (after VoxCPM2 and OmniVoice) that also designs a voice
  from a description: `[backend.mlx] clone = true` and `instruct = true`,
  no preset voices, 24 kHz output. Unlike the five cloning models above,
  its `generate` raises outright without a transcript alongside the
  reference clip (`breeze_tts.py`: `"Breeze voice cloning requires ref_text
  with ref_audio."`), so its manifest also sets `clone_requires_transcript
  = true`. It does not support `speed` either (no
  `speed` parameter, swallowed silently by its own `**kwargs`, same
  mechanism as Chatterbox/VoxCPM2/IndexTTS/OmniVoice), rejected the same
  way as those four. Unlike those four, its `generate` does declare a
  `streaming_interval` parameter and accepts the sidecar's usual short
  cadence (`STREAMING_INTERVAL = 0.32`) without error — confirmed by
  pulling the model and running `say` end to end (2026-09-26) — so, unlike
  `speed`, no sidecar shim is needed for it. The catalog pins
  mlx-community's full-precision `Breeze-TTS-2-mlx` conversion (~7.6 GB),
  not their 4-bit or 8-bit ones, per task 1446.
- **Cost:** about 150–400 ms extra first-request latency for the process start
  **[assumption]**, then a per-request IPC overhead that is negligible
  compared with decode time.
- **Revisit** mlx-rs if one MLX model becomes a default on the `large`
  profile. At that point the port is a one-time cost with a lasting benefit.

### 5.4 License

auris and kokoro-rs are both GPL-3.0-or-later (auris/Cargo.toml:6,
kokoro-rust/Cargo.toml:6). naru-audio reuses their code, so it defaults to
**GPL-3.0-or-later** unless the owner relicenses. This is listed as an open
question.

## 6. Migration plan

### 6.1 Phases

| Phase | Outcome | Tasks | auris / kokoro-rs |
|---|---|---|---|
| P1 | Daemon, registry and batch STT at parity with auris | 1–8 | unchanged |
| P2 | TTS at parity with kokoro-rs | 9–12 | unchanged |
| P3 | Streaming STT over WebSocket | 13–14 | unchanged |
| P4 | Released: Homebrew formula with a service block | 15 | unchanged |
| P5 | Naru can use naru-audio behind `audio.engine`; default stays `legacy` | 16–21 | still the default |
| P6 | WS dictation ships, then Naru's default flips after a week of daily use on both Macs | 22–24 | explicit opt-in only |
| P7 | MLX sidecar on arm64 | 25 | — |
| P8 | Legacy path removed; auris and kokoro-rs formulae deprecated | 26 | retired |

Two explicit config keys, both shown in Settings, and neither is ever used
as a fallback:
- `audio.engine = "legacy" | "naru-audio"` picks what Naru's **server** runs.
  With `naru-audio` selected and the daemon down, Naru shows the §4.4 error.
- `listen.engine = "server" | "browser"` (default `server`) picks what the
  **page** listens with. `browser` is the Web Speech API. It is kept as a
  deliberate opt-in because it is the only option on a machine with no
  models; this resolves the former open question.

### 6.2 How Naru changes

| Module | Change |
|---|---|
| new `src/core/audio.rs` | An HTTP client for naru-audio. This is a new dependency: Naru's Cargo.toml has axum 0.8 (`ws`, server side) but no HTTP or WebSocket **client** crate. Contents: `probe()` with the TTL cache, the §4.4 state enum and texts, `models()`/`voices()` with TTLs, `transcribe(wav, model)`, `speak(text, voice) -> Stream<Bytes>`, `load(kind)`. |
| `src/core/listen.rs` | `models()`/`transcribe()` dispatch on `audio.engine`. On the legacy branch, auris exit 1 maps to `Ok("")` (the silence contract, §2.2), and a TTL cache replaces the `OnceLock` in `models()` (:68), which fixes L1 on both engines. |
| `src/core/speech.rs` | `voices()`/`start()` dispatch the same way. `fix_wav_sizes` stays as a no-op guard: the daemon already writes `0x7fff0000`, which the function leaves unchanged by design (speech.rs:460–463). |
| `src/core/config.rs` (2194–2208, 2377–2392) | New keys `audio.url`, `audio.engine`, `listen.engine`. `speech.voice`/`listen.model` are validated against the daemon's lists when `audio.engine = naru-audio`. |
| `src/api.rs` | `transcribe_live` (:4822) forwards to `/v1/audio/transcriptions`: it returns `{text:""}` for silence and a 503 with the §4.4 `message` for real errors. `transcribe_available` (:4907) returns the rich state. The speak routes proxy `/v1/audio/speech` (`wav`). New `GET /api/live/listen` WS route: Naru's access gates, then a two-way proxy to `/v1/audio/transcriptions/stream` through a WS client crate (tokio-tungstenite **[assumption: choice made in task 22]**), with **`Origin` stripped** before forwarding, because the daemon rejects any Origin (§2.1). |
| `frontend/src/liveRecognition.ts` (:144–161, :443) | `listenPath()` returns `'unavailable'`, not `'browser'`, unless `listen.engine = "browser"`. `isSilentTranscribe` is deleted. |
| `frontend/src/components/LiveHub.tsx` (:1769, :3136) | Empty text is dropped without saying anything, replacing the stderr match at :1769. Adds the unavailable banner and Retry, `POST /api/load` warm-up on open, and later WS streaming. |
| Settings page | Two selectors, for `audio.engine` and `listen.engine`, plus the current probe state. |
| `liveVad.ts`, `speechStream.ts`, `wavStream.ts` | Unchanged. liveVad does not segment on the streaming path. |
| `naru-ios` `MesaKit/TranscriptionQueue.swift` (:147) | Show the server's `message`. Keep refusing to fall back. |

### 6.3 Follow-up implementation tasks

Each task names its phase and what it depends on ("Deps").

1. **Scaffold the daemon: /health, guards, logging.** (P1; Deps: none)
   Create the crate (Rust 2024, tokio, axum) with `serve` on `127.0.0.1:7870`,
   `--listen`/`--allow-remote`, the §2.1 Host/Origin guards (the Host check is
   skipped under `--allow-remote`), the §2.6 envelope, `X-Request-Id`, and the
   §4.2 log format with self-rotation at 10 MiB.
   *Acceptance:* `/health` returns `api:1`. `Origin: http://evil.test` returns
   403. A foreign `Host` returns 403 by default and 200 with `--allow-remote`.
   A non-loopback `--listen` without the flag exits non-zero. A test proves
   rotation at 10 MiB.
2. **Registry core library: catalog, manifests, verified downloads.** (P1; Deps: 1)
   Embedded TOML catalog plus `catalog.d`. A manifest missing a sha256 is
   rejected. Downloads go `.part` → size → sha256 → rename, with per-model
   locks, `requires`, archive extraction and derived files (`bpe.vocab`). No
   CLI or HTTP surface yet.
   *Acceptance:* against a local HTTP stub, a mutated byte fails and leaves
   nothing in `models/`. Two concurrent pulls download once.
3. **Registry CLI and API: pull, list, rm, verify, /v1/models.** (P1; Deps: 2)
   Add `pull/list [--names]/rm/verify` (working without a daemon),
   `POST /api/pull` (NDJSON), `DELETE /api/models/{name}` and `GET /v1/models`
   with every §2.5 field, including `x_available`/`x_unavailable_reason`.
   *Acceptance:* `list --names` prints bare names and exits 0 offline. `rm` of
   a required model fails without `--force`. The `/v1/models` fields match §2.5.
4. **Pin STT and VAD catalog entries.** (P1; Deps: 2)
   Write the manifests for `parakeet-tdt-0.6b-v2-int8` and `silero-vad`,
   copying the sizes and hashes from auris/src/model.rs (36, 46–67, 82–89).
   *Acceptance:* on a clean `$NARU_AUDIO_HOME`, `naru-audio pull parakeet-tdt-0.6b-v2-int8`
   pulls both models, and `verify` passes.
5. **Port the auris STT engine as a library.** (P1; Deps: 4)
   Port engine.rs, the audio.rs decode and resample (1 kHz–384 kHz, caps), the
   `is_silent` and Silero gates, and vocabulary.rs (global 3.0, per-term
   `:boost`, `looks_manufactured`) behind `SttModel`.
   *Acceptance:* auris's test fixtures give identical transcripts through the
   library. Silence gives an empty result, not an error.
6. **POST /v1/audio/transcriptions.** (P1; Deps: 3, 5)
   Implement §2.2: multipart, `json`/`text`/`verbose_json`, SSE `stream`, the
   hotwords string, the VAD fields, aliases, and the field accept/ignore/reject
   table.
   *Acceptance:* the `openai` Python client's
   `audio.transcriptions.create(model="whisper-1")` succeeds. Silence gives 200
   `{"text":""}`. Non-WAV gives 415. `language=fr` gives 400.
7. **Model manager: keep_alive, eviction, budget, profile.** (P1; Deps: 3, 5)
   Implement §3.3–3.5: profile detection (RAM, arch, and the Rosetta
   `sysctl.proc_translated` problem in `/health`), budget, measured RSS, LRU
   eviction of idle models only, 507, `/api/ps` and `/api/load`.
   *Acceptance:* with the budget below two models, loading the second evicts
   an idle first model and returns 507 when the first is busy. `keep_alive:0`
   unloads. `/health` answers in < 50 ms during a 60 s decode.
8. **CLI clients: transcribe, health, ps.** (P1; Deps: 6, 7)
   Thin HTTP clients (§3.7). `health` exits 0/3/4 and prints the §4.4 message.
   `transcribe --format json` prints auris-style JSONL.
   *Acceptance:* with the daemon stopped, `transcribe x.wav` exits 3 with the
   "isn't running" text. With it running, the output matches the endpoint.
9. **Spike: Kokoro via sherpa-onnx vs kokoro-rs.** (P2; Deps: none)
   On the i9 and an arm64 Mac, synthesise a fixed 10-sentence corpus for 5
   voices with both engines. Measure time to first audio, RTF and RSS. Test
   (a) sherpa's sentence split plus callback against the §5.1 character
   chunker, and (b) whether sherpa trims Kokoro's padding. Confirm the `lang`,
   lexicon and `dict/` settings and the name → sid table. Record the result in
   `docs/tts-spike.md`.
   *Acceptance:* the doc has a numbers table and a decision on the chunker and
   on trimming. Quality passes only if **Simon signs off** on a blind A/B of
   the 5 voices ("no worse than kokoro-rs"); otherwise plan B (§5.1).
10. **Pin the Kokoro catalog entry.** (P2; Deps: 2, 9)
    Write the `kokoro-v1.0` manifest from the spike's package: model,
    `voices.bin`, tokens, `espeak-ng-data/`, the `lexicon-*.txt` files and
    `dict/`, each sha256-pinned, plus `lang` and the 54-voice id → sid table.
    *Acceptance:* `pull kokoro-v1.0` and `verify` pass on a clean home. The
    manifest lists 54 voices with `af_heart` as the default.
11. **TTS engine: sherpa Kokoro backend with chunker, trim and leveller.** (P2; Deps: 7, 10)
    Implement `TtsModel` on `OfflineTts`. Port level.rs, trim.rs (if the spike
    says sherpa does not trim) and gap reinsertion, and the chunker the spike
    chose.
    *Acceptance:* unit tests cover chunk boundaries, trim and the leveller.
    The first chunk's samples reach the sink before the second chunk's
    synthesis starts.
12. **POST /v1/audio/speech, GET /v1/audio/voices, and `say`.** (P2; Deps: 11)
    Implement §2.3: chunked `wav` (`0x7FFF0000` sizes), `pcm`, `stream=false`,
    mid-stream abort, and voices from the manifest only. Add the `say` CLI client.
    *Acceptance:* the WAV header bytes are `0x7FFF0024`/`0x7FFF0000`. Voices
    answer with the model unloaded. Safari plays the stream via `fetch()` + Web
    Audio. `say -o f.wav && afplay f.wav` works.
13. **Streaming STT over WebSocket, finals only.** (P3; Deps: 6, 7)
    Implement §2.4: the handshake, s16le/f32le frames, server Silero
    segmentation with 300 ms padding, `speech`/`final`/`error`/`done`,
    `flush`/`stop`, `max_segment`, the 30 s backlog and the close codes, plus
    a real-time WAV test client.
    *Acceptance:* the finals on a 60 s fixture match the batch segments. The
    median time from end of speech to `final` is ≤ `min_silence` + 400 ms on
    the i9. Violations close with 1008 after an `error` event.
14. **Optional partials on the WebSocket stream.** (P3; Deps: 13)
    Add `partials:true`: re-decode every 700 ms, skip a tick while busy, stop
    past 8 s, and send the x86_64 `warning`.
    *Acceptance:* no partial arrives after its `final`. With partials off, no
    extra decode runs (counter test). The i9 CPU cost per session is reported.
15. **Release pipeline and Homebrew formula with a service block.** (P4; Deps: 8, 12)
    Build darwin-arm64/amd64 and linux binaries as auris does. Add
    `Formula/naru-audio.rb` with the §4.1 `service do`, log path and caveats.
    *Acceptance:* `brew install … && brew services start naru-audio` gives a
    passing `naru-audio health` on both Macs, with the log in
    `var/log/naru-audio.log`. `brew test` passes.
16. **Naru: audio client, TTL probe, config keys, rich state.** (P5; Deps: 1; can build against a stub)
    Add `src/core/audio.rs` and an HTTP client crate. Add the `audio.url`,
    `audio.engine` and `listen.engine` keys. Replace the `OnceLock`s in
    `listen::models()`/`speech::voices()` with a TTL cache. `transcribe_available`
    returns `{available,state,engine,url,message,checked_at}`, and state
    changes are logged.
    *Acceptance:* against a stub that goes down and comes back, the state goes
    `ready` → `daemon_down` → `ready` within the TTL with no restart. Existing
    legacy tests pass.
17. **Naru: route transcription and speech through naru-audio, with the silence contract.** (P5; Deps: 16, 6, 12)
    With `audio.engine = "naru-audio"`, `transcribe_live` and the three speak
    routes proxy to the daemon, and validation uses the daemon's lists. On
    both engines, silence returns 200 `{text:""}` (legacy maps auris exit 1 to `Ok("")`).
    *Acceptance:* stub-daemon tests cover success, **silence → 200 `{"text":""}`**,
    `daemon_down`, `model_missing` and a mid-stream TTS abort. The legacy
    silence test also returns `{"text":""}`. There are no `auris_bin()`/`kokoro_bin()`
    calls when the engine is `naru-audio`.
18. **Naru frontend: loud unavailable state, drop the stderr matching.** (P5; Deps: 16, 17)
    Add `'unavailable'` to `ListenPath`, and pick `'browser'` only for
    `listen.engine = "browser"`. Add the banner, Retry and copyable command.
    LiveHub drops empty text. Delete `isSilentTranscribe` and its tests.
    *Acceptance:* vitest for `{transcribes:false,captures:true,recognizes:true}`
    returns `'unavailable'`. `grep isSilentTranscribe` finds nothing. A khora
    check with the daemon stopped shows the banner and no "Listening through
    this browser".
19. **Naru Settings: audio.engine and listen.engine selectors.** (P5; Deps: 16)
    Two selectors that write the config keys through the existing
    `/api/config` routes, showing the live probe state next to them.
    *Acceptance:* changing a selector takes effect on the next probe without a
    restart. A vitest covers the form.
20. **Naru: warm the STT model when Live opens.** (P5; Deps: 16, 7)
    When a Live session opens with `audio.engine = "naru-audio"`, Naru sends
    `POST /api/load {"model":"default","kind":"stt"}` without blocking the UI.
    A failure feeds the probe state.
    *Acceptance:* a stub test sees exactly one load call per session open. The
    first-utterance latency with a cold daemon is reported before and after.
21. **naru-ios: show the server's unavailable message.** (P5; Deps: 16)
    Decode `state`/`message` from `GET /api/live/transcribe` and show
    `message` in place of `voiceUnavailableMessage`, keeping the old string for
    `{available}`-only servers.
    *Acceptance:* unit tests for both shapes. Still no on-device fallback.
22. **Naru: WebSocket proxy route /api/live/listen.** (P6; Deps: 13, 16)
    Add a WS client crate and a route that runs Naru's access gates, opens the
    daemon WS **without the browser's `Origin`**, and pumps frames both ways
    with close codes preserved.
    *Acceptance:* an integration test streams a WAV fixture through Naru and
    receives the daemon's finals. A daemon-down open closes with an `error`
    carrying the §4.4 message.
23. **LiveHub: streaming dictation over /api/live/listen.** (P6; Deps: 22, 18)
    Stream AudioWorklet frames as s16le, use `speech` events as the
    silence-timer heartbeat and `final`s as settled text, and on close re-probe
    and show the banner. The POST path remains for `legacy`.
    *Acceptance:* dictation works in Chrome and Safari. Latency after speech
    ends beats the POST path on the same machine (numbers reported). Killing
    the daemon shows the banner within 2 s.
24. **Flip Naru's default audio.engine to naru-audio.** (P6; Deps: 17–23, one week of daily use)
    Change the default and the config migration so existing configs without
    the key get `naru-audio`, with a release note on how to switch back.
    *Acceptance:* a fresh config uses naru-audio. An explicit `legacy` is kept.
25. **MLX sidecar on Apple Silicon with one model.** (P7; Deps: 3, 7)
    Implement §5.3: `mlx setup` (uv venv from an embedded lock), a supervisor
    with backoff, the framed socket protocol, and `stats` feeding the budget.
    Add one entry, `parakeet-tdt-0.6b-v2-mlx`.
    *Acceptance:* the M5 Max transcribes the task 5 fixtures (WER reported).
    Intel lists the model with `x_available:false`. Killing the sidecar gives
    503 and then a successful retry.
26. **Retire auris and kokoro-rs.** (P8; Deps: 24, plus a week)
    Remove the legacy engine from Naru (the child-process code and
    `fix_wav_sizes`). Add deprecation caveats to `Formula/auris.rb` and
    `Formula/kokoro-rs.rb`, and point their READMEs at naru-audio.
    *Acceptance:* Naru builds and its tests pass with no `auris_bin`/`kokoro_bin`.
    `brew info auris` shows the caveat.

### 6.4 Open questions

1. **"Parakeet small".** The only Parakeet shipped today is the 0.6B int8
   (661 MB, RTF 0.06 on the i9), which already fits the Intel machine. Did you
   mean that model, or a smaller variant such as a 110M TDT-CTC
   **[assumption: a sherpa-onnx export exists]**?
2. **mp3/opus output.** OpenAI clients default to mp3. Should we add an
   encoder (a new dependency), or is WAV/PCM-only acceptable?
3. **License.** naru-audio reuses GPL-3.0-or-later code from both
   predecessors. Should it stay GPL, or will you relicense?
4. **Correction pass** (auris/docs/correction.md, name F1 69% → 89%). Should
   it live in naru-audio (a stage after the final), or in Naru, which knows the
   project names?
5. **LAN clients.** Is the Naru WS proxy enough for phones, or should other
   apps reach naru-audio directly on the LAN (`--allow-remote`)?

## 7. Assumptions not verified

- The Naru service's PATH during session 134 (launchd environment) was not captured.
- sherpa-onnx's Kokoro package carries the same v1.0 82M weights and all 54
  voices as thewh1teagle's files, and quality matches (task 9).
- Kokoro's resident memory on sherpa is about 0.5–1 GB, and its speed on the
  i9 is adequate through sherpa (task 9 measures both).
- mlx-rs is still pre-1.0, with no Rust ports of Parakeet or Kokoro.
- The Python packages `parakeet-mlx`, `mlx-audio` and `mlx-whisper` exist
  with the stated coverage, and `mx.get_active_memory()` is the current name
  of the memory API.
- The sidecar's cold start costs about 150–400 ms.
- A sherpa-onnx export of a ~110M Parakeet exists.
- tokio-tungstenite is the right WS client crate for Naru's proxy (task 22 decides).
- Homebrew's `process_type :interactive` meaningfully reduces latency jitter for this daemon.

## 8. Voice prep (naru task 1461)

Cleans a raw uploaded clip into a reusable clone sample: crop to a chosen
span, denoise, trim silence, normalise level. Transcription (word
timestamps) and speaker diarization let a caller pick *which* span and
*whose* voice before cropping. A sample is stored separately from the
existing voice-preview "sample" (`GET /api/voices/{model}/{voice}/sample`,
§2.5) and from a cloned `voices/` entry: it is model-agnostic — cloned into
any TTS model later, never tied to one — and kept distinct in the URL space
(`/v1/audio/prep/clips` for uploads, `/v1/audio/samples` for the cleaned
result) so the two "sample" words never collide in one response.

### 8.1 Pipeline

1. **Transcribe** (`POST /v1/audio/prep/clips/{id}/transcribe`): word-level
   timestamps plus speaker diarization, so a caller can pick a word span or
   a speaker before cropping. Cached as the clip's `transcript.json`;
   `GET .../transcript` reads the cache without re-running anything (§2.5:
   a `GET` never does work), 409 if it has never run.
2. **Crop** (`POST /v1/audio/samples`, part of building a sample): a
   `start`/`end` in seconds, or a diarized `speaker` index resolved against
   the cached transcript's speaker spans.
3. **Isolate**: source separation, voice from music/noise —
   `source-separation-spleeter-2stems-int8` (Deezer's Spleeter, MIT),
   keeping the vocals stem. The `sherpa-onnx` crate (1.13.6) does not wrap
   this C API, but the prebuilt static lib it links already exports the
   symbols (`libsherpa-onnx-c-api.a`, checked with `nm -g` on both the
   macOS arm64 and Linux x64 v1.13.6 static libs); `src/prep/isolate.rs`
   is a hand-written `extern "C"` binding against `sherpa-onnx/c-api/
   c-api.h` (tag v1.13.6) plus a safe wrapper, the same shape as
   `diarize.rs`/`denoise.rs`. **Always pass two input channels, even for
   mono audio**: the C++ (`offline-source-separation-spleeter-impl.h`'s
   `ComputeStft`) calls `exit(-1)` on the whole process, not a recoverable
   error, when `num_channels` is 1 — checked against the real header and
   confirmed on a real pulled model before this shipped. `isolate`
   defaults to `true`, the same as `denoise`/`trim_silence`/`normalize`:
   it is a documented pipeline step, and an unpulled model already gives
   a clear 409 `model_not_pulled` rather than a silent skip — the same
   protection the other stages have, so there is no reason to single
   `isolate` out as off-by-default. `isolate:false` is the explicit skip.
4. **Clean**: denoise (GTCRN, MIT — `speech-denoiser-gtcrn`), trim silence
   (reusing the existing Silero VAD gate, `stt::vad`: the span from the
   first detected speech to the last), normalise (peak, to −1 dBFS).

Pipeline order: crop → isolate → denoise → trim silence → normalise.
Isolate runs first among the cleaning steps because Spleeter's own output
is at 44.1 kHz (`Isolator::output_sample_rate()`), not the pipeline's
16 kHz working rate; `prep::resample_to_16k` brings it back in line
before denoise (GTCRN, 16 kHz) runs.

Each step has its own request flag (`isolate`, `denoise`, `trim_silence`,
`normalize`), default `true` for all four. `isolate`/`denoise`/
`trim_silence` fail with a 409 naming the unpulled model when their flag
is `true` and the model is not pulled (`registry_error(NotPulled)`, the
same error `GET /v1/audio/transcriptions` gives for an unpulled STT
model) — never a silent skip. Setting the flag `false` is the explicit
skip.

#### Step chains (`src/prep/pipeline.rs`, `src/prep/dsp.rs`)

Past the crop, the cleaning is an ordered list of steps, each
`{"type": "...", "enabled": true, ...params}`; every param has a serde
default, unknown types and unknown params are 400s, and a value outside its
range is a 400 `invalid_request` (`param: "steps"`) whose message names it
(`steps[1].amount must be a number in 0..=1 ratio`). One table of ranges
feeds both validation and `GET /v1/audio/prep/steps`, which returns every
type's params (`default`, `min`, `max`, `unit`; `eq` also its `bands` item
params) and the `default_chain`, so a UI builds its controls from the server.
The crop (`start`/`end`/`speaker`) is not a step: the range defines the
sample and is applied first. Steps run on 16 kHz mono `f32`; at most 32
steps, at most 16 EQ bands. All DSP is pure Rust (hand-written biquads,
WSOLA): no ffmpeg or other system dependency.

| type | params (default) | implementation |
|---|---|---|
| `isolate` | `model`, `bleed` 0..1 (0) | Spleeter vocals; `bleed` mixes that fraction of the input back |
| `denoise` | `model`, `amount` 0..1 (1) | GTCRN; `amount` is the wet/dry mix |
| `trim_silence` | `threshold` 0.01..0.99 (0.2, the VAD default), `pad_ms` 0..2000 (0) | Silero spans; keeps `pad_ms` before the first / after the last span, clamped to the buffer |
| `normalize` | `peak_db` (-1) | peak normalise |
| `highpass` | `freq_hz` 20..1000 (80) | 4th-order Butterworth |
| `eq` | `bands: [{kind: peak\|low_shelf\|high_shelf, freq_hz, gain_db, q}]` (none) | RBJ-cookbook biquads in order |
| `deess` | `freq_hz` (6000), `threshold_db` (-30), `ratio` (4) | Linkwitz-Riley split; only the high band is attenuated by a peak-held detector |
| `loudness` | `target_lufs` (-18), `peak_ceiling_db` (-1) | BS.1770-4 integrated loudness gain, then a 5 ms look-ahead limiter so peak <= ceiling (a peaky take can land under target) |
| `compressor` | `threshold_db`, `ratio`, `attack_ms`, `release_ms`, `makeup_db` | feed-forward, dB-domain detector |
| `pitch` | `semitones` -12..12 | WSOLA stretch + cubic resample: duration kept, **formants move with the pitch** (no formant preservation) |
| `speed` | `rate` 0.5..2 | WSOLA, pitch preserved |
| `telephone` | none | tanh saturation, then 300-3400 Hz 4th-order band-pass |
| `reverb` | `room_size`, `wet` 0..1 | Schroeder (4 damped combs, 2 all-passes), level-matched to the dry RMS; output keeps the input length (tail cut) |

`POST /v1/audio/samples` takes an optional `steps` array. When present it
*is* the chain and the four flags (and `denoise_model`/`isolation_model`)
are ignored. When absent the flags build the default chain
`[isolate, denoise, trim_silence, normalize(-1)]`, each carrying its flag as
`enabled`: audio identical to the former fixed chain. The 409
`model_not_pulled` preflight (and the `silero-vad` check) is derived from the
enabled steps only. The sample's JSON gains `steps` (empty for a sample made
before steps existed) and `analysis` (`null` likewise), both persisted in
`meta.json`; `engines`/`warnings` are as before, for the model-backed steps
that ran. An optional `transcript` string is stored as the sample's
`transcript.txt` in place of the clip's words in the range (the studio's
corrected, as-spoken text).

`analysis` is `{integrated_lufs, peak_dbfs, noise_floor_dbfs, speech_secs,
duration_secs}` of the finished audio. `integrated_lufs` is ITU-R BS.1770-4
(K-weighting biquads derived for 16 kHz from the standard's analogue
prototype, 400 ms blocks at 75 % overlap, -70 LUFS absolute and -10 LU
relative gates; a clip under 400 ms is one block). `peak_dbfs` is sample
peak. `noise_floor_dbfs` is the 10th percentile of 50 ms frame RMS levels
(speech has pauses, so the quietest tenth is the background). `speech_secs`
is the sum of Silero spans, `null` if `silero-vad` is not pulled. Digital
silence is `null`, never `-inf`.

`POST /v1/audio/prep/clips/{id}/render` auditions a chain without creating a
sample: body `{start?, end?, speaker?, steps, until?: N, solo?: N}`. No
`until`/`solo`: crop + every enabled step. `until: N`: crop + steps `0..=N`.
`solo: N`: crop + only step `N`. `until` with `solo`, or an index past the
chain, is a 400. Disabled steps are skipped, except that `solo` applies its
step even if disabled (soloing a step means hearing it). The response is an
`audio/wav` body (16-bit, 16 kHz, mono) with the analysis in
`x-naru-loudness-lufs`, `x-naru-peak-dbfs`, `x-naru-noise-floor-dbfs`,
`x-naru-speech-secs` (a null value omits its header) and the whole struct as
JSON in `x-naru-analysis`. 404 for an unknown clip, 409 `model_not_pulled`
for a missing model, as `POST /v1/audio/samples` — but only the steps that
run need theirs.

### 8.2 Storage

```
$NARU_AUDIO_HOME/prep/
  clips/<id>/
    raw.<ext>          verbatim upload
    working.wav        16 kHz mono, afconvert + stt::audio::decode_with
                        (Limits::PREP: 4 h, not the STT endpoints' 10 min)
    transcript.json     { words (+ confidence), speakers, overlaps,
                          stt_model, diarization_model }
    project.json        a saved Sample Studio edit (below), if any
    meta.json
  samples/<id>/
    clean.wav           denoised, trimmed, normalised
    cropped.wav          the same span before cleaning, for A/B
    transcript.txt       the cropped span's words, joined (surfaced as
                          the sample's JSON "transcript" field, null if
                          the file is absent); after §8.5's transcribe,
                          the verbatim spoken text instead
    transcript.json       §8.5's verbatim transcript, once transcribed
    meta.json             source clip, range, speaker, engines + licences, name,
                          steps (the chain), analysis (measurements of clean.wav)
```

Sample Studio (naru task 1576) opens a raw clip of any length. The upload
(`POST /v1/audio/prep/clips`, 4 GiB body cap) streams to `raw.<ext>` and is
never buffered whole; `working.wav` and every re-read of it use
`Limits::PREP`. Further endpoints, all under `/v1/audio/prep/`:

- `GET clips/{id}/peaks?start&end&buckets` — `{sample_rate, duration, start,
  end, buckets, min[], max[]}`, read straight from `working.wav` for the
  range (buckets 1-8192, default 2000), so a browser never downloads a long
  clip to draw it.
- `render` and `POST /v1/audio/samples` take `segments: [{start, end}]`
  instead of `start`/`end`/`speaker`: validated, sorted, merged, spliced in
  order with a 10 ms crossfade, then the step chain runs. The sample's
  `meta.json` keeps `segments`; its `transcript.txt` is the corrected
  transcript, else the words inside the segments.
- `transcript.json`'s `overlaps: [{start, end}]` are the ranges where two
  different speakers' diarized spans overlap (crosstalk); absent in an older
  file it loads as empty. `words[].confidence` is MLX Whisper's per-word
  probability (absent for Parakeet).
- `POST clips/{id}/takes` `{speaker?, segments?, target_min = 10,
  target_max = 20, exclude_overlaps = true}` — the speaker's speech inside
  the kept region, minus overlaps, cut at word gaps into 1-8 s takes, each
  scored in 0-1 (noise floor, clipping, overlap, confidence, pace) by
  `prep::takes`, the best picked until the total reaches `target_min`
  without passing `target_max`. 409 `transcript_not_run` without a
  transcript. Returns `{takes: [{start, end, score, picked, metrics}],
  picked_secs}`.
- `GET|PUT|DELETE clips/{id}/project` and `GET projects`: a project is the
  clip plus its edit (`name, speaker, segments, cuts, exclude_overlaps,
  steps, transcript, takes, created_at, updated_at`), stored as
  `project.json` (404 `no_project`; steps validated; body at most 1 MiB).
  Deleting the clip deletes its project; the raw upload is never altered.

`<id>` is a fresh opaque string ([`prep::new_id`]), not a client-chosen
name — `meta.json`'s `name` is what `PATCH /v1/audio/samples/{id}` renames.
Both model directories load fresh per request, like the existing Silero
VAD gate (`stt::vad::Vad`): tens of megabytes, cheap next to decoding a
whole clip, and voice-prep is not a hot path that needs `ModelManager`'s
residency, load queue or eviction.

### 8.3 Licence policy

Every default engine here is a clean licence: `speaker-diarization-en`
(pyannote segmentation, MIT; 3D-Speaker CAM++ embedding, Apache-2.0),
`speech-denoiser-gtcrn` (GTCRN, MIT) and `source-separation-spleeter-
2stems-int8` (Deezer's Spleeter, MIT) — three new `catalog/*.toml`
entries, `[model] kind = "diarization"` / `"denoise"` / `"separation"`
(new `Kind` variants). A request naming a different, `non_commercial`-
flagged model for any of the three stages (the catalog's existing flag,
reused rather than a new one) gets a `warnings` entry in the sample's
response and `meta.json`, naming the model and its licence — not a
refusal: naru-audio ships no non-commercial model as a *default*, but
still supports one, warned.

### 8.4 Word timestamps: sherpa-onnx's tokens, and mlx-whisper on MLX

`POST /v1/audio/transcriptions` refuses `timestamp_granularities[]=word`
(§2.2) because nothing in naru-audio produced word timestamps at all. This
still holds for the OpenAI-compatible route; voice-prep does not touch it.
Instead, `SttModel` gained `decode_words` (`src/stt.rs`), giving the same
utterance gate as `decode_each` but keeping `OfflineRecognizerResult`'s
per-token `tokens`/`timestamps` (`stt::engine::Recognizer::decode_with_
tokens`) and merging tokens into words at the SentencePiece `▁` boundary
marker (`stt::sherpa::merge_words`). `OfflineWhisperModelConfig::
enable_token_timestamps` does not apply here — that field exists only for
Whisper's attention-based timestamps; NeMo transducer models (Parakeet
TDT) report per-token timestamps from the transducer's own frame
alignment, with nothing to enable. Two backends implement
word timestamps: `sherpa-onnx`'s Parakeet, and the MLX sidecar's
`mlx-whisper` (naru task 1466, `whisper-large-v3-turbo-mlx`). The default
is `Err(SttError::WordTimestampsUnsupported)`, so a request naming the MLX
Parakeet (whose sidecar answer carries no words) still gets a clear 400
naming a model to use instead.

The Whisper path extends the sidecar protocol additively (§5.3,
`protocol.py`): `transcribe` takes an optional `language` hint and
`words: true`; the answer's segments then carry `words`, and it carries
`language`, the one decoded in. The sidecar picks `mlx-whisper` for a model
named `whisper*` (as it picks IndexTTS by name) and loads the pulled
directory with `mlx_whisper.load_models.load_model`. The daemon still gates
with Silero and decodes per utterance; each utterance's word times are
offset by its slice's start. Without a `language` hint the first
utterance's detected language is the hint for the rest. The trait grew two
defaulted methods, `decode_each_in` and `decode_words_in`, taking that hint
and returning the language used, so no other `SttModel` changes. The
response's `language` (`/v1/audio/transcriptions` `verbose_json`, prep's
`Transcript.language`) is the request's, else the detected one, else the
manifest's first entry for a model with no notion of language. Prep's
transcribe body takes an optional `language` to pass through. The cost is
`mlx-whisper`'s dependency on `torch` (its tokenizer and timing code
import it), which enlarges the sidecar's venv.

### 8.5 Verbatim sample transcript (naru task 1569)

A clone prompt's text must match its audio, fillers and all, so
`POST /v1/audio/samples/{id}/transcribe` transcribes the sample's
`clean.wav` verbatim. Body (all optional): `{"stt_model", "language",
"verbatim": true}`. It writes `samples/<id>/transcript.json` and replaces
`transcript.txt` with `spoken_text`, so the clone screen's prefill is the
spoken form. The answer, and `GET /v1/audio/samples/{id}/transcript` (409
`transcript_not_run` before the first POST; 404 `sample_not_found`):

```
{"words": [{"start", "end", "text", "spoken", "differs", "filler"}],
 "text": "...", "spoken_text": "...", "language", "stt_model",
 "verbatim": true, "verbatim_supported": true}
```

`text` is the words as written, `spoken_text` as spoken. `differs` is a
word whose spoken form is not its written one (case and punctuation
ignored: "42." is "forty-two."), `filler` an um/uh/er/ah/hmm/mm-type
word. The model must report word timestamps (§8.4), as for a clip.

*Decoding.* The sidecar's `transcribe` takes `"verbatim": true`
(`protocol.py`): `mlx-whisper` gets a filler-rich `initial_prompt` ("Um,
uh, so, I- I mean, like, you know, hmm. Uh-huh."), which Whisper continues
in the style of, and `condition_on_previous_text=False`, which stops a
repetition loop feeding itself. No token suppression is touched, so
fillers are not removed. `SttModel::decode_words_verbatim` (default: a
normal decode, answered `false`) carries it; only the MLX Whisper
overrides it. A model that cannot (sherpa Parakeet, which already emits
what it hears, without the cleaning Whisper applies, but cannot be told to)
answers `verbatim_supported: false` and the request is not refused. This
is a bias, not a guarantee: Whisper may still drop a filler.

*Spoken form.* `stt::spoken` converts per word, leaving leading and
trailing punctuation outside: integers and `,`-grouped numbers to
cardinals ("1,000" "one thousand"); a bare 1100..=2099 as a year ("1999"
"nineteen ninety-nine", "2025" "twenty twenty-five", 2000..=2009 "two
thousand [five]"); decimals digit by digit after "point" ("2.5" "two point
five"); versions, three or more dotted parts or any `v`-prefixed ("v2.1"
"version two point one", "3.0.1" "three point oh point one", a part of
exactly 0 being "oh"); ordinals ("3rd" "third"); `%` `$` `&` `+` `@` (also
"$2.50" "two dollars and fifty cents", "50%", "@name", "-5") and ranges
("10-20" "ten to twenty"). Anything else, such as "AT&T", "3D" or "3:30",
is left as written. The year rule is a guess: "1500" the quantity reads
"fifteen hundred".
