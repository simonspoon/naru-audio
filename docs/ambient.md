# Ambient decision gate and state machine (naru task 2031)

Ported from naru-ios (`NaruAmbientGate/InternDecisionGate.swift`, `NaruKit/AmbientMachine.swift`) so the
server can own ambient mode. Wired into the live socket (naru task 2208, below); the clients switch over in
their own tasks (naru-mac 2057). No API key; everything runs locally.

## Modules (`src/ambient/`)

- `machine.rs`: pure, no I/O, no clock (`now_ms` is passed in), like `live_mode.rs`. `TranscriptLog` (2 min /
  500 lines), `phrases` (wake word, custom wake words, sound-alike with the phonetic edit distance, end phrase)
  and `AmbientMachine` (`ambient` <-> `conversation`, 120 s silence return, 30 s event context, 10 s
  continuation of a short untagged line, 1 s echo tail after the agent speaks). Swift's `await gate.evaluate`
  is a `GateRequest` in `Ingest`; run the gate off the async runtime and pass the verdict to `gate_event`, or
  use `ingest_with` with a synchronous gate.
- `intern.rs`: Intern-Decision-4B Q4_K_M, prompt `g1`. The chunk (cut to 4000 characters) is rendered as
  serde_json-pretty JSON in the ChatML prompt, one decode step reads the logits of `A`/`B`/`C` at the last
  `<decision>` marker, divided by temperature 1.99241824 and softmaxed to (idea, can_help, ignore). Flag when
  `1 - P(ignore) > 0.9`; idea when `P(idea) >= P(can_help)`, else can_help. Any failure is ignore, score 0.
- `llama.rs` (cargo feature `llama`, off by default): the model on its own worker thread behind a channel.
  Build with `cargo build --features llama` (needs cmake; Metal on macOS).

`ambient::load_gate(Some(path))` fails open: no path, no `llama` feature, a missing file or a load error logs
and returns `NullGate`, so the server always starts.

## Not done

- No catalog entry: `models pull` needs the GGUF's sha256 and size and a new `Kind`/backend, neither of which
  is available without downloading the file. Point `load_gate` at a local GGUF for now.
- The server reports a wake; it does not switch its own mode. The client decides whether to go `live`.
- The machine's `mode_changed` events and wake candidates are not sent; the gate path that needs a model has
  no socket test (`NullGate` never flags).

## In the live socket (`server/live.rs`, naru task 2208)

Each audio socket has its own `AmbientMachine`, on a clock of milliseconds since the session began. In `ambient`
mode every admitted segment (the one that becomes an `ambient_flag`) is also ingested:

- **Speaker.** With a voice library the segment is scored against every person in ambient too (whether or not the live
  guard is on): the owner wins whenever their score clears the threshold, else the best match at or above it: the owner is `Him`, another named person is `Other` (the
  line carries the name), scored and matching nobody is an unnamed `Other`; no enrollment, a clip under 1 s or no model is `Unknown` (which may still wake, as
  the machine allows).
- **Wake words.** The built-in spellings plus `ambient.wake_words` from naru's config file, read once at session
  start: `NARU_AUDIO_WAKE_CONFIG`, else `~/.naru/config.json` when `~/.naru` exists, else `~/.mesa/config.json`.
  A missing or unreadable file means no custom words.
- **Gate.** `ambient::load_gate` on the GGUF named by `NARU_AUDIO_AMBIENT_GATE`, loaded once per process on first
  use; unset or failing, it is `NullGate`. It runs under `spawn_blocking`.
- **Events.** A wake is a `wake` event, a flagged gate verdict an `ambient_verdict` (`docs/live-protocol.md`). In
  a conversation the socket sends its own and every other socket of the conversation gets the same one.
- Entering `ambient` from another mode ends the machine's conversation, so the next wake word wakes again.

## Tests

`tests/live.rs` checks the socket: one `wake` for `wake.wav`, none for `plain.wav`, and the same `wake` on a
second socket of the conversation. `tests/ambient.rs` ports the scenarios of `AmbientMachineTests.swift` (wake, custom wake words, sound-alike,
echo tail, silence return, gate routing). `src/ambient/intern.rs` tests the prompt, escaping, calibration and
the 0.9 threshold edge. There is no Swift route test or fixture file to diff against, so parity is by
the ported scenarios, not a shared fixture. The model-backed path is not run in CI.
