# Ambient decision gate and state machine (naru task 2031)

Ported from naru-ios (`NaruAmbientGate/InternDecisionGate.swift`, `NaruKit/AmbientMachine.swift`) so the
server can own ambient mode. **Inert**: nothing in the server calls it yet, and naru-ios keeps its own
copy until the switch task. No API key; everything runs locally.

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
- Not wired into `server/live.rs`; the `ambient_verdict` socket event is left for the switch task.

## Tests

`tests/ambient.rs` ports the scenarios of `AmbientMachineTests.swift` (wake, custom wake words, sound-alike,
echo tail, silence return, gate routing). `src/ambient/intern.rs` tests the prompt, escaping, calibration and
the 0.9 threshold edge. There is no Swift route test or fixture file to diff against, so parity is by
the ported scenarios, not a shared fixture. The model-backed path is not run in CI.
