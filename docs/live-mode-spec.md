# Live mode state machine (spec)

Extracted from naru-mac `Sources/NaruMac/NativeLive.swift` (+ `SystemUnmuteGuard.swift`) for naru task 2027, the
first step of making naru-audio the audio server with naru-mac/naru-ios as thin clients. Implemented as pure Rust in
`src/live_mode.rs` with unit tests. **No client is wired to it yet**: `GET /v1/audio/live` (naru task 2029, `docs/live-protocol.md`) now drives it over a socket, but naru-mac keeps its own copy until the
switch task, so nothing existing changes. Swift test cases (`PauseTests`, `SegmentGateTests`,
`SystemUnmuteGuardTests`) are ported one for one.

Time is never read inside the machine: every window takes `now_ms` (any monotonic millisecond origin).

## Inputs

| Input | Source | Notes |
| --- | --- | --- |
| `live` | page `state` | a live session exists |
| `joined` | page `state` | this client is in the conversation |
| `paused` | page `state` (absent = false) | Pause (task 1786) |
| `speech_muted` | page `state` (absent = false) | speaker mute (task 1787) |
| `muted` | `mic` message / media keys / system stem | microphone mute |
| `ambient` | `setAmbient` | only honoured while `live` |
| `speaking_turn` | run loop | the reply currently sounding |
| `other_voice` | derived | another client holds the voice with an unplayed reply, or finished one < 1 s ago |
| `vad_speaking` | VAD | rising edge records `speech_started_open` |

## Modes

`mode()` derives one of four; pause and speaker-mute are **overlays**, not modes.

| live | ambient | joined | muted | mode |
| --- | --- | --- | --- | --- |
| no | – | – | – | idle |
| yes | yes | – | – | ambient |
| yes | no | no | – | idle |
| yes | no | yes | yes | muted |
| yes | no | yes | no | live |

`wants_active = live && joined && !ambient`. The pipeline (session, speaker claim, mute) is held exactly while this
is true. **Pause does not enter it**: ambient listens whenever the live pipeline is inactive, so a pause must never
release it. `stop()` (active going false) resets `muted` (a rejoin is never silently muted), `pause_applied`,
the speaking turn, VAD flag and system-mute bookkeeping.

## Gates

- `admits(active, paused, speaking, other_voice) = active && !paused && !speaking && !other_voice`.
- `admits_segment(open_at_start, ...)`: false unless `active && !paused`; then `open_at_start || (!speaking &&
  !other_voice)`. A reply starting, or another voice's hold, while the person is mid-utterance does not clip it;
  speech that *began* inside a closed gate is dropped.
- `speech_started_open` is set on the VAD rising edge to `admits(...) && !muted`; initial value true.
- A segment reaches the speaker guard iff `admits_segment && !muted`.
- `other_voice`: true while the speaker claim names another client and an assistant turn with text is unplayed
  (each such evaluation pushes the deadline to `now + 1 s`); otherwise `now < deadline`.
- Idle (auto-send) clock runs only while `active && !paused && !muted && no turn playing && !other_voice`, and
  is held at zero while the VAD hears speech.

## Replies

- `should_speak = !speech_muted`; a muted speaker marks the reply played without sounding.
- `counts_as_played(played, cut_by_mute) = played || cut_by_mute`. The cut turn is recorded **at the cut**
  (`speech_cut_turn`) so a quick unmute cannot lose the verdict. Muting the speaker cuts the sounding reply
  only while active; unmuting needs nothing (the run loop speaks the next unplayed reply).

## Pause

`pause_action(paused, applied, speaking, remembered)`: `None` when `paused == applied`.

| Transition | Action |
| --- | --- |
| pause | `cut` (stop voice unstamped, capture, run loop; drop held audio, reset queue/cue), `remember = speaking` |
| resume | `release = remembered` (turn becomes the oldest unplayed again, spoken from its start), `reclaim` the voice |

Pause is reconciled only while `active` (else `pause_applied = false`). A pause that arrives with the join is
already in force: `pause_applied` is set at start, nothing is cut, and resume only reclaims. A paused client does
not claim the speaker (it finishes a pending leave instead).

## Ambient

`set_ambient(on)` is ignored unless `live` and the value changes. On: `leaving_ambient = false`; if active, the
open utterance is cut and sent, the queue gets up to ~4 s to land it, then capture goes down (I/O, caller's job);
else capture is reconciled at once. Off: `leaving_ambient = true`. `page_ambient = ambient || leaving_ambient`: the
page is told ambient until the voice claim after leaving has landed *or failed* (or the claim is skipped because
inactive/paused), so it never speaks too. `live` dropping clears both flags.

## Mute and the system (AirPods stem) mute

`set_muted(m)`: no-op if unchanged. Otherwise it supersedes any pending unmute (`pending`, `frames_expected` cleared,
fallback generation bumped). With the system mute armed and out of step: muting sets the system mute; unmuting
**defers** it (`system_unmute_pending`) until capture delivers a frame (the chime must not play before capture is up);
if capture is already running (a turn is speaking) `frames_expected` is true and the 1 s fallback is armed at once,
else it is armed when capture reports started (`capture_started_after_unmute`). First frame → unmute the system.
Fallback (generation still current, still pending/expected/unmuted/active/armed, 1 s) → unmute anyway, so the
system is never left muted while the app shows unmuted. Muting also cuts the open utterance and sends it with
what was held (not while paused); unmuting claims the voice if active and not paused.

`system_unmute_swallows(should_mute, muted, speaking, cold_cue_recent, hold_settling, device_changing)`:
true if `device_changing && should_mute != muted`; else `!should_mute && muted && (speaking || cold_cue_recent ||
hold_settling)`. A swallowed report re-asserts the app's state. A report while an unmute awaits its first frame
with `should_mute` true is the held state echoing back and is ignored.

Windows: `hold_settling` = hold engine starting, or finished < 1 s ago; `device_changing` = default input changed
< 2 s ago.

## Not extracted (I/O-entangled, stays in the client until the switch)

AVAudioEngine/hold start-stop sequencing and the audio chain, the queue/transcription flush, claim retry (+2 s) and refresh (5 s), the epoch counter that drops stale async work,
the thinking cue gate, auto-send config load, and the cold-cue timestamp. The epoch is a caller concern: effects
returned by this machine are valid only for the state that produced them.

## Speaker guard

Scoring lives in `src/speaker.rs` (naru task 2028), not in the state machine. The `wespeaker-sv-en` catalog model
(WeSpeaker ResNet34 via sherpa-onnx, `kind = "speaker"`) embeds 16 kHz mono audio; an enrollment is the L2-normalised
mean of several clips' embeddings, stored as `$NARU_AUDIO_HOME/speaker/enrollment.json` tagged with the model name
(embeddings of different models are not comparable, so a mismatched enrollment is ignored). A clip is "him" iff its
cosine score against the enrollment is at least the threshold (`DEFAULT_THRESHOLD` = 0.7, overridable per request).
On the TTS fixtures in `tests/fixtures/speaker/` the same voice scored 0.94-0.99 and a different voice 0.59-0.62;
re-measure against a real enrollment before the switch.

**Fails open** (short replies such as "yes" or a number must never be dropped): no enrollment, a model that is not
pulled, a mismatched enrollment, or a clip under 1 s (`MIN_SAMPLES`) all answer `is_him: true`, with `reason`
`no_enrollment`, `model_unavailable`, `model_mismatch` or `too_short` (`scored` otherwise).

The machine's side is `LiveState::segment_fate(other_voice, is_him)`: `Admit` iff `segment_goes_to_guard` and the
guard said him, else `Drop`.

Routes (opt-in, nothing calls them until the switch; multipart like `/v1/audio/transcriptions`):
`POST /v1/audio/speaker/enroll` (one or more `file` clips, replaces the enrollment; 400 when none is 1 s or longer),
`POST /v1/audio/speaker/verify` (`file`, optional `threshold`; answers `{score, is_him, reason, threshold}`),
`DELETE /v1/audio/speaker/enrollment`. A request that needs the model while it is not pulled is 409
`model_not_pulled` (§2.6); a verify with no enrollment, or a clip too short to score, never needs it.
