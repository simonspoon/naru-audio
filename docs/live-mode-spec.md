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

Scoring lives in `src/speaker.rs` (naru tasks 2028, 2260, 2354), not in the state machine. The `campplus-sv-en`
catalog model (3D-Speaker CAM++ via sherpa-onnx, `kind = "speaker"`; `wespeaker-sv-en` until task 2354) embeds 16 kHz
mono audio. CAM++ pools in 2 s segments and mis-scores a short last one, so each clip is tiled (repeated from its
start) to a whole number of 2 s segments before it is embedded. The voice library is
`$NARU_AUDIO_HOME/speaker/people.json`: `{model, live_guard, people: [{id, name, owner, clips: [{id, created_at,
seconds, embedding, has_audio}]}]}`, tagged with the model name (embeddings of different models are never compared).
A library made with another model fails open until the current model is loaded (a live segment to score, a `verify`,
or a new clip), which re-embeds every clip from its kept WAV and drops a clip that has no audio. Ids are integers. Each clip's audio is kept as a 16 kHz mono WAV in
`speaker/clips/<clip id>.wav` (`has_audio`). At most one person is the owner: "him". When `people.json` is absent, an
old `speaker/enrollment.json` (else `enrollment.json.off-20261022`) is migrated into an owner named "Me" with one clip
(`has_audio: false`, `seconds: 0`); the old file is left in place.

A person's score is the cosine of the clip to the L2-normalised mean of their clips' embeddings (max-over-clips scored
a 0.333 leave-one-out margin on the fixtures, the mean 0.336). A clip matches the owner when their score is at or above the threshold (even if someone else scores higher), else the best-scoring person at or above it (`DEFAULT_THRESHOLD` = 0.35, overridable per `verify` request). On real
read speech (mini LibriSpeech dev-clean-2, 26 speakers, two enrollment clips each, test clips cut to 1.5-8 s) no
same-voice clip of 2.5 s or more scored under 0.35 (2.6% at 1.5 s) while 3-7% of other voices cleared it; WeSpeaker at
0.7 dropped 3.5% of same-voice 2.5 s clips. On the TTS fixtures in `tests/fixtures/speaker/` the same voice scores
0.90 or more and a different voice under 0.19. Every `live_segment_dropped` log line carries `score` (`-` when not
scored) and `threshold`, and every admitted segment logs the same as `live_segment_admitted` at `debug`, so the bar
can be re-measured on a real enrollment.

**Fails open** (short replies such as "yes" or a number must never be dropped): no people with clips, a model that is not
pulled, a mismatched library, or a clip under 1 s (`MIN_SAMPLES`) all answer `is_him: true`, with `reason`
`no_enrollment`, `model_unavailable`, `model_mismatch` or `too_short` (`scored` otherwise).

The **live guard** is a separate switch, `live_guard` (default off). It drops a segment in `live` mode only when
`live_guard` is true, the owner has at least one usable clip, the segment is at least 2.0 s long, and the match
is not the owner (another person, or nobody). Shorter segments are always admitted; with the switch off `live` never
drops a segment for speaker reasons. Scoring still runs for the `speaker` tag. A running session reloads
`people.json` when it changes, so library edits and the switch apply without a restart. Ambient tagging scores every
person regardless of the switch: the owner is `Him`, another named person is `Other` with the person's name, a scored
line matching nobody is `Other` unnamed, an unscored one is `Unknown`.

The machine's side is `LiveState::segment_fate(other_voice, is_him)`: `Admit` iff `segment_goes_to_guard` and the
guard said him, else `Drop`.

Routes (opt-in, nothing calls them until the switch). A request that needs the model while it is not pulled is 409
`model_not_pulled` (§2.6); an unknown person or clip is 404 `not_found`. JSON bodies unless noted; clips are multipart
like `/v1/audio/transcriptions`.

| Route | |
|---|---|
| `GET /v1/audio/speaker/people` | `{model, live_guard, people: [{id, name, owner, clips: [{id, created_at, seconds, has_audio}]}]}`; no embeddings |
| `POST /v1/audio/speaker/people` | `{name, owner?}` makes a person (the same shape as one list entry) |
| `PATCH /v1/audio/speaker/people/{id}` | `{name?, owner?}`; `owner: true` clears it on the others |
| `DELETE /v1/audio/speaker/people/{id}` | `{deleted: true}`; removes its clips' WAVs |
| `POST /v1/audio/speaker/people/{id}/clips` | multipart `file` WAV; normalizes its speech to -20 dBFS (peak held under -1 dBFS), then embeds and keeps it; answers the clip `{id, created_at, seconds, has_audio}`; 400 under 1 s; 400 `clip_too_quiet` (re-record) when it needs over +24 dB or its speech is under 10 dB above its noise floor (naru task 2413) |
| `GET /v1/audio/speaker/people/{id}/clips/{clip}/audio` | `audio/wav`; 404 when the clip has no audio |
| `DELETE /v1/audio/speaker/people/{id}/clips/{clip}` | `{deleted: true}` |
| `GET` / `PUT /v1/audio/speaker/guard` | `{live: bool}` (PUT takes the same body) |

Legacy single-voice routes, same response shapes, now over the owner: `POST /v1/audio/speaker/enroll` (one or more
`file` clips replace the owner's clips, creating the owner "Me" if there is none, and keep their audio; 400 when none is
1 s or longer; answers `{model, count, dim}`), `POST /v1/audio/speaker/verify` (`file`, optional `threshold`; scored
against the owner only; answers `{score, is_him, reason, threshold}`), `DELETE /v1/audio/speaker/enrollment` (removes
the owner's clips). A verify with no enrolled owner, or a clip too short to score, never needs the model.
