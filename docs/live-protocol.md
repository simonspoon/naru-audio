# `GET /v1/audio/live` — the live audio socket (naru task 2029)

One WebSocket carries a whole live conversation's audio. The client captures the microphone, plays what it is
sent, and forwards hotkeys and gestures as messages. The server owns the mode, the gates, the VAD, the speaker
guard, speech-to-text and speech.

**Status: inert.** Nothing calls this route until the clients switch (the follow-up "switch" task); naru-mac and
naru-ios keep their own copies of this logic until then. The route is new, so no existing route changes.

It is built on the transcription stream (`docs/design.md` §2.4): the same Silero segmenter and decode threads, the
same frame rules, the same close codes. The state machine is `src/live_mode.rs` (`docs/live-mode-spec.md`), the
speaker guard `src/speaker.rs`, and `speak` goes through the `POST /v1/audio/speech` handler itself. The code is
`src/server/live.rs`; the tests are `tests/live.rs`.

## Handshake

A plain `GET` with `Upgrade: websocket`. The Origin and Host guards of every route apply (§2.1); a foreign
`Origin` is refused before the upgrade. A request without the upgrade is `400 invalid_request`.

The first frame must be a text `start`; anything else is a fatal `invalid_request` (1008).

```json
{"type":"start","mode":"live","rate":16000,"session":"mac-1","model":"default","partials":true,
 "hotwords":"Naru, Kokoro","vad":{"threshold":0.5},"keep_alive":"5m"}
```

| Field | | |
| --- | --- | --- |
| `mode` | required unless resuming | `idle`, `live`, `muted` or `ambient` (the spec's mode table) |
| `rate` | required | the microphone's sample rate: exactly `16000`; resample on the client (`unsupported_value`) |
| `format` | `s16le` | the only microphone format (`unsupported_value` otherwise) |
| `session` | server-made | a client id, 1 to 128 characters; reconnecting with it resumes (below) |
| `model` | `default` | the STT model, as in §2.4 |
| `partials` | `true` | send `partial` events while an utterance is open |
| `hotwords`, `vad`, `keep_alive` | | exactly as §2.4's `start` |

The server answers `ready`, then the first `state`. A model that is not pulled or will not load is the §2.6
error, then a close. `loading` precedes `ready` when the model is not resident.

```json
{"type":"ready","session":"mac-1","model":"parakeet-tdt-0.6b-v2-int8","load_ms":0,"resumed":false}
{"type":"state","mode":"live","seq":1,"agent_speaking":false,"enrolled":false,"enrolling":false,
 "speaking":[],"resumed":false}
```

## Client messages

Binary frames are microphone audio: s16le mono 16 kHz, at most 64 KiB, a whole number of samples. Text frames:

| Message | Meaning |
| --- | --- |
| `{"type":"mode","mode":"muted"}` | switch mode. A change cuts the open segment, then emits `state`; the same mode is ignored |
| `{"type":"agent_speaking","value":true}` | the client is playing a reply. While true the microphone is shut; on `false` it stays shut for 400 ms more (the reply's tail is in the room), and any speak still being made is cancelled. Emits `state` on a change |
| `{"type":"speak","id":"t1","text":"…","voice":"af_heart","model":"kokoro-v1.0"}` | synthesise `text`; `voice` and `model` are optional and mean what they do for `POST /v1/audio/speech`. `id` is 1 to 128 characters and unique among running speaks. A newer `speak` cancels the older ones |
| `{"type":"enroll","action":"begin"}` | record the microphone (up to 120 s) as the enrollment clip; the VAD hears silence meanwhile |
| `{"type":"enroll","action":"finish"}` | embed the recording (at least 1 s) with `wespeaker-sv-en` and store it, replacing any earlier enrollment |
| `{"type":"enroll","action":"cancel"}` / `"clear"` | drop the recording / delete the stored enrollment |
| `{"type":"flush"}` | close the open segment now |
| `{"type":"stop"}` | finish: every pending `utterance`, then `done`, then close 1000 |

## Server events

| Event | Fields |
| --- | --- |
| `state` | `mode`, `seq`, `agent_speaking`, `enrolled`, `enrolling`, `speaking` (running speak ids), `resumed`, and on a resume `previous_mode`, `lost_speaks`. `seq` rises by one on every `state`, so a client can spot a gap |
| `level` | `rms` (the loudest frame since the last event, 0 to 1), `gate` (whether the microphone is reaching the VAD). At most every 100 ms, only while frames arrive |
| `speech` | `active`, `at`: the VAD's edges, as §2.4 |
| `partial` | `segment`, `start`, `text`: the open utterance so far. Only in `live` mode and while the agent is not speaking. An empty `text` clears a partial whose segment was then dropped |
| `utterance` | `id` (0, 1, 2… per session), `segment`, `text`, `start`, `end` (seconds since `start`), `decode_ms`, and `speaker` (`score`, `is_him`, `reason`) when an enrollment exists |
| `ambient_flag` | the same fields as `utterance`, for speech heard in `ambient` mode. The server only flags it; no ambient classification exists yet (next step) |
| `speak_audio` | `id`, `seq` (from 0), `rate`, `final`, `data` (base64 s16le mono, at most 16 KiB raw per chunk). The last event of an id has `final: true` and empty `data`; `cancelled: true` when it was cut |
| `error` | `code`, `message`, `fatal`, and `id` for a speak. `fatal: true` is followed by the close; `false` leaves the session open |
| `done` | after `stop`, before the 1000 close |
| `loading`, `warning` | as §2.4 |

## What the gates do

The microphone reaches the VAD only when the mode's gate is open: `live` — live, joined, not muted, the agent not
speaking and out of the echo hold (`LiveState::mic_open`); `ambient` — the agent not speaking; `muted` and `idle`
never. A shut gate feeds the VAD **silence of the same length**, not nothing, so its clock and any open segment
carry on. A finished segment then goes through `LiveState::segment_fate`: in `live` it must also pass the speaker
guard (`speaker::verify`). The guard **fails open** — no enrollment, a clip under 1 s, or the model not pulled is
"him" — so a short "yes" is never dropped. A dropped segment sends nothing but a clearing `partial`.

## Reconnect and resync

The server keeps the last `state` of every session id for 60 seconds after an unclean end (a dropped socket, a
protocol failure, a server-side error). A `start` with that `session` resumes it: `ready.resumed` and
`state.resumed` are `true`, `seq` carries on, `previous_mode` is the mode when it dropped, and `mode` may be
omitted to keep it (or given to change it). `lost_speaks` lists the speaks that were still running — they were
cancelled with the socket, so the client re-sends the ones it wants. The server's `state` comes before any audio
is accepted. A clean `stop` forgets the session, as does a restart of the daemon or the 60 seconds running out;
`resumed` is then `false` and `seq` starts at 1.

The enrollment is the stored one on disk, so it survives both.

## Errors and close codes

The close codes of §2.4: 1000 after `done`; 1008 for a protocol violation (`invalid_request` — an unreadable or
unknown message, a bad frame, a second `start`, a bad `mode`; `unsupported_value` — `rate`, `format`); 1011 for a
decode or load failure; 1013 for `backlog` (more than 30 s of undecoded audio) and `insufficient_memory`. Each is
preceded by an `error` event with `fatal: true`.

These leave the session open (`fatal: false`): an `enroll` out of order or too short, a missing speaker model, a
`speak` the speech handler refuses (unknown model or voice, text over its cap, a failed synthesis).

## Limits

Frame 64 KiB; message 1 MiB; 30 s of undecoded audio; enrollment 120 s; one running speak (a new one cancels the
last); the resync table lives in the daemon's memory.
