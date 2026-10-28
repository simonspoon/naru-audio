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
| `mode` | required unless resuming or joining a conversation | `idle`, `live`, `muted` or `ambient` (the spec's mode table) |
| `rate` | required for `audio` | the microphone's sample rate: exactly `16000`; resample on the client (`unsupported_value`) |
| `role` | `audio` | `audio`, or `control`: mode messages only, no microphone (Conversations, below) |
| `conversation` | none | a key, 1 to 128 characters; every socket with the same key shares one mode and `seq` (below) |
| `format` | `s16le` | the only microphone format (`unsupported_value` otherwise) |
| `session` | server-made | a client id, 1 to 128 characters; reconnecting with it resumes (below) |
| `model` | `default` | the STT model, as in §2.4 |
| `partials` | `true` | send `partial` events while an utterance is open |
| `hotwords`, `vad`, `keep_alive` | | exactly as §2.4's `start` |
| `typed_frames` | `false` | binary frames lead with a type byte (below); `ready` echoes it |

The server answers `ready`, then the first `state`. A model that is not pulled or will not load is the §2.6
error, then a close. `loading` precedes `ready` when the model is not resident.

```json
{"type":"ready","session":"mac-1","model":"parakeet-tdt-0.6b-v2-int8","load_ms":0,"resumed":false,"typed_frames":false}
{"type":"state","mode":"live","seq":1,"agent_speaking":false,"enrolled":false,"enrolling":false,
 "speaking":[],"resumed":false}
```

## Client messages

Binary frames are microphone audio: s16le mono 16 kHz, at most 64 KiB, a whole number of samples.
With `typed_frames: true` the first byte of every binary frame is its type and the rest is s16le mono 16 kHz
(the 64 KiB limit counts the whole frame, the whole-sample rule the payload):

| Type | Meaning |
| --- | --- |
| `0x00` | microphone audio, exactly as an untyped frame |
| `0x01` | speaker reference: what the client is playing. Never heard by the VAD or recorded for enrollment |
| other, or an empty frame | fatal `invalid_request` (1008) |

**Echo guard** (naru task 2030). The client's own voice processing stays its first defence; this is the second.
Stream the reference as it plays, about when the microphone could hear it. For each microphone frame the server
correlates the last 200 ms of microphone audio against the reference over lags of 0 to 400 ms (normalised
cross-correlation, 4 kHz decimated); at 0.5 or more it is the agent's own voice and the VAD is fed silence for that
frame, as a shut gate does. A finished segment is checked once more the same way and dropped (a clearing `partial`,
no `utterance`) if it is the reference heard back. The guard needs correlation, never loudness alone, and fails open:
no reference, one older than 600 ms, under 100 ms of microphone audio, or a quiet signal pass unchanged, so a short
"yes" over the agent's voice is kept. Untyped sessions have no reference and are unchanged. Frames that arrive faster
than real time are placed back to back, so send the reference at playback pace.

Text frames:

| Message | Meaning |
| --- | --- |
| `{"type":"mode","mode":"muted"}` | switch mode. A change cuts the open segment, then emits `state`; the same mode is ignored |
| `{"type":"agent_speaking","value":true}` | the client is playing a reply. While true the microphone is shut; on `false` it stays shut for 400 ms more (the reply's tail is in the room), and any speak still being made is cancelled. Emits `state` on a change |
| `{"type":"speak","id":"t1","text":"…","voice":"af_heart","model":"kokoro-v1.0"}` | synthesise `text`; `voice` and `model` are optional and mean what they do for `POST /v1/audio/speech`. `id` is 1 to 128 characters and unique among running speaks. A newer `speak` cancels the older ones |
| `{"type":"enroll","action":"begin","person":<id>}` | record the microphone (up to 120 s) as a clip for `person` (optional; the owner, created as "Me" if there is none); the VAD hears silence meanwhile |
| `{"type":"enroll","action":"finish"}` | level-normalize the recording (at least 1 s; a too-faint one is a soft `clip_too_quiet` error) and embed it with `campplus-sv-en`, then ADD it, with its audio, to that person's clips |
| `{"type":"enroll","action":"cancel"}` / `"clear"` | drop the recording / delete the owner's clips |
| `{"type":"flush"}` | close the open segment now |
| `{"type":"stop"}` | finish: every pending `utterance`, then `done`, then close 1000 |

## Server events

| Event | Fields |
| --- | --- |
| `state` | `mode`, `seq`, `agent_speaking`, `enrolled` (the owner has clips), `enrolling`, `live_guard`, `speaking` (running speak ids), `resumed`, and on a resume `previous_mode`, `lost_speaks`. `seq` rises by one on every `state`, so a client can spot a gap. In a conversation `seq` is the conversation's, shared by every socket in it |
| `level` | `rms` (the loudest frame since the last event, 0 to 1), `gate` (whether the microphone is reaching the VAD). At most every 100 ms, only while frames arrive. Sent to every socket of the conversation |
| `speech` | `active`, `at`: the VAD's edges, as §2.4 |
| `partial` | `segment`, `start`, `text`: the open utterance so far. Only in `live` mode and while the agent is not speaking. An empty `text` clears a partial whose segment was then dropped |
| `utterance` | `id` (0, 1, 2… per session), `segment`, `text`, `start`, `end` (seconds since `start`), `decode_ms`, and `speaker` (`score`, `is_him`, `reason`, `person`: `{id, name}` of the best match at or above the threshold, else `null`) when the library has clips |
| `ambient_flag` | the same fields as `utterance`, for every segment heard in `ambient` mode (the speaker guard scores it too when an enrollment exists). The `wake` or `ambient_verdict` it leads to, if any, follows it |
| `wake` | `text` (the line that woke), `remainder` (the words after the wake word, or `null`), `at_ms` (ms since the session began), `context` (the last ~30 s before it, oldest first: `speaker` `him`/`other`/`unknown`, `person` (the name, only on an `other` line matched to a named person), `text`, `at_ms`). The wake word was heard in `ambient` mode (Ambient, below). Sent to every socket of the conversation |
| `ambient_verdict` | `kind` (`idea` or `can_help`), `score` (0 to 1: 1 − P(ignore), above 0.9), `text`, `at_ms`, `context` as `wake`. The ambient gate flagged a line of his. Sent to every socket of the conversation |
| `speak_audio` | `id`, `seq` (from 0), `rate`, `final`, `data` (base64 s16le mono, at most 16 KiB raw per chunk). The last event of an id has `final: true` and empty `data`; `cancelled: true` when it was cut |
| `error` | `code`, `message`, `fatal`, and `id` for a speak. `fatal: true` is followed by the close; `false` leaves the session open |
| `done` | after `stop`, before the 1000 close |
| `loading`, `warning` | as §2.4 |

## What the gates do

The microphone reaches the VAD only when the mode's gate is open: `live` — live, joined, not muted, the agent not
speaking and out of the echo hold (`LiveState::mic_open`); `ambient` — the agent not speaking; `muted` and `idle`
never. A shut gate feeds the VAD **silence of the same length**, not nothing, so its clock and any open segment
carry on. A finished segment then goes through `LiveState::segment_fate`: in `live` it must also pass the speaker
guard (`speaker::identify`), which is off unless `live_guard` is set (`docs/live-mode-spec.md`, Speaker guard). The guard
**fails open** — no owner clips, a segment under 2 s, or the model not pulled is "him" — so a short "yes" is never dropped. A dropped segment sends nothing but a clearing `partial`.

## Ambient (naru task 2208)

In `ambient` mode each segment also goes through the ambient machine and gate (`src/ambient/`, `docs/ambient.md`).
The server only reports; it never changes the mode itself, so a client that wants to talk after a `wake` sends
`{"type":"mode","mode":"live"}`.

- **Wake.** A line that opens with the wake word — the built-in "Naru" spellings (`neru`, `niru`, …), a bare
  sound-alike such as "Hey Naro", or one of naru's custom `ambient.wake_words` — is a `wake`. Custom words are read
  at session start from `NARU_AUDIO_WAKE_CONFIG`, else `~/.naru/config.json` (when `~/.naru` exists), else
  `~/.mesa/config.json`. A line tagged as another voice by the speaker guard never wakes; an untagged one (no
  enrollment, under 1 s) may. After a wake the machine is in its conversation and sends no second `wake` until it
  returns: "thanks, that's all", 120 s without his voice, or the socket entering `ambient` again from another mode.
- **Verdict.** Outside that conversation a line of his (tagged by the speaker guard, so an enrollment is needed) is
  judged by the gate, the model named by `NARU_AUDIO_AMBIENT_GATE` (a GGUF; needs the `llama` build). Unset or
  failing to load, nothing is ever flagged. A flagged line is an `ambient_verdict`.
- **Conversations.** The socket that heard it sends its own `wake` or `ambient_verdict`; every other socket of its
  conversation, control sockets included, gets the same event once.
- **The server's own listener** (naru task 2372, `POST /ambient`, `docs/ambient.md`) is a member of a conversation
  with the local mic in place of a socket. Every socket of that conversation gets its `ambient_flag`, `utterance`,
  `wake`, `ambient_verdict` and `level` events (an `ambient_flag` or `utterance` from it is the only kind a socket
  receives that it did not hear itself). It follows the conversation's mode, and its microphone is shut while any
  audio socket of the conversation has `agent_speaking` true, and 400 ms after.

## Reconnect and resync

The server keeps the last `state` of every session id for 60 seconds after an unclean end (a dropped socket, a
protocol failure, a server-side error). A `start` with that `session` resumes it: `ready.resumed` and
`state.resumed` are `true`, `seq` carries on, `previous_mode` is the mode when it dropped, and `mode` may be
omitted to keep it (or given to change it). `lost_speaks` lists the speaks that were still running — they were
cancelled with the socket, so the client re-sends the ones it wants. The server's `state` comes before any audio
is accepted. A clean `stop` forgets the session, as does a restart of the daemon or the 60 seconds running out;
`resumed` is then `false` and `seq` starts at 1.

The enrollment is the stored one on disk, so it survives both.

## Conversations (naru task 2178)

The mode — above all the microphone mute — has one owner, the server. Every socket whose `start` names the same
`conversation` shares **one mode and one `seq`**; the page, the Mac app and a headset only send intents, and the
server decides and tells them all. Without `conversation` a socket is alone, exactly as above.

- **The intent** is the ordinary `{"type":"mode","mode":"muted"}` (or back to `"live"`, or any mode), from any
  socket of the conversation. A change bumps the conversation's `seq` and sends **one `state` to every socket in
  it**, the sender included (the sender gets no second one). Changing to the mode it already has is ignored. On an
  audio socket a change acts exactly as its own `mode` message would: the open segment is cut and the gate follows
  the new mode, so a mute from the page really shuts the microphone.
- An audio socket's `agent_speaking` or `enroll` change also bumps the shared `seq` and sends every socket a
  `state`; each socket's `agent_speaking`, `enrolled`, `enrolling` and `speaking` are still its own.
- **Joining.** The first socket's `mode` begins the conversation at `seq` 1 (a first socket without `mode` is
  `invalid_request`). A socket that joins later takes the current mode and `seq` as its first `state`, whatever
  its own `start.mode` says, so a reloaded page sees the truth. With `session` resume as well, the conversation's
  mode and `seq` win over the resumed ones.
- **A gap in `seq`** means the socket fell behind; the `state` it got is the current one, so take it as the
  truth.
- **Lifetime.** A conversation lives while it has a socket. When the last one ends (stop, drop or error) it is
  forgotten, and the next socket to name it begins it afresh at `seq` 1. It is not part of the 60-second resume.

**`role: "control"`** is a socket with no microphone, model or VAD, for a client that only shows or sets the mute
(the page through naru serve's proxy). It needs `conversation`; `rate`, `model`, `vad` and the like are not
needed. It is answered by

```json
{"type":"ready","session":"page-1","role":"control","conversation":"c1","resumed":false}
{"type":"state","mode":"live","seq":4,"agent_speaking":false,"enrolled":false,"enrolling":false,
 "speaking":[],"resumed":false}
```

and then gets every `state` of the conversation, and every `wake`, `ambient_verdict` and `level`. It accepts only `mode` and `stop` (`done`, then close 1000).
Binary frames, `agent_speaking`, `speak`, `enroll`, `flush` and a second `start` are fatal `invalid_request`
(1008). Its `state` always has `agent_speaking`, `enrolled` and `enrolling` false and `speaking` empty.

## Errors and close codes

The close codes of §2.4: 1000 after `done`; 1008 for a protocol violation (`invalid_request` — an unreadable or
unknown message, a bad frame, a second `start`, a bad `mode`, a bad `role` or `conversation`, an audio `start`
without `rate`, a message a control socket does not take; `unsupported_value` — `rate`, `format`); 1011 for a
decode or load failure; 1013 for `backlog` (more than 30 s of undecoded audio) and `insufficient_memory`. Each is
preceded by an `error` event with `fatal: true`.

These leave the session open (`fatal: false`): an `enroll` out of order or too short, a missing speaker model, a
`speak` the speech handler refuses (unknown model or voice, text over its cap, a failed synthesis).

## Limits

Frame 64 KiB; message 1 MiB; 30 s of undecoded audio; enrollment 120 s; one running speak (a new one cancels the
last); the resync table lives in the daemon's memory.
