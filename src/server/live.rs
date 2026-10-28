//! `GET /v1/audio/live` (naru task 2029): the live conversation's audio over
//! one WebSocket. The client sends microphone frames and a few control
//! messages; the server owns the mode, the gates, the VAD, the speaker guard,
//! speech-to-text and speech. The protocol is `docs/live-protocol.md`.
//!
//! Opt-in: nothing calls this route until the clients switch (naru task
//! 2029's "switch" follow-up), so no existing behaviour changes.
//!
//! It is built on the transcription stream (`stream.rs`): the same Silero
//! segmenter thread and decode thread run the audio, and this module owns
//! what sits around them. Microphone frames the gate refuses are replaced by
//! silence, not dropped, so the VAD's clock and any open segment keep going.
//! A finished segment is then judged by [`crate::live_mode`] and the speaker
//! guard ([`crate::speaker`]) before it becomes an `utterance`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code};
use axum::extract::{Extension, State};
use axum::response::Response;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::task::JoinHandle;

use super::stream::{
    CLOSE_LINGER, Config, Event, Fail, Input, MAX_BACKLOG_SAMPLES, MAX_FRAME_BYTES,
    MAX_MESSAGE_BYTES, RATE, Segments, Start, decode, next, segment, send, validate, violation,
    warn_truncated,
};
use super::transcriptions::{decode_error, internal, kind_manifest, stt_manifest};
use super::{AppState, RequestId, manager_error};
use crate::ambient::machine::{self, AmbientMachine, EventKind, LogLine};
use crate::ambient::{self, AmbientGate};
use crate::echo::EchoCanceller;
use crate::error::ApiError;
use crate::live_mode::{LiveState, Mode, SegmentFate};
use crate::registry::manifest::Kind;
use crate::speaker::{self, DEFAULT_MODEL, DEFAULT_THRESHOLD, Identity, Library, SpeakerVerifier};
use crate::stt::vad::Vad;
use crate::turn;

pub mod checkin;
pub mod host;

/// Typed binary frames: the first byte.
const FRAME_MIC: u8 = 0x00;
const FRAME_REFERENCE: u8 = 0x01;

/// A `level` event is sent at most this often.
const LEVEL_EVERY: Duration = Duration::from_millis(100);

/// The microphone stays shut this long after `agent_speaking` goes false:
/// the tail of the reply is still in the room.
const ECHO_HOLD: Duration = Duration::from_millis(400);

/// How long a dropped session's state can be resumed.
const RESUME_TTL: Duration = Duration::from_secs(60);

/// Audio kept for scoring a finished segment: the backlog plus the decode lag.
const RING_SAMPLES: usize = 45 * RATE;

/// A segment shorter than this is always admitted by the live guard: a "yes"
/// or a number is too short to tell whose voice it is.
const GUARD_MIN_SECONDS: f64 = 2.0;

/// The most one enrollment recording keeps.
const ENROLL_MAX_SAMPLES: usize = 120 * RATE;

/// Raw s16le bytes in one `speak_audio` chunk.
const SPEAK_CHUNK_BYTES: usize = 16 * 1024;

/// The longest `session` or speak `id`.
const MAX_ID_CHARS: usize = 128;

/// What a reconnecting client is told about the session it lost.
struct Snapshot {
    mode: Mode,
    seq: u64,
    /// Speaks that were still running when the socket dropped.
    speaks: Vec<String>,
    updated: Instant,
}

/// Dropped sessions by client `session` id. Process-wide rather than in
/// `AppState`, which tests build by hand; entries older than [`RESUME_TTL`]
/// are removed whenever the table is touched.
static RESUME: LazyLock<Mutex<HashMap<String, Snapshot>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn remember(session: &str, snapshot: Snapshot) {
    let mut table = RESUME.lock().unwrap();
    table.retain(|_, s| s.updated.elapsed() < RESUME_TTL);
    table.insert(session.to_string(), snapshot);
}

fn recall(session: &str) -> Option<Snapshot> {
    let mut table = RESUME.lock().unwrap();
    table.retain(|_, s| s.updated.elapsed() < RESUME_TTL);
    table.remove(session)
}

fn forget(session: &str) {
    RESUME.lock().unwrap().remove(session);
}

/// The ambient gate, loaded on first use from the GGUF named by
/// `NARU_AUDIO_AMBIENT_GATE`; unset, or any load failure, is the
/// [`ambient::NullGate`] (`ambient::load_gate` fails open). Process-wide for
/// the same reason as [`RESUME`].
static GATE: OnceLock<Arc<dyn AmbientGate>> = OnceLock::new();

fn gate() -> Arc<dyn AmbientGate> {
    GATE.get_or_init(|| {
        let path = std::env::var_os("NARU_AUDIO_AMBIENT_GATE")
            .filter(|p| !p.is_empty())
            .map(std::path::PathBuf::from);
        ambient::load_gate(path.as_deref())
    })
    .clone()
}

/// Naru's custom wake words, `ambient.wake_words` in its config file:
/// `NARU_AUDIO_WAKE_CONFIG`, else `~/.naru/config.json` when `~/.naru`
/// exists, else `~/.mesa/config.json` (naru's own `config_file`). A missing
/// or unreadable file means none.
fn wake_words() -> Vec<String> {
    let path = match std::env::var_os("NARU_AUDIO_WAKE_CONFIG").filter(|p| !p.is_empty()) {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let Some(home) = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(std::path::PathBuf::from)
            else {
                return Vec::new();
            };
            let naru = home.join(".naru");
            let dot = if naru.is_dir() {
                naru
            } else {
                home.join(".mesa")
            };
            dot.join("config.json")
        }
    };
    std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| serde_json::from_value(v["ambient"]["wake_words"].clone()).ok())
        .unwrap_or_default()
}

/// Changes a conversation keeps for a slow member before it lags.
const CONVERSATION_BACKLOG: usize = 16;

/// The state every socket of one `conversation` shares (naru task 2178).
struct Conv {
    mode: Mode,
    /// The `seq` of the conversation's latest `state`.
    seq: u64,
    members: usize,
    /// Each change as `(seq, mode)`, to every member.
    tx: broadcast::Sender<(u64, Mode)>,
    /// `wake`, `ambient_verdict` and `level` events as `(sender's member id, event)`,
    /// to every other member; the sender sends its own.
    events: broadcast::Sender<(u64, Value)>,
    /// Members whose client is playing a reply, so the server's own
    /// listener (`host.rs`) shuts its microphone too.
    speaking: HashSet<u64>,
}

/// Member ids, so a member skips the events it published itself.
static MEMBER_IDS: AtomicU64 = AtomicU64::new(0);

/// Conversations by key, while they have members. Process-wide for the same
/// reason as [`RESUME`].
static CONVERSATIONS: LazyLock<Mutex<HashMap<String, Conv>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// One socket's membership of a conversation; dropping it leaves.
struct Member {
    key: String,
    id: u64,
    rx: broadcast::Receiver<(u64, Mode)>,
    events: broadcast::Receiver<(u64, Value)>,
    /// The latest `seq` this socket was told.
    last: u64,
}

/// What a member hears from its conversation.
enum Shared {
    /// A `state` change: `(seq, mode)`.
    State(u64, Mode),
    /// Another member's `wake`, `ambient_verdict` or `level`.
    Event(Value),
}

/// Joins `key`, creating it in `seed` when it is new; the current mode and
/// seq. Subscribing under the lock means no change falls in between.
fn join(key: &str, seed: Option<Mode>) -> Result<(Member, Mode, u64), Fail> {
    let mut table = CONVERSATIONS.lock().unwrap();
    let conv = match table.get_mut(key) {
        Some(conv) => conv,
        None => {
            let Some(mode) = seed else {
                return Err(violation(
                    "invalid_request",
                    "start needs a mode (idle, live, muted or ambient) to begin a conversation",
                ));
            };
            let (tx, _) = broadcast::channel(CONVERSATION_BACKLOG);
            let (events, _) = broadcast::channel(CONVERSATION_BACKLOG);
            table.entry(key.to_string()).or_insert(Conv {
                mode,
                seq: 1,
                members: 0,
                tx,
                events,
                speaking: HashSet::new(),
            })
        }
    };
    conv.members += 1;
    let member = Member {
        key: key.to_string(),
        id: MEMBER_IDS.fetch_add(1, Ordering::Relaxed),
        rx: conv.tx.subscribe(),
        events: conv.events.subscribe(),
        last: conv.seq,
    };
    Ok((member, conv.mode, conv.seq))
}

/// Moves the conversation to `mode` and tells every member; the same mode
/// is ignored. The conversation's `seq` after it, if it still exists.
fn publish_mode(key: &str, mode: Mode) -> Option<u64> {
    let mut table = CONVERSATIONS.lock().unwrap();
    let conv = table.get_mut(key)?;
    if conv.mode != mode {
        conv.mode = mode;
        conv.seq += 1;
        let _ = conv.tx.send((conv.seq, mode));
    }
    Some(conv.seq)
}

/// One line per received `mode` frame.
fn log_mode(st: &AppState, req_id: &str, conv: Option<&str>, mode: Mode, seq: Option<u64>) {
    let seq = seq.map_or("-".to_string(), |s| s.to_string());
    st.log.info(
        Some(req_id),
        &format!(
            "live_mode conversation={} mode={} seq={seq}",
            conv.unwrap_or("-"),
            mode_name(mode)
        ),
    );
}

/// Records whether member `id` of `key` is playing a reply, then publishes a
/// `state` as [`publish_state`] does.
fn publish_speaking(key: &str, id: u64, speaking: bool) {
    if let Some(conv) = CONVERSATIONS.lock().unwrap().get_mut(key) {
        if speaking {
            conv.speaking.insert(id);
        } else {
            conv.speaking.remove(&id);
        }
    }
    publish_state(key);
}

/// Whether any member of `key` is playing a reply.
fn anyone_speaking(key: &str) -> bool {
    CONVERSATIONS
        .lock()
        .unwrap()
        .get(key)
        .is_some_and(|c| !c.speaking.is_empty())
}

/// A new `state` for every member with the mode unchanged (one member's
/// `agent_speaking` or enrollment changed).
fn publish_state(key: &str) {
    let mut table = CONVERSATIONS.lock().unwrap();
    if let Some(conv) = table.get_mut(key) {
        conv.seq += 1;
        let _ = conv.tx.send((conv.seq, conv.mode));
    }
}

/// Sends `event` (a `wake`, `ambient_verdict` or `level`) to every other member of
/// `key`; `from` sends it itself.
fn publish_event(key: &str, from: u64, event: Value) {
    let table = CONVERSATIONS.lock().unwrap();
    if let Some(conv) = table.get(key) {
        let _ = conv.events.send((from, event));
    }
}

impl Member {
    /// The next change this socket has not been told, or another member's
    /// event. After a lag the conversation's current state is the truth, and
    /// older ones are skipped; lagged events are lost.
    async fn changed(&mut self) -> Shared {
        loop {
            tokio::select! {
                change = self.rx.recv() => {
                    let (seq, mode) = match change {
                        Ok(change) => change,
                        // The sender lives in the table while this member
                        // does, so this is a lag.
                        Err(_) => {
                            let table = CONVERSATIONS.lock().unwrap();
                            let conv = &table[&self.key];
                            (conv.seq, conv.mode)
                        }
                    };
                    if seq > self.last {
                        self.last = seq;
                        return Shared::State(seq, mode);
                    }
                }
                event = self.events.recv() => {
                    if let Ok((from, event)) = event
                        && from != self.id
                    {
                        return Shared::Event(event);
                    }
                }
            }
        }
    }
}

impl Drop for Member {
    fn drop(&mut self) {
        let mut table = CONVERSATIONS.lock().unwrap();
        if let Some(conv) = table.get_mut(&self.key) {
            conv.members -= 1;
            // A member that left mid-reply no longer shuts anyone's mic.
            if conv.speaking.remove(&self.id) {
                conv.seq += 1;
                let _ = conv.tx.send((conv.seq, conv.mode));
            }
            if conv.members == 0 {
                table.remove(&self.key);
            }
        }
    }
}

/// A change for a socket that may be in a conversation; never, if not.
async fn changed(member: &mut Option<Member>) -> Shared {
    match member {
        Some(m) => m.changed().await,
        None => std::future::pending().await,
    }
}

/// The first message.
#[derive(Deserialize)]
struct LiveStart {
    /// Required unless `session` names a session that can be resumed.
    mode: Option<String>,
    /// Sample rate of the microphone frames: 16000. Required for `audio`.
    rate: Option<u64>,
    session: Option<String>,
    /// Sockets with the same key share one mode and `seq`.
    conversation: Option<String>,
    /// `audio` (the default) or `control`: mode messages only, no audio.
    role: Option<String>,
    model: Option<String>,
    /// Partial utterances; on unless `false`.
    partials: Option<bool>,
    #[serde(default = "s16le")]
    format: String,
    vad: Option<Value>,
    hotwords: Option<String>,
    keep_alive: Option<Value>,
    /// Binary frames carry a leading type byte (`0x00` microphone, `0x01`
    /// speaker reference); off unless `true`.
    #[serde(default)]
    typed_frames: bool,
}

fn s16le() -> String {
    "s16le".to_string()
}

/// Every client text frame.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Start(Box<LiveStart>),
    Mode {
        mode: String,
    },
    AgentSpeaking {
        value: bool,
    },
    Enroll {
        action: String,
        /// The person a `begin`ning recording is for; the owner when absent.
        person: Option<u64>,
    },
    Speak {
        id: String,
        text: String,
        voice: Option<String>,
        model: Option<String>,
    },
    Flush,
    Stop,
}

fn parse(text: &str) -> Result<ClientMessage, Fail> {
    serde_json::from_str(text)
        .map_err(|e| violation("invalid_request", format!("unreadable message: {e}")))
}

fn parse_mode(s: &str) -> Result<Mode, Fail> {
    match s {
        "idle" => Ok(Mode::Idle),
        "live" => Ok(Mode::Live),
        "muted" => Ok(Mode::Muted),
        "ambient" => Ok(Mode::Ambient),
        other => Err(violation(
            "invalid_request",
            format!("mode {other:?} is not one of idle, live, muted, ambient"),
        )),
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Idle => "idle",
        Mode::Live => "live",
        Mode::Muted => "muted",
        Mode::Ambient => "ambient",
    }
}

fn check_id(what: &str, id: &str) -> Result<(), Fail> {
    if id.is_empty() || id.chars().count() > MAX_ID_CHARS {
        return Err(violation(
            "invalid_request",
            format!("{what} must be 1 to {MAX_ID_CHARS} characters"),
        ));
    }
    Ok(())
}

/// An `error` that leaves the session open.
fn soft(code: &str, message: impl Into<String>, id: Option<&str>) -> Value {
    let mut v = json!({"type": "error", "code": code, "message": message.into(), "fatal": false});
    if let Some(id) = id {
        v["id"] = json!(id);
    }
    v
}

/// The session's state on the server.
struct Live {
    state: LiveState,
    seq: u64,
    hold_until: Option<Instant>,
    /// The voice library, reloaded when `people.json` changes.
    library: Arc<Library>,
    library_stamp: Option<(std::time::SystemTime, u64)>,
    verifier: Option<Arc<SpeakerVerifier>>,
    verifier_tried: bool,
    /// Re-embedding a library of another model failed; not retried per segment.
    migrate_failed: bool,
    /// Raw microphone audio of a running enrollment recording.
    enrolling: Option<Vec<f32>>,
    /// Who the running recording is for; the owner when `None`.
    enroll_person: Option<u64>,
    /// The audio fed to the VAD, as `Segments` positions it.
    ring: VecDeque<f32>,
    ring_start: usize,
    received: usize,
    peak: f32,
    last_level: Instant,
    utterance: u64,
    speaks: HashMap<String, JoinHandle<()>>,
    /// Binary frames lead with a type byte.
    typed: bool,
    echo: EchoCanceller,
    began: Instant,
    /// The wake word and the gate's routing for `ambient` mode; its clock is
    /// `began`.
    ambient: AmbientMachine,
}

impl Live {
    fn new(mode: Mode, seq: u64, library: Library, typed: bool) -> Self {
        let mut live = Live {
            state: LiveState::new(),
            seq,
            hold_until: None,
            library: Arc::new(library),
            library_stamp: None,
            verifier: None,
            verifier_tried: false,
            migrate_failed: false,
            enrolling: None,
            enroll_person: None,
            ring: VecDeque::new(),
            ring_start: 0,
            received: 0,
            peak: 0.0,
            last_level: Instant::now(),
            utterance: 0,
            speaks: HashMap::new(),
            typed,
            echo: EchoCanceller::new(),
            began: Instant::now(),
            ambient: AmbientMachine::new(),
        };
        live.set_mode(mode);
        live
    }

    /// Picks up a changed `people.json`, so library edits reach a running
    /// session; an unreadable file keeps the library as it was.
    fn refresh_library(&mut self, home: &Path) {
        // A cheap stat; the file is read only when it changed.
        if Library::stamp(home) != self.library_stamp {
            let (library, stamp) = Library::load_stamped(home);
            if let Ok(library) = library {
                self.library = Arc::new(library);
            }
            self.library_stamp = stamp;
        }
    }

    /// The owner has recorded something.
    fn enrolled(&self) -> bool {
        self.library.owner().is_some_and(|o| !o.clips.is_empty())
    }

    /// Milliseconds since the session began: the ambient machine's clock.
    fn now_ms(&self) -> u64 {
        self.began.elapsed().as_millis() as u64
    }

    /// Drives [`LiveState`] to `mode`: live and joined unless idle, ambient
    /// as asked, muted only while the pipeline is wanted.
    fn set_mode(&mut self, mode: Mode) {
        if mode == Mode::Ambient && self.mode() != Mode::Ambient {
            // Back in ambient, the next wake word wakes again.
            let now = self.now_ms();
            self.ambient.end_conversation(now);
        }
        let s = &mut self.state;
        s.live = mode != Mode::Idle;
        s.joined = mode != Mode::Idle;
        s.ambient = mode == Mode::Ambient;
        s.leaving_ambient = false;
        s.apply_capture();
        s.muted = mode == Mode::Muted && s.active;
    }

    fn mode(&self) -> Mode {
        self.state.mode()
    }

    fn agent_speaking(&self) -> bool {
        self.state.speaking_turn.is_some()
    }

    fn set_agent_speaking(&mut self, speaking: bool) {
        self.state.speaking_turn = speaking.then_some(0);
        // The tail of a reply is still in the room.
        self.hold_until = (!speaking).then(|| Instant::now() + ECHO_HOLD);
    }

    fn in_hold(&self) -> bool {
        self.hold_until.is_some_and(|t| Instant::now() < t)
    }

    /// Whether microphone audio reaches the VAD: the live gates for a live
    /// conversation; ambient listens whenever the agent is not speaking.
    fn gate_open(&self) -> bool {
        if self.enrolling.is_some() || self.in_hold() {
            return false;
        }
        match self.mode() {
            Mode::Live => self.state.mic_open(false),
            Mode::Ambient => !self.agent_speaking(),
            Mode::Idle | Mode::Muted => false,
        }
    }

    fn state_event(&mut self, resumed: Option<&Snapshot>) -> Value {
        self.seq += 1;
        self.state_value(resumed)
    }

    /// A `state` at a conversation's `seq`.
    fn shared_state(&mut self, seq: u64, resumed: Option<&Snapshot>) -> Value {
        self.seq = seq;
        self.state_value(resumed)
    }

    fn state_value(&self, resumed: Option<&Snapshot>) -> Value {
        let mut ids: Vec<&String> = self.speaks.keys().collect();
        ids.sort();
        let mut event = json!({
            "type": "state",
            "mode": mode_name(self.mode()),
            "seq": self.seq,
            "agent_speaking": self.agent_speaking(),
            "enrolled": self.enrolled(),
            "enrolling": self.enrolling.is_some(),
            "live_guard": self.library.live_guard,
            "speaking": ids,
            "resumed": resumed.is_some(),
        });
        if let Some(prev) = resumed {
            event["previous_mode"] = json!(mode_name(prev.mode));
            event["lost_speaks"] = json!(prev.speaks);
        }
        event
    }

    fn push_ring(&mut self, fed: &[f32]) {
        self.ring.extend(fed);
        self.received += fed.len();
        if self.ring.len() > RING_SAMPLES {
            let drop = self.ring.len() - RING_SAMPLES;
            self.ring.drain(..drop);
            self.ring_start += drop;
        }
    }

    /// The fed audio of `start..end` seconds, if the ring still has it.
    fn clip(&self, start: f64, end: f64) -> Option<Vec<f32>> {
        let from = ((start * RATE as f64) as usize).max(self.ring_start);
        let to = ((end * RATE as f64) as usize).min(self.received);
        (from < to).then(|| {
            self.ring
                .iter()
                .skip(from - self.ring_start)
                .take(to - from)
                .copied()
                .collect()
        })
    }

    /// Microphone samples as the VAD gets them: metered, recorded by a
    /// running enrollment, then silence while the gate is shut or the frame
    /// is the reference heard back (the VAD keeps its clock). Kept in the ring.
    fn hear(&mut self, samples: Vec<f32>) -> Vec<f32> {
        self.note_level(&samples);
        if let Some(clip) = &mut self.enrolling {
            let room = ENROLL_MAX_SAMPLES.saturating_sub(clip.len());
            clip.extend_from_slice(&samples[..samples.len().min(room)]);
        }
        let echo = self.echo.is_echo(&samples, self.began.elapsed());
        let fed = if !echo && self.gate_open() {
            samples
        } else {
            vec![0.0; samples.len()]
        };
        self.push_ring(&fed);
        fed
    }

    fn note_level(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        self.peak = self.peak.max(rms);
    }

    fn level_event(&mut self) -> Option<Value> {
        if self.last_level.elapsed() < LEVEL_EVERY {
            return None;
        }
        self.last_level = Instant::now();
        let rms = std::mem::take(&mut self.peak);
        Some(json!({"type": "level", "rms": rms, "gate": self.gate_open()}))
    }

    fn abort_speaks(&mut self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for (id, handle) in self.speaks.drain() {
            handle.abort();
            ids.push(id);
        }
        ids.sort();
        ids
    }
}

pub(super) async fn live(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Result<Response, ApiError> {
    let upgrade =
        upgrade.map_err(|e| ApiError::new(e.status(), "invalid_request", e.body_text()))?;
    Ok(upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| run(st, req_id, socket)))
}

/// How a session that did not fail ended.
enum Ended {
    /// `stop`: every utterance and `done` were sent.
    Stopped,
    /// The client closed or dropped the connection.
    Gone,
}

/// One session, then its close: an `error` event always precedes a non-1000
/// close.
async fn run(st: Arc<AppState>, req_id: String, mut socket: WebSocket) {
    let started = Instant::now();
    let (close, reason) = match session(&st, &req_id, &mut socket).await {
        Ok(Ended::Gone) => {
            st.log.info(Some(&req_id), "live_end client_gone");
            return;
        }
        Ok(Ended::Stopped) => (close_code::NORMAL, "done"),
        Err(fail) => {
            st.log.line(
                "warn",
                Some(&req_id),
                &format!(
                    "live_error code={} close={} {}",
                    fail.code, fail.close, fail.message
                ),
            );
            let event = json!({
                "type": "error", "code": fail.code, "message": fail.message, "fatal": true,
            });
            if send(&mut socket, event).await.is_err() {
                return;
            }
            (fail.close, fail.code)
        }
    };
    st.log.info(
        Some(&req_id),
        &format!(
            "live_end close={close} ms={}",
            started.elapsed().as_millis()
        ),
    );
    let frame = CloseFrame {
        code: close,
        reason: reason.into(),
    };
    if socket.send(Message::Close(Some(frame))).await.is_err() {
        return;
    }
    // As the stream socket does: read until the client's close so the
    // `error` and the close are not lost to a reset.
    let _ = tokio::time::timeout(CLOSE_LINGER, async {
        while let Some(Ok(msg)) = socket.recv().await {
            if matches!(msg, Message::Close(_)) {
                break;
            }
        }
    })
    .await;
}

async fn session(st: &Arc<AppState>, req_id: &str, socket: &mut WebSocket) -> Result<Ended, Fail> {
    let start = match next(socket).await? {
        None => return Ok(Ended::Gone),
        Some(Message::Text(text)) => match parse(&text)? {
            ClientMessage::Start(start) => *start,
            _ => {
                return Err(violation(
                    "invalid_request",
                    "the first message must be {\"type\":\"start\"}",
                ));
            }
        },
        Some(_) => {
            return Err(violation(
                "invalid_request",
                "the first frame must be a text {\"type\":\"start\"} message",
            ));
        }
    };
    if let Some(key) = &start.conversation {
        check_id("conversation", key)?;
    }
    let control = match start.role.as_deref() {
        None | Some("audio") => false,
        Some("control") => true,
        Some(other) => {
            return Err(violation(
                "invalid_request",
                format!("role {other:?} is not one of audio, control"),
            ));
        }
    };
    if control {
        return control_session(st, req_id, socket, start).await;
    }
    let Some(rate) = start.rate else {
        return Err(violation(
            "invalid_request",
            "start needs a rate (16000) for an audio socket",
        ));
    };
    if start.format != "s16le" {
        return Err(violation(
            "unsupported_value",
            format!("format {:?} is not supported; use s16le", start.format),
        ));
    }
    let session_id = match &start.session {
        Some(id) => {
            check_id("session", id)?;
            id.clone()
        }
        None => req_id.to_string(),
    };
    let requested = start.mode.as_deref().map(parse_mode).transpose()?;
    // Everything else is the stream's own validation.
    let mut fields = json!({"sample_rate": rate, "partials": start.partials.unwrap_or(true)});
    for (key, value) in [
        ("model", start.model.as_ref().map(|m| json!(m))),
        ("hotwords", start.hotwords.as_ref().map(|h| json!(h))),
        ("vad", start.vad.clone()),
        ("keep_alive", start.keep_alive.clone()),
    ] {
        if let Some(value) = value {
            fields[key] = value;
        }
    }
    let stream_start: Start = serde_json::from_value(fields)
        .map_err(|e| violation("invalid_request", format!("unreadable start: {e}")))?;
    let cfg = validate(stream_start, &st.models.stt_default())?;
    if let Some(v) = &cfg.hotwords {
        warn_truncated(st, req_id, v);
    }

    let previous = recall(&session_id);
    let mode = match (requested, &previous) {
        (Some(mode), _) => Some(mode),
        (None, Some(prev)) => Some(prev.mode),
        // A conversation that exists has a mode.
        (None, None) if start.conversation.is_some() => None,
        (None, None) => {
            return Err(violation(
                "invalid_request",
                "start needs a mode (idle, live, muted or ambient)",
            ));
        }
    };

    let manifest = {
        let (st, name) = (st.clone(), cfg.model.clone());
        tokio::task::spawn_blocking(move || stt_manifest(&st, &name))
    }
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))??;
    let loaded = st
        .models
        .ps()
        .iter()
        .any(|m| m.name == cfg.model && !m.loading);
    if !loaded && send(socket, json!({"type": "loading"})).await.is_err() {
        return Ok(Ended::Gone);
    }
    let load_started = Instant::now();
    let (model, vad) = acquire(st, req_id, manifest, &cfg).await?;
    let load_ms = if loaded {
        0
    } else {
        load_started.elapsed().as_millis() as u64
    };
    let (library, library_stamp) = load_library(st, req_id).await?;

    // A conversation's mode and seq win over the resumed ones.
    let (mut member, mode, shared_seq) = match &start.conversation {
        Some(key) => {
            let (member, mode, seq) = join(key, mode)?;
            (Some(member), mode, Some(seq))
        }
        None => (None, mode.expect("checked above"), None),
    };
    let mut live = Live::new(
        mode,
        previous.as_ref().map_or(0, |p| p.seq),
        library,
        start.typed_frames,
    );
    live.library_stamp = library_stamp;
    live.ambient.set_wake_words(&wake_words());
    let ready = json!({
        "type": "ready", "session": session_id, "model": cfg.model, "load_ms": load_ms,
        "resumed": previous.is_some(), "typed_frames": start.typed_frames,
    });
    if send(socket, ready).await.is_err() {
        return Ok(Ended::Gone);
    }
    let first = match shared_seq {
        Some(seq) => live.shared_state(seq, previous.as_ref()),
        None => live.state_event(previous.as_ref()),
    };
    remember_state(&session_id, &live);
    if send(socket, first).await.is_err() {
        return Ok(Ended::Gone);
    }

    let result = drive(
        st,
        req_id,
        socket,
        &session_id,
        &mut live,
        &mut member,
        cfg,
        vad,
        model,
    )
    .await;
    let lost = live.abort_speaks();
    match &result {
        Ok(Ended::Stopped) => forget(&session_id),
        _ => remember(
            &session_id,
            Snapshot {
                mode: live.mode(),
                seq: live.seq,
                speaks: lost,
                updated: Instant::now(),
            },
        ),
    }
    result
}

/// The speech-to-text model held for the session, and its VAD.
async fn acquire(
    st: &Arc<AppState>,
    req_id: &str,
    manifest: crate::registry::manifest::Manifest,
    cfg: &Config,
) -> Result<(crate::manager::Guard, Vad), Fail> {
    let model = st
        .models
        .acquire(manifest, cfg.keep_alive)
        .await
        .map_err(|e| manager_error(st, req_id, e))?;
    let Some(vad_model) = model.stt().vad_model().map(Path::to_path_buf) else {
        return Err(Fail {
            code: "backend_unavailable",
            message: format!("the model \"{}\" cannot stream: it has no VAD", cfg.model),
            close: close_code::ERROR,
        });
    };
    let max_speech = 2.0 * cfg.max_segment as f32 / RATE as f32;
    let vad = {
        let vad_cfg = cfg.vad.clone();
        tokio::task::spawn_blocking(move || {
            Vad::load_with_max_speech(&vad_model, &vad_cfg, max_speech)
        })
    }
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))?
    .map_err(|e| Fail {
        code: "model_load_failed",
        message: e.to_string(),
        close: close_code::ERROR,
    })?;
    Ok((model, vad))
}

/// The voice library and its stamp. A missing or unreadable library is an
/// empty one: the guard fails open.
async fn load_library(
    st: &Arc<AppState>,
    req_id: &str,
) -> Result<(Library, Option<(std::time::SystemTime, u64)>), Fail> {
    let (library, library_stamp) = {
        let home = st.registry.home().to_path_buf();
        tokio::task::spawn_blocking(move || Library::load_stamped(&home))
    }
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))?;
    let library = library.unwrap_or_else(|e| {
        st.log.line(
            "warn",
            Some(req_id),
            &format!("live_enrollment_unreadable {e}"),
        );
        Library::default()
    });
    Ok((library, library_stamp))
}

/// A `control` socket: joins `start.conversation` and carries its mode
/// both ways, with no audio, model or VAD.
async fn control_session(
    st: &AppState,
    req_id: &str,
    socket: &mut WebSocket,
    start: LiveStart,
) -> Result<Ended, Fail> {
    let Some(key) = start.conversation else {
        return Err(violation(
            "invalid_request",
            "a control socket needs a conversation",
        ));
    };
    let session_id = match &start.session {
        Some(id) => {
            check_id("session", id)?;
            id.clone()
        }
        None => req_id.to_string(),
    };
    let requested = start.mode.as_deref().map(parse_mode).transpose()?;
    let (mut member, mode, seq) = join(&key, requested)?;
    let state = |mode: Mode, seq: u64| {
        json!({
            "type": "state", "mode": mode_name(mode), "seq": seq, "agent_speaking": false,
            "enrolled": false, "enrolling": false, "speaking": [], "resumed": false,
        })
    };
    let ready = json!({
        "type": "ready", "session": session_id, "role": "control", "conversation": key,
        "resumed": false,
    });
    if send(socket, ready).await.is_err() || send(socket, state(mode, seq)).await.is_err() {
        return Ok(Ended::Gone);
    }
    loop {
        tokio::select! {
            msg = next(socket) => {
                let Some(msg) = msg? else { return Ok(Ended::Gone) };
                match msg {
                    Message::Text(text) => match parse(&text)? {
                        ClientMessage::Mode { mode } => {
                            let mode = parse_mode(&mode)?;
                            log_mode(st, req_id, Some(&key), mode, publish_mode(&key, mode));
                        }
                        ClientMessage::Stop => {
                            if send(socket, json!({"type": "done"})).await.is_err() {
                                return Ok(Ended::Gone);
                            }
                            return Ok(Ended::Stopped);
                        }
                        ClientMessage::Start(_) => {
                            return Err(violation("invalid_request", "the session has already started"));
                        }
                        _ => {
                            return Err(violation(
                                "invalid_request",
                                "a control socket accepts only mode and stop",
                            ));
                        }
                    },
                    Message::Binary(_) => {
                        return Err(violation("invalid_request", "a control socket takes no audio"));
                    }
                    _ => {}
                }
            }
            shared = member.changed() => {
                let event = match shared {
                    Shared::State(seq, mode) => state(mode, seq),
                    Shared::Event(event) => event,
                };
                if send(socket, event).await.is_err() {
                    return Ok(Ended::Gone);
                }
            }
        }
    }
}

/// The state a reconnect would find, with the speaks running now.
fn remember_state(session: &str, live: &Live) {
    let mut speaks: Vec<String> = live.speaks.keys().cloned().collect();
    speaks.sort();
    remember(
        session,
        Snapshot {
            mode: live.mode(),
            seq: live.seq,
            speaks,
            updated: Instant::now(),
        },
    );
}

/// The segmenter and decode threads: their input, their events, and the
/// samples sent but not yet decoded.
fn workers(
    st: &AppState,
    cfg: &Config,
    vad: Vad,
    model: crate::manager::Guard,
) -> (
    UnboundedSender<Input>,
    UnboundedReceiver<Event>,
    Arc<AtomicUsize>,
) {
    let backlog = Arc::new(AtomicUsize::new(0));
    let busy = Arc::new(AtomicBool::new(false));
    let (input_tx, input_rx) = unbounded_channel();
    let (job_tx, job_rx) = unbounded_channel();
    let (event_tx, events) = unbounded_channel();
    {
        let (events, backlog, busy) = (event_tx.clone(), backlog.clone(), busy.clone());
        let segments = Segments::new(cfg, job_tx, events, backlog, busy);
        let (log, dir) = (st.log.clone(), st.registry.model_dir(turn::DEFAULT_MODEL));
        // Loaded on this thread, once per process, so a first session's
        // audio waits in the channel rather than blocking the socket.
        tokio::task::spawn_blocking(move || {
            let segments = segments.with_turn(turn::shared(&dir), log);
            segment(vad, segments, input_rx)
        });
    }
    {
        let (vad, backlog) = (cfg.vad.clone(), backlog.clone());
        tokio::task::spawn_blocking(move || decode(model, vad, job_rx, event_tx, backlog, busy));
    }
    (input_tx, events, backlog)
}

#[allow(clippy::too_many_arguments)]
async fn drive(
    st: &Arc<AppState>,
    req_id: &str,
    socket: &mut WebSocket,
    session_id: &str,
    live: &mut Live,
    member: &mut Option<Member>,
    cfg: Config,
    vad: Vad,
    model: crate::manager::Guard,
) -> Result<Ended, Fail> {
    let (input_tx, mut events, backlog) = workers(st, &cfg, vad, model);
    // Speak tasks write here; `out_tx` stays alive so `recv` never ends.
    let (out_tx, mut out) = unbounded_channel::<Value>();

    let mut stopping = false;
    loop {
        tokio::select! {
            msg = next(socket), if !stopping => {
                let Some(msg) = msg? else { return Ok(Ended::Gone) };
                match msg {
                    Message::Binary(bytes) => {
                        if bytes.len() > MAX_FRAME_BYTES {
                            return Err(violation(
                                "invalid_request",
                                format!("a binary frame is at most {MAX_FRAME_BYTES} bytes, got {}", bytes.len()),
                            ));
                        }
                        let (kind, payload) = if live.typed {
                            match bytes.split_first() {
                                Some((&kind @ (FRAME_MIC | FRAME_REFERENCE), rest)) => (kind, rest),
                                Some((kind, _)) => {
                                    return Err(violation(
                                        "invalid_request",
                                        format!("frame type {kind:#04x} is not 0x00 (microphone) or 0x01 (reference)"),
                                    ));
                                }
                                None => {
                                    return Err(violation(
                                        "invalid_request",
                                        "a typed frame needs its type byte",
                                    ));
                                }
                            }
                        } else {
                            (FRAME_MIC, &bytes[..])
                        };
                        if payload.len() % cfg.format.width() != 0 {
                            return Err(violation(
                                "invalid_request",
                                format!("a {}-byte frame is not a whole number of samples", payload.len()),
                            ));
                        }
                        let samples = cfg.format.samples(payload);
                        if kind == FRAME_REFERENCE {
                            // What the client is playing: kept for the echo
                            // check, never heard by the VAD.
                            live.echo.push_reference(&samples, live.began.elapsed());
                            continue;
                        }
                        let fed = live.hear(samples);
                        let queued = backlog.fetch_add(fed.len(), Ordering::SeqCst) + fed.len();
                        if queued > MAX_BACKLOG_SAMPLES {
                            return Err(Fail {
                                code: "backlog",
                                message: format!(
                                    "more than {} s of audio is waiting to be decoded",
                                    MAX_BACKLOG_SAMPLES / RATE
                                ),
                                close: close_code::AGAIN,
                            });
                        }
                        let _ = input_tx.send(Input::Audio(fed));
                        if let Some(level) = live.level_event() {
                            // Every other member gets it too, at the same rate.
                            if let Some(m) = member.as_ref() {
                                publish_event(&m.key, m.id, level.clone());
                            }
                            if send(socket, level).await.is_err() {
                                return Ok(Ended::Gone);
                            }
                        }
                    }
                    Message::Text(text) => {
                        let conv = member.as_ref().map(|m| (m.key.as_str(), m.id));
                        let sent = client_message(
                            st, req_id, session_id, live, conv, &out_tx, &input_tx, &mut stopping,
                            &text,
                        )
                        .await?;
                        for event in sent {
                            if send(socket, event).await.is_err() {
                                return Ok(Ended::Gone);
                            }
                        }
                    }
                    _ => {}
                }
            }
            event = events.recv() => match event {
                Some(Event::Send(v)) => {
                    let conv = member.as_ref().map(|m| (m.key.as_str(), m.id));
                    for event in worker_event(st, live, conv, v).await {
                        if send(socket, event).await.is_err() {
                            return Ok(Ended::Gone);
                        }
                    }
                }
                Some(Event::Done) => {
                    if send(socket, json!({"type": "done"})).await.is_err() {
                        return Ok(Ended::Gone);
                    }
                    return Ok(Ended::Stopped);
                }
                Some(Event::Fail(e)) => return Err(decode_error(st, req_id, e).into()),
                None => {
                    return Err(internal(st, req_id, "the session's workers stopped".to_string()).into());
                }
            },
            shared = changed(member) => {
                let (seq, mode) = match shared {
                    Shared::State(seq, mode) => (seq, mode),
                    Shared::Event(event) => {
                        if send(socket, event).await.is_err() {
                            return Ok(Ended::Gone);
                        }
                        continue;
                    }
                };
                // As a `mode` message does: the old mode's audio is cut first.
                if mode != live.mode() {
                    let _ = input_tx.send(Input::Flush);
                    live.set_mode(mode);
                }
                let state = live.shared_state(seq, None);
                remember_state(session_id, live);
                if send(socket, state).await.is_err() {
                    return Ok(Ended::Gone);
                }
            }
            Some(event) = out.recv() => {
                // A speak that ended is no longer running.
                let ended = match event["type"].as_str() {
                    Some("speak_audio") => event["final"] == true,
                    Some("error") => true,
                    _ => false,
                };
                if ended && let Some(id) = event["id"].as_str() {
                    live.speaks.remove(id);
                }
                if send(socket, event).await.is_err() {
                    return Ok(Ended::Gone);
                }
            }
        }
    }
}

/// One client text frame after `start`; the events to send back. In a
/// conversation a `state` is published instead, and every member, this one
/// included, sends it when it arrives.
#[allow(clippy::too_many_arguments)]
async fn client_message(
    st: &Arc<AppState>,
    req_id: &str,
    session_id: &str,
    live: &mut Live,
    conv: Option<(&str, u64)>,
    out: &UnboundedSender<Value>,
    input: &UnboundedSender<Input>,
    stopping: &mut bool,
    text: &str,
) -> Result<Vec<Value>, Fail> {
    let mut sent = Vec::new();
    match parse(text)? {
        ClientMessage::Start(_) => {
            return Err(violation(
                "invalid_request",
                "the session has already started",
            ));
        }
        ClientMessage::Mode { mode } => {
            let mode = parse_mode(&mode)?;
            if let Some((key, _)) = conv {
                log_mode(st, req_id, Some(key), mode, publish_mode(key, mode));
            } else {
                log_mode(st, req_id, None, mode, None);
                if mode != live.mode() {
                    // What the old mode heard is the old mode's: cut it first.
                    let _ = input.send(Input::Flush);
                    live.set_mode(mode);
                    sent.push(live.state_event(None));
                    remember_state(session_id, live);
                }
            }
        }
        ClientMessage::AgentSpeaking { value } => {
            if value != live.agent_speaking() {
                if value {
                    let _ = input.send(Input::Flush);
                }
                live.set_agent_speaking(value);
                if !value {
                    // Playback is over: whatever was still being made is not wanted.
                    for id in live.abort_speaks() {
                        sent.push(json!({
                            "type": "speak_audio", "id": id, "seq": 0, "final": true,
                            "cancelled": true, "data": "",
                        }));
                    }
                }
                match conv {
                    Some((key, id)) => publish_speaking(key, id, value),
                    None => sent.push(live.state_event(None)),
                }
                remember_state(session_id, live);
            }
        }
        ClientMessage::Flush => {
            let _ = input.send(Input::Flush);
        }
        ClientMessage::Stop => {
            *stopping = true;
            let _ = input.send(Input::Stop);
        }
        ClientMessage::Enroll { action, person } => {
            let mut events = enroll(st, req_id, live, &action, person).await;
            if let Some((key, _)) = conv {
                let before = events.len();
                events.retain(|e| e["type"] != "state");
                if events.len() < before {
                    publish_state(key);
                }
            }
            sent.extend(events);
            remember_state(session_id, live);
        }
        ClientMessage::Speak {
            id,
            text,
            voice,
            model,
        } => {
            check_id("speak id", &id)?;
            if live.speaks.contains_key(&id) {
                sent.push(soft(
                    "invalid_request",
                    format!("a speak with id {id:?} is already running"),
                    Some(&id),
                ));
            } else {
                // A newer speak replaces the older ones.
                for old in live.abort_speaks() {
                    sent.push(json!({
                        "type": "speak_audio", "id": old, "seq": 0, "final": true,
                        "cancelled": true, "data": "",
                    }));
                }
                let handle = tokio::spawn(speak(
                    st.clone(),
                    req_id.to_string(),
                    id.clone(),
                    text,
                    voice,
                    model,
                    out.clone(),
                ));
                live.speaks.insert(id, handle);
                remember_state(session_id, live);
            }
        }
    }
    Ok(sent)
}

/// Synthesises `text` through `POST /v1/audio/speech`'s own handler and
/// streams it as `speak_audio` events; the last has `final: true`.
async fn speak(
    st: Arc<AppState>,
    req_id: String,
    id: String,
    text: String,
    voice: Option<String>,
    model: Option<String>,
    out: UnboundedSender<Value>,
) {
    let mut body = json!({"input": text, "response_format": "pcm", "stream": true});
    if let Some(voice) = voice {
        body["voice"] = json!(voice);
    }
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    let response = super::speech::speech(
        State(st),
        Extension(RequestId(req_id)),
        Bytes::from(body.to_string()),
    )
    .await;
    let response = match response {
        Ok(r) => r,
        Err(e) => {
            let _ = out.send(soft(e.code, e.message, Some(&id)));
            return;
        }
    };
    let rate: u64 = response
        .headers()
        .get("x-audio-sample-rate")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut stream = Body::into_data_stream(response.into_body());
    let mut seq: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                let _ = out.send(soft(
                    "internal",
                    format!("synthesis failed: {e}"),
                    Some(&id),
                ));
                return;
            }
        };
        for part in bytes.chunks(SPEAK_CHUNK_BYTES) {
            let event = json!({
                "type": "speak_audio", "id": id, "seq": seq, "rate": rate,
                "final": false, "data": STANDARD.encode(part),
            });
            if out.send(event).is_err() {
                return;
            }
            seq += 1;
        }
    }
    let _ = out.send(json!({
        "type": "speak_audio", "id": id, "seq": seq, "rate": rate, "final": true, "data": "",
    }));
}

/// `enroll` actions: `begin` records the microphone (for `person`, else the
/// owner), `finish` adds the recording to that person as a new clip, `cancel`
/// drops it, `clear` deletes the owner's clips.
async fn enroll(
    st: &Arc<AppState>,
    req_id: &str,
    live: &mut Live,
    action: &str,
    person: Option<u64>,
) -> Vec<Value> {
    match action {
        "begin" => {
            live.enrolling = Some(Vec::new());
            live.enroll_person = person;
            vec![live.state_event(None)]
        }
        "cancel" => {
            live.enrolling = None;
            vec![live.state_event(None)]
        }
        "finish" => {
            let Some(mut clip) = live.enrolling.take() else {
                return vec![soft(
                    "invalid_request",
                    "no enrollment recording is running",
                    None,
                )];
            };
            let target = live.enroll_person.take();
            let st2 = st.clone();
            let result = tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
                if clip.len() < speaker::MIN_SAMPLES {
                    return Err(ApiError::new(
                        axum::http::StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "an enrollment recording needs at least 1 s of audio",
                    ));
                }
                super::speaker::normalize_level(&mut clip)?;
                let verifier = load_verifier(&st2)?;
                let embedding = verifier
                    .embed(&clip)
                    .map_err(|e| internal(&st2, "live", e.to_string()))?;
                let home = st2.registry.home();
                Library::update(home, |l| {
                    let id = match target {
                        Some(id) => l.person(id)?.id,
                        None => l.ensure_owner(),
                    };
                    l.reembed(home, |pcm| verifier.embed(pcm))?;
                    l.add_clip(home, id, &clip, embedding).map(|_| ())
                })
                .map_err(|e| match e {
                    speaker::SpeakerError::NotFound(m) => {
                        ApiError::new(axum::http::StatusCode::NOT_FOUND, "not_found", m)
                    }
                    e => internal(&st2, "live", e.to_string()),
                })
            })
            .await;
            match result {
                Ok(Ok(())) => {
                    live.refresh_library(st.registry.home());
                    live.verifier = None;
                    live.verifier_tried = false;
                    vec![live.state_event(None)]
                }
                Ok(Err(e)) => {
                    let mut events = vec![soft(e.code, e.message, None)];
                    events.push(live.state_event(None));
                    events
                }
                Err(e) => {
                    st.log
                        .line("error", Some(req_id), &format!("live_enroll_panicked {e}"));
                    vec![
                        soft("internal", "enrollment failed", None),
                        live.state_event(None),
                    ]
                }
            }
        }
        "clear" => {
            let home = st.registry.home().to_path_buf();
            let cleared = tokio::task::spawn_blocking({
                let home = home.clone();
                move || {
                    Library::update(&home, |l| match l.owner().map(|o| o.id) {
                        Some(owner) => l.clear_clips(&home, owner),
                        None => Ok(false),
                    })
                }
            })
            .await;
            match cleared {
                Ok(Ok(_)) => {
                    live.refresh_library(&home);
                    live.verifier = None;
                    vec![live.state_event(None)]
                }
                _ => vec![soft(
                    "internal",
                    "the enrollment could not be deleted",
                    None,
                )],
            }
        }
        other => vec![soft(
            "invalid_request",
            format!("enroll action {other:?} is not one of begin, finish, cancel, clear"),
            None,
        )],
    }
}

/// The pulled speaker model loaded, or why not.
fn load_verifier(st: &AppState) -> Result<SpeakerVerifier, ApiError> {
    kind_manifest(st, DEFAULT_MODEL, Kind::Speaker)?;
    let dir = st.registry.model_dir(DEFAULT_MODEL);
    SpeakerVerifier::load(&dir).map_err(|e| {
        ApiError::new(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_load_failed",
            e.to_string(),
        )
    })
}

/// An event from the VAD or decode thread, as the events to send. `conv` is
/// the socket's conversation key and member id, if it is in one.
async fn worker_event(
    st: &Arc<AppState>,
    live: &mut Live,
    conv: Option<(&str, u64)>,
    v: Value,
) -> Vec<Value> {
    match v["type"].as_str() {
        Some("speech") => {
            let active = v["active"].as_bool().unwrap_or(false);
            live.state.vad_changed(active, false);
            vec![v]
        }
        // A partial is the open utterance, so only a live conversation shows it.
        Some("partial") => {
            if live.mode() == Mode::Live && !live.agent_speaking() {
                vec![v]
            } else {
                Vec::new()
            }
        }
        Some("final") => finish_segment(st, live, conv, &v).await,
        _ => vec![v],
    }
}

/// A decoded segment: gated, scored, then an `utterance` (live), an
/// `ambient_flag` (ambient) or nothing. In ambient the line also goes through
/// the ambient machine, which may add a `wake` or an `ambient_verdict`.
async fn finish_segment(
    st: &Arc<AppState>,
    live: &mut Live,
    conv: Option<(&str, u64)>,
    v: &Value,
) -> Vec<Value> {
    let text = v["text"].as_str().unwrap_or_default();
    let (start, end) = (
        v["start"].as_f64().unwrap_or(0.0),
        v["end"].as_f64().unwrap_or(0.0),
    );
    if text.is_empty() {
        // Closes the partials the segment had.
        return vec![json!({"type": "partial", "segment": v["segment"], "text": ""})];
    }
    let mode = live.mode();
    // Backstop behind the frame-level check: a segment that is the
    // reference heard back, whatever its first 100 ms looked like.
    let echo = mode != Mode::Idle
        && live.clip(start, end).is_some_and(|clip| {
            let back = live.received.saturating_sub((end * RATE as f64) as usize);
            live.echo.clip_is_echo(&clip, back)
        });
    live.refresh_library(st.registry.home());
    let scoring = live.library.people.iter().any(|p| !p.clips.is_empty());
    let identity = if matches!(mode, Mode::Live | Mode::Ambient) && scoring {
        score(st, live, start, end).await
    } else {
        None
    };
    let verdict = identity.as_ref().map(|i| i.verdict);
    // The live guard is opt-in, needs the owner's voice, and never drops a
    // short reply; scoring above is only a tag when it is off.
    let guard = live.library.live_guard
        && live.library.owner_ready(DEFAULT_MODEL)
        && end - start >= GUARD_MIN_SECONDS;
    let admitted = match mode {
        _ if echo => false,
        Mode::Live => {
            live.state
                .segment_fate(false, !guard || verdict.is_none_or(|d| d.is_him))
                == SegmentFate::Admit
        }
        Mode::Ambient => !live.agent_speaking() && !live.in_hold(),
        Mode::Idle | Mode::Muted => false,
    };
    // The score and the bar it is held to, so the threshold can be
    // calibrated from the log; `-` when the segment was not scored.
    let scored = format!(
        "mode={} start={start:.2} end={end:.2} score={} threshold={DEFAULT_THRESHOLD}",
        mode_name(mode),
        verdict
            .and_then(|d| d.score)
            .map_or("-".to_string(), |s| format!("{s:.3}")),
    );
    if !admitted {
        st.log.info(None, &format!("live_segment_dropped {scored}"));
        return vec![json!({"type": "partial", "segment": v["segment"], "text": ""})];
    }
    st.log
        .line("debug", None, &format!("live_segment_admitted {scored}"));
    let id = live.utterance;
    live.utterance += 1;
    let mut event = json!({
        "type": if mode == Mode::Ambient { "ambient_flag" } else { "utterance" },
        "id": id, "segment": v["segment"], "text": text, "start": start, "end": end,
        "decode_ms": v["decode_ms"],
    });
    if let Some(Identity { verdict: d, person }) = &identity {
        event["speaker"] = json!({
            "score": d.score, "is_him": d.is_him, "reason": d.reason, "person": person,
        });
    }
    let mut sent = vec![event];
    if mode == Mode::Ambient {
        let heard = ambient_line(live, text, identity.as_ref()).await;
        // The sender sends its own, so it gets each exactly once and before
        // a `done`; the other members get it through the conversation.
        if let Some((key, id)) = conv {
            for event in &heard {
                publish_event(key, id, event.clone());
            }
        }
        sent.extend(heard);
    }
    sent
}

/// Who the ambient machine is told said a line: the best matching person when
/// the library scored one (the owner is him, anyone else is other, named), a
/// scored line matching nobody is an unnamed other, else unknown (no
/// enrollment, too short, no model).
fn ambient_speaker(identity: Option<&Identity>) -> machine::Speaker {
    match identity {
        Some(i) if i.verdict.reason == speaker::Reason::Scored && i.verdict.is_him => {
            machine::Speaker::Him
        }
        Some(i) if i.verdict.reason == speaker::Reason::Scored => machine::Speaker::Other,
        _ => machine::Speaker::Unknown,
    }
}

fn context_json(context: &[LogLine]) -> Value {
    let lines: Vec<Value> = context
        .iter()
        .map(|l| {
            let who = match l.speaker {
                machine::Speaker::Him => "him",
                machine::Speaker::Other => "other",
                machine::Speaker::Unknown => "unknown",
            };
            let mut line = json!({"speaker": who, "text": l.text, "at_ms": l.at_ms});
            if let Some(name) = &l.person {
                line["person"] = json!(name);
            }
            line
        })
        .collect();
    json!(lines)
}

/// One ambient line through the machine, and the gate when it asks: the
/// `wake` and `ambient_verdict` events it makes. `mode_changed` and the wake
/// candidates are the machine's own bookkeeping and are not sent.
async fn ambient_line(live: &mut Live, text: &str, identity: Option<&Identity>) -> Vec<Value> {
    let now = live.now_ms();
    let similarity = identity.and_then(|i| i.verdict.score);
    // Only another person is named; the owner is "him".
    let name = identity
        .filter(|i| !i.verdict.is_him)
        .and_then(|i| i.person.as_ref())
        .map(|p| p.name.clone());
    live.ambient.set_next_person(name.clone());
    let machine::Ingest { output, gate: req } =
        live.ambient
            .ingest(now, ambient_speaker(identity), text, similarity);
    let mut sent: Vec<Value> = output
        .events
        .iter()
        .filter(|e| e.kind == EventKind::Wake)
        .map(|e| {
            json!({
                "type": "wake", "text": e.text, "remainder": output.utterance,
                "at_ms": e.at_ms, "context": context_json(&e.context),
            })
        })
        .collect();
    if let Some(req) = req {
        // The gate is blocking (a model, when one is configured).
        let judged = tokio::task::spawn_blocking(move || {
            let verdict = gate().evaluate(&req.text, &req.context);
            machine::gate_event(&req, verdict).map(|e| (e, verdict.score))
        })
        .await
        .ok()
        .flatten();
        if let Some((event, score)) = judged {
            let kind = if event.kind == EventKind::IdeaFlagged {
                "idea"
            } else {
                "can_help"
            };
            sent.push(json!({
                "type": "ambient_verdict", "kind": kind, "score": score, "text": event.text,
                "at_ms": event.at_ms, "context": context_json(&event.context),
            }));
        }
    }
    sent
}

/// Who the speaker library says spoke the segment; `None` when the audio is
/// gone. Fails open (`speaker::judge`): a missing model or a short clip is
/// "him".
async fn score(st: &Arc<AppState>, live: &mut Live, start: f64, end: f64) -> Option<Identity> {
    let library = live.library.clone();
    let pcm = live.clip(start, end)?;
    if live.verifier.is_none() && !live.verifier_tried && pcm.len() >= speaker::MIN_SAMPLES {
        live.verifier_tried = true;
        let st2 = st.clone();
        live.verifier = tokio::task::spawn_blocking(move || load_verifier(&st2).ok())
            .await
            .ok()
            .flatten()
            .map(Arc::new);
    }
    let verifier = live.verifier.clone();
    let home = st.registry.home().to_path_buf();
    let migrate = !live.migrate_failed;
    let (identity, failed) = tokio::task::spawn_blocking(move || {
        // A library made with another model is re-embedded once this one is
        // here; until then it fails open.
        let (library, failed) = match &verifier {
            Some(v) if migrate && library.model != DEFAULT_MODEL => {
                match speaker::migrate(&home, v) {
                    Ok(l) => (Arc::new(l), None),
                    Err(e) => (library, Some(e)),
                }
            }
            _ => (library, None),
        };
        let identity = speaker::identify(
            verifier.as_deref(),
            Some(&library),
            false,
            &pcm,
            DEFAULT_THRESHOLD,
        );
        (Some(identity), failed)
    })
    .await
    .unwrap_or((None, None));
    if let Some(e) = failed {
        live.migrate_failed = true;
        st.log
            .line("error", None, &format!("speaker_migrate_failed {e}"));
    }
    // Picks up a migration, so the guard below judges the new library.
    live.refresh_library(st.registry.home());
    identity
}
