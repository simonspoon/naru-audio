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

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
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
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task::JoinHandle;

use super::stream::{
    CLOSE_LINGER, Config, Event, Fail, Input, MAX_BACKLOG_SAMPLES, MAX_FRAME_BYTES,
    MAX_MESSAGE_BYTES, RATE, Segments, Start, decode, next, segment, send, validate, violation,
    warn_truncated,
};
use super::transcriptions::{decode_error, internal, kind_manifest, stt_manifest};
use super::{AppState, RequestId, manager_error};
use crate::error::ApiError;
use crate::live_mode::{LiveState, Mode, SegmentFate};
use crate::registry::manifest::Kind;
use crate::speaker::{self, DEFAULT_MODEL, DEFAULT_THRESHOLD, Enrollment, SpeakerVerifier};
use crate::stt::vad::Vad;

/// A `level` event is sent at most this often.
const LEVEL_EVERY: Duration = Duration::from_millis(100);

/// The microphone stays shut this long after `agent_speaking` goes false:
/// the tail of the reply is still in the room.
const ECHO_HOLD: Duration = Duration::from_millis(400);

/// How long a dropped session's state can be resumed.
const RESUME_TTL: Duration = Duration::from_secs(60);

/// Audio kept for scoring a finished segment: the backlog plus the decode lag.
const RING_SAMPLES: usize = 45 * RATE;

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

/// The first message.
#[derive(Deserialize)]
struct LiveStart {
    /// Required unless `session` names a session that can be resumed.
    mode: Option<String>,
    /// Sample rate of the microphone frames: 16000.
    rate: u64,
    session: Option<String>,
    model: Option<String>,
    /// Partial utterances; on unless `false`.
    partials: Option<bool>,
    #[serde(default = "s16le")]
    format: String,
    vad: Option<Value>,
    hotwords: Option<String>,
    keep_alive: Option<Value>,
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
    enrolled: Option<Arc<Enrollment>>,
    verifier: Option<Arc<SpeakerVerifier>>,
    verifier_tried: bool,
    /// Raw microphone audio of a running enrollment recording.
    enrolling: Option<Vec<f32>>,
    /// The audio fed to the VAD, as `Segments` positions it.
    ring: VecDeque<f32>,
    ring_start: usize,
    received: usize,
    peak: f32,
    last_level: Instant,
    utterance: u64,
    speaks: HashMap<String, JoinHandle<()>>,
}

impl Live {
    fn new(mode: Mode, seq: u64, enrolled: Option<Enrollment>) -> Self {
        let mut live = Live {
            state: LiveState::new(),
            seq,
            hold_until: None,
            enrolled: enrolled.map(Arc::new),
            verifier: None,
            verifier_tried: false,
            enrolling: None,
            ring: VecDeque::new(),
            ring_start: 0,
            received: 0,
            peak: 0.0,
            last_level: Instant::now(),
            utterance: 0,
            speaks: HashMap::new(),
        };
        live.set_mode(mode);
        live
    }

    /// Drives [`LiveState`] to `mode`: live and joined unless idle, ambient
    /// as asked, muted only while the pipeline is wanted.
    fn set_mode(&mut self, mode: Mode) {
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
        let mut ids: Vec<&String> = self.speaks.keys().collect();
        ids.sort();
        let mut event = json!({
            "type": "state",
            "mode": mode_name(self.mode()),
            "seq": self.seq,
            "agent_speaking": self.agent_speaking(),
            "enrolled": self.enrolled.is_some(),
            "enrolling": self.enrolling.is_some(),
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
    let mut fields = json!({"sample_rate": start.rate, "partials": start.partials.unwrap_or(true)});
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
        (Some(mode), _) => mode,
        (None, Some(prev)) => prev.mode,
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
    let model = st
        .models
        .acquire(manifest, cfg.keep_alive)
        .await
        .map_err(|e| manager_error(st, req_id, e))?;
    let load_ms = if loaded {
        0
    } else {
        load_started.elapsed().as_millis() as u64
    };
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

    // A missing or unreadable enrollment is "not enrolled": the guard fails open.
    let enrolled = {
        let home = st.registry.home().to_path_buf();
        tokio::task::spawn_blocking(move || Enrollment::load(&home))
    }
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))?
    .unwrap_or_else(|e| {
        st.log.line(
            "warn",
            Some(req_id),
            &format!("live_enrollment_unreadable {e}"),
        );
        None
    });

    let mut live = Live::new(mode, previous.as_ref().map_or(0, |p| p.seq), enrolled);
    let ready = json!({
        "type": "ready", "session": session_id, "model": cfg.model, "load_ms": load_ms,
        "resumed": previous.is_some(),
    });
    if send(socket, ready).await.is_err() {
        return Ok(Ended::Gone);
    }
    let first = live.state_event(previous.as_ref());
    remember_state(&session_id, &live);
    if send(socket, first).await.is_err() {
        return Ok(Ended::Gone);
    }

    let result = drive(st, req_id, socket, &session_id, &mut live, cfg, vad, model).await;
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

#[allow(clippy::too_many_arguments)]
async fn drive(
    st: &Arc<AppState>,
    req_id: &str,
    socket: &mut WebSocket,
    session_id: &str,
    live: &mut Live,
    cfg: Config,
    vad: Vad,
    model: crate::manager::Guard,
) -> Result<Ended, Fail> {
    let backlog = Arc::new(AtomicUsize::new(0));
    let busy = Arc::new(AtomicBool::new(false));
    let (input_tx, input_rx) = unbounded_channel();
    let (job_tx, job_rx) = unbounded_channel();
    let (event_tx, mut events) = unbounded_channel();
    {
        let (events, backlog, busy) = (event_tx.clone(), backlog.clone(), busy.clone());
        let segments = Segments::new(&cfg, job_tx, events, backlog, busy);
        tokio::task::spawn_blocking(move || segment(vad, segments, input_rx));
    }
    {
        let (vad, backlog) = (cfg.vad.clone(), backlog.clone());
        tokio::task::spawn_blocking(move || decode(model, vad, job_rx, event_tx, backlog, busy));
    }
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
                        if bytes.len() % cfg.format.width() != 0 {
                            return Err(violation(
                                "invalid_request",
                                format!("a {}-byte frame is not a whole number of samples", bytes.len()),
                            ));
                        }
                        let samples = cfg.format.samples(&bytes);
                        live.note_level(&samples);
                        if let Some(clip) = &mut live.enrolling {
                            let room = ENROLL_MAX_SAMPLES.saturating_sub(clip.len());
                            clip.extend_from_slice(&samples[..samples.len().min(room)]);
                        }
                        // A shut gate feeds silence: the VAD keeps its clock.
                        let fed = if live.gate_open() { samples } else { vec![0.0; samples.len()] };
                        live.push_ring(&fed);
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
                        if let Some(level) = live.level_event()
                            && send(socket, level).await.is_err()
                        {
                            return Ok(Ended::Gone);
                        }
                    }
                    Message::Text(text) => {
                        let sent = client_message(
                            st, req_id, session_id, live, &out_tx, &input_tx, &mut stopping, &text,
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
                    for event in worker_event(st, live, v).await {
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

/// One client text frame after `start`; the events to send back.
#[allow(clippy::too_many_arguments)]
async fn client_message(
    st: &Arc<AppState>,
    req_id: &str,
    session_id: &str,
    live: &mut Live,
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
            if mode != live.mode() {
                // What the old mode heard is the old mode's: cut it first.
                let _ = input.send(Input::Flush);
                live.set_mode(mode);
                sent.push(live.state_event(None));
                remember_state(session_id, live);
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
                sent.push(live.state_event(None));
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
        ClientMessage::Enroll { action } => {
            sent.extend(enroll(st, req_id, live, &action).await);
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

/// `enroll` actions: `begin` records the microphone, `finish` enrolls the
/// recording, `cancel` drops it, `clear` deletes the stored enrollment.
async fn enroll(st: &Arc<AppState>, req_id: &str, live: &mut Live, action: &str) -> Vec<Value> {
    match action {
        "begin" => {
            live.enrolling = Some(Vec::new());
            vec![live.state_event(None)]
        }
        "cancel" => {
            live.enrolling = None;
            vec![live.state_event(None)]
        }
        "finish" => {
            let Some(clip) = live.enrolling.take() else {
                return vec![soft(
                    "invalid_request",
                    "no enrollment recording is running",
                    None,
                )];
            };
            let st2 = st.clone();
            let result = tokio::task::spawn_blocking(move || -> Result<Enrollment, ApiError> {
                if clip.len() < speaker::MIN_SAMPLES {
                    return Err(ApiError::new(
                        axum::http::StatusCode::BAD_REQUEST,
                        "invalid_request",
                        "an enrollment recording needs at least 1 s of audio",
                    ));
                }
                let verifier = load_verifier(&st2)?;
                let enrollment = verifier
                    .enroll(&[clip])
                    .map_err(|e| internal(&st2, "live", e.to_string()))?;
                enrollment
                    .save(st2.registry.home())
                    .map_err(|e| internal(&st2, "live", e.to_string()))?;
                Ok(enrollment)
            })
            .await;
            match result {
                Ok(Ok(enrollment)) => {
                    live.enrolled = Some(Arc::new(enrollment));
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
            let deleted = tokio::task::spawn_blocking(move || Enrollment::delete(&home)).await;
            match deleted {
                Ok(Ok(_)) => {
                    live.enrolled = None;
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

/// An event from the VAD or decode thread, as the events to send.
async fn worker_event(st: &Arc<AppState>, live: &mut Live, v: Value) -> Vec<Value> {
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
        Some("final") => finish_segment(st, live, &v).await,
        _ => vec![v],
    }
}

/// A decoded segment: gated, scored, then an `utterance` (live), an
/// `ambient_flag` (ambient) or nothing.
async fn finish_segment(st: &Arc<AppState>, live: &mut Live, v: &Value) -> Vec<Value> {
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
    let verdict = if mode == Mode::Live && live.enrolled.is_some() {
        score(st, live, start, end).await
    } else {
        None
    };
    let admitted = match mode {
        Mode::Live => {
            live.state
                .segment_fate(false, verdict.is_none_or(|d| d.is_him))
                == SegmentFate::Admit
        }
        Mode::Ambient => !live.agent_speaking() && !live.in_hold(),
        Mode::Idle | Mode::Muted => false,
    };
    if !admitted {
        st.log.info(
            None,
            &format!(
                "live_segment_dropped mode={} start={start:.2} end={end:.2}",
                mode_name(mode)
            ),
        );
        return vec![json!({"type": "partial", "segment": v["segment"], "text": ""})];
    }
    let id = live.utterance;
    live.utterance += 1;
    let mut event = json!({
        "type": if mode == Mode::Ambient { "ambient_flag" } else { "utterance" },
        "id": id, "segment": v["segment"], "text": text, "start": start, "end": end,
        "decode_ms": v["decode_ms"],
    });
    if let Some(d) = verdict {
        event["speaker"] = json!({"score": d.score, "is_him": d.is_him, "reason": d.reason});
    }
    vec![event]
}

/// The speaker guard's verdict on the segment; `None` when the audio is gone.
/// The guard fails open (`speaker::verify`): a missing model or a short
/// clip is "him".
async fn score(
    st: &Arc<AppState>,
    live: &mut Live,
    start: f64,
    end: f64,
) -> Option<speaker::Verdict> {
    let enrollment = live.enrolled.clone()?;
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
    tokio::task::spawn_blocking(move || {
        speaker::verify(
            verifier.as_deref(),
            Some(&enrollment),
            &pcm,
            DEFAULT_THRESHOLD,
        )
    })
    .await
    .ok()
}
