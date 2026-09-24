//! §2.4 `GET /v1/audio/transcriptions/stream`: speech-to-text over a
//! WebSocket: finals, and partials when the `start` asks for them.
//!
//! A session runs in three stages, so a decode never stalls the socket or
//! the VAD: the socket task checks frames and writes events; a VAD thread
//! segments the audio with Silero (authoritative, §2.4 "VAD placement");
//! a decode thread decodes each closed segment in order through the model's
//! `decode`, as `POST /v1/audio/transcriptions` does, so the energy and
//! Silero gates run on every segment and a gated one is dropped. A partial
//! is a decode of the open segment's window so far, queued to the same
//! thread, so it always comes before its segment's final.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code};
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio_tungstenite::tungstenite;

use super::transcriptions::{internal, join, stt_manifest};
use super::{AppState, RequestId, manager_error};
use crate::error::ApiError;
use crate::manager::{Guard, KeepAlive};
use crate::stt::audio::TARGET_SAMPLE_RATE;
use crate::stt::vad::{Segmenter, Span, Vad};
use crate::stt::{VadConfig, Vocabulary};

const RATE: usize = TARGET_SAMPLE_RATE as usize;

/// §2.4: the largest binary frame.
const MAX_FRAME_BYTES: usize = 64 * 1024;

/// What the socket reads at all. A frame over [`MAX_FRAME_BYTES`] is
/// refused either way; past this the library refuses it before it is read.
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

/// How long a closing session waits for the client's close.
const CLOSE_LINGER: Duration = Duration::from_secs(1);

/// §2.4 backpressure: undecoded audio a session may queue.
const MAX_BACKLOG_SAMPLES: usize = 30 * RATE;

/// Seconds of the backlog kept free of any one segment.
const BACKLOG_HEADROOM: f32 = 2.0;

/// Silero's frame; the VAD is fed one at a time, so `detected` is checked
/// after every window.
const VAD_WINDOW: usize = 512;

/// sherpa-onnx starts a span this far, plus `min_speech`, before the
/// window that detected it.
const DETECT_LAG: usize = 2 * VAD_WINDOW;

/// `max_segment` cuts at the quietest point of this trailing stretch.
const CUT_SEARCH: usize = RATE;

/// The quietest-point search's window: 20 ms.
const CUT_WINDOW: usize = RATE / 50;

/// §2.4 "Optional partials": the open segment is re-decoded after every
/// 700 ms of its audio...
const PARTIAL_EVERY: usize = 7 * RATE / 10;

/// ...until its window is longer than this.
const PARTIAL_MAX: usize = 8 * RATE;

/// The first message: §2.4 "Handshake". Unknown fields are ignored.
#[derive(Deserialize)]
struct Start {
    model: Option<String>,
    #[serde(default)]
    partials: bool,
    #[serde(default = "s16le")]
    format: String,
    sample_rate: u64,
    #[serde(default)]
    vad: VadFields,
    hotwords: Option<String>,
    keep_alive: Option<Value>,
}

fn s16le() -> String {
    "s16le".to_string()
}

#[derive(Default, Deserialize)]
struct VadFields {
    threshold: Option<f32>,
    min_speech: Option<f32>,
    min_silence: Option<f32>,
    max_segment: Option<f32>,
    pad: Option<f32>,
}

/// Every client text frame.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ClientMessage {
    Start(Box<Start>),
    Flush,
    Stop,
    Hotwords { hotwords: String },
}

#[derive(Clone, Copy)]
enum Format {
    S16le,
    F32le,
}

impl Format {
    fn width(self) -> usize {
        match self {
            Format::S16le => 2,
            Format::F32le => 4,
        }
    }

    /// Whole samples, scaled to [-1, 1) as `audio::decode` scales 16-bit WAV.
    fn samples(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            Format::S16le => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect(),
            Format::F32le => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        }
    }
}

/// A validated `start`. Lengths are in samples.
struct Config {
    model: String,
    format: Format,
    vad: VadConfig,
    max_segment: usize,
    pad: usize,
    hotwords: Option<Arc<Vocabulary>>,
    keep_alive: Option<KeepAlive>,
    partials: bool,
}

/// Why a session ends other than normally: sent as an `error` event, then
/// the close `code` (§2.4 "Close codes").
struct Fail {
    code: &'static str,
    message: String,
    close: u16,
}

impl From<ApiError> for Fail {
    fn from(e: ApiError) -> Self {
        let close = if e.status == StatusCode::INSUFFICIENT_STORAGE {
            close_code::AGAIN
        } else if e.status.is_client_error() {
            close_code::POLICY
        } else {
            close_code::ERROR
        };
        Fail {
            code: e.code,
            message: e.message,
            close,
        }
    }
}

fn violation(code: &'static str, message: impl Into<String>) -> Fail {
    Fail {
        code,
        message: message.into(),
        close: close_code::POLICY,
    }
}

/// How a session that did not fail ended.
enum Ended {
    /// `stop`: every final and `done` were sent.
    Stopped,
    /// The client closed or dropped the connection.
    Gone,
}

/// Socket task to VAD thread.
enum Input {
    Audio(Vec<f32>),
    Flush,
    Stop,
    Hotwords(Option<Arc<Vocabulary>>),
}

/// VAD thread to decode thread, in order.
enum Job {
    Decode {
        pcm: Vec<f32>,
        /// The speech, in seconds since `start`; the padding is not in it.
        start: f64,
        end: f64,
        hotwords: Option<Arc<Vocabulary>>,
    },
    /// The open segment's window so far, for a `partial`.
    Partial {
        pcm: Vec<f32>,
        start: f64,
        hotwords: Option<Arc<Vocabulary>>,
    },
    /// After the last segment of a `stop`.
    Stop,
}

/// Either thread to the socket task.
enum Event {
    Send(Value),
    /// A decode failed: `internal`, 1011.
    Fail(String),
    /// Every final of a `stop` has been sent.
    Done,
}

pub(super) async fn stream(
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

/// One session, then its close: an `error` event always precedes a
/// non-1000 close.
async fn run(st: Arc<AppState>, req_id: String, mut socket: WebSocket) {
    let started = Instant::now();
    let (close, reason) = match session(&st, &req_id, &mut socket).await {
        Ok(Ended::Gone) => {
            st.log.info(Some(&req_id), "stream_end client_gone");
            return;
        }
        Ok(Ended::Stopped) => (close_code::NORMAL, "done"),
        Err(fail) => {
            st.log.line(
                "warn",
                Some(&req_id),
                &format!(
                    "stream_error code={} close={} {}",
                    fail.code, fail.close, fail.message
                ),
            );
            let event = json!({"type": "error", "code": fail.code, "message": fail.message});
            if send(&mut socket, event).await.is_err() {
                return;
            }
            (fail.close, fail.code)
        }
    };
    st.log.info(
        Some(&req_id),
        &format!(
            "stream_end close={close} ms={}",
            started.elapsed().as_millis()
        ),
    );
    let frame = CloseFrame {
        code: close,
        reason: reason.into(),
    };
    if let Err(e) = socket.send(Message::Close(Some(frame))).await {
        st.log
            .line("warn", Some(&req_id), &format!("stream_close_failed {e}"));
        return;
    }
    // Frames the client is still sending would sit unread and turn the
    // drop into a TCP reset, which loses the `error` and the close on the
    // client's side: read and discard until its close, or [`CLOSE_LINGER`].
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
    let cfg = validate(start, &st.models.settings().stt_default)?;
    if let Some(v) = &cfg.hotwords {
        warn_truncated(st, req_id, v);
    }

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
    // The detector's own force-close stays well past `max_segment`, so the
    // quietest-point cut is the one that happens.
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

    let ready = json!({"type": "ready", "session": req_id, "model": cfg.model, "load_ms": load_ms});
    if send(socket, ready).await.is_err() {
        return Ok(Ended::Gone);
    }
    // An x86_64 build under Rosetta is x86_64 too (§3.3).
    if cfg.partials && st.models.settings().profile.arch == "x86_64" {
        let warning = json!({
            "type": "warning",
            "code": "partials_expensive",
            "message": "partials re-decode the open segment every 700 ms; on x86_64 that costs up to about 0.5 s of CPU each and competes with TTS",
        });
        if send(socket, warning).await.is_err() {
            return Ok(Ended::Gone);
        }
    }

    let backlog = Arc::new(AtomicUsize::new(0));
    // A partial is queued or decoding.
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

    let mut stopping = false;
    loop {
        tokio::select! {
            msg = next(socket), if !stopping => {
                let input = match msg? {
                    None => return Ok(Ended::Gone),
                    Some(Message::Binary(bytes)) => {
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
                        let queued = backlog.fetch_add(samples.len(), Ordering::SeqCst) + samples.len();
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
                        Input::Audio(samples)
                    }
                    Some(Message::Text(text)) => match parse(&text)? {
                        ClientMessage::Start(_) => {
                            return Err(violation("invalid_request", "the session has already started"));
                        }
                        ClientMessage::Flush => Input::Flush,
                        ClientMessage::Stop => {
                            stopping = true;
                            Input::Stop
                        }
                        ClientMessage::Hotwords { hotwords } => {
                            let v = hotwords_of(&hotwords)?;
                            if let Some(v) = &v {
                                warn_truncated(st, req_id, v);
                            }
                            Input::Hotwords(v)
                        }
                    },
                    Some(_) => continue,
                };
                // Fails only once the VAD thread has gone, which its
                // event reports.
                let _ = input_tx.send(input);
            }
            event = events.recv() => match event {
                Some(Event::Send(v)) => {
                    if send(socket, v).await.is_err() {
                        return Ok(Ended::Gone);
                    }
                }
                Some(Event::Done) => {
                    if send(socket, json!({"type": "done"})).await.is_err() {
                        return Ok(Ended::Gone);
                    }
                    return Ok(Ended::Stopped);
                }
                Some(Event::Fail(message)) => return Err(internal(st, req_id, message).into()),
                // Both threads ended without `Done`: one panicked.
                None => {
                    return Err(internal(st, req_id, "the session's workers stopped".to_string()).into());
                }
            },
        }
    }
}

/// The next data frame; `None` once the client has closed or dropped the
/// connection. A frame the library refuses (over [`MAX_MESSAGE_BYTES`], or
/// malformed) is a protocol violation. Pings are answered by the library.
async fn next(socket: &mut WebSocket) -> Result<Option<Message>, Fail> {
    loop {
        let Some(msg) = socket.recv().await else {
            return Ok(None);
        };
        match msg {
            Ok(Message::Ping(_) | Message::Pong(_)) => continue,
            Ok(Message::Close(_)) => return Ok(None),
            Ok(msg) => return Ok(Some(msg)),
            Err(e) => {
                let e = e.into_inner();
                return match e.downcast_ref::<tungstenite::Error>() {
                    Some(tungstenite::Error::Capacity(_) | tungstenite::Error::Protocol(_)) => Err(
                        violation("invalid_request", format!("unreadable frame: {e}")),
                    ),
                    _ => Ok(None),
                };
            }
        }
    }
}

async fn send(socket: &mut WebSocket, event: Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(event.to_string().into())).await
}

fn parse(text: &str) -> Result<ClientMessage, Fail> {
    serde_json::from_str(text)
        .map_err(|e| violation("invalid_request", format!("unreadable message: {e}")))
}

/// An empty string clears the hotwords.
fn hotwords_of(s: &str) -> Result<Option<Arc<Vocabulary>>, Fail> {
    let v = Vocabulary::parse(s).map_err(|e| violation("invalid_request", e.to_string()))?;
    Ok((!v.terms.is_empty()).then(|| Arc::new(v)))
}

fn warn_truncated(st: &AppState, req_id: &str, v: &Vocabulary) {
    if v.terms_dropped > 0 {
        st.log.line(
            "warn",
            Some(req_id),
            &format!(
                "hotwords_truncated dropped={} kept={}",
                v.terms_dropped,
                v.terms.len()
            ),
        );
    }
}

/// `default_model` is what `default` resolves to (§3.3).
fn validate(start: Start, default_model: &str) -> Result<Config, Fail> {
    let format = match start.format.as_str() {
        "s16le" => Format::S16le,
        "f32le" => Format::F32le,
        other => {
            return Err(violation(
                "unsupported_value",
                format!("format {other:?} is not supported; use s16le or f32le"),
            ));
        }
    };
    if start.sample_rate != TARGET_SAMPLE_RATE as u64 {
        return Err(violation(
            "unsupported_value",
            format!(
                "sample_rate must be {TARGET_SAMPLE_RATE}, got {}; resample on the client",
                start.sample_rate
            ),
        ));
    }

    // The same bounds as §2.2's `vad_*` fields, plus the two §2.4 adds.
    let bad = |field: &str, rule: &str, v: f32| {
        violation(
            "invalid_request",
            format!("vad.{field} must be {rule}, got {v}"),
        )
    };
    let fields = start.vad;
    let mut vad = VadConfig::default();
    if let Some(v) = fields.threshold {
        if !(0.0..=1.0).contains(&v) {
            return Err(bad("threshold", "between 0.0 and 1.0", v));
        }
        vad.threshold = v;
    }
    if let Some(v) = fields.min_speech {
        if !v.is_finite() || v < 0.0 {
            return Err(bad("min_speech", "a non-negative number of seconds", v));
        }
        vad.min_speech = v;
    }
    if let Some(v) = fields.min_silence {
        // Strictly positive: zero trailing silence never closes a span.
        if !v.is_finite() || v <= 0.0 {
            return Err(bad("min_silence", "a positive number of seconds", v));
        }
        vad.min_silence = v;
    }
    // At least the cut's search window.
    let max_segment = fields.max_segment.unwrap_or(20.0);
    if !(1.0..=(MAX_BACKLOG_SAMPLES / RATE) as f32).contains(&max_segment) {
        return Err(bad("max_segment", "between 1 and 30 seconds", max_segment));
    }
    let pad = fields.pad.unwrap_or(0.3);
    if !(0.0..=1.0).contains(&pad) {
        return Err(bad("pad", "between 0 and 1 second", pad));
    }
    // One padded segment must fit in the backlog with [`BACKLOG_HEADROOM`]
    // to spare for the frames that arrive while it decodes, or the session
    // ends in `backlog`.
    let longest = max_segment + 2.0 * pad;
    let limit = (MAX_BACKLOG_SAMPLES / RATE) as f32 - BACKLOG_HEADROOM;
    if longest > limit {
        return Err(violation(
            "invalid_request",
            format!("vad.max_segment + 2 * vad.pad must be at most {limit} seconds, got {longest}"),
        ));
    }

    let hotwords = start
        .hotwords
        .as_deref()
        .map(hotwords_of)
        .transpose()?
        .flatten();
    let keep_alive = match start.keep_alive {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(KeepAlive::parse(&s)),
        Some(Value::Number(n)) => Some(KeepAlive::from_secs(n.as_f64().unwrap_or(f64::NAN))),
        Some(_) => Some(Err(
            "keep_alive must be a duration string or a number of seconds".to_string(),
        )),
    }
    .transpose()
    .map_err(|e| violation("invalid_request", e))?;
    let model = match start.model.as_deref() {
        None | Some("default") => default_model,
        Some(m) => m,
    };

    Ok(Config {
        model: model.to_string(),
        format,
        vad,
        max_segment: (max_segment * RATE as f32) as usize,
        pad: (pad * RATE as f32) as usize,
        hotwords,
        keep_alive,
        partials: start.partials,
    })
}

/// The VAD thread: until `stop`, or until the socket task goes.
fn segment(vad: Vad, mut segments: Segments, mut input: UnboundedReceiver<Input>) {
    let mut segmenter = vad.segmenter();
    while let Some(input) = input.blocking_recv() {
        match input {
            Input::Audio(samples) => {
                segments.push(&mut segmenter, &samples);
                segments.backlog.fetch_sub(samples.len(), Ordering::SeqCst);
            }
            Input::Flush => segments.cut(segments.received),
            Input::Stop => {
                segments.cut(segments.received);
                // Silero never closes this span: the stream ends first.
                if segments.open.is_some() {
                    segments.speech(false, segments.received);
                }
                let _ = segments.jobs.send(Job::Stop);
                return;
            }
            Input::Hotwords(v) => segments.hotwords = v,
        }
    }
}

/// The VAD thread's state. Positions are samples since `start`.
///
/// Silero's spans are authoritative: a span that closes (`min_silence`)
/// is a segment. Two things cut a segment early without touching the
/// detector: `max_segment`, at the quietest point of its last second, and
/// `flush`/`stop`, at the latest sample. `floor` is where the last segment
/// ended, so audio after a cut goes to the next segment, never both.
struct Segments {
    /// The recent audio: enough for `pad` before any span that can still
    /// open, and the whole open one.
    ring: VecDeque<f32>,
    ring_start: usize,
    received: usize,
    floor: usize,
    /// Where the open span started, estimated when it was detected.
    open: Option<usize>,
    min_speech: usize,
    max_segment: usize,
    pad: usize,
    hotwords: Option<Arc<Vocabulary>>,
    partials: bool,
    /// Where the last partial tick was, queued or skipped.
    partial_at: usize,
    busy: Arc<AtomicBool>,
    jobs: UnboundedSender<Job>,
    events: UnboundedSender<Event>,
    backlog: Arc<AtomicUsize>,
}

impl Segments {
    fn new(
        cfg: &Config,
        jobs: UnboundedSender<Job>,
        events: UnboundedSender<Event>,
        backlog: Arc<AtomicUsize>,
        busy: Arc<AtomicBool>,
    ) -> Self {
        Segments {
            ring: VecDeque::new(),
            ring_start: 0,
            received: 0,
            floor: 0,
            open: None,
            min_speech: (cfg.vad.min_speech * RATE as f32) as usize,
            max_segment: cfg.max_segment,
            pad: cfg.pad,
            hotwords: cfg.hotwords.clone(),
            partials: cfg.partials,
            partial_at: 0,
            busy,
            jobs,
            events,
            backlog,
        }
    }

    fn push(&mut self, segmenter: &mut Segmenter<'_>, samples: &[f32]) {
        for window in samples.chunks(VAD_WINDOW) {
            self.ring.extend(window);
            self.received += window.len();
            segmenter.accept(window);
            while let Some(span) = segmenter.next_span() {
                self.close(span);
            }
            if self.open.is_none() && segmenter.detected() {
                let start = self.received.saturating_sub(DETECT_LAG + self.min_speech);
                self.open = Some(start);
                self.speech(true, start);
            }
            if let Some(open) = self.open {
                let start = open.max(self.floor);
                // Searched up to `max_segment` exactly, so no segment is longer.
                let limit = start + self.max_segment;
                if self.received >= limit {
                    let from = limit - CUT_SEARCH;
                    let cut = from + quietest(self.slice(from, limit));
                    self.cut(cut);
                }
            }
            self.tick();
        }
        // Keep `pad` before the earliest start a span can still report:
        // the open one's, or one detected from the next window on.
        let anchor = self.open.map_or(self.received, |o| o.max(self.floor));
        let keep = anchor.saturating_sub(self.pad + self.min_speech + DETECT_LAG + VAD_WINDOW);
        if keep > self.ring_start {
            self.ring.drain(..keep - self.ring_start);
            self.ring_start = keep;
        }
    }

    /// Silero closed `span`: whatever of it a cut has not already taken.
    fn close(&mut self, span: Span) {
        let end = span.start + span.len;
        if self.open.take().is_some() {
            self.speech(false, end);
            // The next segment's partials count from its own start.
            self.partial_at = 0;
        }
        let start = span.start.max(self.floor);
        // A remainder shorter than `min_speech` after a cut is not speech
        // Silero would have kept on its own.
        if end < start + self.min_speech.max(1) {
            self.floor = self.floor.max(end);
            return;
        }
        self.queue(start, end, end + self.pad);
    }

    /// Ends the open segment at `at`; Silero's span goes on, and whatever
    /// of it follows `at` is the next segment.
    fn cut(&mut self, at: usize) {
        let Some(open) = self.open else { return };
        let start = open.max(self.floor);
        if at > start {
            self.queue(start, at, at);
        }
    }

    /// Queues the speech `start..end` for decode, padded to
    /// `start - pad..audio_end` with audio no earlier segment has taken.
    fn queue(&mut self, start: usize, end: usize, audio_end: usize) {
        let from = start
            .saturating_sub(self.pad)
            .max(self.floor)
            .max(self.ring_start);
        let pcm = self.slice(from, audio_end.min(self.received)).to_vec();
        self.floor = end;
        self.backlog.fetch_add(pcm.len(), Ordering::SeqCst);
        let _ = self.jobs.send(Job::Decode {
            pcm,
            start: start as f64 / RATE as f64,
            end: end as f64 / RATE as f64,
            hotwords: self.hotwords.clone(),
        });
    }

    /// A partial of the open segment every [`PARTIAL_EVERY`] of its audio
    /// while its decoded window, `pad` included, is at most
    /// [`PARTIAL_MAX`]; a tick while the last one is still queued or
    /// decoding is skipped. Not counted in the backlog: at most one is ever
    /// waiting.
    fn tick(&mut self) {
        let Some(open) = self.open.filter(|_| self.partials) else {
            return;
        };
        let start = open.max(self.floor);
        let from = start
            .saturating_sub(self.pad)
            .max(self.floor)
            .max(self.ring_start);
        if self.received < self.partial_at.max(start) + PARTIAL_EVERY
            || self.received - from > PARTIAL_MAX
        {
            return;
        }
        self.partial_at = self.received;
        if self.busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let pcm = self.slice(from, self.received).to_vec();
        let _ = self.jobs.send(Job::Partial {
            pcm,
            start: start as f64 / RATE as f64,
            hotwords: self.hotwords.clone(),
        });
    }

    fn slice(&mut self, from: usize, to: usize) -> &[f32] {
        &self.ring.make_contiguous()[from - self.ring_start..to - self.ring_start]
    }

    /// auris's `speech` line.
    fn speech(&self, active: bool, at: usize) {
        let at = at as f64 / RATE as f64;
        let _ = self.events.send(Event::Send(
            json!({"type": "speech", "active": active, "at": at}),
        ));
    }
}

/// The middle of the quietest [`CUT_WINDOW`] in `pcm`, by mean square.
fn quietest(pcm: &[f32]) -> usize {
    pcm.as_chunks::<CUT_WINDOW>()
        .0
        .iter()
        .map(|w| w.iter().map(|s| s * s).sum::<f32>())
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(pcm.len(), |(i, _)| i * CUT_WINDOW + CUT_WINDOW / 2)
}

/// The decode thread: one segment at a time, in order. Holding `model`
/// keeps it loaded for the session (§3.4).
///
/// §2.4's decode FIFO shared by every session of a model is not
/// implemented: sessions decode concurrently, as `POST` requests do.
fn decode(
    model: Guard,
    vad: VadConfig,
    mut jobs: UnboundedReceiver<Job>,
    events: UnboundedSender<Event>,
    backlog: Arc<AtomicUsize>,
    busy: Arc<AtomicBool>,
) {
    let mut index = 0;
    while let Some(job) = jobs.blocking_recv() {
        // The client has gone: stop, and let the model go.
        if events.is_closed() {
            return;
        }
        let (pcm, start, end, hotwords) = match job {
            Job::Decode {
                pcm,
                start,
                end,
                hotwords,
            } => (pcm, start, end, hotwords),
            Job::Partial {
                pcm,
                start,
                hotwords,
            } => {
                // Anything queued after it means its segment has closed:
                // the final is next, and replaces it.
                let result = jobs
                    .is_empty()
                    .then(|| model.stt().decode(&pcm, hotwords.as_deref(), Some(&vad)));
                busy.store(false, Ordering::SeqCst);
                let text = match result {
                    None => continue,
                    Some(Ok(segments)) => join(&segments),
                    Some(Err(e)) => {
                        let _ = events.send(Event::Fail(e.to_string()));
                        return;
                    }
                };
                if !text.is_empty() {
                    // `index` is the one its segment's final will take.
                    let _ = events.send(Event::Send(json!({
                        "type": "partial",
                        "segment": index,
                        "start": start,
                        "text": text,
                    })));
                }
                continue;
            }
            Job::Stop => {
                let _ = events.send(Event::Done);
                return;
            }
        };
        let started = Instant::now();
        let result = model.stt().decode(&pcm, hotwords.as_deref(), Some(&vad));
        backlog.fetch_sub(pcm.len(), Ordering::SeqCst);
        let text = match result {
            Ok(segments) => join(&segments),
            Err(e) => {
                let _ = events.send(Event::Fail(e.to_string()));
                return;
            }
        };
        // A gated segment has no text and no `final` (§2.4).
        if text.is_empty() {
            continue;
        }
        let _ = events.send(Event::Send(json!({
            "type": "final",
            "segment": index,
            "start": start,
            "end": end,
            "text": text,
            "decode_ms": started.elapsed().as_millis() as u64,
        })));
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quietest_is_the_middle_of_the_lowest_energy_window() {
        let mut pcm = vec![0.5f32; CUT_SEARCH];
        let quiet = 17 * CUT_WINDOW;
        pcm[quiet..quiet + CUT_WINDOW].fill(0.01);
        assert_eq!(quietest(&pcm), quiet + CUT_WINDOW / 2);
    }

    #[test]
    fn s16le_and_f32le_decode_to_the_same_samples() {
        let s16: Vec<u8> = [0i16, 16384, -32768]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let f32: Vec<u8> = [0.0f32, 0.5, -1.0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(Format::S16le.samples(&s16), [0.0, 0.5, -1.0]);
        assert_eq!(Format::F32le.samples(&f32), [0.0, 0.5, -1.0]);
    }

    /// Segments with a segment open from `pad`, so the whole pad is in
    /// its window, and its job queue.
    fn open_segment(partials: bool) -> (Segments, UnboundedReceiver<Job>, Arc<AtomicBool>) {
        let start: Start =
            serde_json::from_value(json!({"sample_rate": 16000, "partials": partials})).unwrap();
        let Ok(cfg) = validate(start, "m") else {
            panic!("a valid start")
        };
        let (jobs, rx) = unbounded_channel();
        let busy = Arc::new(AtomicBool::new(false));
        let mut segments = Segments::new(
            &cfg,
            jobs,
            unbounded_channel().0,
            Arc::new(AtomicUsize::new(0)),
            busy.clone(),
        );
        segments.open = Some(cfg.pad);
        (segments, rx, busy)
    }

    /// Feeds `windows` VAD windows of the open segment, as `push` does;
    /// the lengths of the partials queued. `decoded` clears `busy` after
    /// each, as the decode thread does.
    fn feed(
        segments: &mut Segments,
        jobs: &mut UnboundedReceiver<Job>,
        busy: &AtomicBool,
        windows: usize,
        decoded: bool,
    ) -> Vec<usize> {
        let mut lens = Vec::new();
        for _ in 0..windows {
            segments.ring.extend([0.0; VAD_WINDOW]);
            segments.received += VAD_WINDOW;
            segments.tick();
            while let Ok(job) = jobs.try_recv() {
                let Job::Partial { pcm, .. } = job else {
                    panic!("only partials are queued")
                };
                lens.push(pcm.len());
                if decoded {
                    busy.store(false, Ordering::SeqCst);
                }
            }
        }
        lens
    }

    fn partials_over_10_s(partials: bool, decoded: bool) -> Vec<usize> {
        let (mut segments, mut jobs, busy) = open_segment(partials);
        feed(
            &mut segments,
            &mut jobs,
            &busy,
            10 * RATE / VAD_WINDOW,
            decoded,
        )
    }

    /// The window starts `pad` before the speech, and it is the window
    /// that stops at 8 s.
    #[test]
    fn partials_tick_every_700_ms_until_8_s() {
        let lens = partials_over_10_s(true, true);
        assert_eq!(lens.len(), 10, "{lens:?}");
        let mut last = 0;
        for len in lens {
            assert!(len >= last + PARTIAL_EVERY, "{len} after {last}");
            assert!(len <= PARTIAL_MAX, "{len}");
            last = len;
        }
    }

    #[test]
    fn a_tick_while_a_partial_is_busy_is_skipped() {
        // The first window at or past 700 ms of speech, after the pad.
        assert_eq!(partials_over_10_s(true, false), [32 * VAD_WINDOW]);
    }

    /// A segment that opens before the last one's final tick still gets
    /// its first partial 700 ms after its own start.
    #[test]
    fn a_new_segment_ticks_from_its_own_start() {
        let (mut segments, mut jobs, busy) = open_segment(true);
        let pad = segments.pad;
        // One tick, at 16 384.
        assert_eq!(
            feed(&mut segments, &mut jobs, &busy, 32, true),
            [32 * VAD_WINDOW]
        );
        // Silero closes the span at 12 800, before that tick.
        segments.close(Span {
            start: pad,
            len: 8000,
        });
        assert!(matches!(jobs.try_recv(), Ok(Job::Decode { .. })));
        segments.open = Some(13_000);
        // 13 000 + 700 ms is first reached at 24 576, not 700 ms after
        // the old tick (27 648); the window starts at the floor, 12 800.
        let lens = feed(&mut segments, &mut jobs, &busy, 24, true);
        assert_eq!(lens.first(), Some(&(48 * VAD_WINDOW - 12_800)), "{lens:?}");
    }

    #[test]
    fn partials_off_queue_nothing() {
        assert!(partials_over_10_s(false, true).is_empty());
    }
}
