//! The server's own member of a live conversation (naru task 2372): the local
//! mic (`crate::mic`) through the same pipeline an audio socket's frames take
//! (VAD, speech-to-text, speaker scoring, the ambient machine and gate), so
//! ambient works with no client attached.
//!
//! Off until `POST /ambient {"on":true}` and never on at startup; nothing is
//! persisted, so a restarted daemon is off. What it hears goes to the other
//! members of its conversation and into an in-memory log, `GET /ambient/log`.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

use super::{
    Event, Fail, Input, Live, MAX_BACKLOG_SAMPLES, Member, Mode, Shared, acquire, anyone_speaking,
    check_id, join, load_library, publish_event, wake_words, worker_event,
};
use crate::ambient::machine::TranscriptLog;
use crate::error::ApiError;
use crate::log::Logger;
use crate::mic::{Mic, MicStatus};
use crate::server::stream::{Start, validate};
use crate::server::transcriptions::stt_manifest;
use crate::server::{AppState, RequestId};

/// The conversation the listener joins when `POST /ambient` names none.
pub const DEFAULT_CONVERSATION: &str = "ambient";

/// The listener's microphone: the stream, kept open while it runs (`None`
/// for frames from elsewhere, as in tests), and its 16 kHz mono frames.
pub struct Capture {
    pub mic: Option<Mic>,
    pub frames: mpsc::Receiver<Vec<f32>>,
}

type Open = dyn Fn() -> Result<Capture, String> + Send + Sync;

/// What `AppState::mic` holds: how to open the microphone, and the listener
/// while it is on.
pub struct AmbientHost {
    open: Box<Open>,
    /// One `POST /ambient` at a time; held while a model loads.
    switch: tokio::sync::Mutex<()>,
    running: Mutex<Option<Listener>>,
    log: Arc<Mutex<EventLog>>,
}

struct Listener {
    conversation: String,
    /// Taken and closed by whichever comes first: `stop` or the task ending.
    mic: SharedMic,
    task: JoinHandle<()>,
}

type SharedMic = Arc<Mutex<Option<Mic>>>;

/// Closes the stream, if still open, on a blocking thread (dropping a `Mic`
/// joins its thread).
async fn close(mic: &SharedMic) {
    let open = mic.lock().unwrap().take();
    if let Some(mic) = open {
        let _ = tokio::task::spawn_blocking(move || drop(mic)).await;
    }
}

impl AmbientHost {
    /// Opens the local input device of `crate::mic::configured_device`
    /// (read at each `on`) when switched on.
    pub fn device(home: PathBuf, log: Arc<Logger>) -> Self {
        Self::new(move || {
            let device = crate::mic::configured_device(&home)?;
            let (tx, frames) = mpsc::channel(64);
            let mic = Mic::start(device, log.clone(), tx).map_err(|e| e.to_string())?;
            Ok(Capture {
                mic: Some(mic),
                frames,
            })
        })
    }

    /// Calls `open` for the frames each time it is switched on.
    pub fn new(open: impl Fn() -> Result<Capture, String> + Send + Sync + 'static) -> Self {
        Self {
            open: Box::new(open),
            switch: tokio::sync::Mutex::new(()),
            running: Mutex::new(None),
            log: Arc::new(Mutex::new(EventLog::default())),
        }
    }

    /// The open microphone's status, while the listener is on.
    pub fn mic_status(&self) -> Option<MicStatus> {
        let running = self.running.lock().unwrap();
        running
            .as_ref()
            .filter(|l| !l.task.is_finished())
            .and_then(|l| l.mic.lock().unwrap().as_ref().map(Mic::status))
    }

    fn status(&self) -> Value {
        let running = self.running.lock().unwrap();
        let on = running.as_ref().filter(|l| !l.task.is_finished());
        json!({
            "on": on.is_some(),
            "conversation": on.map(|l| l.conversation.clone()),
            "mic": on.and_then(|l| l.mic.lock().unwrap().as_ref().map(Mic::status)),
        })
    }

    /// Ends the listener, if any, and waits for it: its conversation
    /// membership and workers go with the task, then the stream closes.
    async fn stop(&self) {
        let listener = self.running.lock().unwrap().take();
        if let Some(l) = listener {
            l.task.abort();
            let _ = l.task.await;
            close(&l.mic).await;
        }
    }
}

/// What the listener heard, as the events it published: bounded like
/// [`TranscriptLog`] (2 min, 500 entries). In memory only.
#[derive(Default)]
struct EventLog {
    events: VecDeque<(Instant, Value)>,
}

impl EventLog {
    fn push(&mut self, event: Value) {
        self.prune();
        self.events.push_back((Instant::now(), event));
        while self.events.len() > TranscriptLog::CAPACITY {
            self.events.pop_front();
        }
    }

    fn recent(&mut self) -> Vec<Value> {
        self.prune();
        self.events.iter().map(|(_, e)| e.clone()).collect()
    }

    fn prune(&mut self) {
        let lifetime = Duration::from_millis(TranscriptLog::LIFETIME_MS);
        self.events.retain(|(at, _)| at.elapsed() < lifetime);
    }
}

fn host(st: &AppState) -> Result<&AmbientHost, ApiError> {
    st.mic.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_unavailable",
            "this daemon has no local microphone",
        )
    })
}

/// `GET /ambient`: `{"on","conversation","mic"}`.
pub(in crate::server) async fn status(
    State(st): State<Arc<AppState>>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(host(&st)?.status()))
}

/// `GET /ambient/log`: `{"events":[…]}`, oldest first.
pub(in crate::server) async fn log(
    State(st): State<Arc<AppState>>,
) -> Result<Json<Value>, ApiError> {
    let events = host(&st)?.log.lock().unwrap().recent();
    Ok(Json(json!({ "events": events })))
}

/// `POST /ambient {"on":true|false,"conversation"?}`: switches the listener;
/// answers as `GET /ambient`. On while on (in any conversation) changes
/// nothing.
pub(in crate::server) async fn switch(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let body: Value = serde_json::from_slice(&body).unwrap_or_default();
    let bad = |message: &str| ApiError::new(StatusCode::BAD_REQUEST, "invalid_request", message);
    let Some(on) = body.get("on").and_then(Value::as_bool) else {
        return Err(bad("\"on\" must be true or false"));
    };
    let key = match body.get("conversation") {
        None | Some(Value::Null) => DEFAULT_CONVERSATION.to_string(),
        Some(Value::String(k)) => k.clone(),
        Some(_) => return Err(bad("\"conversation\" must be a string")),
    };
    check_id("conversation", &key).map_err(|f| bad(&f.message))?;
    let host = host(&st)?;
    let _one = host.switch.lock().await;
    if !on {
        host.stop().await;
        st.log.info(Some(&req_id), "ambient_off");
        return Ok(Json(host.status()));
    }
    if host.status()["on"] == true {
        return Ok(Json(host.status()));
    }
    // A listener that ended on its own.
    host.stop().await;
    let fail = |f: Fail| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, f.code, f.message);
    let cfg = validate(
        serde_json::from_value::<Start>(json!({"sample_rate": crate::mic::RATE}))
            .expect("a valid start"),
        &st.models.stt_default(),
    )
    .map_err(fail)?;
    let manifest = {
        let (st, name) = (st.clone(), cfg.model.clone());
        tokio::task::spawn_blocking(move || stt_manifest(&st, &name))
    }
    .await
    .map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            e.to_string(),
        )
    })??;
    let (model, vad) = acquire(&st, &req_id, manifest, &cfg).await.map_err(fail)?;
    let (library, library_stamp) = load_library(&st, &req_id).await.map_err(fail)?;
    let capture = (host.open)()
        .map_err(|e| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "backend_unavailable", e))?;
    let (member, mode, seq) = join(&key, Some(Mode::Ambient)).map_err(fail)?;
    let mut live = Live::new(mode, seq, library, false);
    live.library_stamp = library_stamp;
    // A member may already be playing a reply.
    if anyone_speaking(&key) {
        live.set_agent_speaking(true);
    }
    live.ambient.set_wake_words(&wake_words());
    let (input, events, backlog) = super::workers(&st, &cfg, vad, model);
    host.log.lock().unwrap().events.clear();
    let mic: SharedMic = Arc::new(Mutex::new(capture.mic));
    let task = tokio::spawn(listen(
        mic.clone(),
        st.clone(),
        live,
        member,
        capture.frames,
        input,
        events,
        backlog,
        host.log.clone(),
    ));
    *host.running.lock().unwrap() = Some(Listener {
        conversation: key.clone(),
        mic,
        task,
    });
    st.log.info(
        Some(&req_id),
        &format!("ambient_on conversation={key} model={}", cfg.model),
    );
    Ok(Json(host.status()))
}

/// The events the listener publishes, and logs.
fn heard(event: &Value) -> bool {
    matches!(
        event["type"].as_str(),
        Some("ambient_flag" | "utterance" | "wake" | "ambient_verdict")
    )
}

/// The listener: `drive` with the mic in place of a socket. It follows its
/// conversation's mode, and shuts the mic while any member's client plays a
/// reply (and `ECHO_HOLD` after). Ends when the frames or the workers do.
#[allow(clippy::too_many_arguments)]
async fn listen(
    mic: SharedMic,
    st: Arc<AppState>,
    mut live: Live,
    mut member: Member,
    mut frames: mpsc::Receiver<Vec<f32>>,
    input: UnboundedSender<Input>,
    mut events: UnboundedReceiver<Event>,
    backlog: Arc<AtomicUsize>,
    log: Arc<Mutex<EventLog>>,
) {
    let key = member.key.clone();
    let mut behind = false;
    let why = loop {
        tokio::select! {
            frame = frames.recv() => {
                let Some(samples) = frame else { break "mic_closed" };
                // Decoding cannot keep up: drop the frame rather than grow.
                let queued = backlog.load(Ordering::SeqCst) + samples.len();
                if queued > MAX_BACKLOG_SAMPLES {
                    if !behind {
                        st.log.line("warn", None, "ambient_backlog dropping mic frames");
                    }
                    behind = true;
                    continue;
                }
                behind = false;
                let fed = live.hear(samples);
                backlog.fetch_add(fed.len(), Ordering::SeqCst);
                let _ = input.send(Input::Audio(fed));
                if let Some(level) = live.level_event() {
                    publish_event(&key, member.id, level);
                }
            }
            event = events.recv() => match event {
                Some(Event::Send(v)) => {
                    for event in worker_event(&st, &mut live, None, v).await {
                        if heard(&event) {
                            log.lock().unwrap().push(event.clone());
                            publish_event(&key, member.id, event);
                        }
                    }
                }
                Some(Event::Fail(e)) => {
                    st.log.line("error", None, &format!("ambient_decode_failed {e}"));
                    break "decode_failed";
                }
                Some(Event::Done) | None => break "workers_stopped",
            },
            shared = member.changed() => {
                if let Shared::State(seq, mode) = shared {
                    if mode != live.mode() {
                        let _ = input.send(Input::Flush);
                        live.set_mode(mode);
                    }
                    let speaking = anyone_speaking(&key);
                    if speaking != live.agent_speaking() {
                        if speaking {
                            let _ = input.send(Input::Flush);
                        }
                        live.set_agent_speaking(speaking);
                    }
                    live.seq = seq;
                }
            }
        }
    };
    st.log
        .line("warn", None, &format!("ambient_listener_end reason={why}"));
    // Not left recording while `GET /ambient` says off.
    close(&mic).await;
}
