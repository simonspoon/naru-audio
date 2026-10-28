//! naru-audio checks in to naru serve as the ambient host (naru task 2373),
//! the role the Naru Mac app's page played.
//!
//! While naru has a live session, every `TICK` it reports whether the
//! listener (`super::host`) is on (`POST /api/live/route {"ambient"}`, stale
//! in naru after 10 s), and obeys the session's `ambient_request` once per
//! `ambient_request_seq`, so `naru live ambient on|off` switches it. The
//! listener's `wake` and `ambient_verdict` events go to naru's ambient engine
//! (`POST /api/workflows/events`). Checking in never switches the listener
//! on. naru being down is logged, never fatal.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::host::{self, DEFAULT_CONVERSATION};
use crate::server::AppState;

/// naru serve's address when neither `NARU_AUDIO_NARU_URL` nor config.toml's
/// `[naru] url` names one.
pub const DEFAULT_NARU_URL: &str = "http://127.0.0.1:7777";

/// How often the host reports; well inside naru's 10 s staleness.
const TICK: Duration = Duration::from_secs(2);

/// Per request to naru serve.
const TIMEOUT: Duration = Duration::from_secs(2);

/// `NARU_AUDIO_NARU_URL`, then config.toml's `[naru] url`, then
/// [`DEFAULT_NARU_URL`]; without a trailing `/`.
pub fn naru_url(home: &Path) -> Result<String, String> {
    let url = match std::env::var("NARU_AUDIO_NARU_URL")
        .ok()
        .filter(|u| !u.is_empty())
    {
        Some(url) => url,
        None => match crate::registry::read_config(home)?
            .get("naru")
            .and_then(|n| n.get("url"))
        {
            None => DEFAULT_NARU_URL.to_string(),
            Some(toml::Value::String(url)) if !url.is_empty() => url.clone(),
            Some(_) => {
                return Err(format!(
                    "{}: naru.url must be a URL",
                    crate::registry::config_path(home).display()
                ));
            }
        },
    };
    Ok(url.trim_end_matches('/').to_string())
}

/// The running check-in; [`Checkin::shutdown`] clears the report.
pub struct Checkin {
    task: JoinHandle<()>,
    url: String,
    /// The session last reported to, for the shutdown's `null`.
    reported: Arc<std::sync::Mutex<Option<i64>>>,
}

/// Starts checking in to naru serve at `url`. Needs `st.mic`.
pub fn spawn(st: Arc<AppState>, url: String) -> Checkin {
    let (tx, events) = mpsc::unbounded_channel();
    if let Some(host) = st.mic.as_ref() {
        host.forward_to(tx);
    }
    let reported = Arc::new(std::sync::Mutex::new(None));
    let task = tokio::spawn(run(st, url.clone(), events, reported.clone()));
    Checkin {
        task,
        url,
        reported,
    }
}

impl Checkin {
    /// Stops checking in and reports `null` (no host) to the session last
    /// reported to, so naru does not wait out the staleness.
    pub async fn shutdown(self) {
        self.task.abort();
        let _ = self.task.await;
        if self.reported.lock().unwrap().is_some() {
            let url = format!("{}/api/live/route", self.url);
            let _ = tokio::task::spawn_blocking(move || post(&url, &json!({ "ambient": null })))
                .await;
        }
    }
}

async fn run(
    st: Arc<AppState>,
    url: String,
    mut events: mpsc::UnboundedReceiver<Value>,
    reported: Arc<std::sync::Mutex<Option<i64>>>,
) {
    st.log.info(None, &format!("naru_checkin url={url}"));
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The `(session, ambient_request_seq)` last acted on, the switch it
    // started, and whether naru answered last time (logged on change only).
    let mut acted: Option<(i64, i64)> = None;
    let mut switching: Option<JoinHandle<()>> = None;
    let mut up: Option<bool> = None;
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let ok = check(&st, &url, &mut acted, &mut switching, &reported).await;
                if let Err(e) = &ok {
                    if up != Some(false) {
                        st.log.line("warn", None, &format!("naru_checkin_failed {e}"));
                    }
                } else if up == Some(false) {
                    st.log.info(None, "naru_checkin_ok");
                }
                up = Some(ok.is_ok());
            }
            Some(event) = events.recv() => {
                // Its own task: a slow naru never holds up the report.
                tokio::spawn(emit(st.clone(), url.clone(), event));
            }
        }
    }
}

/// One tick: read naru's live session, act on a new `ambient_request`, then
/// report the listener's state.
async fn check(
    st: &Arc<AppState>,
    url: &str,
    acted: &mut Option<(i64, i64)>,
    switching: &mut Option<JoinHandle<()>>,
    reported: &std::sync::Mutex<Option<i64>>,
) -> Result<(), String> {
    // A cursor past every turn: only the session is wanted.
    let live = format!("{url}/api/live?after={}", i64::MAX);
    let body = tokio::task::spawn_blocking(move || get(&live))
        .await
        .map_err(|e| e.to_string())??;
    let session = &body["session"];
    let Some(id) = session["id"].as_i64() else {
        *reported.lock().unwrap() = None;
        return Ok(());
    };
    let Some(host) = st.mic.as_ref() else {
        return Ok(());
    };
    let seq = session["ambient_request_seq"].as_i64().unwrap_or(0);
    // A session first seen (this daemon just started, or a new session):
    // whatever request it holds predates this host, so it is not obeyed —
    // a restarted daemon stays off until asked again.
    if acted.is_none_or(|(sid, _)| sid != id) {
        *acted = Some((id, seq));
    }
    let busy = switching.as_ref().is_some_and(|t| !t.is_finished());
    if let Some(want) = session["ambient_request"].as_bool()
        && *acted != Some((id, seq))
        && !busy
    {
        *acted = Some((id, seq));
        if want != host.on() {
            // Its own task: a model load must not hold up the reports.
            let st = st.clone();
            *switching = Some(tokio::spawn(async move {
                let req = format!("naru_ambient_request_{id}_{seq}");
                if let Err(e) = host::set(&st, &req, want, DEFAULT_CONVERSATION.to_string()).await {
                    st.log.line(
                        "error",
                        Some(&req),
                        &format!("naru_ambient_request_failed on={want} {}", e.message),
                    );
                }
            }));
        }
    }
    let route = format!("{url}/api/live/route");
    let report = json!({ "ambient": host.on() });
    tokio::task::spawn_blocking(move || post(&route, &report))
        .await
        .map_err(|e| e.to_string())??;
    *reported.lock().unwrap() = Some(id);
    Ok(())
}

/// The ambient-engine event for a listener event, as the Mac app sent it:
/// `wake` (speaker unknown), `can-help` or `idea` (speaker him).
fn engine_event(event: &Value) -> Option<Value> {
    let (name, speaker) = match (event["type"].as_str()?, event["kind"].as_str()) {
        ("wake", _) => ("wake", "unknown"),
        ("ambient_verdict", Some("can_help")) => ("can-help", "him"),
        ("ambient_verdict", _) => ("idea", "him"),
        _ => return None,
    };
    Some(json!({ "event": name, "speaker": speaker, "text": event["text"] }))
}

async fn emit(st: Arc<AppState>, url: String, event: Value) {
    let Some(body) = engine_event(&event) else {
        return;
    };
    let name = body["event"].as_str().unwrap_or_default().to_string();
    let events = format!("{url}/api/workflows/events");
    let sent = tokio::task::spawn_blocking(move || post(&events, &body)).await;
    match sent {
        Ok(Ok(_)) => st.log.info(None, &format!("naru_event event={name}")),
        Ok(Err(e)) => st
            .log
            .line("warn", None, &format!("naru_event_failed event={name} {e}")),
        Err(e) => st
            .log
            .line("warn", None, &format!("naru_event_failed event={name} {e}")),
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .build()
        .into()
}

fn get(url: &str) -> Result<Value, String> {
    let text = agent()
        .get(url)
        .call()
        .and_then(|mut r| r.body_mut().read_to_string())
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

fn post(url: &str, body: &Value) -> Result<(), String> {
    agent()
        .post(url)
        .header("content-type", "application/json")
        .send(body.to_string())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_events_map_as_the_mac_app_sent_them() {
        let wake = json!({"type": "wake", "text": "hey naru", "remainder": ""});
        assert_eq!(
            engine_event(&wake),
            Some(json!({"event": "wake", "speaker": "unknown", "text": "hey naru"}))
        );
        let help = json!({"type": "ambient_verdict", "kind": "can_help", "text": "t"});
        assert_eq!(engine_event(&help).unwrap()["event"], "can-help");
        let idea = json!({"type": "ambient_verdict", "kind": "idea", "text": "t"});
        assert_eq!(
            engine_event(&idea),
            Some(json!({"event": "idea", "speaker": "him", "text": "t"}))
        );
        assert_eq!(engine_event(&json!({"type": "utterance"})), None);
    }
}
