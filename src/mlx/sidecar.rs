//! §5.3 the sidecar process and its framed protocol. The frame format and
//! the ops are documented in `mlx/naru_audio_mlx/protocol.py`; [`encode`]
//! and [`read_frame`] are its Rust side.
//!
//! One [`Sidecar`] per home: it starts `python -m naru_audio_mlx` on the
//! first MLX load, sends it one request at a time, and stops it when the
//! last MLX model unloads. [`Sidecar::synth`] is the one request with more
//! than one answer frame: it reads the stream to its end frame, cancelled
//! or not, so the next request starts in step. The child exits when its stdin, a pipe from the
//! daemon, closes, so it never outlives the daemon, however that ends.
//!
//! A request that finds the sidecar dead, or loses it mid-request, gets
//! [`SttError::BackendUnavailable`] (503 `backend_unavailable`). The
//! sidecar then restarts after 1, 2 and 4 s, reloading every loaded model,
//! and after that waits for the next request to start it.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::registry::manifest::Kind;
use crate::stt::SttError;

/// The largest header either side accepts: `MAX_HEADER` in `protocol.py`.
pub const MAX_HEADER: usize = 1 << 20;
/// The most samples one frame carries, an hour at 16 kHz: `MAX_SAMPLES`
/// in `protocol.py`.
pub const MAX_SAMPLES: usize = 16_000 * 60 * 60;

/// How long a starting sidecar has to listen on its socket: it imports MLX
/// first.
const READY_TIMEOUT: Duration = Duration::from_secs(60);
/// §5.3 restart backoff after a crash, in seconds.
const BACKOFF: [u64; 3] = [1, 2, 4];

/// One frame: `header` with `"samples"` set to their count when there are
/// samples. Over either cap is `InvalidInput`, and nothing is encoded.
pub fn encode(header: &Value, samples: Option<&[f32]>) -> io::Result<Vec<u8>> {
    let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    let mut header = header.clone();
    let Some(object) = header.as_object_mut() else {
        return Err(invalid("a frame's header must be a JSON object".into()));
    };
    if let Some(samples) = samples {
        if samples.len() > MAX_SAMPLES {
            return Err(invalid(format!(
                "{} samples is over {MAX_SAMPLES}",
                samples.len()
            )));
        }
        object.insert("samples".into(), json!(samples.len()));
    }
    let body = serde_json::to_vec(&header)?;
    if body.len() > MAX_HEADER {
        return Err(invalid(format!(
            "a header of {} bytes is over {MAX_HEADER}",
            body.len()
        )));
    }
    let samples = samples.unwrap_or_default();
    let mut frame = Vec::with_capacity(4 + body.len() + 4 * samples.len());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(&body);
    for s in samples {
        frame.extend_from_slice(&s.to_le_bytes());
    }
    Ok(frame)
}

/// Reads one frame: its header, and its samples when it has any. Over
/// either cap, or a header that is not a JSON object, is `InvalidData`.
pub fn read_frame(r: &mut impl Read) -> io::Result<(Value, Option<Vec<f32>>)> {
    let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let mut prefix = [0; 4];
    r.read_exact(&mut prefix)?;
    let len = u32::from_le_bytes(prefix) as usize;
    if len > MAX_HEADER {
        return Err(invalid(format!(
            "a header of {len} bytes is over {MAX_HEADER}"
        )));
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    let header: Value = serde_json::from_slice(&body)?;
    if !header.is_object() {
        return Err(invalid("a frame's header must be a JSON object".into()));
    }
    let Some(count) = header.get("samples") else {
        return Ok((header, None));
    };
    let count = count
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= MAX_SAMPLES)
        .ok_or_else(|| {
            invalid(format!(
                "samples {count} is not a count up to {MAX_SAMPLES}"
            ))
        })?;
    let mut bytes = vec![0; 4 * count];
    r.read_exact(&mut bytes)?;
    let samples = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((header, Some(samples)))
}

static SIDECARS: Mutex<Option<HashMap<PathBuf, Arc<Sidecar>>>> = Mutex::new(None);

/// The sidecar of `home`, which may not be running.
pub fn for_home(home: &Path) -> Arc<Sidecar> {
    let mut sidecars = SIDECARS.lock().unwrap_or_else(|e| e.into_inner());
    sidecars
        .get_or_insert_with(HashMap::new)
        .entry(home.to_path_buf())
        .or_insert_with(|| {
            Arc::new(Sidecar {
                home: home.to_path_buf(),
                state: Mutex::new(State::default()),
                reason: Mutex::new(None),
            })
        })
        .clone()
}

/// §2.5 `/health` `backends[mlx].reason`: why `home`'s sidecar is down
/// after a crash, if it is.
pub fn reason(home: &Path) -> Option<String> {
    let sidecars = SIDECARS.lock().unwrap_or_else(|e| e.into_inner());
    let sidecar = sidecars.as_ref()?.get(home)?;
    sidecar
        .reason
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub struct Sidecar {
    home: PathBuf,
    /// Held for a whole request, so requests go one at a time.
    state: Mutex<State>,
    /// Never held across I/O, so `/health` never waits.
    reason: Mutex<Option<String>>,
}

#[derive(Default)]
struct State {
    process: Option<Process>,
    /// Loaded models to their kinds and directories, reloaded into a
    /// restarted sidecar, and the instance of each that [`Sidecar::load`]
    /// returned.
    models: BTreeMap<String, (Kind, PathBuf, u64)>,
    /// The last instance handed out.
    instances: u64,
    /// A restart thread is waiting out the backoff.
    restarting: bool,
}

struct Process {
    /// Shared with [`Sidecar::watch`], which reaps it if it exits alone.
    child: Arc<Mutex<Child>>,
    stream: UnixStream,
    /// Set under `child`'s lock by [`State::stop`], so the watcher does not
    /// report a stop as a crash.
    stopped: Arc<AtomicBool>,
}

impl State {
    /// Kills the sidecar if it is still running, and reaps it.
    fn stop(&mut self) -> Option<ExitStatus> {
        let process = self.process.take()?;
        let mut child = process.child.lock().unwrap_or_else(|e| e.into_inner());
        process.stopped.store(true, Ordering::SeqCst);
        // Both no-ops once the watcher has reaped it: `Child` keeps the
        // status, so a reused pid is never signalled.
        let _ = child.kill();
        child.wait().ok()
    }
}

impl Sidecar {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_reason(&self, reason: Option<String>) {
        *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = reason;
    }

    /// Loads `model`, a `kind` model, from `dir`, starting the sidecar if
    /// need be: the instance to [`Sidecar::unload`], the bytes the load
    /// added to MLX's active memory (§3.5), and the `load` answer.
    ///
    /// The model manager drops an unloaded model off the runtime, so a
    /// reload can come before the unload of the instance it replaces. The
    /// reload unloads that instance first, so its delta is its own, and the
    /// late unload then names an instance that is gone and does nothing.
    pub fn load(
        self: &Arc<Self>,
        model: &str,
        kind: Kind,
        dir: &Path,
    ) -> Result<(u64, u64, Value), SttError> {
        let mut state = self.lock();
        let replaced = state.models.remove(model).is_some();
        let ready = if state.process.is_none() {
            self.start(&mut state)
        } else if replaced {
            self.request(&mut state, json!({"op": "unload", "model": model}), None)
                .map(drop)
        } else {
            Ok(())
        };
        match ready.and_then(|()| self.load_one(&mut state, model, kind, dir)) {
            Ok((bytes, answer)) => {
                state.instances += 1;
                let instance = state.instances;
                state
                    .models
                    .insert(model.to_string(), (kind, dir.to_path_buf(), instance));
                Ok((instance, bytes, answer))
            }
            Err(e) => {
                if state.models.is_empty() {
                    state.stop();
                }
                Err(e)
            }
        }
    }

    fn load_one(
        self: &Arc<Self>,
        state: &mut State,
        model: &str,
        kind: Kind,
        dir: &Path,
    ) -> Result<(u64, Value), SttError> {
        let before = self.active_bytes(state)?;
        let answer = self.request(
            state,
            json!({"op": "load", "model": model, "kind": kind.as_str(), "dir": dir}),
            None,
        )?;
        Ok((self.active_bytes(state)?.saturating_sub(before), answer))
    }

    fn active_bytes(self: &Arc<Self>, state: &mut State) -> Result<u64, SttError> {
        let answer = self.request(state, json!({"op": "stats"}), None)?;
        answer["active_bytes"]
            .as_u64()
            .ok_or_else(|| SttError::Sidecar("`stats` has no active_bytes".to_string()))
    }

    /// The texts of the segments `model` transcribes `samples` (16 kHz) to.
    pub fn transcribe(
        self: &Arc<Self>,
        model: &str,
        samples: &[f32],
    ) -> Result<Vec<String>, SttError> {
        let mut state = self.lock();
        if state.process.is_none() {
            self.start(&mut state)?;
        }
        let answer = self.request(
            &mut state,
            json!({"op": "transcribe", "model": model}),
            Some(samples),
        )?;
        Ok(answer["segments"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["text"].as_str())
            .map(str::to_string)
            .collect())
    }

    /// Synthesises with `model`: `request` is the `synth` header's other
    /// keys (`protocol.py`), and `sink` gets each chunk of samples as it
    /// comes. A `false` from the sink cancels: the sidecar is told to stop,
    /// and the chunks already on their way are read and dropped up to the
    /// end frame. A panic in the sink does the same, then resumes.
    pub fn synth(
        self: &Arc<Self>,
        model: &str,
        mut request: Value,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), SttError> {
        let mut state = self.lock();
        if state.process.is_none() {
            self.start(&mut state)?;
        }
        request["op"] = json!("synth");
        request["model"] = json!(model);
        let frame = encode(&request, None).map_err(|e| SttError::Sidecar(e.to_string()))?;
        let cancel = encode(&json!({"op": "cancel"}), None).expect("a small header");
        let Some(process) = &mut state.process else {
            return Err(not_running());
        };
        let mut panic = None;
        let end = process.stream.write_all(&frame).and_then(|()| {
            let mut cancelled = false;
            loop {
                let (header, samples) = read_frame(&mut process.stream)?;
                if header.get("end").is_some() {
                    return Ok(header);
                }
                if cancelled {
                    continue;
                }
                let samples = samples.unwrap_or_default();
                let more =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink(&samples)))
                        .unwrap_or_else(|payload| {
                            panic = Some(payload);
                            false
                        });
                if !more {
                    cancelled = true;
                    process.stream.write_all(&cancel)?;
                }
            }
        });
        let result = match end {
            Ok(end) => answer(end).map(drop),
            Err(e) => Err(self.crashed(&mut state, e)),
        };
        if let Some(payload) = panic {
            drop(state);
            std::panic::resume_unwind(payload);
        }
        result
    }

    /// Unloads `instance` of `model`, unless it has been replaced; the last
    /// model out stops the sidecar.
    pub fn unload(self: &Arc<Self>, model: &str, instance: u64) {
        let mut state = self.lock();
        if state
            .models
            .get(model)
            .is_none_or(|(_, _, i)| *i != instance)
        {
            return;
        }
        state.models.remove(model);
        if state.models.is_empty() {
            state.stop();
            self.set_reason(None);
        } else if state.process.is_some() {
            // A failure here is a crash, handled like any other.
            let _ = self.request(&mut state, json!({"op": "unload", "model": model}), None);
        }
    }

    /// Starts the sidecar, waits until it listens, and reloads every model
    /// in `state.models`.
    fn start(self: &Arc<Self>, state: &mut State) -> Result<(), SttError> {
        let unavailable = |reason: String| SttError::BackendUnavailable {
            backend: "mlx".to_string(),
            reason,
        };
        let python = super::python(&self.home).map_err(unavailable)?;
        let dir = super::dir(&self.home);
        let socket = dir.join("sidecar.sock");
        let _ = std::fs::remove_file(&socket);
        let mut child = Command::new(&python)
            .args(["-m", "naru_audio_mlx", "--socket"])
            .arg(&socket)
            .current_dir(&dir)
            // Models are loaded from their pulled directories only.
            .env("HF_HUB_OFFLINE", "1")
            // The sidecar exits when this pipe closes (`protocol.py`).
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(|e| unavailable(format!("cannot start {}: {e}", python.display())))?;
        let started = Instant::now();
        let stream = loop {
            if let Ok(stream) = UnixStream::connect(&socket) {
                break stream;
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(unavailable(format!(
                    "the sidecar exited as it started ({status}); see `naru-audio mlx status`"
                )));
            }
            if started.elapsed() > READY_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err(unavailable(format!(
                    "the sidecar did not listen within {} s",
                    READY_TIMEOUT.as_secs()
                )));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let child = Arc::new(Mutex::new(child));
        let stopped = Arc::new(AtomicBool::new(false));
        self.watch(child.clone(), stopped.clone());
        state.process = Some(Process {
            child,
            stream,
            stopped,
        });
        let models: Vec<(String, Kind, PathBuf)> = state
            .models
            .iter()
            .map(|(m, (k, d, _))| (m.clone(), *k, d.clone()))
            .collect();
        for (model, kind, dir) in models {
            if let Err(e) = self.load_one(state, &model, kind, &dir) {
                state.stop();
                return Err(e);
            }
        }
        self.set_reason(None);
        Ok(())
    }

    /// One request and its answer. An answer with an error is
    /// [`SttError::Sidecar`]; a broken connection is a crash.
    fn request(
        self: &Arc<Self>,
        state: &mut State,
        header: Value,
        samples: Option<&[f32]>,
    ) -> Result<Value, SttError> {
        let frame = encode(&header, samples).map_err(|e| SttError::Sidecar(e.to_string()))?;
        let Some(process) = &mut state.process else {
            return Err(not_running());
        };
        let reply = process
            .stream
            .write_all(&frame)
            .and_then(|()| read_frame(&mut process.stream));
        match reply {
            Ok((reply, _)) => answer(reply),
            Err(e) => Err(self.crashed(state, e)),
        }
    }

    /// Waits on the thread for `child` to exit, and when it exits alone,
    /// reaps it and says so on `/health` at once, before any request finds
    /// it dead (which then handles it as [`Sidecar::crashed`]).
    fn watch(self: &Arc<Self>, child: Arc<Mutex<Child>>, stopped: Arc<AtomicBool>) {
        let pid = child.lock().unwrap_or_else(|e| e.into_inner()).id();
        let this = self.clone();
        std::thread::spawn(move || {
            // `WNOWAIT`: wait without reaping, so the pid stays the child's
            // until `Child` reaps it and records its status.
            // SAFETY: `info` is a valid out-pointer for the call.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            while unsafe {
                libc::waitid(libc::P_PID, pid, &mut info, libc::WEXITED | libc::WNOWAIT)
            } != 0
            {
                if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    // Already reaped by `stop`, or no way to wait.
                    return;
                }
            }
            let mut child = child.lock().unwrap_or_else(|e| e.into_inner());
            if stopped.load(Ordering::SeqCst) {
                return;
            }
            // Under `child`'s lock: a `stop`, and so the restart that
            // clears the reason, comes after this.
            if let Ok(Some(status)) = child.try_wait() {
                this.set_reason(Some(format!(
                    "the sidecar died ({status}); it restarts on the next request"
                )));
            }
        });
    }

    /// The connection broke: reap the sidecar, say so on `/health`, and
    /// restart it with backoff while models are loaded.
    fn crashed(self: &Arc<Self>, state: &mut State, e: io::Error) -> SttError {
        let died = match state.stop() {
            Some(status) => format!("the sidecar died ({status}): {e}"),
            None => format!("the sidecar died: {e}"),
        };
        if !state.models.is_empty() {
            self.set_reason(Some(format!("{died}; restarting")));
            if !state.restarting {
                state.restarting = true;
                let this = self.clone();
                std::thread::spawn(move || this.restart());
            }
        } else {
            self.set_reason(Some(format!("{died}; it restarts on the next request")));
        }
        SttError::BackendUnavailable {
            backend: "mlx".to_string(),
            reason: died,
        }
    }

    /// §5.3 backoff: restart after 1, 2 and 4 s unless a request has
    /// already, then leave it to the next request.
    fn restart(self: Arc<Self>) {
        for (i, delay) in BACKOFF.iter().enumerate() {
            std::thread::sleep(Duration::from_secs(*delay));
            // `restarting` is cleared under the same lock as the last
            // look, so a crash after it always starts a new thread.
            let mut state = self.lock();
            if state.process.is_some() || state.models.is_empty() {
                state.restarting = false;
                return;
            }
            match self.start(&mut state) {
                Ok(()) => {
                    state.restarting = false;
                    return;
                }
                Err(e) => self.set_reason(Some(match BACKOFF.get(i + 1) {
                    Some(next) => format!("{e}; restarting in {next} s"),
                    None => {
                        state.restarting = false;
                        format!("{e}; it restarts on the next request")
                    }
                })),
            }
        }
    }
}

fn not_running() -> SttError {
    SttError::BackendUnavailable {
        backend: "mlx".to_string(),
        reason: "the sidecar is not running".to_string(),
    }
}

/// An answer header: itself when `ok`, else [`SttError::Sidecar`].
fn answer(header: Value) -> Result<Value, SttError> {
    if header["ok"].as_bool() == Some(true) {
        Ok(header)
    } else {
        Err(SttError::Sidecar(
            header["error"]
                .as_str()
                .unwrap_or("no reason given")
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(frame: &[u8]) -> io::Result<(Value, Option<Vec<f32>>)> {
        read_frame(&mut io::Cursor::new(frame))
    }

    #[test]
    fn a_frame_is_a_length_prefixed_header_then_f32le_samples() {
        let frame = encode(&json!({"op": "transcribe"}), Some(&[0.5, -1.0])).unwrap();
        let body = br#"{"op":"transcribe","samples":2}"#;
        assert_eq!(frame[..4], (body.len() as u32).to_le_bytes());
        assert_eq!(&frame[4..4 + body.len()], body);
        assert_eq!(
            frame[4 + body.len()..4 + body.len() + 4],
            0.5f32.to_le_bytes()
        );
        assert_eq!(frame.len(), 4 + body.len() + 8);
        assert_eq!(
            decode(&frame).unwrap(),
            (
                json!({"op": "transcribe", "samples": 2}),
                Some(vec![0.5, -1.0])
            )
        );
    }

    #[test]
    fn a_frame_without_samples_is_the_header_alone() {
        let frame = encode(&json!({"op": "stats"}), None).unwrap();
        assert_eq!(frame.len(), 4 + br#"{"op":"stats"}"#.len());
        assert_eq!(decode(&frame).unwrap(), (json!({"op": "stats"}), None));
    }

    #[test]
    fn frames_over_the_caps_are_refused() {
        let kind = |r: io::Result<Vec<u8>>| r.unwrap_err().kind();
        assert_eq!(
            kind(encode(&json!({"pad": "x".repeat(MAX_HEADER)}), None)),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            kind(encode(&json!({}), Some(&vec![0.0; MAX_SAMPLES + 1]))),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(kind(encode(&json!([1]), None)), io::ErrorKind::InvalidInput);

        let kind = |frame: &[u8]| decode(frame).unwrap_err().kind();
        let mut huge = ((MAX_HEADER + 1) as u32).to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert_eq!(kind(&huge), io::ErrorKind::InvalidData);
        for header in [
            format!(r#"{{"samples":{}}}"#, MAX_SAMPLES + 1),
            r#"{"samples":-1}"#.to_string(),
            "[1]".to_string(),
        ] {
            let mut frame = (header.len() as u32).to_le_bytes().to_vec();
            frame.extend_from_slice(header.as_bytes());
            assert_eq!(kind(&frame), io::ErrorKind::InvalidData, "{header}");
        }
    }

    #[test]
    fn a_truncated_frame_is_an_unexpected_eof() {
        let frame = encode(&json!({"op": "transcribe"}), Some(&[0.25; 4])).unwrap();
        for cut in [0, 2, 6, frame.len() - 1] {
            assert_eq!(
                decode(&frame[..cut]).unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof,
                "{cut}"
            );
        }
    }
}
