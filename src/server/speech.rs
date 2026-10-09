//! §2.3 `POST /v1/audio/speech`, §2.5 `GET`/`POST /v1/audio/voices` and
//! `GET`/`DELETE /v1/audio/voices/{name}`.
//!
//! Speech checks run cheapest first: the JSON fields, then the model
//! (404/409/400) and the voice against its manifest, and only then the load.
//! The response starts once the first sentence is synthesised, so a failure
//! before that still gets its status code; one after it aborts the chunked
//! body (§2.3).

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Extension, Multipart, Path, Query, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{AppendHeaders, IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};

use super::transcriptions::{bad_request, internal, kind_manifest, multipart_error, not_kind};
use super::{AppState, RequestId, manager_error};
use crate::error::ApiError;
use crate::manager::{Guard, KeepAlive};
use crate::registry::manifest::{Kind, Manifest, Voice};
use crate::tts::{SynthOptions, TtsError};
use crate::voices::{self, AddError};

/// OpenAI model names that stand for `default`, so stock clients work.
const DEFAULT_ALIASES: [&str; 4] = ["default", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"];

/// §2.3 `input` cap, in characters.
pub(super) const MAX_INPUT_CHARS: usize = 16_384;

/// §2.3 streaming `wav`: the length is unknown when the header goes out.
const STREAM_DATA_SIZE: u32 = 0x7FFF_0000;

/// Sentences of audio buffered for a slow client before synthesis waits.
const QUEUED_PIECES: usize = 4;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Wav,
    Pcm,
}

/// A validated request.
struct Job {
    model: String,
    /// `model` was asked for by name, not defaulted from no `"model"` (or
    /// one of the OpenAI aliases): a cloned voice only picks its own model
    /// ([`voices::say_model`]) when this is false, and only an explicit
    /// mismatched `model` is refused (§5.3 "refuse mismatch").
    explicit_model: bool,
    input: String,
    voice: Option<String>,
    format: Format,
    stream: bool,
    options: SynthOptions,
    keep_alive: Option<KeepAlive>,
    /// `"knobs"` (naru task 1458), parsed as finite numbers but not yet
    /// checked against a model: `validate` runs before the manifest is
    /// fetched, so `speech` resolves them once it has one (`apply_knobs`).
    knobs: HashMap<String, f64>,
}

pub(super) async fn speech(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let mut job = validate(&body, &st.models.tts_default())?;
    // §5.3: with no explicit `model`, a cloned voice picks its own model
    // (`say_model`, generalised from the CLI's) instead of the daemon's
    // `tts_default`; with an explicit one, a cloned voice made for another
    // model is refused rather than silently ignored or 404 "unknown_voice".
    // A voice that is one of `name`'s own built-ins always wins over a
    // same-named cloned voice made for some other model — Pocket's own
    // "bria" is not shadowed by a clone somebody happened to also call
    // "bria" for Qwen3-TTS Base.
    let (manifest, voices) = {
        let (st, name, voice, explicit) = (
            st.clone(),
            job.model.clone(),
            job.voice.clone(),
            job.explicit_model,
        );
        tokio::task::spawn_blocking(move || {
            let home = st.registry.home();
            let manifest = kind_manifest(&st, &name, Kind::Tts)?;
            let built_in = voice
                .as_deref()
                .is_some_and(|v| manifest.voices.iter().any(|mv| mv.id == v));
            let manifest = if built_in {
                manifest
            } else {
                let made_for = voice.as_deref().and_then(|v| voices::model_of(home, v));
                match (&made_for, explicit) {
                    (Some(made_for), false) if made_for != &name => {
                        kind_manifest(&st, made_for, Kind::Tts)?
                    }
                    (Some(made_for), true) if made_for != &name => {
                        return Err(voice_model_mismatch(
                            voice.as_deref().unwrap(),
                            made_for,
                            &name,
                        ));
                    }
                    _ => manifest,
                }
            };
            let voices = voices::of(&manifest, home);
            Ok::<_, ApiError>((manifest, voices))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    apply_knobs(&manifest, job.knobs, &mut job.options)?;
    let default_voice = st.models.default_voice(&manifest.model.name);
    let voice = voice(
        &manifest,
        &voices,
        job.voice.as_deref(),
        job.options.instructions.as_deref(),
        default_voice.as_deref(),
    )?;
    // Only a model that takes `instructions` sees them.
    if !manifest.instructs() {
        job.options.instructions = None;
    }
    let reference = job
        .options
        .reference
        .as_ref()
        .map(|(wav, _)| wav.clone())
        .or_else(|| {
            voices::find(st.registry.home(), &voice)
                .filter(|c| c.model == manifest.model.name)
                .map(|c| c.wav)
        });
    // A sampling knob the request did not send is the backend's own default.
    let knob = |name: &str| {
        job.options
            .knobs
            .get(name)
            .map_or_else(|| "default".to_string(), |v| v.to_string())
    };
    let recent_id = recent_id();
    st.log.line(
        "info",
        Some(&req_id),
        &format!(
            "tts_request id={recent_id} voice={voice} reference={} model={} temperature={} top_p={} seed={} chars={} sentences={}",
            reference.as_ref().map_or("none".into(), |p| p.display().to_string()),
            manifest.model.name,
            knob("temperature"),
            knob("top_p"),
            job.options.seed,
            job.input.chars().count(),
            sentence_count(&job.input),
        ),
    );
    // Only a model that takes `exaggeration` sees it.
    if !manifest.exaggerates() {
        job.options.exaggeration = None;
    }
    let model = st
        .models
        .acquire(manifest, job.keep_alive)
        .await
        .map_err(|e| manager_error(&st, &req_id, e))?;
    let sample_rate = model.tts().sample_rate();

    let recent = Some(Recent {
        home: st.registry.home().to_path_buf(),
        id: recent_id,
        log: st.log.clone(),
        req_id: req_id.clone(),
    });
    let mut rx = synthesise(model, job.input, voice, job.options, recent);
    if !job.stream {
        let mut pcm = Vec::new();
        while let Some(piece) = rx.recv().await {
            pcm.extend_from_slice(&piece.map_err(|e| failed(&st, &req_id, e))?);
        }
        let body = match job.format {
            Format::Wav => {
                let mut wav = wav_header(sample_rate, pcm.len() as u32).to_vec();
                wav.extend_from_slice(&pcm);
                wav
            }
            Format::Pcm => pcm,
        };
        // A full body: axum sets `Content-Length`.
        return Ok(audio_response(job.format, sample_rate, body));
    }

    // Held back until the first sentence is out of the model, so a failure
    // before any audio is still a status code.
    let first = match rx.recv().await {
        Some(Ok(piece)) => Some(piece),
        Some(Err(e)) => return Err(failed(&st, &req_id, e)),
        None => None,
    };
    let head = match job.format {
        Format::Wav => Bytes::copy_from_slice(&wav_header(sample_rate, STREAM_DATA_SIZE)),
        Format::Pcm => Bytes::new(),
    };
    let first = [head]
        .into_iter()
        .chain(first)
        .filter(|b| !b.is_empty())
        .map(Ok::<_, std::io::Error>);
    let rest = futures_util::stream::unfold((rx, st, req_id), |(mut rx, st, req_id)| async move {
        match rx.recv().await? {
            Ok(piece) => Some((Ok(piece), (rx, st, req_id))),
            Err(e) => {
                // Pending once first, so hyper flushes the audio already
                // written: an error in the same write loop drops it unsent.
                tokio::task::yield_now().await;
                // An `Err` makes hyper drop the connection without the
                // terminating zero-length chunk.
                st.log.line(
                    "error",
                    Some(&req_id),
                    &format!("tts_stream_aborted {}", e.message),
                );
                Some((Err(std::io::Error::other(e.message)), (rx, st, req_id)))
            }
        }
    });
    let chunks = futures_util::stream::StreamExt::chain(futures_util::stream::iter(first), rest);
    Ok((
        AppendHeaders(headers(job.format, sample_rate)),
        Body::from_stream(chunks),
    )
        .into_response())
}

/// Synthesises on the blocking pool, sending each sentence as s16le bytes,
/// or the failure's text last. When the receiver is dropped (the client went
/// away and hyper dropped the body) the next send fails and the sink cancels
/// the rest; the model is released when synthesis returns. With `recent`
/// (`home`, id), what was synthesised, even if cut short, is then written
/// to `<home>/recent/<id>.wav` ([`save_recent`]).
fn synthesise(
    model: Guard,
    input: String,
    voice: String,
    options: SynthOptions,
    recent: Option<Recent>,
) -> tokio::sync::mpsc::Receiver<Result<Bytes, Failure>> {
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUED_PIECES);
    tokio::task::spawn_blocking(move || {
        let sample_rate = model.tts().sample_rate();
        let sink_tx = tx.clone();
        // The audio so far as s16le, kept for `save_recent` only.
        let kept = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let sink_kept = recent.is_some().then(|| kept.clone());
        let sink = Box::new(move |samples: &[f32]| {
            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
                .collect();
            if let Some(kept) = &sink_kept {
                kept.lock().unwrap().extend_from_slice(&bytes);
            }
            sink_tx.blocking_send(Ok(Bytes::from(bytes))).is_ok()
        });
        let result = catch_unwind(AssertUnwindSafe(|| {
            model
                .tts()
                .synth(&input, &voice, &options, sink)
                .map_err(|e| Failure {
                    unavailable: matches!(e, TtsError::BackendUnavailable { .. }),
                    message: e.to_string(),
                })
        }))
        .unwrap_or_else(|_| {
            Err(Failure {
                unavailable: false,
                message: "the synthesis panicked".to_string(),
            })
        });
        if let Err(e) = result {
            let _ = tx.blocking_send(Err(e));
        }
        // The response ends and the model is free before the file is written.
        drop(tx);
        drop(model);
        if let Some(r) = recent {
            let pcm = std::mem::take(&mut *kept.lock().unwrap());
            // No audio, nothing worth evicting a recording for.
            if pcm.is_empty() {
                return;
            }
            if let Err(e) = save_recent(&r.home, &r.id, sample_rate, &pcm) {
                r.log.line(
                    "error",
                    Some(&r.req_id),
                    &format!("recent_save_failed id={} {e}", r.id),
                );
            }
        }
    });
    rx
}

/// How many sentences the MLX sidecar's `split_sentences` makes of `text`:
/// split after `. ! ?` before whitespace, after `。！？`, and at newlines;
/// a piece under 12 characters joins the next (a short last one, the
/// previous), so it only ever counts once.
fn sentence_count(text: &str) -> usize {
    const MIN_SENTENCE: usize = 12;
    let mut pieces = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\n' {
            pieces.push(std::mem::take(&mut cur));
            continue;
        }
        cur.push(c);
        let ends = match c {
            '。' | '！' | '？' => true,
            '.' | '!' | '?' => chars.peek().is_some_and(|n| n.is_whitespace()),
            _ => false,
        };
        if ends {
            pieces.push(std::mem::take(&mut cur));
        }
    }
    pieces.push(cur);
    let (mut count, mut carry) = (0, 0);
    for piece in pieces.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        // A joined piece is the carry, a space, and this piece.
        let len = if carry > 0 { carry + 1 } else { 0 } + piece.chars().count();
        if len < MIN_SENTENCE {
            carry = len;
        } else {
            count += 1;
            carry = 0;
        }
    }
    // A short last piece joins the previous sentence, or is the only one.
    if carry > 0 && count == 0 {
        count = 1;
    }
    count
}

/// Where and as what [`synthesise`] saves a request's audio.
struct Recent {
    home: PathBuf,
    id: String,
    log: Arc<crate::log::Logger>,
    req_id: String,
}

/// How many wavs `<home>/recent` keeps.
const RECENT_KEPT: usize = 50;

/// A short unique id for one speech request: unix seconds and a counter.
fn recent_id() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{secs}-{n}")
}

/// Writes `pcm` (s16le mono) as `<home>/recent/<id>.wav`, then deletes all
/// but the newest [`RECENT_KEPT`] wavs there.
fn save_recent(
    home: &std::path::Path,
    id: &str,
    sample_rate: u32,
    pcm: &[u8],
) -> std::io::Result<()> {
    let dir = home.join("recent");
    std::fs::create_dir_all(&dir)?;
    let mut wav = wav_header(sample_rate, pcm.len() as u32).to_vec();
    wav.extend_from_slice(pcm);
    std::fs::write(dir.join(format!("{id}.wav")), wav)?;
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort();
    let excess = files.len().saturating_sub(RECENT_KEPT);
    for (_, path) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

/// A failed synthesis, classified where the [`TtsError`] is still at hand.
struct Failure {
    /// The backend went away mid-synthesis (a crashed MLX sidecar, §5.3).
    unavailable: bool,
    message: String,
}

/// §5.3: 400 `voice_model_mismatch` when a request names both a cloned
/// voice and, explicitly, a model it was not made for.
fn voice_model_mismatch(voice: &str, made_for: &str, requested: &str) -> ApiError {
    ApiError {
        param: Some("model"),
        ..ApiError::new(
            StatusCode::BAD_REQUEST,
            "voice_model_mismatch",
            format!("the voice \"{voice}\" was made for \"{made_for}\", not \"{requested}\""),
        )
    }
}

/// A failed synthesis before any audio, logged: 503 `backend_unavailable`,
/// or a 500.
fn failed(st: &AppState, req_id: &str, e: Failure) -> ApiError {
    if e.unavailable {
        st.log
            .line("warn", Some(req_id), &format!("synth_failed {}", e.message));
        return ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_unavailable",
            e.message,
        );
    }
    internal(st, req_id, e.message)
}

fn headers(format: Format, sample_rate: u32) -> Vec<(&'static str, String)> {
    match format {
        Format::Wav => vec![(CONTENT_TYPE.as_str(), "audio/wav".to_string())],
        Format::Pcm => vec![
            (CONTENT_TYPE.as_str(), "audio/pcm".to_string()),
            ("x-audio-sample-rate", sample_rate.to_string()),
            ("x-audio-channels", "1".to_string()),
            ("x-audio-encoding", "s16le".to_string()),
        ],
    }
}

/// A full (non-streamed) audio body with `headers()`'s headers, exactly
/// once each. `AppendHeaders` appends rather than replaces, and `Vec<u8>`'s
/// own `IntoResponse` already sets a default `Content-Type:
/// application/octet-stream` (axum-core's `impl IntoResponse for Vec<u8>`);
/// combining the two the way the streamed responses combine `AppendHeaders`
/// with `Body::from_stream` (which sets no header of its own) would instead
/// send `Content-Type` twice — naru task 1458, caught by the admin page's
/// browser `<audio>` tag choking on it. Wrapping `body` in `Body` first,
/// which has no default headers of its own, avoids the second one.
fn audio_response(format: Format, sample_rate: u32, body: Vec<u8>) -> Response {
    (
        AppendHeaders(headers(format, sample_rate)),
        Body::from(body),
    )
        .into_response()
}

/// A 44-byte header for 16-bit mono PCM with `data_size` bytes of samples.
fn wav_header(sample_rate: u32, data_size: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    let fields: [(usize, &[u8]); 13] = [
        (0, b"RIFF"),
        (4, &data_size.wrapping_add(36).to_le_bytes()),
        (8, b"WAVE"),
        (12, b"fmt "),
        (16, &16u32.to_le_bytes()),
        (20, &1u16.to_le_bytes()), // PCM
        (22, &1u16.to_le_bytes()), // mono
        (24, &sample_rate.to_le_bytes()),
        (28, &(sample_rate * 2).to_le_bytes()),
        (32, &2u16.to_le_bytes()),
        (34, &16u16.to_le_bytes()),
        (36, b"data"),
        (40, &data_size.to_le_bytes()),
    ];
    for (at, bytes) in fields {
        h[at..at + bytes.len()].copy_from_slice(bytes);
    }
    h
}

/// `default_model` is what `default` and its aliases resolve to (§3.3).
fn validate(body: &[u8], default_model: &str) -> Result<Job, ApiError> {
    let body: serde_json::Map<String, Value> = serde_json::from_slice(body).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("the body must be a JSON object: {e}"),
        )
    })?;
    let field = |name: &str| body.get(name).filter(|v| !v.is_null());
    let string = |name: &'static str| -> Result<Option<&str>, ApiError> {
        field(name)
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    bad_request(name, "invalid_request", format!("{name} must be a string"))
                })
            })
            .transpose()
    };
    let number = |name: &'static str| -> Result<Option<f64>, ApiError> {
        field(name)
            .map(|v| {
                v.as_f64().ok_or_else(|| {
                    bad_request(name, "invalid_request", format!("{name} must be a number"))
                })
            })
            .transpose()
    };
    let boolean = |name: &'static str, default: bool| -> Result<bool, ApiError> {
        field(name).map_or(Ok(default), |v| {
            v.as_bool().ok_or_else(|| {
                bad_request(
                    name,
                    "invalid_request",
                    format!("{name} must be true or false"),
                )
            })
        })
    };

    let input = string("input")?.unwrap_or_default();
    if input.is_empty() {
        return Err(bad_request(
            "input",
            "invalid_request",
            "\"input\" must be a non-empty string",
        ));
    }
    let chars = input.chars().count();
    if chars > MAX_INPUT_CHARS {
        return Err(ApiError {
            param: Some("input"),
            ..ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                format!("input is {chars} characters; the cap is {MAX_INPUT_CHARS}"),
            )
        });
    }
    if input.contains('\0') {
        return Err(bad_request(
            "input",
            "invalid_request",
            "input must not contain a NUL character",
        ));
    }

    let format = match string("response_format")? {
        None | Some("wav") => Format::Wav,
        Some("pcm") => Format::Pcm,
        Some(other) => {
            return Err(bad_request(
                "response_format",
                "unsupported_value",
                format!("response_format {other:?} is not supported; use wav or pcm"),
            ));
        }
    };
    match string("stream_format")? {
        None | Some("audio") => {}
        Some(other) => {
            return Err(bad_request(
                "stream_format",
                "unsupported_value",
                format!("stream_format {other:?} is not supported; use audio"),
            ));
        }
    }

    let mut options = SynthOptions::default();
    if let Some(speed) = number("speed")? {
        if !(0.5..=2.0).contains(&speed) {
            return Err(bad_request(
                "speed",
                "invalid_request",
                format!("speed must be between 0.5 and 2.0, got {speed}"),
            ));
        }
        options.speed = speed as f32;
    }
    if let Some(gap) = number("gap")? {
        if !(0.0..=5.0).contains(&gap) {
            return Err(bad_request(
                "gap",
                "invalid_request",
                format!("gap must be between 0 and 5 seconds, got {gap}"),
            ));
        }
        options.gap = gap as f32;
    }
    options.level = boolean("level", true)?;
    // Read loosely: a model that does not take them ignores them, whatever
    // they are.
    options.instructions = field("instructions")
        .and_then(Value::as_str)
        .filter(|i| !i.trim().is_empty())
        .map(str::to_string);
    if let Some(exaggeration) = number("exaggeration")? {
        if !(0.0..=1.0).contains(&exaggeration) {
            return Err(bad_request(
                "exaggeration",
                "invalid_request",
                format!("exaggeration must be between 0 and 1, got {exaggeration}"),
            ));
        }
        options.exaggeration = Some(exaggeration as f32);
    }
    if let Some(seed) = field("seed") {
        options.seed = seed.as_u64().ok_or_else(|| {
            bad_request(
                "seed",
                "invalid_request",
                "seed must be a non-negative integer",
            )
        })?;
    }
    let stream = boolean("stream", true)?;
    let knobs = parse_knobs(field("knobs"))?;

    let keep_alive = match field("keep_alive") {
        None => None,
        Some(Value::String(s)) => Some(KeepAlive::parse(s)),
        Some(Value::Number(n)) => Some(KeepAlive::from_secs(n.as_f64().unwrap_or(f64::NAN))),
        Some(_) => Some(Err(
            "keep_alive must be a duration string or a number of seconds".to_string(),
        )),
    }
    .transpose()
    .map_err(|e| bad_request("keep_alive", "invalid_request", e))?;

    let explicit_model = match string("model")? {
        None => None,
        Some(m) if DEFAULT_ALIASES.contains(&m) => None,
        Some(m) => Some(m),
    };
    let model = explicit_model.unwrap_or(default_model);
    // Anything unknown is ignored (§2.3).
    Ok(Job {
        model: model.to_string(),
        explicit_model: explicit_model.is_some(),
        input: input.to_string(),
        voice: string("voice")?.map(str::to_string),
        format,
        stream,
        options,
        keep_alive,
        knobs,
    })
}

/// `value` (a `"knobs"` field, from a JSON body or a multipart field
/// decoded the same way) as a name-to-number map, checked only for shape:
/// an object of finite numbers. Not yet checked against any model, since
/// neither caller has fetched a manifest at this point.
fn parse_knobs(value: Option<&Value>) -> Result<HashMap<String, f64>, ApiError> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let object = value
        .as_object()
        .ok_or_else(|| bad_request("knobs", "invalid_request", "\"knobs\" must be an object"))?;
    object
        .iter()
        .map(|(name, v)| {
            let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
                bad_request(
                    "knobs",
                    "invalid_request",
                    format!("knobs.{name} must be a finite number"),
                )
            })?;
            Ok((name.clone(), n))
        })
        .collect()
}

/// Checks `raw` (parsed by [`parse_knobs`]) against `manifest`'s declared
/// knobs (`PromptFormat::knobs`, naru task 1458): a name that is not one of
/// them is 400 `unsupported_value`; a value outside its `min`/`max` is 400
/// `invalid_request`. `exaggeration` and `speed` are also declared knobs
/// for the models that take them, but go to `options`' own fields rather
/// than `options.knobs`, alongside the top-level `exaggeration`/`speed`
/// fields, which run through the same range checks separately.
fn apply_knobs(
    manifest: &Manifest,
    raw: HashMap<String, f64>,
    options: &mut SynthOptions,
) -> Result<(), ApiError> {
    if raw.is_empty() {
        return Ok(());
    }
    let declared = manifest
        .prompt_format()
        .map(|pf| pf.knobs)
        .unwrap_or_default();
    for (name, value) in raw {
        let knob = declared.iter().find(|k| k.name == name).ok_or_else(|| {
            bad_request(
                "knobs",
                "unsupported_value",
                format!("\"{name}\" is not a knob of \"{}\"", manifest.model.name),
            )
        })?;
        if let (Some(min), Some(max)) = (knob.min, knob.max)
            && !(min..=max).contains(&value)
        {
            return Err(bad_request(
                "knobs",
                "invalid_request",
                format!("knobs.{name} must be between {min} and {max}, got {value}"),
            ));
        }
        match name.as_str() {
            "exaggeration" => options.exaggeration = Some(value as f32),
            "speed" => options.speed = value as f32,
            _ => {
                options.knobs.insert(name, value);
            }
        }
    }
    Ok(())
}

/// `voice` if it is one of `voices` (the manifest's, and any cloned ones),
/// else 400 `unknown_voice` with the valid ids; no voice is `default_voice`
/// (`ModelManager::default_voice`, `PUT /api/defaults`' live override) if
/// it names one of `voices`, else the manifest's own `default`, else the
/// first. A cloning model with no cloned voices and no voice asked for is a
/// 400 that says how to add one. A model that instructs (VoiceDesign, and
/// VoxCPM2 alongside cloning) makes its voice up from `instructions`
/// whenever a named voice was not asked for or does not match one of
/// `voices`: any `voice` is taken and ignored, and, if `voices` is empty
/// too (nothing else it could speak in), no `instructions` is a 400. A
/// named voice that does match — VoxCPM2 cloning a voice under
/// $NARU_AUDIO_HOME/voices/ — wins over `instructions` even though the
/// model also instructs.
fn voice(
    manifest: &Manifest,
    voices: &[Voice],
    voice: Option<&str>,
    instructions: Option<&str>,
    default_voice: Option<&str>,
) -> Result<String, ApiError> {
    let named_match = voice.is_some_and(|id| voices.iter().any(|v| v.id == id));
    if manifest.instructs() && !named_match {
        if instructions.is_some() {
            // No voice is "", which `synth` ignores as it does any other.
            return Ok(voice.unwrap_or_default().to_string());
        }
        if voices.is_empty() {
            return Err(bad_request(
                "instructions",
                "invalid_request",
                format!(
                    "the model \"{}\" needs \"instructions\" describing the voice",
                    manifest.model.name
                ),
            ));
        }
    }
    if voice.is_none() && voices.is_empty() && manifest.clones() {
        return Err(bad_request(
            "voice",
            "unknown_voice",
            format!(
                "the model \"{}\" has no cloned voices; add one with naru-audio voice add",
                manifest.model.name
            ),
        ));
    }
    let found = match voice {
        Some(id) => voices.iter().find(|v| v.id == id),
        None => default_voice
            .and_then(|d| voices.iter().find(|v| v.id == d))
            .or_else(|| voices.iter().find(|v| v.default))
            .or_else(|| voices.first()),
    };
    if let Some(v) = found {
        return Ok(v.id.clone());
    }
    let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
    Err(bad_request(
        "voice",
        "unknown_voice",
        format!(
            "the model \"{}\" has no voice {:?}; use one of: {}",
            manifest.model.name,
            voice.unwrap_or_default(),
            ids.join(", ")
        ),
    ))
}

/// §2.5 `GET /v1/audio/voices?model=`: read from the manifest (the pulled
/// one, else the catalog's), and the cloned voices made for that model if it
/// clones; never by loading the model. `cloned` marks the ones from
/// `voices/`. `?model=clones` is not a real model: it lists every cloned
/// voice regardless of the model it was made for, same as before per-model
/// voices, each entry naming its own `model`.
pub(super) async fn voices(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let name = match query.get("model").map(String::as_str) {
        None => st.models.tts_default(),
        Some(m) if DEFAULT_ALIASES.contains(&m) => st.models.tts_default(),
        Some(m) => m.to_string(),
    };
    if name == "clones" {
        let voices: Vec<Value> = {
            let st = st.clone();
            tokio::task::spawn_blocking(move || {
                let home = st.registry.home();
                voices::list(home)
                    .into_iter()
                    .filter_map(|id| {
                        let model = voices::model_of(home, &id)?;
                        let origin = if voices::is_designed(home, &id) {
                            "designed"
                        } else {
                            "cloned"
                        };
                        let has_transcript =
                            voices::find(home, &id).is_some_and(|c| !c.text.is_empty());
                        Some(
                            json!({"id": id, "accent": null, "gender": null, "default": false,
                                     "cloned": true, "model": model, "origin": origin,
                                     "description": voices::description(home, &id),
                                     "duration": voices::duration(home, &id),
                                     "has_transcript": has_transcript}),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        }
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?;
        return Ok(Json(json!({"model": "clones", "voices": voices})));
    }
    // `PUT /api/defaults`' live override, if set, wins over the manifest's
    // own `default` marker.
    let default_override = st.models.default_voice(&name);
    let (model_name, voices) = {
        let (st, default_override) = (st.clone(), default_override);
        tokio::task::spawn_blocking(move || {
            let manifest = match st.registry.catalog().models.get(&name) {
                Some(m) if !st.registry.is_installed(&name) => match m.model.kind {
                    Kind::Tts => Ok(m.clone()),
                    _ => Err(not_kind(&name, Kind::Tts)),
                },
                _ => kind_manifest(&st, &name, Kind::Tts),
            }?;
            let home = st.registry.home();
            let voices: Vec<Value> = voices::of(&manifest, home)
                .iter()
                .map(|v| {
                    // `voices::of` appends the cloned voices no `[[voice]]`
                    // shadows.
                    let cloned = !manifest.voices.iter().any(|m| m.id == v.id);
                    let default = match &default_override {
                        Some(d) => &v.id == d,
                        None => v.default,
                    };
                    // A built-in voice is never in `voices/`, so it is never
                    // designed and has no clip or transcript here to report.
                    let origin = if !cloned {
                        "builtin"
                    } else if voices::is_designed(home, &v.id) {
                        "designed"
                    } else {
                        "cloned"
                    };
                    let has_transcript =
                        voices::find(home, &v.id).is_some_and(|c| !c.text.is_empty());
                    json!({"id": v.id, "accent": v.accent, "gender": v.gender,
                           "default": default, "cloned": cloned, "origin": origin,
                           "description": voices::description(home, &v.id),
                           "duration": voices::duration(home, &v.id),
                           "has_transcript": has_transcript})
                })
                .collect();
            Ok::<_, ApiError>((manifest.model.name, voices))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok(Json(json!({"model": model_name, "voices": voices})))
}

/// §2.5 `GET /v1/audio/voices/{name}`: a cloned voice's transcript, the
/// model it was made for (§5.3 `model_of`, `CLONE_MODEL` if it predates
/// per-model voices) and its `ref.wav`, base64, byte for byte, so it can be
/// added elsewhere with that model. A bad name is a 400, as `POST` has it;
/// no such cloned voice a 404.
pub(super) async fn voice_export(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    voices::check_name(&name).map_err(|m| bad_request("name", "invalid_request", m))?;
    let found = {
        let (home, name) = (st.registry.home().to_path_buf(), name.clone());
        tokio::task::spawn_blocking(move || {
            let Some(voice) = voices::find(&home, &name) else {
                return Ok(None);
            };
            let wav = std::fs::read(&voice.wav)
                .map_err(|e| format!("read {}: {e}", voice.wav.display()))?;
            let description = voices::description(&home, &name);
            Ok::<_, String>(Some((voice.text, voice.model, wav, description)))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|m| internal(&st, &req_id, m))?;
    let (text, model, wav, description) = found.ok_or_else(|| voice_not_found(&name))?;
    Ok(Json(json!({
        "name": name, "text": text, "model": model, "description": description,
        "wav_base64": STANDARD.encode(wav),
    })))
}

/// §2.5 `DELETE /v1/audio/voices/{name}`: removes a cloned voice's
/// directory. A bad name is a 400, as `GET` and
/// `POST` have it. A name that is a built-in voice of some catalog
/// model (never in `voices/`) is a 409: there is nothing cloned to
/// remove, and it is not this route's place to touch a manifest's
/// `[[voice]]`s. Anything else unknown is a 404.
pub(super) async fn delete_voice(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    voices::check_name(&name).map_err(|m| bad_request("name", "invalid_request", m))?;
    if is_builtin_voice(&st, &name) {
        return Err(builtin_voice(&name));
    }
    let (model, removed) = {
        let (home, name) = (st.registry.home().to_path_buf(), name.clone());
        tokio::task::spawn_blocking(move || {
            (voices::model_of(&home, &name), voices::remove(&home, &name))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let removed = removed.map_err(|m| internal(&st, &req_id, m))?;
    if !removed {
        return Err(voice_not_found(&name));
    }
    // §2.6: a configured default voice does not survive its own deletion
    // (naru task 1458). Persisted before it is applied live, so a failed
    // write never leaves memory and `config.toml` disagreeing.
    if let Some(model) = model
        && st.models.default_voice(&model).as_deref() == Some(name.as_str())
    {
        if let Err(e) =
            crate::manager::write_defaults(st.registry.home(), None, None, &[(model.clone(), None)])
        {
            return Err(internal(&st, &req_id, e));
        }
        st.models.set_default_voice(model, None);
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Whether `name` is one of some catalog model's own `[[voice]]`s, never in
/// `voices/` — the built-in check `DELETE`/`PATCH /v1/audio/voices/{name}`
/// share (naru task 1458).
fn is_builtin_voice(st: &AppState, name: &str) -> bool {
    st.registry
        .catalog()
        .models
        .values()
        .any(|m| m.voices.iter().any(|v| v.id == name))
}

/// 409: `name` is a built-in voice, not a cloned one `DELETE`/`PATCH` can
/// touch.
fn builtin_voice(name: &str) -> ApiError {
    ApiError {
        param: Some("name"),
        ..ApiError::new(
            StatusCode::CONFLICT,
            "builtin_voice",
            format!("{name:?} is a built-in voice; only cloned voices can be changed"),
        )
    }
}

/// 404: there is no cloned voice `name`.
fn voice_not_found(name: &str) -> ApiError {
    ApiError {
        param: Some("name"),
        ..ApiError::new(
            StatusCode::NOT_FOUND,
            "voice_not_found",
            format!("there is no cloned voice {name:?}"),
        )
    }
}

/// §2.5 `POST /v1/audio/voices`: the multipart fields `name`, `file` (the
/// clip), `text` (what it says) and `model` (which model it is for; a
/// cloning model, defaulting to `CLONE_MODEL` as every voice was for
/// before per-model voices) add a cloned voice through [`voices::add`], as
/// `naru-audio voice add` does (always `CLONE_MODEL`; it has no `--model`
/// yet). Recreating an exported voice (`GET /v1/audio/voices/{name}`)
/// elsewhere is the same request, with its `model` carried over. The
/// upload lands in `tmp/` and is removed after. 201 with the voice as the
/// listing shows it, plus its `model` and the clip's `duration` in seconds.
pub(super) async fn add_voice(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let mut multipart = multipart.map_err(|e| {
        bad_request(
            "file",
            "invalid_request",
            format!("the body must be multipart/form-data: {}", e.body_text()),
        )
    })?;
    let (mut name, mut file, mut text, mut model, mut description) = (None, None, None, None, None);
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        match field.name().unwrap_or_default().to_owned().as_str() {
            "file" => file = Some(field.bytes().await.map_err(multipart_error)?),
            "name" => name = Some(field.text().await.map_err(multipart_error)?),
            "text" => text = Some(field.text().await.map_err(multipart_error)?),
            "model" => model = Some(field.text().await.map_err(multipart_error)?),
            // §2.6: a designed voice's description (naru task 1458); its
            // mere presence, once stored, is what marks the voice
            // "designed" rather than "cloned".
            "description" => description = Some(field.text().await.map_err(multipart_error)?),
            _ => {}
        }
    }
    let required = |param: &'static str| {
        bad_request(
            param,
            "invalid_request",
            format!("the \"{param}\" field is required"),
        )
    };
    let name = name.ok_or_else(|| required("name"))?;
    let file = file.ok_or_else(|| required("file"))?;
    let model = model.unwrap_or_else(|| voices::CLONE_MODEL.to_string());

    // The model must be one that clones, checked against the catalog (a
    // model need not be pulled to record a voice for it, as `GET
    // /v1/audio/voices?model=` does not require it either).
    let manifest = {
        let (st, model) = (st.clone(), model.clone());
        tokio::task::spawn_blocking(move || clone_manifest(&st, &model))
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    // `text` is required only when the model's manifest actually needs a
    // transcript alongside the clip (`clone_requires_transcript`, mesa task
    // 1455): a model that clones from audio alone has no use for one, and a
    // caller with no transcript to give must not be blocked by a field the
    // model doesn't read. A *present but blank* transcript still falls
    // through to `voices::add`'s own check, whose message ("the transcript
    // is empty") is unchanged for a model that does require one.
    let require_text = manifest.clone_requires_transcript();
    if require_text && text.is_none() {
        return Err(required("text"));
    }
    let text = text.unwrap_or_default();

    let result = {
        let (home, name, model, description) = (
            st.registry.home().to_path_buf(),
            name.clone(),
            model.clone(),
            description.clone(),
        );
        tokio::task::spawn_blocking(move || {
            let upload = voices::scratch(&home, "upload");
            let result = std::fs::create_dir_all(home.join("tmp"))
                .and_then(|()| std::fs::write(&upload, &file))
                .map_err(|e| AddError::Io(format!("write {}: {e}", upload.display())))
                .and_then(|()| {
                    voices::add(
                        &home,
                        &name,
                        &upload,
                        &text,
                        &model,
                        require_text,
                        description.as_deref(),
                    )
                });
            // Also after a write that failed partway.
            let _ = std::fs::remove_file(&upload);
            result
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let secs = result.map_err(|e| match e {
        AddError::Name(m) => bad_request("name", "invalid_request", m),
        AddError::Exists(m) => ApiError {
            param: Some("name"),
            ..ApiError::new(StatusCode::CONFLICT, "voice_exists", m)
        },
        AddError::Text => bad_request("text", "invalid_request", e.to_string()),
        AddError::Clip(_) => ApiError {
            param: Some("file"),
            ..ApiError::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                CLIP_UNREADABLE,
            )
        },
        AddError::Length(_) => bad_request("file", "invalid_request", e.to_string()),
        AddError::Io(m) => internal(&st, &req_id, m),
    })?;
    let description = description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty());
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": name,
            "accent": null,
            "gender": null,
            "default": false,
            "cloned": true,
            "origin": if description.is_some() { "designed" } else { "cloned" },
            "description": description,
            "model": model,
            "duration": secs,
        })),
    )
        .into_response())
}

/// The manifest (pulled, or the catalog's if not) of the TTS model `name`,
/// as `voices()` reads it: 404 `model_not_found` unknown, 409
/// `model_not_pulled` never applies here (a catalog model need not be
/// pulled to record a voice for it), 400 `invalid_request` not TTS, 400
/// `model_does_not_clone` a TTS model that never speaks in cloned voices.
/// Used to check a voice's `model` before it is stored, so a name that
/// will never be usable is refused up front rather than only at
/// `/v1/audio/speech`.
fn clone_manifest(st: &AppState, name: &str) -> Result<Manifest, ApiError> {
    let manifest = match st.registry.catalog().models.get(name) {
        Some(m) if !st.registry.is_installed(name) => match m.model.kind {
            Kind::Tts => m.clone(),
            _ => return Err(not_kind(name, Kind::Tts)),
        },
        _ => kind_manifest(st, name, Kind::Tts)?,
    };
    if !manifest.clones() {
        return Err(ApiError {
            param: Some("model"),
            ..ApiError::new(
                StatusCode::BAD_REQUEST,
                "model_does_not_clone",
                format!("the model \"{name}\" does not clone voices"),
            )
        });
    }
    Ok(manifest)
}

/// §2.6 `PATCH /v1/audio/voices/{name}` `{"name"?, "text"?, "description"?}`
/// (naru task 1458): renames a cloned voice and/or overwrites its
/// transcript or description, each only where the body actually gives it.
/// A built-in voice is 409 `builtin_voice`, as `DELETE` has it; an unknown
/// `name` is 404 `voice_not_found`; a taken new name is 409 `voice_exists`.
/// A configured default voice (`PUT /api/defaults`, `config.toml
/// [defaults.voices]`) follows the rename. Returns 200 with the voice as
/// `GET /v1/audio/voices` lists it.
pub(super) async fn patch_voice(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(name): Path<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    voices::check_name(&name).map_err(|m| bad_request("name", "invalid_request", m))?;
    if is_builtin_voice(&st, &name) {
        return Err(builtin_voice(&name));
    }
    let body: Value = serde_json::from_slice(&body).unwrap_or_default();
    let string = |key: &'static str| -> Result<Option<&str>, ApiError> {
        body.get(key)
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    bad_request(key, "invalid_request", format!("{key} must be a string"))
                })
            })
            .transpose()
    };
    let new_name = string("name")?;
    let text = string("text")?;
    let description = string("description")?;
    if let Some(n) = new_name {
        voices::check_name(n).map_err(|m| bad_request("name", "invalid_request", m))?;
    }

    let (home, old_name, new_name_owned, text_owned, description_owned) = (
        st.registry.home().to_path_buf(),
        name.clone(),
        new_name.map(str::to_string),
        text.map(str::to_string),
        description.map(str::to_string),
    );
    let (model_before, updated) = {
        let (home, old_name, new_name_owned, text_owned, description_owned) = (
            home.clone(),
            old_name.clone(),
            new_name_owned.clone(),
            text_owned.clone(),
            description_owned.clone(),
        );
        tokio::task::spawn_blocking(move || {
            let model_before = voices::model_of(&home, &old_name);
            let updated = voices::update(
                &home,
                &old_name,
                new_name_owned.as_deref(),
                text_owned.as_deref(),
                description_owned.as_deref(),
            );
            (model_before, updated)
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let final_name = updated.map_err(|e| match e {
        voices::UpdateError::Name(m) => bad_request("name", "invalid_request", m),
        voices::UpdateError::NotFound => voice_not_found(&name),
        voices::UpdateError::Exists(m) => ApiError {
            param: Some("name"),
            ..ApiError::new(StatusCode::CONFLICT, "voice_exists", m)
        },
        voices::UpdateError::Io(m) => internal(&st, &req_id, m),
    })?;

    // §2.6: a configured default voice follows its own rename, persisted
    // before it is applied live (as `DELETE` above does too), so a failed
    // write never leaves memory and `config.toml` disagreeing.
    if final_name != name
        && let Some(model) = model_before
        && st.models.default_voice(&model).as_deref() == Some(name.as_str())
    {
        if let Err(e) = crate::manager::write_defaults(
            st.registry.home(),
            None,
            None,
            &[(model.clone(), Some(final_name.clone()))],
        ) {
            return Err(internal(&st, &req_id, e));
        }
        st.models.set_default_voice(model, Some(final_name.clone()));
    }

    let listing = {
        let (st, final_name) = (st.clone(), final_name.clone());
        tokio::task::spawn_blocking(move || {
            let home = st.registry.home();
            let voice = voices::find(home, &final_name)?;
            let default = st.models.default_voice(&voice.model).as_deref() == Some(final_name.as_str());
            let has_transcript = !voice.text.is_empty();
            Some(json!({
                "id": final_name, "accent": null, "gender": null, "default": default,
                "cloned": true, "model": voice.model,
                "origin": if voices::is_designed(home, &final_name) { "designed" } else { "cloned" },
                "description": voices::description(home, &final_name),
                "duration": voices::duration(home, &final_name),
                "has_transcript": has_transcript,
            }))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .ok_or_else(|| internal(&st, &req_id, "the voice vanished after its own update".to_string()))?;
    Ok(Json(listing))
}

/// `model`/`voice`, resolved and checked for `GET`/`POST
/// /api/voices/{model}/{voice}/sample`: the model's manifest (from the
/// catalog if not installed, so an uninstalled built-in still answers) and
/// whether `voice` is a cloned one rather than the manifest's own. 404
/// `voice_not_found` for a `voice` that is neither.
async fn sample_lookup(
    st: &Arc<AppState>,
    req_id: &str,
    model: &str,
    voice: &str,
) -> Result<(Manifest, bool), ApiError> {
    voices::check_name(model).map_err(|m| bad_request("model", "invalid_request", m))?;
    voices::check_name(voice).map_err(|m| bad_request("voice", "invalid_request", m))?;
    let (blocking, model, voice) = (st.clone(), model.to_string(), voice.to_string());
    tokio::task::spawn_blocking(move || {
        let manifest = match blocking.registry.catalog().models.get(&model) {
            Some(m) if !blocking.registry.is_installed(&model) => match m.model.kind {
                Kind::Tts => Ok(m.clone()),
                _ => Err(not_kind(&model, Kind::Tts)),
            },
            _ => kind_manifest(&blocking, &model, Kind::Tts),
        }?;
        let voices = voices::of(&manifest, blocking.registry.home());
        if !voices.iter().any(|v| v.id == voice) {
            return Err(voice_not_found_of(&model, &voice));
        }
        let cloned = !manifest.voices.iter().any(|v| v.id == voice);
        Ok::<_, ApiError>((manifest, cloned))
    })
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))?
}

/// A cloned voice's own `ref.wav`, byte for byte.
async fn cloned_sample(
    st: &Arc<AppState>,
    req_id: &str,
    model: &str,
    voice: &str,
) -> Result<Response, ApiError> {
    let (home, voice_id, model_name) = (
        st.registry.home().to_path_buf(),
        voice.to_string(),
        model.to_string(),
    );
    let wav = tokio::task::spawn_blocking(move || {
        voices::find(&home, &voice_id).and_then(|c| std::fs::read(&c.wav).ok())
    })
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))?
    .ok_or_else(|| voice_not_found_of(&model_name, voice))?;
    Ok(audio_response(Format::Wav, 0, wav))
}

/// §2.6 `GET /api/voices/{model}/{voice}/sample` (naru task 1458): a
/// cloned or designed voice's own `ref.wav`, byte for byte; a built-in
/// voice's cached preview at `state/previews/<model>/<voice>.wav` under
/// the home, if there is one. Read-only, so it never loads a model or
/// synthesises: a cross-site `<audio src>` sends no `Origin` header, which
/// the Host/Origin guard (§2.1) would otherwise let straight through on a
/// `GET`, so generating belongs on `POST /api/voices/{model}/{voice}/sample`
/// instead, covered by that guard like any other mutating request. 404
/// `voice_not_found` for a `voice` that is none of the model's own or
/// cloned voices; 404 `preview_not_cached` for a built-in voice with
/// nothing cached — a built-in voice is never generated unasked, on `GET`
/// or `POST`.
pub(super) async fn voice_sample(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path((model, voice)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let (_manifest, cloned) = sample_lookup(&st, &req_id, &model, &voice).await?;
    if cloned {
        return cloned_sample(&st, &req_id, &model, &voice).await;
    }
    let cache = preview_cache_path(st.registry.home(), &model, &voice);
    let cached = {
        let cache = cache.clone();
        tokio::task::spawn_blocking(move || std::fs::read(&cache).ok())
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    match cached {
        Some(bytes) => Ok(audio_response(Format::Wav, 0, bytes)),
        None => Err(ApiError {
            param: Some("voice"),
            ..ApiError::new(
                StatusCode::NOT_FOUND,
                "preview_not_cached",
                format!(
                    "no cached preview for \"{voice}\" of \"{model}\"; \
                     POST this same path to generate one"
                ),
            )
        }),
    }
}

/// §2.6 `POST /api/voices/{model}/{voice}/sample` (naru task 1458): the
/// same cached preview `GET` serves if there is one, else synthesises a
/// fixed line, caches it and returns that — the load-and-synthesise half
/// of the old `GET ...?generate=true`, moved to `POST` so the Origin guard
/// covers it (see `voice_sample`). A cloned voice has nothing to
/// generate, so it answers its own `ref.wav`, same as `GET`.
pub(super) async fn generate_voice_sample(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path((model, voice)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let (manifest, cloned) = sample_lookup(&st, &req_id, &model, &voice).await?;
    if cloned {
        return cloned_sample(&st, &req_id, &model, &voice).await;
    }

    let cache = preview_cache_path(st.registry.home(), &model, &voice);
    let cached = {
        let cache = cache.clone();
        tokio::task::spawn_blocking(move || std::fs::read(&cache).ok())
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    if let Some(bytes) = cached {
        return Ok(audio_response(Format::Wav, 0, bytes));
    }

    let guard = st
        .models
        .acquire(manifest, None)
        .await
        .map_err(|e| manager_error(&st, &req_id, e))?;
    let sample_rate = guard.tts().sample_rate();
    let mut rx = synthesise(
        guard,
        PREVIEW_TEXT.to_string(),
        voice.clone(),
        SynthOptions::default(),
        None,
    );
    let mut pcm = Vec::new();
    while let Some(piece) = rx.recv().await {
        pcm.extend_from_slice(&piece.map_err(|e| failed(&st, &req_id, e))?);
    }
    let mut wav = wav_header(sample_rate, pcm.len() as u32).to_vec();
    wav.extend_from_slice(&pcm);
    {
        let (cache, bytes) = (cache, wav.clone());
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(parent) = cache.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&cache, &bytes)
        })
        .await;
    }
    Ok(audio_response(Format::Wav, 0, wav))
}

/// 404: the model `model` has no voice `voice`, built in or cloned.
fn voice_not_found_of(model: &str, voice: &str) -> ApiError {
    ApiError {
        param: Some("voice"),
        ..ApiError::new(
            StatusCode::NOT_FOUND,
            "voice_not_found",
            format!("the model {model:?} has no voice {voice:?}"),
        )
    }
}

/// A fixed line synthesised for `GET /api/voices/{model}/{voice}/sample`'s
/// `generate=true` and cached for next time.
const PREVIEW_TEXT: &str = "This is a preview of this voice.";

/// Where a built-in voice's generated preview is cached, under `home`'s
/// `state/`; `DELETE /api/models/{name}` removes the whole `<model>/`
/// directory here. The caller checks `model`/`voice` with
/// `voices::check_name` first (as `voice_sample` does); the `debug_assert`s
/// catch a future caller that forgets to, before it ever writes outside
/// `previews/` in release.
pub(super) fn preview_cache_path(home: &std::path::Path, model: &str, voice: &str) -> PathBuf {
    debug_assert!(
        voices::check_name(model).is_ok(),
        "bad model name {model:?}"
    );
    debug_assert!(
        voices::check_name(voice).is_ok(),
        "bad voice name {voice:?}"
    );
    home.join("state")
        .join("previews")
        .join(model)
        .join(format!("{voice}.wav"))
}

/// §2.6 `POST /api/voices/preview` (naru task 1458): `multipart/form-data`
/// fields `model` (a TTS model that clones), `file` (the reference clip),
/// `text` (its transcript) and `input` (what to say) synthesise a one-off
/// clip in `model`'s voice without saving anything to `voices/`. The clip
/// is converted the same way `POST /v1/audio/voices` converts an upload
/// ([`voices::convert_to_tmp`]) and always removed once synthesis ends,
/// whether it succeeds or not. Returns `audio/wav`, not streamed.
pub(super) async fn preview_voice(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let mut multipart = multipart.map_err(|e| {
        bad_request(
            "file",
            "invalid_request",
            format!("the body must be multipart/form-data: {}", e.body_text()),
        )
    })?;
    let (mut model, mut file, mut text, mut input, mut knobs) = (None, None, None, None, None);
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        match field.name().unwrap_or_default().to_owned().as_str() {
            "model" => model = Some(field.text().await.map_err(multipart_error)?),
            "file" => file = Some(field.bytes().await.map_err(multipart_error)?),
            "text" => text = Some(field.text().await.map_err(multipart_error)?),
            "input" => input = Some(field.text().await.map_err(multipart_error)?),
            "knobs" => knobs = Some(field.text().await.map_err(multipart_error)?),
            _ => {}
        }
    }
    let required = |param: &'static str| {
        bad_request(
            param,
            "invalid_request",
            format!("the \"{param}\" field is required"),
        )
    };
    let model = model.ok_or_else(|| required("model"))?;
    let file = file.ok_or_else(|| required("file"))?;
    let input = input.ok_or_else(|| required("input"))?;
    if input.trim().is_empty() {
        return Err(bad_request(
            "input",
            "invalid_request",
            "\"input\" must be a non-empty string",
        ));
    }
    let text = text.unwrap_or_default();
    let knobs = knobs
        .map(|raw| {
            serde_json::from_str::<Value>(&raw).map_err(|e| {
                bad_request(
                    "knobs",
                    "invalid_request",
                    format!("\"knobs\" must be JSON: {e}"),
                )
            })
        })
        .transpose()?;
    let knobs = parse_knobs(knobs.as_ref())?;

    let manifest = {
        let (st, model) = (st.clone(), model.clone());
        tokio::task::spawn_blocking(move || clone_manifest(&st, &model))
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;

    let converted = {
        let (home, file) = (st.registry.home().to_path_buf(), file.clone());
        tokio::task::spawn_blocking(move || {
            let upload = voices::scratch(&home, "preview-upload");
            let result = std::fs::create_dir_all(home.join("tmp"))
                .and_then(|()| std::fs::write(&upload, &file))
                .map_err(|e| AddError::Io(format!("write {}: {e}", upload.display())))
                .and_then(|()| voices::convert_to_tmp(&home, "preview", &upload));
            let _ = std::fs::remove_file(&upload);
            result
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|e| match e {
        AddError::Clip(_) => ApiError {
            param: Some("file"),
            ..ApiError::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                CLIP_UNREADABLE,
            )
        },
        AddError::Length(_) => bad_request("file", "invalid_request", e.to_string()),
        _ => internal(&st, &req_id, e.to_string()),
    })?;
    let (wav_path, _secs) = converted;
    let tmp_dir = wav_path.parent().map(std::path::Path::to_path_buf);
    let cleanup = |tmp_dir: &Option<PathBuf>| {
        if let Some(dir) = tmp_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    };

    let mut options = SynthOptions {
        reference: Some((wav_path, text)),
        ..SynthOptions::default()
    };
    if let Err(e) = apply_knobs(&manifest, knobs, &mut options) {
        cleanup(&tmp_dir);
        return Err(e);
    }
    let guard = match st.models.acquire(manifest, None).await {
        Ok(guard) => guard,
        Err(e) => {
            cleanup(&tmp_dir);
            return Err(manager_error(&st, &req_id, e));
        }
    };
    let sample_rate = guard.tts().sample_rate();
    let mut rx = synthesise(guard, input, String::new(), options, None);
    let mut pcm = Vec::new();
    let mut failure = None;
    while let Some(piece) = rx.recv().await {
        match piece {
            Ok(p) => pcm.extend_from_slice(&p),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    cleanup(&tmp_dir);
    if let Some(e) = failure {
        return Err(failed(&st, &req_id, e));
    }
    let mut wav = wav_header(sample_rate, pcm.len() as u32).to_vec();
    wav.extend_from_slice(&pcm);
    Ok(audio_response(Format::Wav, sample_rate, wav))
}

#[cfg(target_os = "macos")]
const CLIP_UNREADABLE: &str = "the clip is not audio afconvert can read, such as WAV or MP3";
#[cfg(not(target_os = "macos"))]
const CLIP_UNREADABLE: &str = "the clip is not audio this build can decode, such as WAV or MP3";

#[cfg(test)]
mod tests {
    use super::sentence_count;

    #[test]
    fn sentences_are_counted_as_the_sidecar_splits_them() {
        // "Dr." is short and joins the next piece; "3.14" has no space.
        assert_eq!(sentence_count("Dr. Smith paid 3.14 dollars."), 1);
        assert_eq!(
            sentence_count("The first one. Is this second? Yes, it is! Done"),
            3
        );
        assert_eq!(
            sentence_count("今天天气非常好啊。我们去公园玩吧！\n\nA line of text here"),
            2
        );
        assert_eq!(sentence_count("Hi."), 1);
        assert_eq!(sentence_count("Hi. This is a longer sentence.\n \nOk."), 1);
        assert_eq!(sentence_count(" \n "), 0);
    }
}
