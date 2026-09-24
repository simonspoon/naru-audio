//! §2.2 `POST /v1/audio/transcriptions`.
//!
//! Checks run cheapest first: the form fields, then the model (404/409),
//! `language` against its manifest, the audio (415/413), and only then the
//! model manager's load (§3.4), unless the model is already loaded.

use std::convert::Infallible;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::{Extension, Multipart, State};
use axum::http::StatusCode;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::{AppState, RequestId, manager_error, registry_error};
use crate::error::ApiError;
use crate::manager::{Guard, KeepAlive};
use crate::registry::RegistryError;
use crate::registry::manifest::{Kind, Manifest};
use crate::stt::audio::{self, AudioError, TARGET_SAMPLE_RATE};
use crate::stt::{Segment, SttError, VadConfig, Vocabulary};

/// The request body cap: the §2.2 256 MiB audio cap plus room for the other
/// fields and the multipart framing.
pub(super) const MAX_BODY_BYTES: usize = audio::MAX_INPUT_BYTES as usize + 1024 * 1024;

/// OpenAI model names that stand for `default`, so stock clients work.
const DEFAULT_ALIASES: [&str; 4] = [
    "default",
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
];

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Json,
    Text,
    VerboseJson,
}

/// The multipart fields, as sent.
#[derive(Default)]
struct Form {
    file: Option<Bytes>,
    model: Option<String>,
    response_format: Option<String>,
    language: Option<String>,
    timestamp_granularities: Vec<String>,
    stream: Option<String>,
    hotwords: Option<String>,
    vad: Option<String>,
    vad_threshold: Option<String>,
    vad_min_speech: Option<String>,
    vad_min_silence: Option<String>,
    keep_alive: Option<String>,
}

/// A validated request.
struct Job {
    audio: Bytes,
    model: String,
    format: Format,
    language: Option<String>,
    stream: bool,
    hotwords: Option<Vocabulary>,
    vad: Option<VadConfig>,
    keep_alive: Option<KeepAlive>,
}

/// The checked request, before the load.
struct Prepared {
    manifest: Manifest,
    pcm: Vec<f32>,
    language: Option<String>,
}

/// Everything the decode needs, once the model has loaded. Dropping it ends
/// the request for the model manager (§3.4).
struct Loaded {
    model: Guard,
    pcm: Vec<f32>,
    language: Option<String>,
}

pub(super) async fn transcriptions(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let multipart = multipart.map_err(|e| {
        bad_request(
            "file",
            "invalid_request",
            format!("the body must be multipart/form-data: {}", e.body_text()),
        )
    })?;
    let job = validate(
        read_form(multipart).await?,
        &st.models.settings().stt_default,
    )?;
    if let Some(v) = &job.hotwords
        && v.terms_dropped > 0
    {
        st.log.line(
            "warn",
            Some(&req_id),
            &format!(
                "hotwords_truncated dropped={} kept={}",
                v.terms_dropped,
                v.terms.len()
            ),
        );
    }

    let Job {
        audio,
        model,
        format,
        language,
        stream,
        hotwords,
        vad,
        keep_alive,
    } = job;
    let prepare = {
        let st = st.clone();
        let model = model.clone();
        move || prepare(&st, &model, language, &audio)
    };
    let Prepared {
        manifest,
        pcm,
        language,
    } = tokio::task::spawn_blocking(prepare)
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    let loaded = Loaded {
        model: st
            .models
            .acquire(manifest, keep_alive)
            .await
            .map_err(|e| manager_error(&st, &req_id, e))?,
        pcm,
        language,
    };

    if stream {
        return Ok(sse(st, req_id, loaded, hotwords, vad));
    }

    let started = Instant::now();
    let (loaded, result) = tokio::task::spawn_blocking(move || {
        let result = loaded
            .model
            .stt()
            .decode(&loaded.pcm, hotwords.as_ref(), vad.as_ref());
        (loaded, result)
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let segments = result.map_err(|e| decode_error(&st, &req_id, e))?;
    let text = join(&segments);

    Ok(match format {
        Format::Json => Json(json!({"text": text})).into_response(),
        Format::Text => ([(CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response(),
        Format::VerboseJson => Json(json!({
            "task": "transcribe",
            "language": loaded.language,
            "duration": loaded.pcm.len() as f64 / TARGET_SAMPLE_RATE as f64,
            "text": text,
            "segments": segments
                .iter()
                .enumerate()
                .map(|(id, s)| json!({"id": id, "start": s.start, "end": s.end, "text": s.text}))
                .collect::<Vec<_>>(),
            "x_model": model,
            "x_decode_ms": started.elapsed().as_millis() as u64,
        }))
        .into_response(),
    })
}

async fn read_form(mut multipart: Multipart) -> Result<Form, ApiError> {
    let mut form = Form::default();
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" {
            form.file = Some(field.bytes().await.map_err(multipart_error)?);
            continue;
        }
        let value = field.text().await.map_err(multipart_error)?;
        let slot = match name.as_str() {
            "model" => &mut form.model,
            "response_format" => &mut form.response_format,
            "language" => &mut form.language,
            "stream" => &mut form.stream,
            "hotwords" => &mut form.hotwords,
            "vad" => &mut form.vad,
            "vad_threshold" => &mut form.vad_threshold,
            "vad_min_speech" => &mut form.vad_min_speech,
            "vad_min_silence" => &mut form.vad_min_silence,
            "keep_alive" => &mut form.keep_alive,
            "timestamp_granularities[]" => {
                form.timestamp_granularities.push(value);
                continue;
            }
            // `prompt`, `temperature`, `chunking_strategy` and `include[]`
            // are ignored (§2.2), as is anything unknown.
            _ => continue,
        };
        *slot = Some(value);
    }
    Ok(form)
}

/// `default_model` is what `default` and its aliases resolve to (§3.3).
fn validate(form: Form, default_model: &str) -> Result<Job, ApiError> {
    let audio = form
        .file
        .ok_or_else(|| bad_request("file", "invalid_request", "the \"file\" field is required"))?;

    let format = match form.response_format.as_deref() {
        None | Some("json") => Format::Json,
        Some("text") => Format::Text,
        Some("verbose_json") => Format::VerboseJson,
        Some(other) => {
            return Err(bad_request(
                "response_format",
                "unsupported_value",
                format!(
                    "response_format {other:?} is not supported; use json, text or verbose_json"
                ),
            ));
        }
    };

    for g in &form.timestamp_granularities {
        let message = match g.as_str() {
            "segment" if format == Format::VerboseJson => continue,
            "segment" => "segment timestamps need response_format=verbose_json".to_string(),
            "word" => "word timestamps are not supported; use segment".to_string(),
            other => format!("timestamp granularity {other:?} is not supported; use segment"),
        };
        return Err(bad_request(
            "timestamp_granularities[]",
            "unsupported_value",
            message,
        ));
    }

    let stream = parse_bool("stream", form.stream.as_deref(), false)?;

    let hotwords = form
        .hotwords
        .map(|h| Vocabulary::parse(&h))
        .transpose()
        .map_err(|e| bad_request("hotwords", "invalid_request", e.to_string()))?;

    // The auris flag checks (auris src/cli.rs).
    let mut cfg = VadConfig::default();
    if let Some(v) = parse_f32("vad_threshold", form.vad_threshold.as_deref())? {
        if !(0.0..=1.0).contains(&v) {
            return Err(bad_request(
                "vad_threshold",
                "invalid_request",
                format!("vad_threshold must be between 0.0 and 1.0, got {v}"),
            ));
        }
        cfg.threshold = v;
    }
    if let Some(v) = parse_f32("vad_min_speech", form.vad_min_speech.as_deref())? {
        if !v.is_finite() || v < 0.0 {
            return Err(bad_request(
                "vad_min_speech",
                "invalid_request",
                format!("vad_min_speech must be a non-negative number of seconds, got {v}"),
            ));
        }
        cfg.min_speech = v;
    }
    if let Some(v) = parse_f32("vad_min_silence", form.vad_min_silence.as_deref())? {
        // Strictly positive: zero trailing silence never closes a span.
        if !v.is_finite() || v <= 0.0 {
            return Err(bad_request(
                "vad_min_silence",
                "invalid_request",
                format!("vad_min_silence must be a positive number of seconds, got {v}"),
            ));
        }
        cfg.min_silence = v;
    }
    let vad = parse_bool("vad", form.vad.as_deref(), true)?.then_some(cfg);

    let keep_alive = form
        .keep_alive
        .map(|k| KeepAlive::parse(&k))
        .transpose()
        .map_err(|e| bad_request("keep_alive", "invalid_request", e))?;

    let model = match form.model.as_deref() {
        None => default_model,
        Some(m) if DEFAULT_ALIASES.contains(&m) => default_model,
        Some(m) => m,
    };

    Ok(Job {
        audio,
        model: model.to_string(),
        format,
        language: form.language,
        stream,
        hotwords,
        vad,
        keep_alive,
    })
}

/// The pulled manifest of the STT model `name`: 404 unknown, 409 not
/// pulled, 400 not STT. Blocking.
pub(super) fn stt_manifest(st: &AppState, name: &str) -> Result<Manifest, ApiError> {
    // Only a catalog name or a pulled one reaches the filesystem, so a name
    // like `../x` or `/tmp/x` is 404, as in `Registry::lock_pulled`.
    let known = st.registry.catalog().models.contains_key(name)
        || st
            .registry
            .pulled()
            .map_err(registry_error)?
            .iter()
            .any(|n| n == name);
    if !known {
        return Err(registry_error(RegistryError::UnknownModel(
            name.to_string(),
        )));
    }
    // Never auto-pulls (§2.6 `model_not_pulled`).
    let manifest = st.registry.pulled_manifest(name).map_err(registry_error)?;
    if manifest.model.kind != Kind::Stt {
        return Err(bad_request(
            "model",
            "invalid_request",
            format!("the model \"{name}\" is not a speech-to-text model"),
        ));
    }
    Ok(manifest)
}

/// The model's manifest (404/409), `language` (400), then the audio
/// (415/413). Blocking.
fn prepare(
    st: &AppState,
    name: &str,
    language: Option<String>,
    wav: &[u8],
) -> Result<Prepared, ApiError> {
    let manifest = stt_manifest(st, name)?;
    let languages = languages(&manifest);
    let language = match language {
        None => languages.first().cloned(),
        Some(l) if languages.contains(&l) => Some(l),
        Some(l) => {
            return Err(bad_request(
                "language",
                "unsupported_value",
                format!(
                    "the model \"{name}\" does not support language {l:?}; it supports: {}",
                    languages.join(", ")
                ),
            ));
        }
    };

    let pcm = match audio::decode(wav) {
        Ok(pcm) => pcm,
        // A valid WAV with no samples has nothing to transcribe (§2.2).
        Err(AudioError::Empty) => Vec::new(),
        Err(e) => return Err(audio_error(e)),
    };
    Ok(Prepared {
        manifest,
        pcm,
        language,
    })
}

/// `model.languages` from the manifest; the struct does not model it.
fn languages(manifest: &Manifest) -> Vec<String> {
    manifest
        .raw
        .get("model")
        .and_then(|m| m.get("languages"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// §2.2 `stream=true`: a `transcript.text.delta` per segment as it is
/// decoded, then `transcript.text.done`. A decode failure (or panic) after
/// the stream has started ends it with an `error` event.
fn sse(
    st: Arc<AppState>,
    req_id: String,
    loaded: Loaded,
    hotwords: Option<Vocabulary>,
    vad: Option<VadConfig>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::task::spawn_blocking(move || {
        // A send fails only once the client has gone; the decode still finishes.
        let send = |v: Value| {
            let _ = tx.send(format!("data: {v}\n\n"));
        };
        let mut text = String::new();
        // A panic would otherwise drop `tx` and end the stream with no
        // `done` or `error` event.
        let result = catch_unwind(AssertUnwindSafe(|| {
            loaded
                .model
                .stt()
                .decode_each(&loaded.pcm, hotwords.as_ref(), vad.as_ref(), &mut |s| {
                    let delta = if text.is_empty() {
                        s.text
                    } else {
                        format!(" {}", s.text)
                    };
                    text.push_str(&delta);
                    send(json!({"type": "transcript.text.delta", "delta": delta}));
                })
                .map_err(|e| (decode_code(&e), e.to_string()))
        }))
        .unwrap_or_else(|_| Err(("internal", "the decode panicked".to_string())));
        match result {
            Ok(()) => send(json!({"type": "transcript.text.done", "text": text})),
            Err((code, e)) => {
                st.log
                    .line("error", Some(&req_id), &format!("decode_failed {e}"));
                send(json!({"type": "error", "error": {
                    "message": e,
                    "type": "server_error",
                    "code": code,
                    "param": null,
                }}));
            }
        }
    });
    let events = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, Infallible>(event), rx))
    });
    (
        [
            (CONTENT_TYPE, "text/event-stream"),
            (CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(events),
    )
        .into_response()
}

/// Segment texts joined with a space.
pub(super) fn join(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn bad_request(param: &'static str, code: &'static str, message: impl Into<String>) -> ApiError {
    ApiError {
        param: Some(param),
        ..ApiError::new(StatusCode::BAD_REQUEST, code, message)
    }
}

/// §2.6 code for a failed decode: the backend went away mid-decode (a
/// crashed MLX sidecar, §5.3), or anything else.
fn decode_code(e: &SttError) -> &'static str {
    match e {
        SttError::BackendUnavailable { .. } => "backend_unavailable",
        _ => "internal",
    }
}

/// A failed decode, logged: 503 `backend_unavailable`, or a 500.
pub(super) fn decode_error(st: &AppState, req_id: &str, e: SttError) -> ApiError {
    if let SttError::BackendUnavailable { .. } = e {
        st.log
            .line("warn", Some(req_id), &format!("decode_failed {e}"));
        return ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_unavailable",
            e.to_string(),
        );
    }
    internal(st, req_id, e.to_string())
}

/// A 500, logged: the log line carries the request id (§2.6).
pub(super) fn internal(st: &AppState, req_id: &str, message: String) -> ApiError {
    st.log
        .line("error", Some(req_id), &format!("internal {message}"));
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
}

fn parse_bool(param: &'static str, value: Option<&str>, default: bool) -> Result<bool, ApiError> {
    match value {
        None => Ok(default),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(bad_request(
            param,
            "invalid_request",
            format!("{param} must be true or false, not {other:?}"),
        )),
    }
}

fn parse_f32(param: &'static str, value: Option<&str>) -> Result<Option<f32>, ApiError> {
    value
        .map(|v| {
            v.trim().parse::<f32>().map_err(|_| {
                bad_request(
                    param,
                    "invalid_request",
                    format!("{param} must be a number, not {v:?}"),
                )
            })
        })
        .transpose()
}

/// The body cap trips as 413; any other framing error is 400.
fn multipart_error(e: MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError {
            param: Some("file"),
            ..ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                format!(
                    "the request exceeds the {} MiB cap",
                    audio::MAX_INPUT_BYTES / 1024 / 1024
                ),
            )
        };
    }
    bad_request(
        "file",
        "invalid_request",
        format!("malformed multipart body: {}", e.body_text()),
    )
}

/// §2.2/§2.6: not WAV, or a WAV variant outside §2.2, is 415; over a cap is
/// 413; a broken WAV is 400.
fn audio_error(e: AudioError) -> ApiError {
    let (status, code) = match &e {
        AudioError::NotWav(_)
        | AudioError::UnsupportedRate(_)
        | AudioError::Wav(hound::Error::Unsupported) => {
            (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type")
        }
        AudioError::TooLarge | AudioError::TooLong => {
            (StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large")
        }
        AudioError::Io(_) | AudioError::Resample(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "internal")
        }
        AudioError::NoInput | AudioError::Wav(_) | AudioError::Empty => {
            (StatusCode::BAD_REQUEST, "invalid_request")
        }
    };
    ApiError {
        param: Some("file"),
        ..ApiError::new(status, code, e.to_string())
    }
}
