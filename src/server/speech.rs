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

    let mut rx = synthesise(model, job.input, voice, job.options);
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
        return Ok((AppendHeaders(headers(job.format, sample_rate)), body).into_response());
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
/// the rest; the model is released when synthesis returns.
fn synthesise(
    model: Guard,
    input: String,
    voice: String,
    options: SynthOptions,
) -> tokio::sync::mpsc::Receiver<Result<Bytes, Failure>> {
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUED_PIECES);
    tokio::task::spawn_blocking(move || {
        let sink_tx = tx.clone();
        let sink = Box::new(move |samples: &[f32]| {
            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
                .collect();
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
    });
    rx
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
    let stream = boolean("stream", true)?;

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
    })
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
                        Some(
                            json!({"id": id, "accent": null, "gender": null, "default": false,
                                     "cloned": true, "model": model}),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        }
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?;
        return Ok(Json(json!({"model": "clones", "voices": voices})));
    }
    let (manifest, voices) = {
        let st = st.clone();
        tokio::task::spawn_blocking(move || {
            let manifest = match st.registry.catalog().models.get(&name) {
                Some(m) if !st.registry.is_installed(&name) => match m.model.kind {
                    Kind::Tts => Ok(m.clone()),
                    _ => Err(not_kind(&name, Kind::Tts)),
                },
                _ => kind_manifest(&st, &name, Kind::Tts),
            }?;
            let voices = voices::of(&manifest, st.registry.home());
            Ok::<_, ApiError>((manifest, voices))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    // `PUT /api/defaults`' live override, if set, wins over the manifest's
    // own `default` marker.
    let default_override = st.models.default_voice(&manifest.model.name);
    let voices: Vec<Value> = voices
        .iter()
        .map(|v| {
            // `voices::of` appends the cloned voices no `[[voice]]` shadows.
            let cloned = !manifest.voices.iter().any(|m| m.id == v.id);
            let default = match &default_override {
                Some(d) => &v.id == d,
                None => v.default,
            };
            json!({"id": v.id, "accent": v.accent, "gender": v.gender, "default": default,
                   "cloned": cloned})
        })
        .collect();
    Ok(Json(
        json!({"model": manifest.model.name, "voices": voices}),
    ))
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
            Ok::<_, String>(Some((voice.text, voice.model, wav)))
        })
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|m| internal(&st, &req_id, m))?;
    let (text, model, wav) = found.ok_or_else(|| ApiError {
        param: Some("name"),
        ..ApiError::new(
            StatusCode::NOT_FOUND,
            "voice_not_found",
            format!("there is no cloned voice {name:?}"),
        )
    })?;
    Ok(Json(
        json!({"name": name, "text": text, "model": model, "wav_base64": STANDARD.encode(wav)}),
    ))
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
    if st
        .registry
        .catalog()
        .models
        .values()
        .any(|m| m.voices.iter().any(|v| v.id == name))
    {
        return Err(ApiError {
            param: Some("name"),
            ..ApiError::new(
                StatusCode::CONFLICT,
                "builtin_voice",
                format!("{name:?} is a built-in voice; only cloned voices can be removed"),
            )
        });
    }
    let removed = {
        let (home, name) = (st.registry.home().to_path_buf(), name.clone());
        tokio::task::spawn_blocking(move || voices::remove(&home, &name))
    }
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|m| internal(&st, &req_id, m))?;
    if !removed {
        return Err(ApiError {
            param: Some("name"),
            ..ApiError::new(
                StatusCode::NOT_FOUND,
                "voice_not_found",
                format!("there is no cloned voice {name:?}"),
            )
        });
    }
    Ok(StatusCode::NO_CONTENT)
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
    let (mut name, mut file, mut text, mut model) = (None, None, None, None);
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        match field.name().unwrap_or_default().to_owned().as_str() {
            "file" => file = Some(field.bytes().await.map_err(multipart_error)?),
            "name" => name = Some(field.text().await.map_err(multipart_error)?),
            "text" => text = Some(field.text().await.map_err(multipart_error)?),
            "model" => model = Some(field.text().await.map_err(multipart_error)?),
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
        let (home, name, model) = (
            st.registry.home().to_path_buf(),
            name.clone(),
            model.clone(),
        );
        tokio::task::spawn_blocking(move || {
            let upload = voices::scratch(&home, "upload");
            let result = std::fs::create_dir_all(home.join("tmp"))
                .and_then(|()| std::fs::write(&upload, &file))
                .map_err(|e| AddError::Io(format!("write {}: {e}", upload.display())))
                .and_then(|()| voices::add(&home, &name, &upload, &text, &model, require_text));
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
                "the clip is not audio afconvert can read, such as WAV or MP3",
            )
        },
        AddError::Length(_) => bad_request("file", "invalid_request", e.to_string()),
        AddError::Io(m) => internal(&st, &req_id, m),
    })?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": name,
            "accent": null,
            "gender": null,
            "default": false,
            "cloned": true,
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
