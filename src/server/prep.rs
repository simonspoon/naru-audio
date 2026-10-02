//! §8 (`docs/design.md`) voice-prep HTTP surface: `/v1/audio/prep/clips`
//! (upload, transcript, raw/working audio) and `/v1/audio/samples` (the
//! cleaned clone samples a clip's clips are cropped and cleaned into).
//! Every processing step is a `POST` (§2.5 "a GET never does work"); every
//! blocking filesystem or model call runs in `spawn_blocking`, as
//! `transcriptions` and `speech` do.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Extension, Multipart, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{AppendHeaders, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::transcriptions::{bad_request, internal, kind_manifest, multipart_error};
use super::{AppState, RequestId, manager_error, registry_error};
use crate::backend;
use crate::error::ApiError;
use crate::prep::diarize::ClusterOptions;
use crate::prep::edit::{self, Segment};
use crate::prep::pipeline::{self, Analysis, ModelSteps, Select, Step};
use crate::prep::takes::{self, TakeOptions};
use crate::prep::{
    self, ClipMeta, EngineUsed, PrepError, Project, SampleMeta, SampleRange, SpeakerSpan,
    Transcript, TranscriptWord,
};
use crate::registry::manifest::{Kind, Manifest};
use crate::stt::SttError;
use crate::stt::audio::TARGET_SAMPLE_RATE;
use crate::stt::vad::VadConfig;

fn prep_error(st: &AppState, req_id: &str, e: PrepError) -> ApiError {
    internal(st, req_id, e.to_string())
}

fn clip_not_found(id: &str) -> ApiError {
    ApiError {
        param: Some("clip_id"),
        ..ApiError::new(
            StatusCode::NOT_FOUND,
            "clip_not_found",
            format!("there is no clip {id:?}"),
        )
    }
}

fn sample_not_found(id: &str) -> ApiError {
    ApiError {
        param: Some("id"),
        ..ApiError::new(
            StatusCode::NOT_FOUND,
            "sample_not_found",
            format!("there is no sample {id:?}"),
        )
    }
}

fn transcript_not_found(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::CONFLICT,
        "transcript_not_run",
        format!("clip {id:?} has no transcript yet; POST .../transcribe first"),
    )
}

fn load_clip_meta(home: &Path, id: &str) -> Result<ClipMeta, ApiError> {
    prep::check_id(id).map_err(|m| bad_request("clip_id", "invalid_request", m))?;
    let dir = prep::clip_dir(home, id);
    if !dir.join(prep::CLIP_META).is_file() {
        return Err(clip_not_found(id));
    }
    prep::load_json(&dir.join(prep::CLIP_META)).map_err(|_| clip_not_found(id))
}

fn load_sample_meta(home: &Path, id: &str) -> Result<SampleMeta, ApiError> {
    prep::check_id(id).map_err(|m| bad_request("id", "invalid_request", m))?;
    let dir = prep::sample_dir(home, id);
    if !dir.join(prep::SAMPLE_META).is_file() {
        return Err(sample_not_found(id));
    }
    prep::load_json(&dir.join(prep::SAMPLE_META)).map_err(|_| sample_not_found(id))
}

fn clip_json(m: &ClipMeta) -> Value {
    json!({
        "id": m.id,
        "original_filename": m.original_filename,
        "content_type": m.content_type,
        "duration_secs": m.duration_secs,
        "created_at": m.created_at,
    })
}

fn sample_json(home: &Path, m: &SampleMeta) -> Value {
    let transcript =
        std::fs::read_to_string(prep::sample_dir(home, &m.id).join(prep::TRANSCRIPT_TXT)).ok();
    json!({
        "id": m.id,
        "name": m.name,
        "source_clip_id": m.source_clip_id,
        "original_filename": m.original_filename,
        "range": {"start": m.range.start, "end": m.range.end},
        "speaker": m.speaker,
        "engines": m.engines.iter().map(|e| json!({
            "stage": e.stage,
            "model": e.model,
            "license": e.license,
            "non_commercial": e.non_commercial,
        })).collect::<Vec<_>>(),
        "warnings": m.warnings,
        "created_at": m.created_at,
        "transcript": transcript,
        "steps": m.steps,
        "analysis": m.analysis,
        "segments": m.segments,
    })
}

/// A file name's extension (without the dot), sanitised to plain ASCII
/// word characters; `"bin"` for anything else, including no extension.
fn safe_extension(filename: Option<&str>) -> String {
    let ext = filename
        .and_then(|f| f.rsplit_once('.'))
        .map(|(_, ext)| ext)
        .unwrap_or("");
    let ext: String = ext
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();
    if ext.is_empty() {
        "bin".to_string()
    } else {
        ext.to_lowercase()
    }
}

// ---- clips ----------------------------------------------------------

/// `POST /v1/audio/prep/clips`: uploads a raw clip (`file`, any format
/// `afconvert` reads). Streams it to disk verbatim as `raw.<ext>` (never held
/// in memory: a recording can be hours long) next to a decoded `working.wav`
/// (naru task 1461 §8); the clip's length must be positive and within
/// [`crate::stt::audio::Limits::PREP`].
pub(super) async fn upload_clip(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    use tokio::io::AsyncWriteExt;

    let mut multipart = multipart.map_err(|e| {
        bad_request(
            "file",
            "invalid_request",
            format!("the body must be multipart/form-data: {}", e.body_text()),
        )
    })?;
    let home = st.registry.home().to_path_buf();
    let id = prep::new_id();
    let dir = prep::clip_dir(&home, &id);

    let streamed = async {
        let mut saved: Option<(String, Option<String>, Option<String>)> = None;
        while let Some(mut field) = multipart.next_field().await.map_err(upload_error)? {
            if field.name().unwrap_or_default() != "file" {
                continue;
            }
            let filename = field.file_name().map(str::to_string);
            let content_type = field.content_type().map(str::to_string);
            let raw_filename = format!("raw.{}", safe_extension(filename.as_deref()));
            tokio::fs::create_dir_all(&dir)
                .await
                .map_err(|e| internal(&st, &req_id, e.to_string()))?;
            let mut out = tokio::fs::File::create(dir.join(&raw_filename))
                .await
                .map_err(|e| internal(&st, &req_id, e.to_string()))?;
            while let Some(chunk) = field.chunk().await.map_err(upload_error)? {
                out.write_all(&chunk)
                    .await
                    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
            }
            out.flush()
                .await
                .map_err(|e| internal(&st, &req_id, e.to_string()))?;
            saved = Some((raw_filename, filename, content_type));
        }
        saved
            .ok_or_else(|| bad_request("file", "invalid_request", "the \"file\" field is required"))
    }
    .await;
    let (raw_filename, filename, content_type) = match streamed {
        Ok(saved) => saved,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
    };

    let result = tokio::task::spawn_blocking(move || {
        create_clip(&home, id, raw_filename, filename, content_type)
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let meta = result.map_err(|e| match e {
        CreateClipError::Prep(e) => prep_error(&st, &req_id, e),
        CreateClipError::BadAudio(m) => bad_request("file", "invalid_request", m),
    })?;
    Ok((StatusCode::CREATED, Json(clip_json(&meta))).into_response())
}

/// A multipart read error: the body cap trips as 413, anything else is 400.
fn upload_error(e: axum::extract::multipart::MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError {
            param: Some("file"),
            ..ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                format!(
                    "the request exceeds the {} GiB clip upload cap",
                    super::MAX_PREP_UPLOAD_BYTES / 1024 / 1024 / 1024
                ),
            )
        };
    }
    multipart_error(e)
}

enum CreateClipError {
    Prep(PrepError),
    BadAudio(String),
}

/// Decodes the streamed-in `raw_filename` of clip `id` to `working.wav` and
/// writes `meta.json`; the clip's directory is removed on any failure.
fn create_clip(
    home: &Path,
    id: String,
    raw_filename: String,
    filename: Option<String>,
    content_type: Option<String>,
) -> Result<ClipMeta, CreateClipError> {
    let dir = prep::clip_dir(home, &id);
    let raw_path = dir.join(&raw_filename);
    let working = dir.join(prep::WORKING_WAV);
    let duration_secs = match prep::ingest_to_wav(&raw_path, &working) {
        Ok(secs) if secs > 0.0 => secs,
        Ok(_) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(CreateClipError::BadAudio(
                "the clip decoded to no audio".to_string(),
            ));
        }
        Err(PrepError::BadAudio(m)) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(CreateClipError::BadAudio(m));
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(CreateClipError::Prep(e));
        }
    };
    let meta = ClipMeta {
        id: id.clone(),
        raw_filename,
        original_filename: filename,
        content_type,
        duration_secs,
        created_at: prep::now_rfc3339(),
    };
    prep::save_json(&dir.join(prep::CLIP_META), &meta).map_err(CreateClipError::Prep)?;
    Ok(meta)
}

/// `GET /v1/audio/prep/clips`.
pub(super) async fn list_clips(State(st): State<Arc<AppState>>) -> Json<Value> {
    let home = st.registry.home().to_path_buf();
    let ids = tokio::task::spawn_blocking(move || prep::list_clips(&home))
        .await
        .unwrap_or_default();
    Json(json!({"clips": ids}))
}

/// `GET /v1/audio/prep/clips/{id}`.
pub(super) async fn get_clip(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let meta = tokio::task::spawn_blocking(move || load_clip_meta(&home, &id))
        .await
        .map_err(|e| {
            ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
        })??;
    Ok(Json(clip_json(&meta)))
}

/// `DELETE /v1/audio/prep/clips/{id}`.
pub(super) async fn delete_clip(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    // A path segment can still decode to `..` or embed a `/` (axum
    // percent-decodes the captured value after matching, so `%2F` in
    // `{id}` becomes a literal `/`); without this check `clip_dir` joins
    // it straight into a `remove_dir_all` (naru task 1461 review).
    prep::check_id(&id).map_err(|_| clip_not_found(&id))?;
    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let existed = tokio::task::spawn_blocking(move || {
        let dir = prep::clip_dir(&home, &id2);
        let existed = dir.join(prep::CLIP_META).is_file();
        if existed {
            let _ = std::fs::remove_dir_all(&dir);
        }
        existed
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    if !existed {
        return Err(clip_not_found(&id));
    }
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /v1/audio/prep/clips/{id}/audio`: the decoded `working.wav`
/// (16 kHz mono). Its raw upload is available un-decoded from the same
/// route with `?raw=true`, so a client can compare either against a
/// sample's own raw/clean pair.
pub(super) async fn clip_audio(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let home = st.registry.home().to_path_buf();
    let raw = q.get("raw").map(String::as_str) == Some("true");
    let id2 = id.clone();
    let outcome =
        tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, String), ClipAudioError> {
            let meta = load_clip_meta_blocking(&home, &id2)?;
            let dir = prep::clip_dir(&home, &id2);
            let (path, content_type) = if raw {
                (
                    dir.join(&meta.raw_filename),
                    meta.content_type
                        .unwrap_or_else(|| "application/octet-stream".to_string()),
                )
            } else {
                (dir.join(prep::WORKING_WAV), "audio/wav".to_string())
            };
            let bytes = std::fs::read(&path).map_err(|e| ClipAudioError::Io(e.to_string()))?;
            Ok((bytes, content_type))
        })
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?;
    let (bytes, content_type) = outcome.map_err(|e| match e {
        ClipAudioError::NotFound => clip_not_found(&id),
        ClipAudioError::Io(m) => internal(&st, &req_id, m),
    })?;
    Ok((
        AppendHeaders([(CONTENT_TYPE, content_type)]),
        axum::body::Body::from(bytes),
    )
        .into_response())
}

enum ClipAudioError {
    NotFound,
    Io(String),
}

/// [`load_clip_meta`] without an `ApiError`, for a `spawn_blocking` closure
/// that maps its own errors after the join.
fn load_clip_meta_blocking(home: &Path, id: &str) -> Result<ClipMeta, ClipAudioError> {
    if prep::check_id(id).is_err() {
        return Err(ClipAudioError::NotFound);
    }
    let dir = prep::clip_dir(home, id);
    if !dir.join(prep::CLIP_META).is_file() {
        return Err(ClipAudioError::NotFound);
    }
    prep::load_json(&dir.join(prep::CLIP_META)).map_err(|e| ClipAudioError::Io(e.to_string()))
}

// ---- transcript -------------------------------------------------------

#[derive(Debug)]
struct TranscribeRequest {
    stt_model: Option<String>,
    diarize: bool,
    diarization_model: Option<String>,
    /// Exact speaker count, when the caller knows it; fixes real speakers
    /// getting merged by the clustering threshold. Overrides
    /// `cluster_threshold` when set.
    num_speakers: Option<u32>,
    /// Cosine-distance clustering threshold; ignored when `num_speakers`
    /// is set. Sherpa-onnx's own default is 0.5.
    cluster_threshold: Option<f32>,
    /// A multilingual `stt_model`'s language, else it detects one.
    language: Option<String>,
}

fn word_json(w: &TranscriptWord) -> Value {
    let mut v = json!({"start": w.start, "end": w.end, "text": w.text});
    if let Some(c) = w.confidence {
        v["confidence"] = json!(c);
    }
    v
}

fn speaker_json(s: &SpeakerSpan) -> Value {
    json!({"start": s.start, "end": s.end, "speaker": s.speaker})
}

fn transcript_json(t: &Transcript) -> Value {
    json!({
        "words": t.words.iter().map(word_json).collect::<Vec<_>>(),
        "speakers": t.speakers.iter().map(speaker_json).collect::<Vec<_>>(),
        "overlaps": t.overlaps,
        "language": t.language,
        "x_stt_model": t.stt_model,
        "x_diarization_model": t.diarization_model,
    })
}

/// `POST /v1/audio/prep/clips/{id}/transcribe` (naru task 1461 §8
/// pipeline step 1): word timestamps from the default (or requested)
/// speech-to-text model, plus speaker diarization unless `diarize:false`.
/// Only a `sherpa-onnx` Parakeet model and the MLX Whisper model report
/// word timestamps ([`crate::stt::SttModel::decode_words`]'s default is
/// "unsupported"); the MLX Parakeet does not, so the daemon's default
/// `stt_model` may need overriding here even where it serves
/// `/v1/audio/transcriptions` fine. The optional `language` is passed to a
/// multilingual model; the transcript's `language` is the one used.
pub(super) async fn transcribe_clip(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: TranscribeRequest = if body.is_empty() {
        TranscribeRequest {
            stt_model: None,
            diarize: true,
            diarization_model: None,
            num_speakers: None,
            cluster_threshold: None,
            language: None,
        }
    } else {
        #[derive(Deserialize, Default)]
        #[serde(default)]
        struct Raw {
            stt_model: Option<String>,
            diarize: Option<bool>,
            diarization_model: Option<String>,
            num_speakers: Option<i64>,
            cluster_threshold: Option<f32>,
            language: Option<String>,
        }
        let raw: Raw = serde_json::from_slice(&body)
            .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?;
        let num_speakers = match raw.num_speakers {
            Some(n) if n < 1 || n > i32::MAX as i64 => {
                return Err(bad_request(
                    "num_speakers",
                    "invalid_request",
                    format!("\"num_speakers\" must be a positive integer, got {n}"),
                ));
            }
            Some(n) => Some(n as u32),
            None => None,
        };
        if let Some(t) = raw.cluster_threshold
            && (!t.is_finite() || t <= 0.0 || t > 2.0)
        {
            return Err(bad_request(
                "cluster_threshold",
                "invalid_request",
                format!("\"cluster_threshold\" must be in (0, 2], got {t}"),
            ));
        }
        TranscribeRequest {
            stt_model: raw.stt_model,
            diarize: raw.diarize.unwrap_or(true),
            diarization_model: raw.diarization_model,
            num_speakers,
            cluster_threshold: raw.cluster_threshold,
            language: raw.language,
        }
    };

    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    // Shared by the recogniser and the diarizer: a long clip's samples are
    // held once.
    let pcm = tokio::task::spawn_blocking(move || -> Result<Arc<Vec<f32>>, ApiError> {
        let _meta = load_clip_meta(&home, &id2)?;
        prep::read_wav(&prep::clip_dir(&home, &id2).join(prep::WORKING_WAV))
            .map(Arc::new)
            .map_err(prep_error_to_api)
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;

    let stt_model = req.stt_model.unwrap_or_else(|| st.models.stt_default());
    let stt_manifest = {
        let (st2, stt_model2) = (st.clone(), stt_model.clone());
        tokio::task::spawn_blocking(move || kind_manifest(&st2, &stt_model2, Kind::Stt))
            .await
            .map_err(|e| internal(&st, &req_id, e.to_string()))?
    }?;
    let languages = stt_manifest.languages().unwrap_or_default();
    if let Some(l) = &req.language
        && !languages.contains(l)
    {
        return Err(bad_request(
            "language",
            "unsupported_value",
            format!(
                "the model \"{stt_model}\" does not support language {l:?}; it supports: {}",
                languages.join(", ")
            ),
        ));
    }
    let requested = req.language.clone();
    let default_language = languages.first().cloned();
    let guard = st
        .models
        .acquire(stt_manifest, None)
        .await
        .map_err(|e| manager_error(&st, &req_id, e))?;

    let pcm_for_decode = Arc::clone(&pcm);
    let (words, used_language) = tokio::task::spawn_blocking(move || {
        guard.stt().decode_words_in(
            &pcm_for_decode,
            Some(&VadConfig::default()),
            requested.as_deref(),
        )
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|e| match e {
        SttError::WordTimestampsUnsupported => bad_request(
            "stt_model",
            "word_timestamps_unsupported",
            format!(
                "\"{stt_model}\" does not report word timestamps; use a sherpa-onnx Parakeet model, e.g. \"parakeet-tdt-0.6b-v2-int8\", or \"whisper-large-v3-turbo-mlx\" on Apple Silicon"
            ),
        ),
        other => super::transcriptions::decode_error(&st, &req_id, other),
    })?;

    let diarization_model = req
        .diarization_model
        .unwrap_or_else(|| prep::DEFAULT_DIARIZE_MODEL.to_string());
    let speakers: Vec<SpeakerSpan> = if req.diarize {
        let (st2, req_id2, diarization_model2) =
            (st.clone(), req_id.clone(), diarization_model.clone());
        let home = st.registry.home().to_path_buf();
        let pcm_for_diarize = Arc::clone(&pcm);
        let cluster_options = ClusterOptions {
            num_speakers: req.num_speakers,
            threshold: req.cluster_threshold,
        };
        tokio::task::spawn_blocking(move || -> Result<Vec<_>, ApiError> {
            let manifest = kind_manifest(&st2, &diarization_model2, Kind::Diarization)?;
            let dir = home.join("models").join(&diarization_model2);
            backend::load_diarizer(&manifest, &dir, cluster_options)
                .and_then(|d| d.diarize(&pcm_for_diarize))
                .map_err(|e| internal(&st2, &req_id2, e.to_string()))
        })
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))??
        .into_iter()
        .map(|s| SpeakerSpan {
            start: s.start,
            end: s.end,
            speaker: s.speaker,
        })
        .collect()
    } else {
        Vec::new()
    };

    let overlaps = edit::overlaps_of(&speakers);
    let transcript = Transcript {
        words: words
            .into_iter()
            .map(|w| TranscriptWord {
                start: w.start,
                end: w.end,
                text: w.text,
                confidence: w.confidence,
            })
            .collect(),
        speakers,
        overlaps,
        stt_model,
        // The language the model used (the request's, else the one it
        // detected), or for a model with no notion of one, the manifest's.
        language: used_language.or(default_language),
        diarization_model: req_diarize_model_or_none(req.diarize, &diarization_model),
    };

    let home = st.registry.home().to_path_buf();
    let id3 = id.clone();
    let transcript2 = transcript.clone();
    tokio::task::spawn_blocking(move || {
        prep::save_json(
            &prep::clip_dir(&home, &id3).join(prep::TRANSCRIPT_JSON),
            &transcript2,
        )
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|e| prep_error(&st, &req_id, e))?;

    Ok(Json(transcript_json(&transcript)))
}

fn req_diarize_model_or_none(diarize: bool, model: &str) -> Option<String> {
    diarize.then(|| model.to_string())
}

fn prep_error_to_api(e: PrepError) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
}

/// `GET /v1/audio/prep/clips/{id}/transcript`: the cached
/// [`prep::TRANSCRIPT_JSON`] from the last `POST .../transcribe`; 409 if
/// none has run yet (§2.5: a `GET` never runs the pipeline itself).
pub(super) async fn get_transcript(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let transcript = tokio::task::spawn_blocking(move || -> Result<Transcript, ApiError> {
        let _meta = load_clip_meta(&home, &id2)?;
        let path = prep::clip_dir(&home, &id2).join(prep::TRANSCRIPT_JSON);
        if !path.is_file() {
            return Err(transcript_not_found(&id2));
        }
        prep::load_json(&path).map_err(prep_error_to_api)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))??;
    Ok(Json(transcript_json(&transcript)))
}

// ---- peaks, takes -------------------------------------------------------

/// Most buckets one `GET .../peaks` returns.
const MAX_PEAK_BUCKETS: usize = 8192;
/// Buckets when `buckets` is not given.
const DEFAULT_PEAK_BUCKETS: usize = 2000;

fn query_f64(q: &HashMap<String, String>, param: &'static str) -> Result<Option<f64>, ApiError> {
    q.get(param)
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .ok_or_else(|| {
                    bad_request(
                        param,
                        "invalid_request",
                        format!("{param} must be a number, not {v:?}"),
                    )
                })
        })
        .transpose()
}

/// `GET /v1/audio/prep/clips/{id}/peaks?start=&end=&buckets=`: min/max
/// waveform peaks of `working.wav`'s `start..end` (default the whole clip) in
/// `buckets` buckets (default 2000, at most 8192), so a browser can draw a
/// long clip without downloading it. Reads only that range of the file.
pub(super) async fn clip_peaks(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let (start, end) = (query_f64(&q, "start")?, query_f64(&q, "end")?);
    let buckets = match q.get("buckets") {
        None => DEFAULT_PEAK_BUCKETS,
        Some(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=MAX_PEAK_BUCKETS).contains(n))
            .ok_or_else(|| {
                bad_request(
                    "buckets",
                    "invalid_request",
                    format!("buckets must be an integer in 1..={MAX_PEAK_BUCKETS}, not {v:?}"),
                )
            })?,
    };
    if let (Some(s), Some(e)) = (start, end)
        && e <= s
    {
        return Err(bad_request(
            "end",
            "invalid_request",
            format!("the range {s}-{e} is empty"),
        ));
    }
    let home = st.registry.home().to_path_buf();
    let peaks = tokio::task::spawn_blocking(move || -> Result<edit::Peaks, ApiError> {
        load_clip_meta(&home, &id)?;
        edit::peaks(
            &prep::clip_dir(&home, &id).join(prep::WORKING_WAV),
            start,
            end,
            buckets,
        )
        .map_err(|e| match e {
            PrepError::BadAudio(m) => bad_request("start", "invalid_request", m),
            e => prep_error_to_api(e),
        })
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok(Json(json!(peaks)))
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TakesRequest {
    speaker: Option<i32>,
    /// The kept region; the whole clip when absent.
    segments: Option<Vec<Segment>>,
    target_min: Option<f64>,
    target_max: Option<f64>,
    exclude_overlaps: Option<bool>,
}

/// `POST /v1/audio/prep/clips/{id}/takes`: scores the clip's candidate takes
/// and picks the best ones ([`takes::plan_takes`]). Needs the clip's
/// transcript (409 `transcript_not_run` otherwise); a `speaker` the
/// transcript has no spans for is a 400. Defaults: `target_min` 10 s,
/// `target_max` 20 s, `exclude_overlaps` true.
pub(super) async fn clip_takes(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: TakesRequest = if body.is_empty() {
        TakesRequest::default()
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?
    };
    let target_min = req.target_min.unwrap_or(10.0);
    let target_max = req.target_max.unwrap_or(20.0);
    if !(target_min.is_finite() && target_min > 0.0 && target_min <= target_max)
        || !target_max.is_finite()
        || target_max > 3600.0
    {
        return Err(bad_request(
            "target_max",
            "invalid_request",
            "need 0 < target_min <= target_max <= 3600",
        ));
    }
    let exclude_overlaps = req.exclude_overlaps.unwrap_or(true);
    let home = st.registry.home().to_path_buf();
    let out = tokio::task::spawn_blocking(move || -> Result<takes::Takes, ApiError> {
        let meta = load_clip_meta(&home, &id)?;
        let path = prep::clip_dir(&home, &id).join(prep::TRANSCRIPT_JSON);
        if !path.is_file() {
            return Err(transcript_not_found(&id));
        }
        let transcript: Transcript = prep::load_json(&path).map_err(prep_error_to_api)?;
        if let Some(sp) = req.speaker
            && !transcript.speakers.iter().any(|s| s.speaker == sp)
        {
            return Err(bad_request(
                "speaker",
                "invalid_request",
                format!("no diarized speaker {sp} in this clip's transcript"),
            ));
        }
        let kept = match &req.segments {
            Some(raw) => Some(
                edit::normalize_segments(raw, meta.duration_secs)
                    .map_err(|m| bad_request("segments", "invalid_request", m))?,
            ),
            None => None,
        };
        let pcm = prep::read_wav(&prep::clip_dir(&home, &id).join(prep::WORKING_WAV))
            .map_err(prep_error_to_api)?;
        Ok(takes::plan_takes(
            &pcm,
            &transcript.words,
            &transcript.speakers,
            &transcript.overlaps,
            &TakeOptions {
                speaker: req.speaker,
                kept: kept.as_deref(),
                target_min,
                target_max,
                exclude_overlaps,
            },
        ))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok(Json(json!(out)))
}

// ---- projects ---------------------------------------------------------------

/// Largest `PUT .../project` body: 1 MiB (a project holds a transcript and
/// the last takes result, both text).
const MAX_PROJECT_BYTES: usize = 1024 * 1024;
const MAX_PROJECT_NAME_CHARS: usize = 200;

fn no_project(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "no_project",
        format!("clip {id:?} has no saved project"),
    )
}

fn project_json(clip_id: &str, p: &Project) -> Value {
    let mut v = json!(p);
    v["clip_id"] = json!(clip_id);
    v
}

/// `GET /v1/audio/prep/clips/{id}/project`: the clip's saved project, or 404
/// `no_project`.
pub(super) async fn get_project(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let value = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        load_clip_meta(&home, &id)?;
        let path = prep::clip_dir(&home, &id).join(prep::PROJECT_JSON);
        if !path.is_file() {
            return Err(no_project(&id));
        }
        let project: Project = prep::load_json(&path).map_err(prep_error_to_api)?;
        Ok(project_json(&id, &project))
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))??;
    Ok(Json(value))
}

/// The client's side of a [`Project`]; the timestamps are the server's.
#[derive(Debug, Deserialize)]
struct ProjectRequest {
    name: String,
    #[serde(default)]
    speaker: Option<i32>,
    #[serde(default)]
    segments: Vec<Segment>,
    #[serde(default)]
    cuts: Vec<Segment>,
    #[serde(default)]
    exclude_overlaps: bool,
    #[serde(default)]
    steps: Vec<Step>,
    #[serde(default)]
    transcript: Option<String>,
    #[serde(default)]
    takes: Option<Value>,
}

/// `PUT /v1/audio/prep/clips/{id}/project`: creates or replaces the clip's
/// project and returns it. `steps` go through [`pipeline::validate`],
/// `segments` and `cuts` are validated, sorted and merged against the clip's
/// length; `created_at` is kept across a replace. Body at most 1 MiB (413).
pub(super) async fn put_project(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    if body.len() > MAX_PROJECT_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            format!("a project is at most {MAX_PROJECT_BYTES} bytes"),
        ));
    }
    let req: ProjectRequest = serde_json::from_slice(&body)
        .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?;
    let name = req.name.trim().to_string();
    if name.is_empty() || name.chars().count() > MAX_PROJECT_NAME_CHARS {
        return Err(bad_request(
            "name",
            "invalid_request",
            format!("\"name\" is required, at most {MAX_PROJECT_NAME_CHARS} characters"),
        ));
    }
    pipeline::validate(&req.steps).map_err(|m| bad_request("steps", "invalid_request", m))?;
    let home = st.registry.home().to_path_buf();
    let value = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let meta = load_clip_meta(&home, &id)?;
        let segments = edit::normalize_segments(&req.segments, meta.duration_secs)
            .map_err(|m| bad_request("segments", "invalid_request", m))?;
        let cuts = edit::normalize_segments(&req.cuts, meta.duration_secs)
            .map_err(|m| bad_request("cuts", "invalid_request", m))?;
        let path = prep::clip_dir(&home, &id).join(prep::PROJECT_JSON);
        let now = prep::now_rfc3339();
        let created_at = prep::load_json::<Project>(&path)
            .map(|p| p.created_at)
            .unwrap_or_else(|_| now.clone());
        let project = Project {
            name,
            speaker: req.speaker,
            segments,
            cuts,
            exclude_overlaps: req.exclude_overlaps,
            steps: req.steps,
            transcript: req.transcript,
            takes: req.takes,
            updated_at: now,
            created_at,
        };
        prep::save_json(&path, &project).map_err(prep_error_to_api)?;
        Ok(project_json(&id, &project))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok(Json(value))
}

/// `DELETE /v1/audio/prep/clips/{id}/project`: 204, or 404 `no_project`.
pub(super) async fn delete_project(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let home = st.registry.home().to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        load_clip_meta(&home, &id)?;
        let path = prep::clip_dir(&home, &id).join(prep::PROJECT_JSON);
        if !path.is_file() {
            return Err(no_project(&id));
        }
        std::fs::remove_file(&path).map_err(prep_error_to_api_io)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))??;
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /v1/audio/prep/projects`: every saved project, newest `updated_at`
/// first, as `{"projects": [{clip_id, name, duration, updated_at, speaker,
/// segment_count, has_transcript}]}`.
pub(super) async fn list_projects(State(st): State<Arc<AppState>>) -> Json<Value> {
    let home = st.registry.home().to_path_buf();
    let mut rows = tokio::task::spawn_blocking(move || {
        prep::list_projects(&home)
            .into_iter()
            .filter_map(|id| {
                let dir = prep::clip_dir(&home, &id);
                let project: Project = prep::load_json(&dir.join(prep::PROJECT_JSON)).ok()?;
                let meta: ClipMeta = prep::load_json(&dir.join(prep::CLIP_META)).ok()?;
                Some((
                    project.updated_at.clone(),
                    json!({
                        "clip_id": id,
                        "name": project.name,
                        "duration": meta.duration_secs,
                        "updated_at": project.updated_at,
                        "speaker": project.speaker,
                        "segment_count": project.segments.len(),
                        "has_transcript": dir.join(prep::TRANSCRIPT_JSON).is_file(),
                    }),
                ))
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    Json(json!({"projects": rows.into_iter().map(|r| r.1).collect::<Vec<_>>()}))
}

// ---- samples ------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ProcessRequest {
    clip_id: String,
    name: String,
    start: Option<f64>,
    end: Option<f64>,
    speaker: Option<i32>,
    /// Kept regions of the clip, spliced in order (see [`Edit`]); an
    /// alternative to `start`/`end`/`speaker`, which are then ignored.
    segments: Option<Vec<Segment>>,
    #[serde(default = "default_true")]
    denoise: bool,
    #[serde(default = "default_true")]
    isolate: bool,
    #[serde(default = "default_true")]
    trim_silence: bool,
    #[serde(default = "default_true")]
    normalize: bool,
    denoise_model: Option<String>,
    isolation_model: Option<String>,
    /// When present it defines the chain and the four flags above (and the
    /// two model names) are ignored.
    steps: Option<Vec<Step>>,
    /// The sample's transcript as the caller corrected it; replaces the
    /// clip's words in the range (`transcript.txt`) when present.
    transcript: Option<String>,
}

fn default_true() -> bool {
    true
}

/// The engines the enabled `steps` need, resolved up front so a missing
/// model is a 409 before any work starts.
struct Plan {
    isolators: HashMap<String, Manifest>,
    denoisers: HashMap<String, Manifest>,
    vad_path: PathBuf,
    vad_available: bool,
}

/// Resolves the models of `steps` (the ones that will run): each isolate or
/// denoise step's model manifest (409 `model_not_pulled` if it is not
/// pulled), and `silero-vad` for a trim_silence step. Resolving a manifest
/// reads `pulled` and its `manifest.json` (`kind_manifest`) and the VAD check
/// `stat`s a file, none of which belongs on the async runtime thread (naru
/// task 1461 review), so it all runs in one `spawn_blocking`.
async fn plan_models(st: &Arc<AppState>, req_id: &str, steps: Vec<Step>) -> Result<Plan, ApiError> {
    let home = st.registry.home().to_path_buf();
    let st2 = st.clone();
    let plan = tokio::task::spawn_blocking(move || -> Result<Plan, ApiError> {
        let mut plan = Plan {
            isolators: HashMap::new(),
            denoisers: HashMap::new(),
            vad_path: home
                .join("models")
                .join("silero-vad")
                .join(prep::VAD_FILENAME),
            vad_available: false,
        };
        plan.vad_available = plan.vad_path.is_file();
        for step in &steps {
            match step {
                Step::Isolate(s) => {
                    let name = s.model.as_deref().unwrap_or(prep::DEFAULT_SEPARATION_MODEL);
                    if !plan.isolators.contains_key(name) {
                        let m = kind_manifest(&st2, name, Kind::Separation)?;
                        plan.isolators.insert(name.to_string(), m);
                    }
                }
                Step::Denoise(s) => {
                    let name = s.model.as_deref().unwrap_or(prep::DEFAULT_DENOISE_MODEL);
                    if !plan.denoisers.contains_key(name) {
                        let m = kind_manifest(&st2, name, Kind::Denoise)?;
                        plan.denoisers.insert(name.to_string(), m);
                    }
                }
                Step::TrimSilence(_) if !plan.vad_available => {
                    return Err(registry_error(crate::registry::RegistryError::NotPulled(
                        "silero-vad".to_string(),
                    )));
                }
                _ => {}
            }
        }
        Ok(plan)
    })
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))??;
    Ok(plan)
}

/// Runs the model-backed steps against the engines a [`Plan`] resolved,
/// recording which engines ran and any licence warnings (`EngineUsed`).
struct Runner<'a> {
    home: &'a Path,
    plan: &'a Plan,
    engines: Vec<EngineUsed>,
    warnings: Vec<String>,
}

impl Runner<'_> {
    fn record(&mut self, stage: &str, model: &str, manifest: &Manifest) {
        self.engines.push(EngineUsed {
            stage: stage.to_string(),
            model: model.to_string(),
            license: manifest.model.license.clone(),
            non_commercial: manifest.model.non_commercial,
        });
        if manifest.model.non_commercial {
            self.warnings.push(format!(
                "{model} is licensed for non-commercial use only ({})",
                manifest.model.license.as_deref().unwrap_or("unknown")
            ));
        }
    }
}

impl ModelSteps for Runner<'_> {
    fn isolate(&mut self, model: Option<&str>, pcm: &[f32]) -> Result<Vec<f32>, PrepError> {
        let name = model.unwrap_or(prep::DEFAULT_SEPARATION_MODEL);
        let manifest = &self.plan.isolators[name];
        let isolator = backend::load_isolator(manifest, &self.home.join("models").join(name))
            .map_err(|e| PrepError::Engine(e.to_string()))?;
        let vocals = isolator
            .vocals(pcm, TARGET_SAMPLE_RATE as i32)
            .map_err(|e| PrepError::Engine(e.to_string()))?;
        let out = prep::resample_to_16k(&vocals, isolator.output_sample_rate() as u32)?;
        self.record("isolate", name, manifest);
        Ok(out)
    }

    fn denoise(&mut self, model: Option<&str>, pcm: &[f32]) -> Result<Vec<f32>, PrepError> {
        let name = model.unwrap_or(prep::DEFAULT_DENOISE_MODEL);
        let manifest = &self.plan.denoisers[name];
        let denoiser = backend::load_denoiser(manifest, &self.home.join("models").join(name))
            .map_err(|e| PrepError::Engine(e.to_string()))?;
        let out = denoiser.run(pcm, TARGET_SAMPLE_RATE as i32);
        self.record("denoise", name, manifest);
        Ok(out)
    }

    fn speech_spans(
        &mut self,
        pcm: &[f32],
        cfg: &VadConfig,
    ) -> Result<Vec<(usize, usize)>, PrepError> {
        let spans = prep::speech_spans(pcm, &self.plan.vad_path, cfg)?;
        self.engines.push(EngineUsed {
            stage: "trim_silence".to_string(),
            model: "silero-vad".to_string(),
            license: Some("MIT".to_string()),
            non_commercial: false,
        });
        Ok(spans)
    }
}

/// [`pipeline::analyze`] of `pcm`, with `speech_secs` when the VAD is pulled
/// (a VAD failure here leaves it `null` rather than failing the request).
fn analysis_of(pcm: &[f32], plan: &Plan) -> Analysis {
    let speech = plan
        .vad_available
        .then(|| prep::speech_secs(pcm, &plan.vad_path).ok())
        .flatten();
    pipeline::analyze(pcm, speech)
}

/// `POST /v1/audio/samples` (naru task 1461 §8, crop → steps): crops
/// `clip_id` to `start..end` or, given `speaker` instead, to that cached
/// diarized speaker's own span (`POST .../transcribe` must have run first);
/// then runs the chain. `steps` (see [`pipeline`]) defines the chain when
/// present; otherwise the four flags build the default one: isolate
/// (Spleeter, via [`crate::prep::isolate`]'s hand-written FFI) → denoise →
/// trim silence → normalise, each skipped when its flag is `false`. Every
/// enabled step whose engine is not pulled fails with a 409 naming it.
pub(super) async fn create_sample(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let req: ProcessRequest = serde_json::from_slice(&body)
        .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?;
    if req.name.trim().is_empty() {
        return Err(bad_request(
            "name",
            "invalid_request",
            "\"name\" is required",
        ));
    }
    let steps = match req.steps.clone() {
        Some(steps) => steps,
        None => pipeline::default_chain(
            req.isolate,
            req.denoise,
            req.trim_silence,
            req.normalize,
            req.isolation_model.clone(),
            req.denoise_model.clone(),
        ),
    };
    pipeline::validate(&steps).map_err(|m| bad_request("steps", "invalid_request", m))?;
    let home = st.registry.home().to_path_buf();
    let clip_id = req.clip_id.clone();
    let meta = tokio::task::spawn_blocking({
        let home = home.clone();
        let clip_id = clip_id.clone();
        move || load_clip_meta(&home, &clip_id)
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;

    let edit = resolve_edit(
        &st,
        &req_id,
        &home,
        &clip_id,
        meta.duration_secs,
        req.segments.as_deref(),
        req.start,
        req.end,
        req.speaker,
    )
    .await?;
    let (start, end) = edit.bounds();

    let enabled: Vec<Step> = steps.iter().filter(|s| s.enabled()).cloned().collect();
    let plan = plan_models(&st, &req_id, enabled).await?;

    let id = prep::new_id();
    let name = req.name.trim().to_string();

    let build = move || -> Result<Value, PrepError> {
        let working = prep::clip_dir(&home, &clip_id).join(prep::WORKING_WAV);
        let pcm = prep::read_wav(&working)?;
        let cropped = edit.crop(&pcm);

        let dir = prep::sample_dir(&home, &id);
        std::fs::create_dir_all(&dir).map_err(|e| PrepError::Io(e.to_string()))?;

        // Everything past this point can fail mid-pipeline (a missing
        // engine, a bad decode); on any of those the half-written sample
        // directory is removed rather than left behind (naru task 1461
        // review), the same way `create_clip` cleans up its own tmp dir.
        let result = (|| -> Result<SampleMeta, PrepError> {
            prep::write_wav(&dir.join(prep::CROPPED_WAV), &cropped)?;

            let mut runner = Runner {
                home: &home,
                plan: &plan,
                engines: Vec::new(),
                warnings: Vec::new(),
            };
            let clean = pipeline::run(&steps, Select::All, cropped, &mut runner)?;
            let analysis = analysis_of(&clean, &plan);

            prep::write_wav(&dir.join(prep::CLEAN_WAV), &clean)?;

            let transcript_txt = match req.transcript.as_deref().map(str::trim) {
                Some(text) if !text.is_empty() => text.to_string(),
                _ => words_in_segments(&home, &clip_id, &edit.segments()).unwrap_or_default(),
            };
            std::fs::write(dir.join(prep::TRANSCRIPT_TXT), transcript_txt)
                .map_err(|e| PrepError::Io(e.to_string()))?;

            let sample_meta = SampleMeta {
                id: id.clone(),
                name,
                source_clip_id: clip_id.clone(),
                original_filename: meta.original_filename.clone(),
                range: SampleRange { start, end },
                speaker: req.speaker,
                engines: runner.engines,
                warnings: runner.warnings,
                created_at: prep::now_rfc3339(),
                steps,
                analysis: Some(analysis),
                segments: match &edit {
                    Edit::Segments(segs) => segs.clone(),
                    Edit::Range(..) => Vec::new(),
                },
            };
            prep::save_json(&dir.join(prep::SAMPLE_META), &sample_meta)?;
            Ok(sample_meta)
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        result.map(|meta| sample_json(&home, &meta))
    };

    let value = tokio::task::spawn_blocking(build)
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?
        .map_err(|e| prep_error(&st, &req_id, e))?;
    Ok((StatusCode::CREATED, Json(value)).into_response())
}

#[derive(Debug, Deserialize)]
struct RenderRequest {
    start: Option<f64>,
    end: Option<f64>,
    speaker: Option<i32>,
    /// As [`ProcessRequest::segments`].
    segments: Option<Vec<Segment>>,
    steps: Vec<Step>,
    until: Option<usize>,
    solo: Option<usize>,
}

/// `POST /v1/audio/prep/clips/{id}/render`: auditions a chain without
/// creating a sample. Crops like [`create_sample`], then runs `steps` (all
/// of them, `until: N` = steps `0..=N`, or `solo: N` = only step `N`;
/// `until` and `solo` together are a 400, an index past the chain is a 400).
/// Disabled steps are skipped, except that `solo` applies its step even if
/// disabled. Returns the 16-bit 16 kHz mono WAV, with the result's
/// [`Analysis`] in `x-naru-loudness-lufs`, `x-naru-peak-dbfs`,
/// `x-naru-noise-floor-dbfs` and `x-naru-speech-secs` (each omitted when
/// null) plus the whole struct as JSON in `x-naru-analysis`. Same 404/409
/// semantics as [`create_sample`], for the steps that run.
pub(super) async fn render_clip(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let req: RenderRequest = serde_json::from_slice(&body)
        .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?;
    pipeline::validate(&req.steps).map_err(|m| bad_request("steps", "invalid_request", m))?;
    let select = match (req.until, req.solo) {
        (Some(_), Some(_)) => {
            return Err(bad_request(
                "solo",
                "invalid_request",
                "\"until\" and \"solo\" are mutually exclusive",
            ));
        }
        (Some(n), None) => Select::Until(n),
        (None, Some(n)) => Select::Solo(n),
        (None, None) => Select::All,
    };
    let (param, index) = match select {
        Select::Until(n) => ("until", Some(n)),
        Select::Solo(n) => ("solo", Some(n)),
        Select::All => ("steps", None),
    };
    if index.is_some_and(|n| n >= req.steps.len()) {
        return Err(bad_request(
            param,
            "invalid_request",
            format!("\"{param}\" must be a step index below {}", req.steps.len()),
        ));
    }

    let home = st.registry.home().to_path_buf();
    let meta = {
        let (home, id) = (home.clone(), id.clone());
        tokio::task::spawn_blocking(move || load_clip_meta(&home, &id))
            .await
            .map_err(|e| internal(&st, &req_id, e.to_string()))??
    };
    let edit = resolve_edit(
        &st,
        &req_id,
        &home,
        &id,
        meta.duration_secs,
        req.segments.as_deref(),
        req.start,
        req.end,
        req.speaker,
    )
    .await?;

    let running: Vec<Step> = select
        .indices(&req.steps)
        .into_iter()
        .map(|i| req.steps[i].clone())
        .collect();
    let plan = plan_models(&st, &req_id, running).await?;

    let steps = req.steps;
    let (bytes, analysis) =
        tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, Analysis), PrepError> {
            let working = prep::clip_dir(&home, &id).join(prep::WORKING_WAV);
            let cropped = edit.crop(&prep::read_wav(&working)?);
            let mut runner = Runner {
                home: &home,
                plan: &plan,
                engines: Vec::new(),
                warnings: Vec::new(),
            };
            let out = pipeline::run(&steps, select, cropped, &mut runner)?;
            Ok((prep::wav_bytes(&out)?, analysis_of(&out, &plan)))
        })
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?
        .map_err(|e| prep_error(&st, &req_id, e))?;

    let mut headers = vec![
        ("content-type", "audio/wav".to_string()),
        (
            "x-naru-analysis",
            serde_json::to_string(&analysis).unwrap_or_default(),
        ),
    ];
    for (name, v) in [
        ("x-naru-loudness-lufs", analysis.integrated_lufs),
        ("x-naru-peak-dbfs", analysis.peak_dbfs),
        ("x-naru-noise-floor-dbfs", analysis.noise_floor_dbfs),
        ("x-naru-speech-secs", analysis.speech_secs),
    ] {
        if let Some(v) = v {
            headers.push((name, format!("{v:.2}")));
        }
    }
    Ok((AppendHeaders(headers), axum::body::Body::from(bytes)).into_response())
}

/// `GET /v1/audio/prep/steps`: the step catalogue (every type's params with
/// default/min/max/unit) and the default chain, for a UI to build its
/// controls from.
pub(super) async fn list_steps() -> Json<Value> {
    Json(pipeline::catalogue())
}

/// What part of a clip an edit keeps.
enum Edit {
    /// One `start..end` range, a plain crop.
    Range(f64, f64),
    /// Validated, sorted, merged segments, spliced in order with a short
    /// crossfade ([`edit::splice`]).
    Segments(Vec<Segment>),
}

impl Edit {
    /// First start to last end.
    fn bounds(&self) -> (f64, f64) {
        match self {
            Edit::Range(s, e) => (*s, *e),
            Edit::Segments(segs) => (segs[0].start, segs[segs.len() - 1].end),
        }
    }

    fn segments(&self) -> Vec<Segment> {
        match self {
            Edit::Range(start, end) => vec![Segment {
                start: *start,
                end: *end,
            }],
            Edit::Segments(segs) => segs.clone(),
        }
    }

    fn crop(&self, pcm: &[f32]) -> Vec<f32> {
        match self {
            Edit::Range(start, end) => prep::crop(pcm, *start, *end),
            Edit::Segments(segs) => edit::splice(pcm, segs),
        }
    }
}

/// `segments` (validated against the clip's `duration`) when given, else the
/// [`resolve_range`] of `start`/`end`/`speaker`; 400 if the result is empty.
#[allow(clippy::too_many_arguments)]
async fn resolve_edit(
    st: &Arc<AppState>,
    req_id: &str,
    home: &Path,
    clip_id: &str,
    duration: f64,
    segments: Option<&[Segment]>,
    start: Option<f64>,
    end: Option<f64>,
    speaker: Option<i32>,
) -> Result<Edit, ApiError> {
    if let Some(raw) = segments {
        if raw.is_empty() {
            return Err(bad_request(
                "segments",
                "invalid_request",
                "\"segments\" must not be empty",
            ));
        }
        let segs = edit::normalize_segments(raw, duration)
            .map_err(|m| bad_request("segments", "invalid_request", m))?;
        return Ok(Edit::Segments(segs));
    }
    let (start, end) = resolve_range(st, req_id, home, clip_id, start, end, speaker).await?;
    if end <= start {
        return Err(bad_request(
            "end",
            "invalid_request",
            format!("the range {start}-{end} is empty"),
        ));
    }
    Ok(Edit::Range(start, end))
}

/// `start`/`end` when given directly, else `speaker`'s span from the
/// clip's cached diarization (`POST .../transcribe` with `diarize:true`
/// must have run).
async fn resolve_range(
    st: &Arc<AppState>,
    req_id: &str,
    home: &Path,
    clip_id: &str,
    start: Option<f64>,
    end: Option<f64>,
    speaker: Option<i32>,
) -> Result<(f64, f64), ApiError> {
    if let (Some(start), Some(end)) = (start, end) {
        return Ok((start, end));
    }
    let Some(speaker) = speaker else {
        return Err(bad_request(
            "start",
            "invalid_request",
            "give either \"start\" and \"end\", or \"speaker\"",
        ));
    };
    let home = home.to_path_buf();
    let clip_id = clip_id.to_string();
    let transcript = tokio::task::spawn_blocking(move || -> Result<Transcript, ApiError> {
        let path = prep::clip_dir(&home, &clip_id).join(prep::TRANSCRIPT_JSON);
        if !path.is_file() {
            return Err(transcript_not_found(&clip_id));
        }
        prep::load_json(&path).map_err(prep_error_to_api)
    })
    .await
    .map_err(|e| internal(st, req_id, e.to_string()))??;
    let spans: Vec<&SpeakerSpan> = transcript
        .speakers
        .iter()
        .filter(|s| s.speaker == speaker)
        .collect();
    if spans.is_empty() {
        return Err(bad_request(
            "speaker",
            "invalid_request",
            format!("no diarized speaker {speaker} in this clip's transcript"),
        ));
    }
    let start = spans.iter().map(|s| s.start).fold(f64::INFINITY, f64::min);
    let end = spans
        .iter()
        .map(|s| s.end)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((start, end))
}

/// The cached transcript's words inside any of `segments`, joined with a
/// space; `None` if the clip has never been transcribed.
fn words_in_segments(home: &Path, clip_id: &str, segments: &[Segment]) -> Option<String> {
    let path = prep::clip_dir(home, clip_id).join(prep::TRANSCRIPT_JSON);
    let transcript: Transcript = prep::load_json(&path).ok()?;
    Some(
        transcript
            .words
            .iter()
            .filter(|w| {
                segments
                    .iter()
                    .any(|s| w.start >= s.start && w.end <= s.end)
            })
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// `GET /v1/audio/samples`.
pub(super) async fn list_samples(State(st): State<Arc<AppState>>) -> Json<Value> {
    let home = st.registry.home().to_path_buf();
    let ids = tokio::task::spawn_blocking(move || prep::list_samples(&home))
        .await
        .unwrap_or_default();
    Json(json!({"samples": ids}))
}

/// `GET /v1/audio/samples/{id}`.
pub(super) async fn get_sample(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let value = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let meta = load_sample_meta(&home, &id)?;
        Ok(sample_json(&home, &meta))
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))??;
    Ok(Json(value))
}

#[derive(Debug, Deserialize)]
struct RenameSampleRequest {
    name: String,
}

/// `PATCH /v1/audio/samples/{id}`: renames the sample (`meta.json`'s
/// `name`; the id and its directory do not change).
pub(super) async fn rename_sample(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: RenameSampleRequest = serde_json::from_slice(&body)
        .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?;
    if req.name.trim().is_empty() {
        return Err(bad_request(
            "name",
            "invalid_request",
            "\"name\" is required",
        ));
    }
    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let value = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let mut meta = load_sample_meta(&home, &id2)?;
        meta.name = req.name.trim().to_string();
        prep::save_json(
            &prep::sample_dir(&home, &id2).join(prep::SAMPLE_META),
            &meta,
        )
        .map_err(prep_error_to_api)?;
        Ok(sample_json(&home, &meta))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok(Json(value))
}

/// `DELETE /v1/audio/samples/{id}`.
pub(super) async fn delete_sample(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    // See `delete_clip`'s comment: a decoded `id` can still contain `..`
    // or `/`.
    prep::check_id(&id).map_err(|_| sample_not_found(&id))?;
    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let existed = tokio::task::spawn_blocking(move || {
        let dir = prep::sample_dir(&home, &id2);
        let existed = dir.join(prep::SAMPLE_META).is_file();
        if existed {
            let _ = std::fs::remove_dir_all(&dir);
        }
        existed
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    if !existed {
        return Err(sample_not_found(&id));
    }
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /v1/audio/samples/{id}/audio?variant=clean|cropped` (default
/// `clean`), for A/B against the pre-clean crop.
pub(super) async fn sample_audio(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let cropped = q.get("variant").map(String::as_str) == Some("cropped");
    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let _meta = load_sample_meta(&home, &id2)?;
        let file = if cropped {
            prep::CROPPED_WAV
        } else {
            prep::CLEAN_WAV
        };
        std::fs::read(prep::sample_dir(&home, &id2).join(file)).map_err(prep_error_to_api_io)
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))??;
    Ok((
        AppendHeaders([(CONTENT_TYPE, "audio/wav")]),
        axum::body::Body::from(bytes),
    )
        .into_response())
}

fn prep_error_to_api_io(e: std::io::Error) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
}
