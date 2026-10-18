//! Speaker verification routes (naru task 2028, `docs/live-mode-spec.md`
//! "Speaker guard"): `POST /v1/audio/speaker/enroll`, `POST
//! /v1/audio/speaker/verify`, `DELETE /v1/audio/speaker/enrollment`. Opt-in:
//! nothing calls them until the clients switch. The model is `wespeaker-sv-en`
//! and is never auto-pulled (409 `model_not_pulled`, §2.6); a verify with no
//! enrollment answers `is_him: true` without needing the model, failing open.

use std::sync::Arc;

use axum::Json;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Extension, Multipart, State};
use serde_json::{Value, json};

use super::transcriptions::{audio_error, bad_request, internal, kind_manifest, multipart_error};
use super::{AppState, RequestId};
use crate::error::ApiError;
use crate::registry::manifest::Kind;
use crate::speaker::{
    self, DEFAULT_MODEL, DEFAULT_THRESHOLD, Enrollment, SpeakerError, SpeakerVerifier,
};
use crate::stt::audio::{self, AudioError};

struct Form {
    files: Vec<axum::body::Bytes>,
    threshold: Option<String>,
}

async fn read_form(multipart: Result<Multipart, MultipartRejection>) -> Result<Form, ApiError> {
    let mut multipart = multipart.map_err(|e| {
        bad_request(
            "file",
            "invalid_request",
            format!("the body must be multipart/form-data: {}", e.body_text()),
        )
    })?;
    let mut form = Form {
        files: Vec::new(),
        threshold: None,
    };
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        match field.name().unwrap_or_default() {
            "file" => form
                .files
                .push(field.bytes().await.map_err(multipart_error)?),
            "threshold" => form.threshold = Some(field.text().await.map_err(multipart_error)?),
            _ => {}
        }
    }
    Ok(form)
}

fn decode(wav: &[u8]) -> Result<Vec<f32>, ApiError> {
    match audio::decode(wav) {
        Ok(pcm) => Ok(pcm),
        Err(AudioError::Empty) => Ok(Vec::new()),
        Err(e) => Err(audio_error(e)),
    }
}

fn threshold(raw: Option<&str>) -> Result<f32, ApiError> {
    match raw {
        None => Ok(DEFAULT_THRESHOLD),
        Some(s) => s
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|t| (-1.0..=1.0).contains(t))
            .ok_or_else(|| {
                bad_request(
                    "threshold",
                    "invalid_request",
                    "threshold must be a cosine score between -1 and 1",
                )
            }),
    }
}

fn speaker_error(st: &AppState, req_id: &str, e: SpeakerError) -> ApiError {
    match e {
        SpeakerError::NoClips => bad_request(
            "file",
            "invalid_request",
            "enrolling needs at least one \"file\" clip of 1 s or more",
        ),
        e => internal(st, req_id, e.to_string()),
    }
}

/// The pulled model loaded: 404/409/400 from `kind_manifest`, then the files.
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

/// `POST /v1/audio/speaker/enroll`: one or more `file` WAV clips, averaged
/// into the stored enrollment (replacing any earlier one).
pub(super) async fn enroll(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    let form = read_form(multipart).await?;
    let (st2, req_id2) = (st.clone(), req_id.clone());
    tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let clips = form
            .files
            .iter()
            .map(|b| decode(b))
            .collect::<Result<Vec<_>, _>>()?;
        let verifier = load_verifier(&st2)?;
        let enrollment = verifier
            .enroll(&clips)
            .map_err(|e| speaker_error(&st2, &req_id2, e))?;
        enrollment
            .save(st2.registry.home())
            .map_err(|e| speaker_error(&st2, &req_id2, e))?;
        Ok(json!({
            "model": enrollment.model,
            "count": enrollment.count,
            "dim": enrollment.embedding.len(),
        }))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map(Json)
}

/// `POST /v1/audio/speaker/verify`: one `file` clip, optional `threshold`.
pub(super) async fn verify(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    let form = read_form(multipart).await?;
    let wav = form
        .files
        .first()
        .ok_or_else(|| bad_request("file", "invalid_request", "the \"file\" field is required"))?
        .clone();
    let threshold = threshold(form.threshold.as_deref())?;
    let (st2, req_id2) = (st.clone(), req_id.clone());
    tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        let pcm = decode(&wav)?;
        let enrollment =
            Enrollment::load(st2.registry.home()).map_err(|e| speaker_error(&st2, &req_id2, e))?;
        // No enrollment, or a clip too short to score: answered without the
        // model, so neither needs it pulled.
        let needs_model = enrollment.is_some() && pcm.len() >= speaker::MIN_SAMPLES;
        let verifier = if needs_model {
            Some(load_verifier(&st2)?)
        } else {
            None
        };
        let v = speaker::verify(verifier.as_ref(), enrollment.as_ref(), &pcm, threshold);
        Ok(json!({
            "score": v.score,
            "is_him": v.is_him,
            "reason": v.reason,
            "threshold": threshold,
        }))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map(Json)
}

/// `DELETE /v1/audio/speaker/enrollment`.
pub(super) async fn delete_enrollment(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let deleted = tokio::task::spawn_blocking(move || Enrollment::delete(&home))
        .await
        .map_err(|e| internal(&st, &req_id, e.to_string()))?
        .map_err(|e| speaker_error(&st, &req_id, e))?;
    Ok(Json(json!({"deleted": deleted})))
}
