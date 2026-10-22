//! Speaker routes (naru tasks 2028 and 2260, `docs/live-mode-spec.md` "Speaker
//! guard"): the voice library under `/v1/audio/speaker/people` and
//! `/v1/audio/speaker/guard`, and the legacy single-voice `enroll`, `verify`
//! and `enrollment` routes, now over the library's owner. Opt-in: nothing calls
//! them until the clients switch. The model is `wespeaker-sv-en` and is never
//! auto-pulled (409 `model_not_pulled`, §2.6); a verify with no enrollment
//! answers `is_him: true` without needing the model, failing open.

use std::sync::Arc;

use axum::Json;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{Extension, Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::transcriptions::{audio_error, bad_request, internal, kind_manifest, multipart_error};
use super::{AppState, RequestId};
use crate::error::ApiError;
use crate::registry::manifest::Kind;
use crate::speaker::{
    self, DEFAULT_MODEL, DEFAULT_THRESHOLD, Library, Person, SpeakerError, SpeakerVerifier,
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

const NO_CLIPS: &str = "enrolling needs at least one \"file\" clip of 1 s or more";

fn speaker_error(st: &AppState, req_id: &str, e: SpeakerError) -> ApiError {
    match e {
        SpeakerError::NotFound(m) => ApiError::new(StatusCode::NOT_FOUND, "not_found", m),
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

/// Runs `f` off the async threads and maps a panic to a 500.
async fn blocking<T: Send + 'static>(
    st: &Arc<AppState>,
    req_id: &str,
    f: impl FnOnce(&AppState, &str) -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    let (st2, req_id2) = (st.clone(), req_id.to_string());
    tokio::task::spawn_blocking(move || f(&st2, &req_id2))
        .await
        .map_err(|e| internal(st, req_id, e.to_string()))?
}

fn person_json(p: &Person) -> Value {
    json!({
        "id": p.id, "name": p.name, "owner": p.owner,
        "clips": p.clips.iter().map(|c| json!({
            "id": c.id, "created_at": c.created_at, "seconds": c.seconds,
            "has_audio": c.has_audio,
        })).collect::<Vec<_>>(),
    })
}

fn name_of(raw: &str) -> Result<&str, ApiError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(bad_request(
            "name",
            "invalid_request",
            "name must not be empty",
        ));
    }
    Ok(name)
}

/// `GET /v1/audio/speaker/people`: the library, without embeddings.
pub(super) async fn list_people(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
) -> Result<Json<Value>, ApiError> {
    blocking(&st, &req_id, |st, req_id| {
        let lib = Library::load(st.registry.home()).map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({
            "model": lib.model, "live_guard": lib.live_guard,
            "people": lib.people.iter().map(person_json).collect::<Vec<_>>(),
        }))
    })
    .await
    .map(Json)
}

#[derive(Deserialize)]
pub(super) struct PersonBody {
    name: Option<String>,
    owner: Option<bool>,
}

/// `POST /v1/audio/speaker/people` `{name, owner?}`.
pub(super) async fn create_person(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Json(body): Json<PersonBody>,
) -> Result<Json<Value>, ApiError> {
    let name = name_of(body.name.as_deref().unwrap_or_default())?.to_string();
    blocking(&st, &req_id, move |st, req_id| {
        Library::update(st.registry.home(), |l| {
            let id = l.add_person(&name, body.owner.unwrap_or(false));
            Ok(person_json(l.person(id)?))
        })
        .map_err(|e| speaker_error(st, req_id, e))
    })
    .await
    .map(Json)
}

/// `PATCH /v1/audio/speaker/people/{id}` `{name?, owner?}`.
pub(super) async fn patch_person(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(id): Path<u64>,
    Json(body): Json<PersonBody>,
) -> Result<Json<Value>, ApiError> {
    let name = body
        .name
        .as_deref()
        .map(name_of)
        .transpose()?
        .map(str::to_string);
    blocking(&st, &req_id, move |st, req_id| {
        Library::update(st.registry.home(), |l| {
            l.patch_person(id, name.as_deref(), body.owner)?;
            Ok(person_json(l.person(id)?))
        })
        .map_err(|e| speaker_error(st, req_id, e))
    })
    .await
    .map(Json)
}

/// `DELETE /v1/audio/speaker/people/{id}`, with its clips' audio.
pub(super) async fn delete_person(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(id): Path<u64>,
) -> Result<Json<Value>, ApiError> {
    blocking(&st, &req_id, move |st, req_id| {
        let home = st.registry.home();
        Library::update(home, |l| l.remove_person(home, id))
            .map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({"deleted": true}))
    })
    .await
    .map(Json)
}

/// `POST /v1/audio/speaker/people/{id}/clips`: one `file` WAV, embedded and
/// kept as the person's next clip.
pub(super) async fn add_clip(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path(id): Path<u64>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    let form = read_form(multipart).await?;
    let wav = form
        .files
        .first()
        .ok_or_else(|| bad_request("file", "invalid_request", "the \"file\" field is required"))?
        .clone();
    blocking(&st, &req_id, move |st, req_id| {
        let home = st.registry.home();
        // 404 before spending an embedding on a person who is not there
        Library::load(home)
            .and_then(|l| l.person(id).map(|_| ()))
            .map_err(|e| speaker_error(st, req_id, e))?;
        let pcm = decode(&wav)?;
        if pcm.len() < speaker::MIN_SAMPLES {
            return Err(bad_request("file", "invalid_request", NO_CLIPS));
        }
        let embedding = load_verifier(st)?
            .embed(&pcm)
            .map_err(|e| speaker_error(st, req_id, e))?;
        let clip = Library::update(home, |l| l.add_clip(home, id, &pcm, embedding))
            .map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({
            "id": clip.id, "created_at": clip.created_at, "seconds": clip.seconds,
            "has_audio": clip.has_audio,
        }))
    })
    .await
    .map(Json)
}

/// `GET /v1/audio/speaker/people/{id}/clips/{clip}/audio`: 404 without audio.
pub(super) async fn clip_audio(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path((id, clip)): Path<(u64, u64)>,
) -> Result<Response, ApiError> {
    let bytes = blocking(&st, &req_id, move |st, req_id| {
        let home = st.registry.home();
        let lib = Library::load(home).map_err(|e| speaker_error(st, req_id, e))?;
        let has_audio = lib
            .person(id)
            .map_err(|e| speaker_error(st, req_id, e))?
            .clips
            .iter()
            .any(|c| c.id == clip && c.has_audio);
        let missing = || ApiError::new(StatusCode::NOT_FOUND, "not_found", "the clip has no audio");
        if !has_audio {
            return Err(missing());
        }
        std::fs::read(Library::clip_path(home, clip)).map_err(|_| missing())
    })
    .await?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes).into_response())
}

/// `DELETE /v1/audio/speaker/people/{id}/clips/{clip}`.
pub(super) async fn delete_clip(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Path((id, clip)): Path<(u64, u64)>,
) -> Result<Json<Value>, ApiError> {
    blocking(&st, &req_id, move |st, req_id| {
        let home = st.registry.home();
        Library::update(home, |l| l.remove_clip(home, id, clip))
            .map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({"deleted": true}))
    })
    .await
    .map(Json)
}

#[derive(Deserialize)]
pub(super) struct GuardBody {
    live: bool,
}

/// `GET /v1/audio/speaker/guard`: whether the live guard drops other voices.
pub(super) async fn get_guard(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
) -> Result<Json<Value>, ApiError> {
    blocking(&st, &req_id, |st, req_id| {
        let lib = Library::load(st.registry.home()).map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({"live": lib.live_guard}))
    })
    .await
    .map(Json)
}

/// `PUT /v1/audio/speaker/guard` `{live}`.
pub(super) async fn put_guard(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    Json(body): Json<GuardBody>,
) -> Result<Json<Value>, ApiError> {
    blocking(&st, &req_id, move |st, req_id| {
        Library::update(st.registry.home(), |l| {
            l.live_guard = body.live;
            Ok(())
        })
        .map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({"live": body.live}))
    })
    .await
    .map(Json)
}

/// `POST /v1/audio/speaker/enroll` (legacy): one or more `file` WAV clips that
/// replace the owner's clips, creating the owner "Me" if there is none.
pub(super) async fn enroll(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, ApiError> {
    let form = read_form(multipart).await?;
    blocking(&st, &req_id, move |st, req_id| {
        let clips = form
            .files
            .iter()
            .map(|b| decode(b))
            .collect::<Result<Vec<_>, _>>()?;
        let clips: Vec<_> = clips
            .into_iter()
            .filter(|c| c.len() >= speaker::MIN_SAMPLES)
            .collect();
        if clips.is_empty() {
            return Err(bad_request("file", "invalid_request", NO_CLIPS));
        }
        let verifier = load_verifier(st)?;
        let embedded = clips
            .into_iter()
            .map(|pcm| verifier.embed(&pcm).map(|e| (pcm, e)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| speaker_error(st, req_id, e))?;
        let home = st.registry.home();
        let dim = embedded[0].1.len();
        let count = embedded.len();
        Library::update(home, |l| {
            let owner = l.ensure_owner();
            l.clear_clips(home, owner)?;
            for (pcm, embedding) in embedded {
                l.add_clip(home, owner, &pcm, embedding)?;
            }
            Ok(())
        })
        .map_err(|e| speaker_error(st, req_id, e))?;
        Ok(json!({"model": DEFAULT_MODEL, "count": count, "dim": dim}))
    })
    .await
    .map(Json)
}

/// `POST /v1/audio/speaker/verify`: one `file` clip scored against the owner,
/// optional `threshold`.
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
    blocking(&st, &req_id, move |st, req_id| {
        let pcm = decode(&wav)?;
        let library =
            Library::load(st.registry.home()).map_err(|e| speaker_error(st, req_id, e))?;
        // No enrolled owner, or a clip too short to score: answered without
        // the model, so neither needs it pulled.
        let needs_model = library.owner_ready(DEFAULT_MODEL) && pcm.len() >= speaker::MIN_SAMPLES;
        let verifier = if needs_model {
            Some(load_verifier(st)?)
        } else {
            None
        };
        let v = speaker::identify(verifier.as_ref(), Some(&library), true, &pcm, threshold).verdict;
        Ok(json!({
            "score": v.score,
            "is_him": v.is_him,
            "reason": v.reason,
            "threshold": threshold,
        }))
    })
    .await
    .map(Json)
}

/// `DELETE /v1/audio/speaker/enrollment` (legacy): removes the owner's clips.
pub(super) async fn delete_enrollment(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
) -> Result<Json<Value>, ApiError> {
    let deleted = blocking(&st, &req_id, |st, req_id| {
        let home = st.registry.home();
        Library::update(home, |l| match l.owner().map(|o| o.id) {
            Some(owner) => l.clear_clips(home, owner),
            None => Ok(false),
        })
        .map_err(|e| speaker_error(st, req_id, e))
    })
    .await?;
    Ok(Json(json!({"deleted": deleted})))
}
