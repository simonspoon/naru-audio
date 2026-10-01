//! `POST /v1/audio/samples/{id}/transcribe` and `GET .../transcript` (naru
//! task 1569, part 3): a verbatim reference transcript of a processed
//! sample's `clean.wav`, with word timestamps, fillers kept and numbers in
//! spoken form ([`crate::stt::spoken`]), so the text a clone is prompted
//! with matches the audio. `transcript.txt` is overwritten with the spoken
//! text, which is what the clone screen prefills from.

use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Extension, Path as AxumPath, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::transcriptions::{bad_request, internal, kind_manifest};
use super::{AppState, RequestId, manager_error};
use crate::error::ApiError;
use crate::prep;
use crate::registry::manifest::Kind;
use crate::stt::SttError;
use crate::stt::Word;
use crate::stt::spoken::{self, SpokenWord};
use crate::stt::vad::VadConfig;

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Request {
    stt_model: Option<String>,
    language: Option<String>,
    verbatim: bool,
}

impl Default for Request {
    fn default() -> Self {
        Request {
            stt_model: None,
            language: None,
            verbatim: true,
        }
    }
}

/// What is stored as the sample's [`prep::TRANSCRIPT_JSON`] and answered.
#[derive(Debug, Serialize, Deserialize)]
struct SampleTranscript {
    words: Vec<WordJson>,
    /// The words as the model wrote them.
    text: String,
    /// The words as spoken: what `transcript.txt` holds.
    spoken_text: String,
    language: Option<String>,
    stt_model: String,
    /// What was asked for.
    verbatim: bool,
    /// Whether the model could do it; `false` means a normal decode.
    verbatim_supported: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct WordJson {
    start: f64,
    end: f64,
    text: String,
    spoken: String,
    differs: bool,
    filler: bool,
}

impl From<SpokenWord> for WordJson {
    fn from(w: SpokenWord) -> Self {
        WordJson {
            start: w.start,
            end: w.end,
            text: w.text,
            spoken: w.spoken,
            differs: w.differs,
            filler: w.filler,
        }
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

fn check_sample(home: &Path, id: &str) -> Result<(), ApiError> {
    prep::check_id(id).map_err(|_| sample_not_found(id))?;
    if !prep::sample_dir(home, id).join(prep::SAMPLE_META).is_file() {
        return Err(sample_not_found(id));
    }
    Ok(())
}

fn transcript_not_run(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::CONFLICT,
        "transcript_not_run",
        format!("sample {id:?} has no transcript yet; POST .../transcribe first"),
    )
}

fn build(
    words: Vec<Word>,
    language: Option<String>,
    stt_model: String,
    verbatim: bool,
    verbatim_supported: bool,
) -> SampleTranscript {
    let words = spoken::annotate(&words);
    let join = |f: fn(&SpokenWord) -> &str| words.iter().map(f).collect::<Vec<_>>().join(" ");
    SampleTranscript {
        text: join(|w| &w.text),
        spoken_text: join(|w| &w.spoken),
        words: words.into_iter().map(WordJson::from).collect(),
        language,
        stt_model,
        verbatim,
        verbatim_supported,
    }
}

pub(super) async fn transcribe_sample(
    State(st): State<Arc<AppState>>,
    Extension(RequestId(req_id)): Extension<RequestId>,
    AxumPath(id): AxumPath<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: Request = if body.is_empty() {
        Request::default()
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| bad_request("body", "invalid_request", e.to_string()))?
    };

    let home = st.registry.home().to_path_buf();
    let id2 = id.clone();
    let pcm = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, ApiError> {
        check_sample(&home, &id2)?;
        prep::read_wav(&prep::sample_dir(&home, &id2).join(prep::CLEAN_WAV)).map_err(|e| {
            ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
        })
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
    let default_language = languages.first().cloned();
    let guard = st
        .models
        .acquire(stt_manifest, None)
        .await
        .map_err(|e| manager_error(&st, &req_id, e))?;

    let (requested, verbatim) = (req.language, req.verbatim);
    let (words, used_language, verbatim_supported) = tokio::task::spawn_blocking(move || {
        guard.stt().decode_words_verbatim(
            &pcm,
            Some(&VadConfig::default()),
            requested.as_deref(),
            verbatim,
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

    let transcript = build(
        words,
        used_language.or(default_language),
        stt_model,
        verbatim,
        verbatim_supported,
    );
    let home = st.registry.home().to_path_buf();
    let (id3, stored) = (id.clone(), serde_json::to_value(&transcript).unwrap());
    let spoken_text = transcript.spoken_text.clone();
    tokio::task::spawn_blocking(move || -> Result<(), prep::PrepError> {
        let dir = prep::sample_dir(&home, &id3);
        prep::save_json(&dir.join(prep::TRANSCRIPT_JSON), &stored)?;
        std::fs::write(dir.join(prep::TRANSCRIPT_TXT), spoken_text)
            .map_err(|e| prep::PrepError::Io(e.to_string()))
    })
    .await
    .map_err(|e| internal(&st, &req_id, e.to_string()))?
    .map_err(|e| internal(&st, &req_id, e.to_string()))?;

    Ok(Json(serde_json::to_value(&transcript).unwrap()))
}

/// `GET /v1/audio/samples/{id}/transcript`: the stored transcript from the
/// last `POST .../transcribe`; 409 when none has run (a `GET` never does
/// work).
pub(super) async fn get_sample_transcript(
    State(st): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let home = st.registry.home().to_path_buf();
    let transcript = tokio::task::spawn_blocking(move || -> Result<Value, ApiError> {
        check_sample(&home, &id)?;
        let path = prep::sample_dir(&home, &id).join(prep::TRANSCRIPT_JSON);
        if !path.is_file() {
            return Err(transcript_not_run(&id));
        }
        prep::load_json(&path).map_err(|e| {
            ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
        })
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))??;
    Ok(Json(transcript))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_joins_written_and_spoken_text() {
        let word = |t: &str| Word {
            start: 0.0,
            end: 0.1,
            text: t.to_string(),
        };
        let t = build(
            vec![
                word("Um,"),
                word("I"),
                word("have"),
                word("42"),
                word("cats."),
            ],
            Some("en".into()),
            "m".into(),
            true,
            true,
        );
        assert_eq!(t.text, "Um, I have 42 cats.");
        assert_eq!(t.spoken_text, "Um, I have forty-two cats.");
        assert!(t.words[0].filler);
        assert!(t.words[3].differs && !t.words[2].differs);
    }
}
