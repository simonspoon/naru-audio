//! Voice prep (naru task 1461 §6): cleans a raw upload into a reusable
//! clone sample. §8 in `docs/design.md` is the design; this module is the
//! pipeline and on-disk storage, `server::prep` the HTTP surface.
//!
//! **Storage.** `$NARU_AUDIO_HOME/prep/clips/<id>/` holds one upload: the
//! verbatim `raw.<ext>`, a decoded `working.wav` (16 kHz mono, the pipeline's
//! working rate throughout), `meta.json`, and — once transcribed —
//! `transcript.json` (word timestamps and diarized speakers, cached so a
//! second `GET` is free). `$NARU_AUDIO_HOME/prep/samples/<id>/` holds one
//! clone sample, model-agnostic (never tied to a TTS model): `clean.wav`
//! (denoised, trimmed, normalised), `cropped.wav` (the same span before
//! cleaning, for A/B), `transcript.txt` (the cropped span's words, joined),
//! and `meta.json` (source clip, range, speaker, engines used and their
//! licences, `created_at`).
//!
//! Both models load fresh per request, like [`crate::stt::vad::Vad`]: they
//! are tens of megabytes, cheap next to decoding a clip, and voice-prep is
//! not a hot path that needs `ModelManager`'s residency.

pub mod denoise;
pub mod diarize;
pub mod dsp;
pub mod edit;
pub mod isolate;
pub mod pipeline;
pub mod takes;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::stt::audio::TARGET_SAMPLE_RATE;
use crate::stt::vad::{Vad, VadConfig};

pub const MAX_ID_BYTES: usize = 64;

/// A fresh id, unique within and across processes, safe as a directory name.
pub fn new_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "{nanos:x}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// An id is one plain path component, as [`crate::voices::check_name`]
/// checks a voice name.
pub fn check_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return Err(format!("id must be 1-{MAX_ID_BYTES} bytes"));
    }
    if id.starts_with('.') || id.contains(['/', '\\', '\0']) || id.chars().any(char::is_control) {
        return Err(format!("id {id:?} must be a plain name"));
    }
    Ok(())
}

pub fn clips_dir(home: &Path) -> PathBuf {
    home.join("prep").join("clips")
}

pub fn samples_dir(home: &Path) -> PathBuf {
    home.join("prep").join("samples")
}

pub fn clip_dir(home: &Path, id: &str) -> PathBuf {
    clips_dir(home).join(id)
}

pub fn sample_dir(home: &Path, id: &str) -> PathBuf {
    samples_dir(home).join(id)
}

pub const WORKING_WAV: &str = "working.wav";
pub const CLEAN_WAV: &str = "clean.wav";
pub const CROPPED_WAV: &str = "cropped.wav";
pub const TRANSCRIPT_JSON: &str = "transcript.json";
pub const TRANSCRIPT_TXT: &str = "transcript.txt";
pub const CLIP_META: &str = "meta.json";
pub const SAMPLE_META: &str = "meta.json";

/// The catalog's default diarization model (naru task 1461 §8): pyannote
/// segmentation (MIT) + 3D-Speaker's English embedding (Apache-2.0), both
/// clean licences, so this is safe as a default.
pub const DEFAULT_DIARIZE_MODEL: &str = "speaker-diarization-en";

/// The catalog's default denoise model: GTCRN (MIT), also clean.
pub const DEFAULT_DENOISE_MODEL: &str = "speech-denoiser-gtcrn";

/// The catalog's default source-separation model: Spleeter 2-stems (MIT),
/// also clean.
pub const DEFAULT_SEPARATION_MODEL: &str = "source-separation-spleeter-2stems-int8";

/// `silero-vad`'s file within its own pulled model directory
/// ([`crate::stt::sherpa::VAD_FILENAME`], duplicated here rather than
/// making that `pub` for one constant voice-prep also needs).
pub const VAD_FILENAME: &str = "silero_vad.onnx";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipMeta {
    pub id: String,
    /// The verbatim upload's file name within the clip's directory, e.g.
    /// `"raw.mp3"` — never `../` or an absolute path (naru task 1461 §8
    /// derives it from the upload's `filename`, sanitised).
    pub raw_filename: String,
    pub original_filename: Option<String>,
    pub content_type: Option<String>,
    pub duration_secs: f64,
    pub created_at: String,
}

/// One diarized speaker span, as stored in a clip's cached
/// [`TRANSCRIPT_JSON`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSpan {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

/// One transcribed word, as stored in [`TRANSCRIPT_JSON`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The engine's confidence in the word (MLX Whisper's), `None` for an
    /// engine that reports none or a transcript cached before this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub words: Vec<TranscriptWord>,
    pub speakers: Vec<SpeakerSpan>,
    /// The STT model that produced `words` (an "engine" field: Parakeet or
    /// the MLX Whisper, a value here, not a schema change).
    pub stt_model: String,
    /// The language `words` were transcribed in: the request's, else the
    /// one a multilingual model (Whisper) detected; for a model with no
    /// notion of one (Parakeet, English-only), its manifest `[model]
    /// languages` first entry, if it declares any. `#[serde(default)]` so a
    /// `transcript.json` cached before this field existed still loads.
    #[serde(default)]
    pub language: Option<String>,
    /// The diarization model that produced `speakers`, `None` if diarization
    /// was skipped.
    pub diarization_model: Option<String>,
    /// Time ranges where spans of two or more different speakers overlap
    /// (crosstalk). `#[serde(default)]` so a `transcript.json` cached before
    /// this field existed still loads, with none.
    #[serde(default)]
    pub overlaps: Vec<edit::Segment>,
}

/// One engine a sample's pipeline actually ran, for `meta.json`'s
/// accounting (naru task 1461's licence policy: a client can see exactly
/// what produced a sample).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineUsed {
    pub stage: String,
    pub model: String,
    pub license: Option<String>,
    pub non_commercial: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRange {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMeta {
    pub id: String,
    pub name: String,
    pub source_clip_id: String,
    pub original_filename: Option<String>,
    pub range: SampleRange,
    pub speaker: Option<i32>,
    pub engines: Vec<EngineUsed>,
    pub warnings: Vec<String>,
    pub created_at: String,
    /// The chain that produced `clean.wav` (`pipeline::Step`s). Empty for a
    /// sample made before steps existed.
    #[serde(default)]
    pub steps: Vec<pipeline::Step>,
    /// Measurements of `clean.wav` at creation; `None` for a sample made
    /// before analysis existed.
    #[serde(default)]
    pub analysis: Option<pipeline::Analysis>,
    /// The kept segments when the sample was made from a multi-segment edit
    /// (spliced in order); empty for a plain `range`. `range` is then the
    /// first segment's start to the last one's end.
    #[serde(default)]
    pub segments: Vec<edit::Segment>,
}

/// A saved Sample Studio edit of one clip (`project.json` in the clip's
/// directory): everything needed to reopen the studio where it was left.
/// The raw upload and `working.wav` are never touched by an edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    #[serde(default)]
    pub speaker: Option<i32>,
    /// The kept regions of `working.wav`, sorted and merged; empty = whole
    /// clip.
    #[serde(default)]
    pub segments: Vec<edit::Segment>,
    /// Ranges the user cut out (the studio's own bookkeeping).
    #[serde(default)]
    pub cuts: Vec<edit::Segment>,
    #[serde(default)]
    pub exclude_overlaps: bool,
    #[serde(default)]
    pub steps: Vec<pipeline::Step>,
    /// The corrected transcript, if the user fixed it.
    #[serde(default)]
    pub transcript: Option<String>,
    /// The last best-takes result, stored as the client sent it.
    #[serde(default)]
    pub takes: Option<serde_json::Value>,
    pub updated_at: String,
    pub created_at: String,
}

pub const PROJECT_JSON: &str = "project.json";

pub fn now_rfc3339() -> String {
    humantime::format_rfc3339_seconds(SystemTime::now()).to_string()
}

#[derive(Debug)]
pub enum PrepError {
    Io(String),
    /// `afconvert` (or the decode after it) failed on the upload.
    BadAudio(String),
    Json(String),
    /// A model-backed step's engine failed to load or run.
    Engine(String),
}

impl std::fmt::Display for PrepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepError::Io(m) => write!(f, "{m}"),
            PrepError::BadAudio(m) => write!(f, "{m}"),
            PrepError::Json(m) => write!(f, "{m}"),
            PrepError::Engine(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for PrepError {}

fn io_err(what: &str) -> impl FnOnce(std::io::Error) -> PrepError + '_ {
    move |e| PrepError::Io(format!("{what}: {e}"))
}

/// Converts `src` (anything `afconvert` reads) to a 16 kHz mono 16-bit WAV
/// at `dst`. Returns the decoded length in seconds.
pub fn ingest_to_wav(src: &Path, dst: &Path) -> Result<f64, PrepError> {
    let out = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(src)
        .arg(dst)
        .output()
        .map_err(io_err("run afconvert"))?;
    if !out.status.success() {
        return Err(PrepError::BadAudio(format!(
            "afconvert failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    // The length comes from the WAV header; decoding a 4-hour clip just to
    // count it would hold ~1 GB of samples.
    let file = std::fs::File::open(dst).map_err(io_err("open converted wav"))?;
    let wav = hound::WavReader::new(std::io::BufReader::new(file))
        .map_err(|e| PrepError::BadAudio(format!("failed to decode wav: {e}")))?;
    let secs = wav.duration() as f64 / TARGET_SAMPLE_RATE as f64;
    let cap = crate::stt::audio::Limits::PREP.max_seconds;
    if secs > cap as f64 {
        return Err(PrepError::BadAudio(format!(
            "audio exceeds the {}-hour clip cap",
            cap / 3600
        )));
    }
    Ok(secs)
}

/// Reads a 16 kHz mono WAV back into samples, e.g. a clip's `working.wav`
/// or a sample's `cropped.wav`.
pub fn read_wav(path: &Path) -> Result<Vec<f32>, PrepError> {
    let file = std::fs::File::open(path).map_err(io_err("open wav"))?;
    decode_wav(file)
}

/// Decodes under [`Limits::PREP`] (4 hours), not the STT endpoints' 10
/// minutes: a raw clip is the whole recording.
fn decode_wav(reader: impl std::io::Read) -> Result<Vec<f32>, PrepError> {
    use crate::stt::audio::{AudioError, Limits, decode_with};
    decode_with(reader, Limits::PREP).map_err(|e| match e {
        AudioError::TooLong => PrepError::BadAudio(format!(
            "audio exceeds the {}-hour clip cap",
            Limits::PREP.max_seconds / 3600
        )),
        e => PrepError::BadAudio(e.to_string()),
    })
}

/// Writes `samples` (16 kHz mono) as a 16-bit PCM WAV.
pub fn write_wav(path: &Path, samples: &[f32]) -> Result<(), PrepError> {
    std::fs::write(path, wav_bytes(samples)?).map_err(io_err("write wav"))
}

/// `samples` (16 kHz mono) as the bytes of a 16-bit PCM WAV.
pub fn wav_bytes(samples: &[f32]) -> Result<Vec<u8>, PrepError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer =
        hound::WavWriter::new(&mut cursor, spec).map_err(|e| PrepError::Io(e.to_string()))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * i16::MAX as f32).round() as i16;
        writer
            .write_sample(i)
            .map_err(|e| PrepError::Io(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| PrepError::Io(e.to_string()))?;
    Ok(cursor.into_inner())
}

/// Resamples mono `input` at `input_rate` to [`crate::stt::audio::
/// TARGET_SAMPLE_RATE`] (16 kHz), the pipeline's working rate throughout —
/// [`isolate::Isolator::vocals`] hands back 44.1 kHz (Spleeter's own rate,
/// via `output_sample_rate()`), which this brings back in line before
/// denoise/trim/normalise run on it. A no-op copy when the rates already
/// match, as `stt::audio`'s own (private) resampler is.
pub fn resample_to_16k(input: &[f32], input_rate: u32) -> Result<Vec<f32>, PrepError> {
    if input_rate == TARGET_SAMPLE_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    const CHANNELS: usize = 1;
    const CHUNK_SIZE: usize = 1024;
    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        CHUNK_SIZE,
        CHANNELS,
        FixedSync::Both,
    )
    .map_err(|e| PrepError::BadAudio(e.to_string()))?;
    let input_adapter = InterleavedSlice::new(input, CHANNELS, input.len())
        .map_err(|e| PrepError::BadAudio(e.to_string()))?;
    let output = resampler
        .process_all(&input_adapter, input.len(), None)
        .map_err(|e| PrepError::BadAudio(e.to_string()))?;
    Ok(output.take_data())
}

/// `pcm`'s samples from `start_secs` to `end_secs`, clamped to the clip's
/// length; `end_secs <= start_secs` crops to nothing.
pub fn crop(pcm: &[f32], start_secs: f64, end_secs: f64) -> Vec<f32> {
    let rate = TARGET_SAMPLE_RATE as f64;
    let len = pcm.len();
    let from = ((start_secs.max(0.0)) * rate).round() as usize;
    let to = ((end_secs.max(0.0)) * rate).round() as usize;
    let from = from.min(len);
    let to = to.clamp(from, len);
    pcm[from..to].to_vec()
}

/// The speech spans the Silero gate `vad_model` finds in `pcm`, as
/// `(start, end)` sample indices.
pub fn speech_spans(
    pcm: &[f32],
    vad_model: &Path,
    cfg: &VadConfig,
) -> Result<Vec<(usize, usize)>, PrepError> {
    let vad = Vad::load(vad_model, cfg).map_err(|e| PrepError::BadAudio(e.to_string()))?;
    let mut segmenter = vad.segmenter();
    let mut spans = Vec::new();
    for chunk in pcm.chunks(4096) {
        segmenter.accept(chunk);
        while let Some(span) = segmenter.next_span() {
            spans.push((span.start, span.start + span.len));
        }
    }
    segmenter.finish();
    while let Some(span) = segmenter.next_span() {
        spans.push((span.start, span.start + span.len));
    }
    Ok(spans)
}

/// Total seconds of speech [`speech_spans`] finds in `pcm` (default gate).
pub fn speech_secs(pcm: &[f32], vad_model: &Path) -> Result<f64, PrepError> {
    let spans = speech_spans(pcm, vad_model, &VadConfig::default())?;
    let samples: usize = spans
        .iter()
        .map(|(from, to)| to.saturating_sub(*from))
        .sum();
    Ok(samples as f64 / TARGET_SAMPLE_RATE as f64)
}

/// `pcm` from the first span's start to the last span's end, widened by
/// `pad` samples each side and clamped to the buffer; `pcm` unchanged if
/// there are no spans.
pub fn trim_to_spans(pcm: &[f32], spans: &[(usize, usize)], pad: usize) -> Vec<f32> {
    match (spans.first(), spans.last()) {
        (Some(&(from, _)), Some(&(_, to))) => {
            let from = from.saturating_sub(pad).min(pcm.len());
            let to = to.saturating_add(pad).clamp(from, pcm.len());
            pcm[from..to].to_vec()
        }
        _ => pcm.to_vec(),
    }
}

/// Trims leading and trailing silence using the Silero gate `vad_model`
/// already loaded for (naru task 1461 §8 "clean"): the span from the
/// first detected span's start to the last one's end, or `pcm` unchanged
/// if the gate finds no speech in it at all.
pub fn trim_silence(pcm: &[f32], vad_model: &Path, cfg: &VadConfig) -> Result<Vec<f32>, PrepError> {
    Ok(trim_to_spans(pcm, &speech_spans(pcm, vad_model, cfg)?, 0))
}

/// The clip's peak level in dBFS (`-inf` reported as `f32::NEG_INFINITY`
/// for digital silence); the smoke test and `POST .../process`'s response
/// both report this before and after normalising.
pub fn peak_dbfs(pcm: &[f32]) -> f32 {
    let peak = pcm.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// The clip's RMS level in dBFS, as [`peak_dbfs`].
pub fn rms_dbfs(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64;
    let rms = mean_sq.sqrt() as f32;
    if rms <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    }
}

/// Scales `pcm` in place so its peak lands at `target_dbfs` (e.g. `-1.0`),
/// a no-op on digital silence.
pub fn normalize_peak(pcm: &mut [f32], target_dbfs: f32) {
    let peak = pcm.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= 0.0 {
        return;
    }
    let target = 10f32.powf(target_dbfs / 20.0);
    let gain = target / peak;
    for s in pcm.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

pub fn save_json<T: Serialize>(path: &Path, value: &T) -> Result<(), PrepError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| PrepError::Json(e.to_string()))?;
    std::fs::write(path, bytes).map_err(io_err(&format!("write {}", path.display())))
}

pub fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PrepError> {
    let bytes = std::fs::read(path).map_err(io_err(&format!("read {}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| PrepError::Json(e.to_string()))
}

/// The clip ids in `home`, sorted; a directory missing its `meta.json` is
/// skipped rather than failing the listing.
pub fn list_clips(home: &Path) -> Vec<String> {
    list_ids(&clips_dir(home), CLIP_META)
}

/// The ids of clips in `home` that have a saved [`Project`], sorted.
pub fn list_projects(home: &Path) -> Vec<String> {
    let dir = clips_dir(home);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|id| check_id(id).is_ok() && dir.join(id).join(PROJECT_JSON).is_file())
        .collect();
    ids.sort();
    ids
}

/// The sample ids in `home`, sorted, as [`list_clips`].
pub fn list_samples(home: &Path) -> Vec<String> {
    list_ids(&samples_dir(home), SAMPLE_META)
}

fn list_ids(dir: &Path, meta_name: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|id| check_id(id).is_ok() && dir.join(id).join(meta_name).is_file())
        .collect();
    ids.sort();
    ids
}
