//! Speaker verification and the voice library (naru tasks 2028, 2260, 2354),
//! the `campplus-sv-en` catalog model: a 3D-Speaker CAM++ embedding (via
//! sherpa-onnx) turns a clip into a vector, and cosine similarity against a
//! named person's clips says who spoke. The owner is "him", the voice the live
//! guard admits (`docs/live-mode-spec.md`). Loaded fresh per request like
//! `prep::diarize::Diarizer`.
//!
//! **Fails open** everywhere (memory: he often answers with just "yes"): no
//! library, no pulled model, a library made with another model (until
//! [`migrate`] re-embeds it), or a clip shorter than [`MIN_SAMPLES`] all say
//! `is_him = true`, so a guard built on
//! this never drops a short reply or goes silent on a missing model.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

/// The catalog model name; also tagged into the library. Was
/// `wespeaker-sv-en` before naru task 2354; such a library is re-embedded.
pub const DEFAULT_MODEL: &str = "campplus-sv-en";
/// The embedding model's file in the pulled model directory.
pub const EMBEDDING_FILENAME: &str = "embedding.onnx";
/// Sample rate the embedding is computed at.
pub const SAMPLE_RATE: i32 = 16_000;
/// Clips shorter than this (1 s) are not scored: a "yes" or a number is too
/// short for a stable embedding and must never be dropped.
pub const MIN_SAMPLES: usize = 16_000;
/// Cosine score at or above which a clip is "him", for `campplus-sv-en`
/// (naru task 2354). Calibrated on real read speech (mini LibriSpeech
/// dev-clean-2, 26 speakers, each enrolled from two clips, scored on six
/// clips cut to 1.5-8 s against all 26 enrollments): at 0.35 no same-voice
/// clip of 2.5 s or more fell below the bar (2.6% at 1.5 s), while 3-7% of
/// other voices cleared it. Kept that low because wrongly dropping him is
/// worse than admitting someone else. WeSpeaker's 0.7 on the same set dropped
/// 3.5% of his 2.5 s clips and 9.6% at 1.5 s. Every drop logs its score
/// (`live_segment_dropped`), so the bar can be re-measured on his own voice.
pub const DEFAULT_THRESHOLD: f32 = 0.35;
/// CAM++ pools in segments of 200 feature frames (2 s), and its ONNX export,
/// as sherpa-onnx runs it, mis-scores a short last segment: a clip just past
/// a multiple of 2 s embeds so wrongly that two different voices score 0.9.
/// A clip is therefore tiled (repeated from its start) to a whole number of
/// segments before embedding.
const SEGMENT_SAMPLES: usize = 32_000;

#[derive(Debug)]
pub enum SpeakerError {
    MissingModel(PathBuf),
    /// `SpeakerEmbeddingExtractor::create` returned `None`.
    CreateFailed,
    /// The extractor produced no embedding for the clip.
    EmbedFailed,
    /// A person or clip id that does not exist.
    NotFound(String),
    /// The library's file could not be read or written.
    Store(String),
}

impl std::fmt::Display for SpeakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeakerError::MissingModel(path) => {
                write!(f, "speaker model not found: {}", path.display())
            }
            SpeakerError::CreateFailed => write!(f, "failed to create the speaker extractor"),
            SpeakerError::EmbedFailed => write!(f, "the speaker model produced no embedding"),
            SpeakerError::NotFound(m) => write!(f, "{m}"),
            SpeakerError::Store(m) => write!(f, "speaker store: {m}"),
        }
    }
}

impl std::error::Error for SpeakerError {}

pub struct SpeakerVerifier {
    inner: SpeakerEmbeddingExtractor,
}

impl SpeakerVerifier {
    /// `dir` is the pulled [`DEFAULT_MODEL`] directory.
    pub fn load(dir: &Path) -> Result<Self, SpeakerError> {
        let model = dir.join(EMBEDDING_FILENAME);
        if !model.is_file() {
            return Err(SpeakerError::MissingModel(model));
        }
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let inner = SpeakerEmbeddingExtractor::create(&config).ok_or(SpeakerError::CreateFailed)?;
        Ok(SpeakerVerifier { inner })
    }

    /// The L2-normalised embedding of 16 kHz mono `pcm`.
    pub fn embed(&self, pcm: &[f32]) -> Result<Vec<f32>, SpeakerError> {
        let stream = self
            .inner
            .create_stream()
            .ok_or(SpeakerError::EmbedFailed)?;
        let whole = pcm.len().div_ceil(SEGMENT_SAMPLES) * SEGMENT_SAMPLES;
        let tiled: Vec<f32> = pcm.iter().cycle().take(whole).copied().collect();
        stream.accept_waveform(SAMPLE_RATE, &tiled);
        stream.input_finished();
        if !self.inner.is_ready(&stream) {
            return Err(SpeakerError::EmbedFailed);
        }
        let mut v = self
            .inner
            .compute(&stream)
            .ok_or(SpeakerError::EmbedFailed)?;
        normalize(&mut v);
        Ok(v)
    }
}

/// One recorded sample of a person's voice. The audio, when kept, is
/// `speaker/clips/<id>.wav` (16 kHz mono); only the embedding is scored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: u64,
    /// Unix seconds.
    pub created_at: u64,
    /// Length of the audio; 0 when unknown (a migrated enrollment).
    pub seconds: f32,
    /// L2-normalised.
    pub embedding: Vec<f32>,
    pub has_audio: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub id: u64,
    pub name: String,
    /// The voice the live guard admits ("him"); at most one person.
    pub owner: bool,
    pub clips: Vec<Clip>,
}

/// Who a line was attributed to.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PersonRef {
    pub id: u64,
    pub name: String,
}

/// The library of named people and their voice clips, `speaker/people.json`.
/// Tagged with the model, since embeddings of different models are not
/// comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Library {
    pub model: String,
    /// Whether the live guard drops other voices; off until switched on.
    #[serde(default)]
    pub live_guard: bool,
    pub people: Vec<Person>,
    /// The next clip id to hand out; never reused, so a clip's audio URL never
    /// serves another clip.
    #[serde(default)]
    next_clip_id: u64,
    /// Clip WAVs to delete once the library is saved (see [`Library::update`]).
    #[serde(skip)]
    pending_delete: Vec<PathBuf>,
    /// WAVs written since the library was loaded, removed if the update fails.
    #[serde(skip)]
    written: Vec<PathBuf>,
}

impl Default for Library {
    fn default() -> Self {
        Library {
            model: DEFAULT_MODEL.to_string(),
            live_guard: false,
            people: Vec::new(),
            next_clip_id: 0,
            pending_delete: Vec::new(),
            written: Vec::new(),
        }
    }
}

/// A library file's modification time and size.
pub type Stamp = (std::time::SystemTime, u64);

/// Serialises read-modify-write of the library across requests and sessions.
static STORE: Mutex<()> = Mutex::new(());

/// The name of the owner created when nothing names one.
pub const OWNER_NAME: &str = "Me";

fn store_err(e: impl std::fmt::Display) -> SpeakerError {
    SpeakerError::Store(e.to_string())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The pre-library single enrollment, read only to migrate it.
#[derive(Deserialize)]
struct LegacyEnrollment {
    model: String,
    embedding: Vec<f32>,
}

impl Library {
    pub fn path(home: &Path) -> PathBuf {
        home.join("speaker").join("people.json")
    }

    pub fn clip_path(home: &Path, clip: u64) -> PathBuf {
        home.join("speaker")
            .join("clips")
            .join(format!("{clip}.wav"))
    }

    /// Changes whenever the file is rewritten; a live session reloads on it.
    pub fn stamp(home: &Path) -> Option<Stamp> {
        let meta = std::fs::metadata(Self::path(home)).ok()?;
        Some((meta.modified().ok()?, meta.len()))
    }

    /// [`load`](Self::load) with the stamp taken first, so a write landing
    /// during the load leaves the stamp stale and gets reloaded.
    pub fn load_stamped(home: &Path) -> (Result<Self, SpeakerError>, Option<Stamp>) {
        let stamp = Self::stamp(home);
        (Self::load(home), stamp)
    }

    /// An empty library when nothing is stored. With no `people.json`, the old
    /// single enrollment (`enrollment.json`, else the `.off-20261022` copy) is
    /// migrated into an owner named "Me" and left in place.
    pub fn load(home: &Path) -> Result<Self, SpeakerError> {
        let _guard = STORE.lock().unwrap_or_else(|e| e.into_inner());
        Self::load_unlocked(home)
    }

    fn load_unlocked(home: &Path) -> Result<Self, SpeakerError> {
        match std::fs::read(Self::path(home)) {
            Ok(bytes) => return serde_json::from_slice(&bytes).map_err(store_err),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(store_err(e)),
        }
        let dir = home.join("speaker");
        for name in ["enrollment.json", "enrollment.json.off-20261022"] {
            // An unreadable or corrupt candidate is skipped for the next one.
            let Ok(bytes) = std::fs::read(dir.join(name)) else {
                continue;
            };
            let Ok(old) = serde_json::from_slice::<LegacyEnrollment>(&bytes) else {
                continue;
            };
            let library = Library {
                model: old.model,
                next_clip_id: 2,
                people: vec![Person {
                    id: 1,
                    name: OWNER_NAME.to_string(),
                    owner: true,
                    clips: vec![Clip {
                        id: 1,
                        created_at: now_secs(),
                        seconds: 0.0,
                        embedding: old.embedding,
                        has_audio: false,
                    }],
                }],
                ..Library::default()
            };
            library.save(home)?;
            return Ok(library);
        }
        Ok(Library::default())
    }

    /// Atomic: written beside the file, then renamed over it.
    pub fn save(&self, home: &Path) -> Result<(), SpeakerError> {
        let path = Self::path(home);
        std::fs::create_dir_all(path.parent().expect("has a parent")).map_err(store_err)?;
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec(self).map_err(store_err)?;
        std::fs::write(&tmp, json).map_err(store_err)?;
        std::fs::rename(&tmp, &path).map_err(store_err)
    }

    /// Loads, applies `f`, and saves if it succeeded, all under the store lock.
    pub fn update<T>(
        home: &Path,
        f: impl FnOnce(&mut Library) -> Result<T, SpeakerError>,
    ) -> Result<T, SpeakerError> {
        let _guard = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let mut library = Self::load_unlocked(home)?;
        let saved = f(&mut library).and_then(|out| library.save(home).map(|()| out));
        // Files go only once the library no longer lists them, and the ones a
        // failed update wrote are not left behind.
        let files = match saved {
            Ok(_) => &library.pending_delete,
            Err(_) => &library.written,
        };
        for file in files {
            let _ = std::fs::remove_file(file);
        }
        saved
    }

    pub fn owner(&self) -> Option<&Person> {
        self.people.iter().find(|p| p.owner)
    }

    pub fn person(&self, id: u64) -> Result<&Person, SpeakerError> {
        self.people
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| SpeakerError::NotFound(format!("person {id} not found")))
    }

    fn person_mut(&mut self, id: u64) -> Result<&mut Person, SpeakerError> {
        self.people
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| SpeakerError::NotFound(format!("person {id} not found")))
    }

    /// The clips of `person` that can be scored against `model`.
    pub fn usable<'a>(&self, person: &'a Person, model: &str) -> Vec<&'a Clip> {
        if self.model != model {
            return Vec::new();
        }
        person.clips.iter().collect()
    }

    /// Whether the owner has a clip that can be scored against `model`.
    pub fn owner_ready(&self, model: &str) -> bool {
        self.owner()
            .is_some_and(|o| !self.usable(o, model).is_empty())
    }

    /// Adds a person; `owner` makes them the owner, clearing it on the others.
    pub fn add_person(&mut self, name: &str, owner: bool) -> u64 {
        let id = self.people.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        if owner {
            self.people.iter_mut().for_each(|p| p.owner = false);
        }
        self.people.push(Person {
            id,
            name: name.to_string(),
            owner,
            clips: Vec::new(),
        });
        id
    }

    /// The owner's id, creating "Me" when nobody owns the library yet.
    pub fn ensure_owner(&mut self) -> u64 {
        match self.owner() {
            Some(o) => o.id,
            None => self.add_person(OWNER_NAME, true),
        }
    }

    /// Renames and/or re-owns; setting `owner` true clears it on the others.
    pub fn patch_person(
        &mut self,
        id: u64,
        name: Option<&str>,
        owner: Option<bool>,
    ) -> Result<(), SpeakerError> {
        self.person(id)?;
        if owner == Some(true) {
            self.people.iter_mut().for_each(|p| p.owner = false);
        }
        let person = self.person_mut(id)?;
        if let Some(name) = name {
            person.name = name.to_string();
        }
        if let Some(owner) = owner {
            person.owner = owner;
        }
        Ok(())
    }

    /// Removes a person and their clip audio.
    pub fn remove_person(&mut self, home: &Path, id: u64) -> Result<(), SpeakerError> {
        let files: Vec<PathBuf> = (self.person(id)?.clips.iter())
            .map(|c| Self::clip_path(home, c.id))
            .collect();
        self.pending_delete.extend(files);
        self.people.retain(|p| p.id != id);
        Ok(())
    }

    /// Adds a clip to `person`, keeping its 16 kHz audio as a WAV.
    pub fn add_clip(
        &mut self,
        home: &Path,
        person: u64,
        pcm: &[f32],
        embedding: Vec<f32>,
    ) -> Result<Clip, SpeakerError> {
        self.person(person)?;
        if self.people.iter().all(|p| p.clips.is_empty()) {
            self.model = DEFAULT_MODEL.to_string();
        } else if self.model != DEFAULT_MODEL {
            return Err(SpeakerError::Store(format!(
                "the library was built with {}; remove its clips to start over",
                self.model
            )));
        }
        let highest = self
            .people
            .iter()
            .flat_map(|p| &p.clips)
            .map(|c| c.id)
            .max()
            .unwrap_or(0);
        let id = self.next_clip_id.max(highest + 1);
        self.next_clip_id = id + 1;
        let path = Self::clip_path(home, id);
        std::fs::create_dir_all(path.parent().expect("has a parent")).map_err(store_err)?;
        write_wav(&path, pcm)?;
        self.written.push(path);
        let clip = Clip {
            id,
            created_at: now_secs(),
            seconds: pcm.len() as f32 / SAMPLE_RATE as f32,
            embedding,
            has_audio: true,
        };
        self.person_mut(person)?.clips.push(clip.clone());
        Ok(clip)
    }

    pub fn remove_clip(&mut self, home: &Path, person: u64, clip: u64) -> Result<(), SpeakerError> {
        let p = self.person_mut(person)?;
        let before = p.clips.len();
        p.clips.retain(|c| c.id != clip);
        if p.clips.len() == before {
            return Err(SpeakerError::NotFound(format!("clip {clip} not found")));
        }
        self.pending_delete.push(Self::clip_path(home, clip));
        Ok(())
    }

    /// Re-embeds a library made with another model from each clip's kept
    /// audio with `embed`, and tags it with [`DEFAULT_MODEL`]. A clip with no
    /// audio (a migrated enrollment) or whose WAV is gone is dropped, since
    /// embeddings are never compared across models. When no clip has its WAV
    /// the library is left as it is (failing open until someone re-enrolls)
    /// rather than emptied. `true` if it re-embedded.
    pub fn reembed(
        &mut self,
        home: &Path,
        embed: impl Fn(&[f32]) -> Result<Vec<f32>, SpeakerError>,
    ) -> Result<bool, SpeakerError> {
        let has_wav = |c: &Clip| c.has_audio && Self::clip_path(home, c.id).is_file();
        if self.model == DEFAULT_MODEL || !self.people.iter().flat_map(|p| &p.clips).any(has_wav) {
            return Ok(false);
        }
        for person in &mut self.people {
            let mut kept = Vec::new();
            for mut clip in std::mem::take(&mut person.clips) {
                let path = Self::clip_path(home, clip.id);
                if !has_wav(&clip) {
                    self.pending_delete.push(path);
                    continue;
                }
                clip.embedding = embed(&read_wav(&path)?)?;
                kept.push(clip);
            }
            person.clips = kept;
        }
        self.model = DEFAULT_MODEL.to_string();
        Ok(true)
    }

    /// Removes every clip of `person` (their audio too); `true` if there were any.
    pub fn clear_clips(&mut self, home: &Path, person: u64) -> Result<bool, SpeakerError> {
        let clips = std::mem::take(&mut self.person_mut(person)?.clips);
        let had = !clips.is_empty();
        self.pending_delete
            .extend(clips.iter().map(|c| Self::clip_path(home, c.id)));
        Ok(had)
    }
}

/// [`Library::reembed`] on the stored library; the library as it now stands.
pub fn migrate(home: &Path, verifier: &SpeakerVerifier) -> Result<Library, SpeakerError> {
    Library::update(home, |l| l.reembed(home, |pcm| verifier.embed(pcm)))?;
    Library::load(home)
}

/// Speech level every enrolled clip is scaled to (naru task 2413): the loud
/// blocks of the Mac app's clips, which sit near -21 dBFS.
pub const TARGET_DBFS: f32 = -20.0;
/// No sample of a normalized clip goes above this, so it never clips.
pub const PEAK_CEILING_DBFS: f32 = -1.0;
/// A clip needing more gain than this to reach [`TARGET_DBFS`] is refused.
pub const MAX_GAIN_DB: f32 = 24.0;
/// A clip whose speech is less than this above its noise floor is refused.
pub const MIN_SPEECH_OVER_FLOOR_DB: f32 = 10.0;
/// Level blocks are 50 ms.
const LEVEL_BLOCK: usize = SAMPLE_RATE as usize / 20;

/// A clip [`normalize_level`] refuses: too faint to bring up to the target cleanly.
#[derive(Debug, PartialEq)]
pub struct TooQuiet {
    pub speech_dbfs: f32,
    pub floor_dbfs: f32,
}

impl std::fmt::Display for TooQuiet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fix = if self.speech_dbfs - self.floor_dbfs < MIN_SPEECH_OVER_FLOOR_DB {
            "record it again somewhere quieter, with pauses between sentences"
        } else {
            "move closer to the mic or raise its input level, then record it again"
        };
        write!(
            f,
            "the clip is too quiet to use (speech at {:.0} dBFS, noise floor at {:.0} dBFS); {fix}",
            self.speech_dbfs, self.floor_dbfs
        )
    }
}

/// Scales an enrollment clip up or down so its speech (the 90th percentile of
/// its 50 ms block levels, so pauses do not count) lands at [`TARGET_DBFS`],
/// holding its peak at or below [`PEAK_CEILING_DBFS`]. Every ingest path calls
/// this, so a quiet browser mic and a hot one enroll at the same level. Returns
/// the gain applied in dB, or [`TooQuiet`] when the gain would exceed
/// [`MAX_GAIN_DB`] or the speech sits too near the noise floor (the 10th
/// percentile).
pub fn normalize_level(pcm: &mut [f32]) -> Result<f32, TooQuiet> {
    let mut levels: Vec<f32> = pcm
        .as_chunks::<LEVEL_BLOCK>()
        .0
        .iter()
        .map(|b| {
            let ms = b.iter().map(|s| s * s).sum::<f32>() / b.len() as f32;
            10.0 * ms.max(1e-12).log10()
        })
        .collect();
    if levels.is_empty() {
        levels.push(-120.0);
    }
    levels.sort_by(f32::total_cmp);
    let at = |q: f32| levels[(q * (levels.len() - 1) as f32) as usize];
    let (speech_dbfs, floor_dbfs) = (at(0.9), at(0.1));
    if TARGET_DBFS - speech_dbfs > MAX_GAIN_DB
        || speech_dbfs - floor_dbfs < MIN_SPEECH_OVER_FLOOR_DB
    {
        return Err(TooQuiet {
            speech_dbfs,
            floor_dbfs,
        });
    }
    let peak = pcm.iter().fold(0f32, |m, s| m.max(s.abs()));
    let peak_dbfs = 20.0 * peak.max(1e-12).log10();
    let gain_db = (TARGET_DBFS - speech_dbfs).min(PEAK_CEILING_DBFS - peak_dbfs);
    let gain = 10f32.powf(gain_db / 20.0);
    for s in pcm.iter_mut() {
        *s *= gain;
    }
    Ok(gain_db)
}

/// A clip WAV as [`write_wav`] wrote it.
fn read_wav(path: &Path) -> Result<Vec<f32>, SpeakerError> {
    let mut reader = hound::WavReader::open(path).map_err(store_err)?;
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<_, _>>()
        .map_err(store_err)
}

fn write_wav(path: &Path, pcm: &[f32]) -> Result<(), SpeakerError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(store_err)?;
    for s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v).map_err(store_err)?;
    }
    writer.finalize().map_err(store_err)
}

/// How a person's clips combine into one score against a clip's embedding:
/// the cosine to the normalised mean of their clips. Measured against the best
/// (max) clip over leave-one-out fixtures in `tests/speaker.rs`: margins were
/// 0.336 (mean) and 0.333 (max), so the mean, which also smooths a noisy clip.
fn person_score(clips: &[&Clip], embedding: &[f32]) -> Option<f32> {
    let mut mean = vec![0f32; embedding.len()];
    let mut count = 0;
    for c in clips
        .iter()
        .filter(|c| c.embedding.len() == embedding.len())
    {
        mean.iter_mut().zip(&c.embedding).for_each(|(m, x)| *m += x);
        count += 1;
    }
    (count > 0).then(|| cosine(embedding, &mean))
}

/// Why a verdict came out as it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Scored against the library.
    Scored,
    /// Nobody in the library has a clip to score against.
    NoEnrollment,
    /// The library was made with a different model.
    ModelMismatch,
    /// Under [`MIN_SAMPLES`].
    TooShort,
    /// The model is not pulled or failed to embed.
    ModelUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Verdict {
    /// The best person's score.
    pub score: Option<f32>,
    /// The best match is the owner (or nothing could be scored: fail open).
    pub is_him: bool,
    pub reason: Reason,
}

impl Verdict {
    fn open(reason: Reason) -> Self {
        Verdict {
            score: None,
            is_him: true,
            reason,
        }
    }
}

/// A verdict and the person it matched, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct Identity {
    pub verdict: Verdict,
    /// The best-scoring person at or above the threshold.
    pub person: Option<PersonRef>,
}

/// The pure decision. `embedding` is the clip's embedding, `None` when the
/// model could not produce one. Every case but a scored one fails open. With
/// `only_owner` only the owner is a candidate (the legacy `verify`); else every
/// person with a usable clip is scored and the best at or above `threshold`
/// wins. A scored clip matching nobody is `is_him: false` with no person.
pub fn judge(
    library: Option<&Library>,
    only_owner: bool,
    model: &str,
    samples: usize,
    embedding: Option<&[f32]>,
    threshold: f32,
) -> Identity {
    let open = |reason| Identity {
        verdict: Verdict::open(reason),
        person: None,
    };
    let Some(library) = library else {
        return open(Reason::NoEnrollment);
    };
    let candidates: Vec<(&Person, Vec<&Clip>)> = library
        .people
        .iter()
        .filter(|p| p.owner || !only_owner)
        .map(|p| (p, library.usable(p, model)))
        .filter(|(_, clips)| !clips.is_empty())
        .collect();
    if candidates.is_empty() {
        let any = library.people.iter().any(|p| !p.clips.is_empty());
        return open(if any && library.model != model {
            Reason::ModelMismatch
        } else {
            Reason::NoEnrollment
        });
    }
    if samples < MIN_SAMPLES {
        return open(Reason::TooShort);
    }
    let Some(embedding) = embedding else {
        return open(Reason::ModelUnavailable);
    };
    let scored: Vec<(&Person, f32)> = candidates
        .iter()
        .filter_map(|(p, clips)| person_score(clips, embedding).map(|s| (*p, s)))
        .collect();
    // The owner wins whenever they clear the threshold, even if someone else
    // scores higher: wrongly dropping him is worse than admitting another.
    // Someone's clips scored but the owner's could not (a dimension mismatch):
    // fail open rather than call the owner another voice.
    if candidates.iter().any(|(p, _)| p.owner) && !scored.iter().any(|(p, _)| p.owner) {
        return open(Reason::ModelUnavailable);
    }
    let best = scored
        .iter()
        .copied()
        .find(|(p, s)| p.owner && *s >= threshold)
        .or_else(|| scored.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)));
    let Some((person, score)) = best else {
        return open(Reason::ModelUnavailable);
    };
    let matched = score >= threshold;
    Identity {
        verdict: Verdict {
            score: Some(score),
            is_him: matched && person.owner,
            reason: Reason::Scored,
        },
        person: matched.then(|| PersonRef {
            id: person.id,
            name: person.name.clone(),
        }),
    }
}

/// [`judge`] on `pcm`, embedding only when it can matter (someone to score
/// against, the clip long enough, the model present).
pub fn identify(
    verifier: Option<&SpeakerVerifier>,
    library: Option<&Library>,
    only_owner: bool,
    pcm: &[f32],
    threshold: f32,
) -> Identity {
    let embedding = match (verifier, library) {
        (Some(v), Some(_)) if pcm.len() >= MIN_SAMPLES => v.embed(pcm).ok(),
        _ => None,
    };
    judge(
        library,
        only_owner,
        DEFAULT_MODEL,
        pcm.len(),
        embedding.as_deref(),
        threshold,
    )
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Cosine similarity; 0 when either vector is zero or lengths differ.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (
        a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        b.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library of `(name, owner, embeddings)` people.
    fn library(people: &[(&str, bool, &[&[f32]])]) -> Library {
        let mut lib = Library::default();
        let mut clip = 0;
        for (i, (name, owner, clips)) in people.iter().enumerate() {
            lib.people.push(Person {
                id: i as u64 + 1,
                name: name.to_string(),
                owner: *owner,
                clips: clips
                    .iter()
                    .map(|e| {
                        clip += 1;
                        Clip {
                            id: clip,
                            created_at: 0,
                            seconds: 1.0,
                            embedding: e.to_vec(),
                            has_audio: false,
                        }
                    })
                    .collect(),
            });
        }
        lib
    }

    fn id(lib: &Library, emb: &[f32], thr: f32, only_owner: bool) -> Identity {
        judge(
            Some(lib),
            only_owner,
            DEFAULT_MODEL,
            MIN_SAMPLES,
            Some(emb),
            thr,
        )
    }

    /// 4 s of "speech": 250 ms bursts of a 220 Hz tone at `speech_dbfs` RMS
    /// with 250 ms pauses of noise at `floor` amplitude between them.
    fn speechlike(speech_dbfs: f32, floor: f32) -> Vec<f32> {
        let amp = 10f32.powf(speech_dbfs / 20.0) * std::f32::consts::SQRT_2;
        let mut seed = 1u32;
        (0..4 * SAMPLE_RATE as usize)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = floor * ((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0);
                let on = (i / 4_000) % 2 == 0;
                let t = i as f32 / SAMPLE_RATE as f32;
                noise
                    + if on {
                        amp * (t * 220.0 * std::f32::consts::TAU).sin()
                    } else {
                        0.0
                    }
            })
            .collect()
    }

    fn speech_and_peak(pcm: &[f32]) -> (f32, f32) {
        let mut levels: Vec<f32> = pcm
            .as_chunks::<LEVEL_BLOCK>()
            .0
            .iter()
            .map(|b| 10.0 * (b.iter().map(|s| s * s).sum::<f32>() / b.len() as f32).log10())
            .collect();
        levels.sort_by(f32::total_cmp);
        let peak = pcm.iter().fold(0f32, |m, s| m.max(s.abs()));
        (
            levels[(0.9 * (levels.len() - 1) as f32) as usize],
            20.0 * peak.log10(),
        )
    }

    #[test]
    fn normalize_level_brings_a_quiet_clip_up() {
        let mut pcm = speechlike(-37.0, 1e-4);
        let gain = normalize_level(&mut pcm).unwrap();
        let (speech, peak) = speech_and_peak(&pcm);
        assert!((gain - 17.0).abs() < 0.5, "gain {gain}");
        assert!((speech - TARGET_DBFS).abs() < 0.5, "speech {speech}");
        assert!(peak <= PEAK_CEILING_DBFS + 1e-3, "peak {peak}");
    }

    #[test]
    fn normalize_level_brings_a_loud_clip_down() {
        let mut pcm = speechlike(-8.0, 1e-3);
        let gain = normalize_level(&mut pcm).unwrap();
        let (speech, _) = speech_and_peak(&pcm);
        assert!((gain + 12.0).abs() < 0.5, "gain {gain}");
        assert!((speech - TARGET_DBFS).abs() < 0.5, "speech {speech}");
    }

    #[test]
    fn normalize_level_holds_the_peak_ceiling() {
        // speech at -30 dBFS with one full-scale click: reaching -20 would
        // put the click 10 dB over full scale, so the ceiling wins
        let mut pcm = speechlike(-30.0, 1e-4);
        pcm[1_000] = 1.0;
        let gain = normalize_level(&mut pcm).unwrap();
        let (_, peak) = speech_and_peak(&pcm);
        assert!((gain - PEAK_CEILING_DBFS).abs() < 1e-3, "gain {gain}");
        assert!(peak <= PEAK_CEILING_DBFS + 1e-3, "peak {peak}");
        assert!(pcm.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn normalize_level_refuses_a_too_faint_clip() {
        let mut pcm = speechlike(-50.0, 1e-5);
        let before = pcm.clone();
        let e = normalize_level(&mut pcm).unwrap_err();
        assert!((e.speech_dbfs + 50.0).abs() < 0.5, "{e:?}");
        assert!(e.to_string().contains("record it again"));
        assert_eq!(pcm, before, "a refused clip is left alone");
        // digital silence too
        assert!(normalize_level(&mut vec![0.0; 16_000]).is_err());
    }

    #[test]
    fn normalize_level_refuses_speech_buried_in_noise() {
        // loud enough, but the "pauses" are as loud as the speech
        let mut pcm = speechlike(-30.0, 0.04);
        let e = normalize_level(&mut pcm).unwrap_err();
        assert!(
            e.speech_dbfs - e.floor_dbfs < MIN_SPEECH_OVER_FLOOR_DB,
            "{e:?}"
        );
        assert!(e.to_string().contains("somewhere quieter"));
    }

    #[test]
    fn cosine_math() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn threshold_decides_a_scored_clip() {
        let lib = library(&[("Me", true, &[&[1.0, 0.0]])]);
        let same = id(&lib, &[1.0, 0.0], 0.5, true);
        assert!(same.verdict.is_him && same.verdict.reason == Reason::Scored);
        assert_eq!(same.person.unwrap().name, "Me");
        let other = id(&lib, &[0.0, 1.0], 0.5, true);
        assert!(
            !other.verdict.is_him && other.verdict.score == Some(0.0) && other.person.is_none()
        );
        // exactly at the threshold admits
        assert!(id(&lib, &[0.6, 0.8], 0.6, true).verdict.is_him);
        assert!(!id(&lib, &[0.6, 0.8], 0.7, true).verdict.is_him);
    }

    #[test]
    fn a_person_scores_against_the_mean_of_their_clips() {
        let lib = library(&[("Me", true, &[&[1.0, 0.0], &[0.0, 1.0]])]);
        // a clip identical to one of the two sits 45 degrees from their mean
        let score = id(&lib, &[0.0, 1.0], 0.5, true).verdict.score.unwrap();
        assert!((score - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        // a clip between them scores 1
        assert!(id(&lib, &[0.6, 0.6], 0.99, true).verdict.is_him);
    }

    #[test]
    fn the_owner_wins_when_they_clear_the_threshold() {
        let lib = library(&[("Me", true, &[&[1.0, 0.0]]), ("Sam", false, &[&[0.6, 0.8]])]);
        // Sam scores 1.0 and Me 0.6, but Me clears 0.5: Me
        let id1 = id(&lib, &[0.6, 0.8], 0.5, false);
        assert!(id1.verdict.is_him && id1.person.unwrap().name == "Me");
        assert_eq!(id1.verdict.score.map(|s| (s * 10.0).round()), Some(6.0));
        // below the owner's threshold the best person wins as before
        let id2 = id(&lib, &[0.6, 0.8], 0.7, false);
        assert!(!id2.verdict.is_him && id2.person.unwrap().name == "Sam");
    }

    #[test]
    fn the_best_person_wins_and_names_the_line() {
        let lib = library(&[("Me", true, &[&[1.0, 0.0]]), ("Sam", false, &[&[0.0, 1.0]])]);
        let me = id(&lib, &[0.9, 0.1], 0.7, false);
        assert!(me.verdict.is_him && me.person.as_ref().unwrap().name == "Me");
        let sam = id(&lib, &[0.1, 0.9], 0.7, false);
        assert!(!sam.verdict.is_him && sam.verdict.reason == Reason::Scored);
        assert_eq!(
            sam.person,
            Some(PersonRef {
                id: 2,
                name: "Sam".into()
            })
        );
        // scored but matches nobody: other, unnamed
        let nobody = id(&lib, &[0.7, -0.7], 0.9, false);
        assert!(!nobody.verdict.is_him && nobody.person.is_none());
        // only the owner is a candidate for the legacy verify
        let legacy = id(&lib, &[0.1, 0.9], 0.7, true);
        assert!(!legacy.verdict.is_him && legacy.person.is_none());
    }

    #[test]
    fn every_fail_open_path_is_him() {
        let lib = library(&[("Me", true, &[&[1.0, 0.0]])]);
        let orthogonal: &[f32] = &[0.0, 1.0];
        let run = |l: Option<&Library>, model, samples, emb| {
            judge(l, true, model, samples, emb, 0.5).verdict
        };
        let v = run(None, DEFAULT_MODEL, MIN_SAMPLES, Some(orthogonal));
        assert!(v.is_him && v.reason == Reason::NoEnrollment);
        let empty = Library::default();
        let v = run(Some(&empty), DEFAULT_MODEL, MIN_SAMPLES, Some(orthogonal));
        assert!(v.is_him && v.reason == Reason::NoEnrollment);
        let v = run(Some(&lib), "other-model", MIN_SAMPLES, Some(orthogonal));
        assert!(v.is_him && v.reason == Reason::ModelMismatch);
        let v = run(Some(&lib), DEFAULT_MODEL, MIN_SAMPLES - 1, Some(orthogonal));
        assert!(v.is_him && v.reason == Reason::TooShort && v.score.is_none());
        let v = run(Some(&lib), DEFAULT_MODEL, MIN_SAMPLES, None);
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
        let v = run(
            Some(&lib),
            DEFAULT_MODEL,
            MIN_SAMPLES,
            Some(&[1.0, 0.0, 0.0]),
        );
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
        // no verifier at all (model not pulled): open, even with a long clip
        let v = identify(None, Some(&lib), true, &vec![0.0; MIN_SAMPLES * 2], 0.5).verdict;
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
    }

    #[test]
    fn library_round_trips_and_edits() {
        let home = tempfile::tempdir().unwrap();
        let h = home.path();
        assert_eq!(Library::load(h).unwrap(), Library::default());
        let (me, sam) = Library::update(h, |l| {
            let me = l.ensure_owner();
            let sam = l.add_person("Sam", false);
            l.add_clip(h, me, &vec![0.1; 16_000], vec![1.0, 0.0])?;
            l.add_clip(h, sam, &vec![0.1; 8_000], vec![0.0, 1.0])?;
            Ok((me, sam))
        })
        .unwrap();
        let lib = Library::load(h).unwrap();
        assert_eq!(lib.owner().unwrap().id, me);
        assert!(lib.owner_ready(DEFAULT_MODEL) && !lib.owner_ready("x"));
        let clip = lib.person(sam).unwrap().clips[0].clone();
        assert!(clip.has_audio && (clip.seconds - 0.5).abs() < 1e-6);
        assert!(Library::clip_path(h, clip.id).is_file());
        // owner moves; the old owner is cleared
        Library::update(h, |l| l.patch_person(sam, Some("Samantha"), Some(true))).unwrap();
        let lib = Library::load(h).unwrap();
        assert_eq!(lib.owner().unwrap().name, "Samantha");
        assert_eq!(lib.people.iter().filter(|p| p.owner).count(), 1);
        // deleting a person removes their audio
        Library::update(h, |l| l.remove_person(h, sam)).unwrap();
        assert!(!Library::clip_path(h, clip.id).exists());
        assert!(matches!(
            Library::update(h, |l| l.remove_clip(h, me, 99)),
            Err(SpeakerError::NotFound(_))
        ));
    }

    #[test]
    fn the_old_enrollment_migrates_to_an_owner_and_stays() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("speaker");
        std::fs::create_dir_all(&dir).unwrap();
        let old = r#"{"model":"wespeaker-sv-en","count":2,"embedding":[0.6,0.8]}"#;
        // the .off copy is used when there is no live file
        std::fs::write(dir.join("enrollment.json.off-20261022"), old).unwrap();
        let lib = Library::load(home.path()).unwrap();
        let me = lib.owner().unwrap();
        assert_eq!((me.name.as_str(), me.clips.len()), ("Me", 1));
        assert_eq!(me.clips[0].embedding, vec![0.6, 0.8]);
        assert!(!me.clips[0].has_audio && !lib.live_guard);
        assert!(Library::path(home.path()).is_file());
        assert!(dir.join("enrollment.json.off-20261022").is_file());
        // once people.json exists the old file is ignored
        std::fs::write(dir.join("enrollment.json"), old.replace("0.6", "0.1")).unwrap();
        assert_eq!(Library::load(home.path()).unwrap(), lib);
        // enrollment.json is preferred when people.json is absent
        std::fs::remove_file(Library::path(home.path())).unwrap();
        let lib = Library::load(home.path()).unwrap();
        assert_eq!(lib.owner().unwrap().clips[0].embedding, vec![0.1, 0.8]);
        assert!(dir.join("enrollment.json").is_file());
    }

    #[test]
    fn an_owner_who_cannot_be_scored_fails_open() {
        let lib = library(&[
            ("Me", true, &[&[1.0, 0.0, 0.0]]),
            ("Sam", false, &[&[0.0, 1.0]]),
        ]);
        let v = id(&lib, &[0.0, 1.0], 0.5, false).verdict;
        assert!(v.is_him && v.reason == Reason::ModelUnavailable);
    }

    #[test]
    fn a_failed_update_keeps_old_audio_and_drops_new() {
        let home = tempfile::tempdir().unwrap();
        let h = home.path();
        let pcm = vec![0.1; 16_000];
        let me = Library::update(h, |l| {
            let me = l.ensure_owner();
            l.add_clip(h, me, &pcm, vec![1.0, 0.0])?;
            Ok(me)
        })
        .unwrap();
        let old = Library::clip_path(h, Library::load(h).unwrap().owner().unwrap().clips[0].id);
        assert!(old.is_file());
        let failed = Library::update(h, |l| {
            l.clear_clips(h, me)?;
            let clip = l.add_clip(h, me, &pcm, vec![0.0, 1.0])?;
            assert!(Library::clip_path(h, clip.id).is_file());
            Err::<(), _>(SpeakerError::Store("boom".into()))
        });
        assert!(failed.is_err());
        assert!(old.is_file(), "old audio survives a failed replace");
        assert_eq!(Library::load(h).unwrap().owner().unwrap().clips.len(), 1);
        let wavs = || std::fs::read_dir(h.join("speaker/clips")).unwrap().count();
        assert_eq!(wavs(), 1, "the new audio is not leaked");
        // a successful replace removes the old audio only after saving
        Library::update(h, |l| {
            l.clear_clips(h, me)?;
            l.add_clip(h, me, &pcm, vec![0.0, 1.0]).map(|_| ())
        })
        .unwrap();
        assert!(!old.exists());
        assert_eq!(wavs(), 1);
    }

    #[test]
    fn a_corrupt_enrollment_falls_through_to_the_next_candidate() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("speaker");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("enrollment.json"), "").unwrap();
        std::fs::write(
            dir.join("enrollment.json.off-20261022"),
            r#"{"model":"wespeaker-sv-en","count":1,"embedding":[0.6,0.8]}"#,
        )
        .unwrap();
        let lib = Library::load(home.path()).unwrap();
        assert_eq!(lib.owner().unwrap().clips[0].embedding, vec![0.6, 0.8]);
        // nothing usable: an empty library, not an error
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join("speaker")).unwrap();
        std::fs::write(home.path().join("speaker/enrollment.json"), "{").unwrap();
        assert_eq!(Library::load(home.path()).unwrap(), Library::default());
    }

    #[test]
    fn clip_ids_are_not_reused() {
        let home = tempfile::tempdir().unwrap();
        let h = home.path();
        let pcm = vec![0.1; 16_000];
        let ids = |l: &mut Library, me| {
            let a = l.add_clip(h, me, &pcm, vec![1.0]).unwrap().id;
            l.remove_clip(h, me, a).unwrap();
            let b = l.add_clip(h, me, &pcm, vec![1.0]).unwrap().id;
            (a, b)
        };
        let (a, b) = Library::update(h, |l| {
            let me = l.ensure_owner();
            Ok(ids(l, me))
        })
        .unwrap();
        assert!(b > a);
        // the counter survives a reload
        let c = Library::update(h, |l| {
            let me = l.ensure_owner();
            l.remove_clip(h, me, b)?;
            Ok(l.add_clip(h, me, &pcm, vec![1.0])?.id)
        })
        .unwrap();
        assert!(c > b);
    }

    #[test]
    fn the_stamp_is_taken_before_the_load() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("speaker");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("enrollment.json"),
            r#"{"model":"wespeaker-sv-en","count":1,"embedding":[1.0]}"#,
        )
        .unwrap();
        // the load migrates and so writes people.json; the stamp predates it
        // and differs from the file's now, so the next check reloads
        let (lib, stamp) = Library::load_stamped(home.path());
        assert!(lib.unwrap().owner().is_some());
        assert_eq!(stamp, None);
        assert_ne!(Library::stamp(home.path()), stamp);
    }

    #[test]
    fn a_library_of_another_model_is_reembedded_from_its_audio() {
        let home = tempfile::tempdir().unwrap();
        let h = home.path();
        let pcm = vec![0.1; 16_000];
        let (kept, gone) = Library::update(h, |l| {
            let me = l.ensure_owner();
            let kept = l.add_clip(h, me, &pcm, vec![1.0, 0.0])?.id;
            let gone = l.add_clip(h, me, &pcm, vec![1.0, 0.0])?.id;
            // a migrated enrollment, which has no audio
            l.people[0].clips.push(Clip {
                id: 99,
                created_at: 0,
                seconds: 0.0,
                embedding: vec![1.0, 0.0],
                has_audio: false,
            });
            l.model = "wespeaker-sv-en".into();
            Ok((kept, gone))
        })
        .unwrap();
        std::fs::remove_file(Library::clip_path(h, gone)).unwrap();
        let embed = |p: &[f32]| Ok(vec![p.len() as f32, 0.5]);
        assert!(Library::update(h, |l| l.reembed(h, embed)).unwrap());
        let lib = Library::load(h).unwrap();
        assert_eq!(lib.model, DEFAULT_MODEL);
        let clips = &lib.owner().unwrap().clips;
        assert_eq!(clips.iter().map(|c| c.id).collect::<Vec<_>>(), [kept]);
        // the WAV round-trips to the same length
        assert_eq!(clips[0].embedding, vec![16_000.0, 0.5]);
        assert!(lib.owner_ready(DEFAULT_MODEL));
        // a library already on the model is left alone
        let mut again = lib.clone();
        assert!(!again.reembed(h, |_| Err(SpeakerError::EmbedFailed)).unwrap());
        // one with no audio at all is kept, not emptied
        again.model = "wespeaker-sv-en".into();
        again.people[0].clips[0].has_audio = false;
        assert!(!again.reembed(h, embed).unwrap());
        assert_eq!((again.model.as_str(), again.people[0].clips.len()), ("wespeaker-sv-en", 1));
    }

    #[test]
    fn missing_model_dir_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            SpeakerVerifier::load(dir.path()),
            Err(SpeakerError::MissingModel(_))
        ));
    }
}
