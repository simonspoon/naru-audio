//! Cloned voices: `$NARU_AUDIO_HOME/voices/<name>/` holds `ref.wav`, a
//! short clip of the voice, `ref.txt`, what it says, and `model.txt`, the
//! model it was cloned or designed for. A model whose manifest sets
//! `[backend.<backend>] clone = true` (Qwen3-TTS Base) speaks in each voice
//! made for it, after its own `[[voice]]`s. A voice with no `model.txt` —
//! everything `add` made before per-model voices existed — is attributed to
//! [`CLONE_MODEL`], what `say_model` always asked for before too. They are
//! read from disk on each request, so a voice added while the daemon runs
//! is usable at once.

use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::registry::manifest::{Manifest, Voice};

pub const REF_WAV: &str = "ref.wav";
pub const REF_TXT: &str = "ref.txt";
pub const MODEL_TXT: &str = "model.txt";
/// A designed voice's description; its mere presence marks the voice as
/// "designed" rather than "cloned" (naru task 1458 §2.6).
pub const DESIGN_TXT: &str = "design.txt";

/// Clip lengths `add` accepts, in seconds; 5–15 s is the target.
pub const MIN_SECS: f64 = 3.0;
pub const MAX_SECS: f64 = 30.0;

/// `ref.wav`'s rate: Qwen3-TTS's (mlx-audio resamples to the model's
/// rate anyway).
const SAMPLE_RATE: u32 = 24_000;

/// The cloning model `say` asks for when its voice is a cloned one.
pub const CLONE_MODEL: &str = "qwen3-tts-0.6b-base-mlx";

/// A cloned voice's clip, transcript and the model it was made for.
#[derive(Debug, PartialEq)]
pub struct Cloned {
    pub wav: PathBuf,
    pub text: String,
    pub model: String,
}

pub fn dir(home: &Path) -> PathBuf {
    home.join("voices")
}

/// The longest voice name, in bytes: well inside a file name's 255 once
/// `add`'s scratch directory wraps it.
pub const MAX_NAME_BYTES: usize = 64;

/// A voice name is one plain path component: not empty, no `/` or `\`,
/// no leading `.` (so neither `.` nor `..`), and at most
/// [`MAX_NAME_BYTES`] long.
pub fn check_name(name: &str) -> Result<(), String> {
    if name.len() > MAX_NAME_BYTES {
        return Err(format!(
            "voice name is {} bytes; the cap is {MAX_NAME_BYTES}",
            name.len()
        ));
    }
    if name.is_empty()
        || name.starts_with('.')
        || name.contains(['/', '\\', '\0'])
        || name.chars().any(char::is_control)
    {
        return Err(format!(
            "voice name {name:?} must be a plain name: no path separators and no leading `.`"
        ));
    }
    Ok(())
}

/// The names of the complete cloned voices in `home`, sorted.
pub fn list(home: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir(home)) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| find(home, n).is_some())
        .collect();
    names.sort();
    names
}

/// The voice `name` in `home`, if it is a valid name with both files.
/// `model` is `model.txt`'s content, trimmed, or [`CLONE_MODEL`] if the
/// voice predates it or `model.txt` is empty or blank (as unreadable, not
/// a model named "").
pub fn find(home: &Path, name: &str) -> Option<Cloned> {
    check_name(name).ok()?;
    let voice = dir(home).join(name);
    let wav = voice.join(REF_WAV);
    let text = std::fs::read_to_string(voice.join(REF_TXT)).ok()?;
    let model = std::fs::read_to_string(voice.join(MODEL_TXT))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| CLONE_MODEL.to_string());
    wav.is_file().then(|| Cloned {
        wav,
        text: text.trim().to_string(),
        model,
    })
}

/// The model voice `name` in `home` was made for, if it exists at all
/// (found or not, as [`find`]).
pub fn model_of(home: &Path, name: &str) -> Option<String> {
    find(home, name).map(|c| c.model)
}

/// `manifest`'s voices, then, if it clones, the cloned voices in `home`
/// made for `manifest`'s model that no `[[voice]]` shadows.
pub fn of(manifest: &Manifest, home: &Path) -> Vec<Voice> {
    let mut voices = manifest.voices.clone();
    if manifest.clones() {
        for id in list(home) {
            if voices.iter().any(|v| v.id == id) {
                continue;
            }
            if model_of(home, &id).as_deref() != Some(manifest.model.name.as_str()) {
                continue;
            }
            voices.push(Voice {
                id,
                sid: 0,
                reference: None,
                accent: None,
                gender: None,
                default: false,
            });
        }
    }
    voices
}

/// The model `say` asks for: `model` if given, else the model `voice` was
/// cloned or designed for if it is a cloned voice in `home` ([`model_of`],
/// [`CLONE_MODEL`] for one that predates per-model voices), else the
/// daemon's `default`.
pub fn say_model(home: Option<&Path>, model: Option<&str>, voice: Option<&str>) -> String {
    match (model, voice) {
        (Some(model), _) => model.to_string(),
        (None, Some(voice)) => home
            .and_then(|h| model_of(h, voice))
            .unwrap_or_else(|| "default".to_string()),
        _ => "default".to_string(),
    }
}

/// Why [`add`] refused a voice; its `Display` is the message.
#[derive(Debug)]
pub enum AddError {
    /// Not a plain name ([`check_name`]).
    Name(String),
    /// A voice of that name already exists.
    Exists(String),
    /// The transcript is empty.
    Text,
    /// `afconvert` cannot read the clip.
    Clip(String),
    /// The clip's length, outside `MIN_SECS..=MAX_SECS`.
    Length(f64),
    /// Anything else: the home's files, or no `afconvert`.
    Io(String),
}

impl std::fmt::Display for AddError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddError::Name(m) | AddError::Exists(m) | AddError::Clip(m) | AddError::Io(m) => {
                f.write_str(m)
            }
            AddError::Text => f.write_str("the transcript is empty"),
            AddError::Length(secs) => write!(
                f,
                "the clip is {secs:.1} s; it must be {MIN_SECS}–{MAX_SECS} s (5–15 s is best)"
            ),
        }
    }
}

/// A fresh path in `home`'s `tmp/`, unique within and across processes.
pub fn scratch(home: &Path, stem: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    home.join("tmp").join(format!(
        "{stem}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Adds the voice `name` from `clip` (anything `afconvert` reads, such as
/// WAV or MP3), converted to 24 kHz mono 16-bit WAV, with `text` as its
/// transcript, recorded for `model` (`model.txt`; the caller checks `model`
/// actually clones — this just writes what it is told). Returns the clip's
/// length in seconds, which must be within `MIN_SECS..=MAX_SECS`. An
/// existing voice is not replaced. Both `voice add` (always [`CLONE_MODEL`],
/// which requires a transcript) and `POST /v1/audio/voices` come here;
/// `require_text` is the caller's `model.clone_requires_transcript()`
/// (mesa task 1455) — a blank transcript is refused only when the model
/// actually needs one, though whatever text is given (even blank) is still
/// stored, so a model with no use for a transcript can still be added from
/// a clip alone. `description`, if given and non-blank, is stored as
/// [`DESIGN_TXT`] (naru task 1458 §2.6): its mere presence, not its
/// content, is what [`is_designed`] reads as "designed" rather than
/// "cloned".
pub fn add(
    home: &Path,
    name: &str,
    clip: &Path,
    text: &str,
    model: &str,
    require_text: bool,
    description: Option<&str>,
) -> Result<f64, AddError> {
    check_name(name).map_err(AddError::Name)?;
    let text = text.trim();
    if require_text && text.is_empty() {
        return Err(AddError::Text);
    }
    let voice = dir(home).join(name);
    let exists = || {
        AddError::Exists(format!(
            "voice `{name}` already exists; remove {} first",
            voice.display()
        ))
    };
    if voice.exists() {
        return Err(exists());
    }
    // Built in tmp/ and renamed into place, so a voice is never half there.
    let tmp = scratch(home, &format!("voice-{name}"));
    std::fs::create_dir_all(&tmp)
        .map_err(|e| AddError::Io(format!("create {}: {e}", tmp.display())))?;
    let result = convert(clip, &tmp.join(REF_WAV)).and_then(|secs| {
        if !(MIN_SECS..=MAX_SECS).contains(&secs) {
            return Err(AddError::Length(secs));
        }
        std::fs::write(tmp.join(REF_TXT), format!("{text}\n"))
            .map_err(|e| AddError::Io(format!("write {REF_TXT}: {e}")))?;
        std::fs::write(tmp.join(MODEL_TXT), format!("{model}\n"))
            .map_err(|e| AddError::Io(format!("write {MODEL_TXT}: {e}")))?;
        if let Some(description) = description.map(str::trim).filter(|d| !d.is_empty()) {
            std::fs::write(tmp.join(DESIGN_TXT), format!("{description}\n"))
                .map_err(|e| AddError::Io(format!("write {DESIGN_TXT}: {e}")))?;
        }
        std::fs::create_dir_all(dir(home))
            .map_err(|e| AddError::Io(format!("create voices/: {e}")))?;
        // A concurrent add of the same name may have won the rename.
        std::fs::rename(&tmp, &voice).map_err(|e| {
            if voice.exists() {
                exists()
            } else {
                AddError::Io(format!("move into {}: {e}", voice.display()))
            }
        })?;
        Ok(secs)
    });
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    result
}

/// Removes the cloned voice `name` from `home`, whether or not it is
/// complete (an `add` left half-done, say). Returns whether it existed.
/// An invalid name, as `find` treats it, is just not found. Used by
/// `DELETE /v1/audio/voices/{name}`; there is no `voice rm` yet, as
/// there is no `voice list` either.
pub fn remove(home: &Path, name: &str) -> Result<bool, String> {
    if check_name(name).is_err() {
        return Ok(false);
    }
    let voice = dir(home).join(name);
    if !voice.is_dir() {
        return Ok(false);
    }
    std::fs::remove_dir_all(&voice)
        .map_err(|e| format!("remove {}: {e}", voice.display()))
        .map(|()| true)
}

/// Whether `name` in `home` is a designed voice: a cloned voice with a
/// [`DESIGN_TXT`] alongside its clip, written from `POST
/// /v1/audio/voices`'s `description` field (naru task 1458 §2.6). A
/// built-in voice, never in `voices/`, is never designed.
pub fn is_designed(home: &Path, name: &str) -> bool {
    check_name(name).is_ok() && dir(home).join(name).join(DESIGN_TXT).is_file()
}

/// `name`'s description in `home` — [`DESIGN_TXT`], trimmed — if it has one
/// and it is not blank; `None` for a voice with no description, including
/// every cloned (not designed) and built-in voice.
pub fn description(home: &Path, name: &str) -> Option<String> {
    check_name(name).ok()?;
    let text = std::fs::read_to_string(dir(home).join(name).join(DESIGN_TXT)).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// `name`'s clip length in `home`, in seconds, read from its `ref.wav`
/// (§2.6 `GET /v1/audio/voices`'s `duration`); `None` for a voice that does
/// not exist or whose `ref.wav` cannot be read, and always `None` for a
/// built-in voice, which has no clip here at all.
pub fn duration(home: &Path, name: &str) -> Option<f64> {
    let wav = find(home, name)?.wav;
    let reader = hound::WavReader::open(&wav).ok()?;
    let spec = reader.spec();
    (spec.sample_rate > 0).then(|| f64::from(reader.duration()) / f64::from(spec.sample_rate))
}

/// Why [`update`] refused a change; its `Display` is the message.
#[derive(Debug)]
pub enum UpdateError {
    /// The new name is not a plain name ([`check_name`]).
    Name(String),
    /// `name` is not a cloned voice (the caller checks it is not a
    /// built-in one first, as its message would be wrong here).
    NotFound,
    /// The new name is already taken.
    Exists(String),
    /// Anything else: the home's files.
    Io(String),
}

impl std::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::Name(m) | UpdateError::Exists(m) | UpdateError::Io(m) => f.write_str(m),
            UpdateError::NotFound => f.write_str("no such cloned voice"),
        }
    }
}

/// `PATCH /v1/audio/voices/{name}` (naru task 1458 §2.6): renames the
/// cloned voice `name` to `new_name` (a no-op if `None` or unchanged),
/// then overwrites [`REF_TXT`] with `text` and [`DESIGN_TXT`] with
/// `description` where given (`None` leaves each alone; the caller passes
/// only the fields the request body actually set). A blank `description`
/// still writes an empty [`DESIGN_TXT`] rather than removing it — clearing
/// a design is out of scope here — so once a voice is designed, `PATCH`
/// cannot make it merely cloned again. Returns the voice's final name.
pub fn update(
    home: &Path,
    name: &str,
    new_name: Option<&str>,
    text: Option<&str>,
    description: Option<&str>,
) -> Result<String, UpdateError> {
    if find(home, name).is_none() {
        return Err(UpdateError::NotFound);
    }
    let mut voice = dir(home).join(name);
    let mut final_name = name.to_string();
    if let Some(new_name) = new_name.filter(|n| *n != name) {
        check_name(new_name).map_err(UpdateError::Name)?;
        let target = dir(home).join(new_name);
        if target.exists() {
            return Err(UpdateError::Exists(format!(
                "voice `{new_name}` already exists; remove {} first",
                target.display()
            )));
        }
        // A concurrent rename onto the same new name may have won.
        std::fs::rename(&voice, &target).map_err(|e| {
            if target.exists() {
                UpdateError::Exists(format!(
                    "voice `{new_name}` already exists; remove {} first",
                    target.display()
                ))
            } else {
                UpdateError::Io(format!(
                    "rename {} to {}: {e}",
                    voice.display(),
                    target.display()
                ))
            }
        })?;
        voice = target;
        final_name = new_name.to_string();
    }
    if let Some(text) = text {
        std::fs::write(voice.join(REF_TXT), format!("{}\n", text.trim()))
            .map_err(|e| UpdateError::Io(format!("write {REF_TXT}: {e}")))?;
    }
    if let Some(description) = description {
        std::fs::write(voice.join(DESIGN_TXT), format!("{}\n", description.trim()))
            .map_err(|e| UpdateError::Io(format!("write {DESIGN_TXT}: {e}")))?;
    }
    Ok(final_name)
}

/// Converts `clip` to a 24 kHz mono WAV at a fresh scratch path under
/// `home`'s `tmp/`, as [`add`] does, but without creating a voice: `POST
/// /api/voices/preview` (naru task 1458 §2.6) clones from an upload it
/// never saves. Returns the temp WAV's path (whose parent directory the
/// caller must remove once done) and its length in seconds, checked
/// against `MIN_SECS..=MAX_SECS` the same way `add` does.
pub fn convert_to_tmp(home: &Path, stem: &str, clip: &Path) -> Result<(PathBuf, f64), AddError> {
    let tmp = scratch(home, stem);
    std::fs::create_dir_all(&tmp)
        .map_err(|e| AddError::Io(format!("create {}: {e}", tmp.display())))?;
    let wav = tmp.join(REF_WAV);
    match convert(clip, &wav) {
        Ok(secs) if (MIN_SECS..=MAX_SECS).contains(&secs) => Ok((wav, secs)),
        Ok(secs) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(AddError::Length(secs))
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(e)
        }
    }
}

/// `afconvert`s `clip` to `wav` and returns its length in seconds.
fn convert(clip: &Path, wav: &Path) -> Result<f64, AddError> {
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("afconvert")
            .args([
                "-f",
                "WAVE",
                "-d",
                &format!("LEI16@{SAMPLE_RATE}"),
                "-c",
                "1",
            ])
            .arg(clip)
            .arg(wav)
            .output()
            .map_err(|e| AddError::Io(format!("cannot run afconvert (macOS only): {e}")))?;
        if !out.status.success() {
            return Err(AddError::Clip(format!(
                "afconvert cannot read {}: {}",
                clip.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
    }
    #[cfg(not(target_os = "macos"))]
    // Cap well above MAX_SECS so an over-long clip still reports its length.
    crate::transcode::to_wav(clip, wav, SAMPLE_RATE, 600)
        .map_err(|e| AddError::Clip(format!("cannot read {}: {e}", clip.display())))?;
    let reader = hound::WavReader::open(wav)
        .map_err(|e| AddError::Io(format!("read the converted clip: {e}")))?;
    Ok(f64::from(reader.duration()) / f64::from(reader.spec().sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_one_plain_component() {
        for ok in ["narrator", "Simon-2", "a b", "voix_é"] {
            assert!(check_name(ok).is_ok(), "{ok}");
        }
        assert!(check_name(&"a".repeat(MAX_NAME_BYTES)).is_ok());
        assert!(check_name(&"a".repeat(MAX_NAME_BYTES + 1)).is_err());
        for bad in [
            "", ".", "..", ".hidden", "a/b", "../x", "a\\b", "a\0b", "a\nb",
        ] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }

    fn voice(home: &Path, name: &str, files: &[&str]) {
        let d = dir(home).join(name);
        std::fs::create_dir_all(&d).unwrap();
        for f in files {
            std::fs::write(d.join(f), " hello there \n").unwrap();
        }
    }

    #[test]
    fn only_complete_voices_are_listed_and_found() {
        let home = tempfile::tempdir().unwrap();
        assert!(list(home.path()).is_empty());
        voice(home.path(), "zed", &[REF_WAV, REF_TXT]);
        voice(home.path(), "amy", &[REF_WAV, REF_TXT]);
        voice(home.path(), "no-text", &[REF_WAV]);
        voice(home.path(), "no-wav", &[REF_TXT]);
        voice(home.path(), ".tmp", &[REF_WAV, REF_TXT]);
        assert_eq!(list(home.path()), ["amy", "zed"]);

        assert_eq!(
            find(home.path(), "amy"),
            Some(Cloned {
                wav: dir(home.path()).join("amy").join(REF_WAV),
                text: "hello there".to_string(),
                // No `model.txt`: predates per-model voices.
                model: CLONE_MODEL.to_string(),
            })
        );
        // Outside voices/, even where both files exist.
        voice(&home.path().join("models"), "x", &[REF_WAV, REF_TXT]);
        for name in ["no-text", "no-wav", "nope", "../models/voices/x", ".tmp"] {
            assert_eq!(find(home.path(), name), None, "{name}");
        }
    }

    /// A blank `model.txt` (empty, or only whitespace) is treated the same
    /// as a missing one — `CLONE_MODEL` — not as a model literally named
    /// "", which nothing would ever resolve.
    #[test]
    fn a_blank_model_txt_is_clone_model_not_the_empty_string() {
        let home = tempfile::tempdir().unwrap();
        voice(home.path(), "empty", &[REF_WAV, REF_TXT]);
        std::fs::write(dir(home.path()).join("empty").join(MODEL_TXT), "").unwrap();
        voice(home.path(), "blank", &[REF_WAV, REF_TXT]);
        std::fs::write(
            dir(home.path()).join("blank").join(MODEL_TXT),
            " 
	",
        )
        .unwrap();
        for name in ["empty", "blank"] {
            assert_eq!(
                model_of(home.path(), name),
                Some(CLONE_MODEL.to_string()),
                "{name}"
            );
            assert_eq!(
                find(home.path(), name).unwrap().model,
                CLONE_MODEL,
                "{name}"
            );
        }
    }

    #[test]
    fn remove_deletes_the_directory_and_ignores_what_is_already_gone() {
        let home = tempfile::tempdir().unwrap();
        voice(home.path(), "amy", &[REF_WAV, REF_TXT]);
        voice(home.path(), "half", &[REF_WAV]);

        assert_eq!(remove(home.path(), "amy"), Ok(true));
        assert!(find(home.path(), "amy").is_none());
        assert!(!dir(home.path()).join("amy").exists());
        // Already gone, and never there: no error either way.
        assert_eq!(remove(home.path(), "amy"), Ok(false));
        assert_eq!(remove(home.path(), "nope"), Ok(false));
        // Incomplete voices are removed all the same.
        assert_eq!(remove(home.path(), "half"), Ok(true));
        // A bad name, as if path traversal were tried: not found either.
        assert_eq!(remove(home.path(), "../etc"), Ok(false));
    }

    #[test]
    fn only_a_cloning_manifest_gets_the_cloned_voices_made_for_it() {
        let home = tempfile::tempdir().unwrap();
        voice(home.path(), "amy", &[REF_WAV, REF_TXT]);
        // Made for "m", the manifest below: it must show up there.
        std::fs::write(dir(home.path()).join("amy").join(MODEL_TXT), "m").unwrap();
        voice(home.path(), "ryan", &[REF_WAV, REF_TXT]);
        // No `model.txt`: attributed to `CLONE_MODEL`, not "m".
        voice(home.path(), "other", &[REF_WAV, REF_TXT]);
        std::fs::write(dir(home.path()).join("other").join(MODEL_TXT), "not-m").unwrap();
        let manifest = |clone: bool| {
            Manifest::parse(
                &format!(
                    "[model]\nname = \"m\"\nkind = \"tts\"\nbackend = \"mlx\"\n\
                     [backend.mlx]\nclone = {clone}\n\
                     [[voice]]\nid = \"ryan\"\nsid = 0\ngender = \"m\"\ndefault = true\n"
                ),
                "test",
            )
            .unwrap()
        };
        let ids = |m: &Manifest| -> Vec<(String, Option<String>, bool)> {
            of(m, home.path())
                .into_iter()
                .map(|v| (v.id, v.gender, v.default))
                .collect()
        };
        let ryan = ("ryan".to_string(), Some("m".to_string()), true);
        assert_eq!(ids(&manifest(false)), std::slice::from_ref(&ryan));
        // The manifest's `ryan` shadows the cloned one; "other", made for a
        // different model, and "ryan"'s own cloned entry (shadowed) do not
        // appear, only "amy".
        assert_eq!(
            ids(&manifest(true)),
            [ryan, ("amy".to_string(), None, false)]
        );
    }

    #[test]
    fn say_asks_for_the_voice_own_model_only_for_a_cloned_voice() {
        let home = tempfile::tempdir().unwrap();
        // No `model.txt`: predates per-model voices, so `CLONE_MODEL`.
        voice(home.path(), "elise", &[REF_WAV, REF_TXT]);
        voice(home.path(), "designed", &[REF_WAV, REF_TXT]);
        std::fs::write(
            dir(home.path()).join("designed").join(MODEL_TXT),
            "breeze-tts-2-mlx",
        )
        .unwrap();
        let h = Some(home.path());
        assert_eq!(say_model(h, None, Some("elise")), CLONE_MODEL);
        // A voice made for a specific model asks for that one.
        assert_eq!(say_model(h, None, Some("designed")), "breeze-tts-2-mlx");
        // A built-in or unknown voice, or none, is the daemon's default.
        assert_eq!(say_model(h, None, Some("af_heart")), "default");
        assert_eq!(say_model(h, None, None), "default");
        assert_eq!(say_model(None, None, Some("elise")), "default");
        // An explicit model wins, even over a voice made for another one.
        assert_eq!(say_model(h, Some("kokoro"), Some("elise")), "kokoro");
        assert_eq!(say_model(h, Some("kokoro"), None), "kokoro");
    }

    /// A clip of `secs` seconds of a 440 Hz tone at 44.1 kHz stereo.
    #[cfg(target_os = "macos")]
    fn clip(path: &Path, secs: f64) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..(secs * 44_100.0) as u32 {
            let s = (f64::from(i) * 440.0 * std::f64::consts::TAU / 44_100.0).sin();
            let s = (s * 8000.0) as i16;
            w.write_sample(s).unwrap();
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn add_converts_to_24k_mono_and_checks_the_length() {
        let home = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let (short, ok, long) = (
            src.path().join("short.wav"),
            src.path().join("ok.wav"),
            src.path().join("long.wav"),
        );
        clip(&short, 2.0);
        clip(&ok, 6.0);
        clip(&long, 31.0);

        let secs = add(
            home.path(),
            "amy",
            &ok,
            "  Hello there.\n",
            "breeze-tts-2-mlx",
            true,
            None,
        )
        .unwrap();
        assert!((secs - 6.0).abs() < 0.05, "{secs}");
        let found = find(home.path(), "amy").unwrap();
        assert_eq!(found.text, "Hello there.");
        assert_eq!(found.model, "breeze-tts-2-mlx");
        let spec = hound::WavReader::open(&found.wav).unwrap().spec();
        assert_eq!((spec.channels, spec.sample_rate), (1, SAMPLE_RATE));

        let err = add(home.path(), "amy", &ok, "again", CLONE_MODEL, true, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("already exists"), "{err}");
        for (name, path) in [("short", &short), ("long", &long)] {
            let err = add(home.path(), name, path, "text", CLONE_MODEL, true, None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("it must be 3–30 s"), "{err}");
        }
        let err = add(
            home.path(),
            "junk",
            &src.path().join("nope.wav"),
            "t",
            CLONE_MODEL,
            true,
            None,
        )
        .unwrap_err()
        .to_string();
        let want = if cfg!(target_os = "macos") {
            "afconvert cannot read"
        } else {
            "cannot read"
        };
        assert!(err.contains(want), "{err}");
        assert!(add(home.path(), "../x", &ok, "t", CLONE_MODEL, true, None).is_err());
        assert!(add(home.path(), "blank", &ok, " \n", CLONE_MODEL, true, None).is_err());
        // Nothing half-made is left behind.
        assert_eq!(list(home.path()), ["amy"]);
        assert_eq!(
            std::fs::read_dir(home.path().join("tmp")).unwrap().count(),
            0
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn add_allows_a_blank_transcript_when_the_model_does_not_require_one() {
        // mesa task 1455: a model whose manifest carries no
        // `clone_requires_transcript` (or sets it false) can be cloned from
        // audio alone — `require_text: false` must not refuse a blank one,
        // though the blank transcript is still what gets stored.
        let home = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let ok = src.path().join("ok.wav");
        clip(&ok, 6.0);

        let secs = add(
            home.path(),
            "silent",
            &ok,
            "  \n",
            "some-model",
            false,
            None,
        )
        .unwrap();
        assert!((secs - 6.0).abs() < 0.05, "{secs}");
        let found = find(home.path(), "silent").unwrap();
        assert_eq!(found.text, "");
        assert_eq!(found.model, "some-model");
    }
}
