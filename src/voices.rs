//! Cloned voices: `$NARU_AUDIO_HOME/voices/<name>/` holds `ref.wav`, a
//! short clip of the voice, and `ref.txt`, what it says. A model whose
//! manifest sets `[backend.<backend>] clone = true` (Qwen3-TTS Base) speaks
//! in each of them as voice `<name>`, after its own `[[voice]]`s. They are
//! read from disk on each request, so a voice added while the daemon runs
//! is usable at once.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::registry::manifest::{Manifest, Voice};

pub const REF_WAV: &str = "ref.wav";
pub const REF_TXT: &str = "ref.txt";

/// Clip lengths `add` accepts, in seconds; 5–15 s is the target.
pub const MIN_SECS: f64 = 3.0;
pub const MAX_SECS: f64 = 30.0;

/// `ref.wav`'s rate: Qwen3-TTS's (mlx-audio resamples to the model's
/// rate anyway).
const SAMPLE_RATE: u32 = 24_000;

/// The cloning model `say` asks for when its voice is a cloned one.
pub const CLONE_MODEL: &str = "qwen3-tts-0.6b-base-mlx";

/// A cloned voice's clip and transcript.
#[derive(Debug, PartialEq)]
pub struct Cloned {
    pub wav: PathBuf,
    pub text: String,
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
pub fn find(home: &Path, name: &str) -> Option<Cloned> {
    check_name(name).ok()?;
    let voice = dir(home).join(name);
    let wav = voice.join(REF_WAV);
    let text = std::fs::read_to_string(voice.join(REF_TXT)).ok()?;
    wav.is_file().then(|| Cloned {
        wav,
        text: text.trim().to_string(),
    })
}

/// `manifest`'s voices, then, if it clones, the cloned voices in `home`
/// that no `[[voice]]` shadows.
pub fn of(manifest: &Manifest, home: &Path) -> Vec<Voice> {
    let mut voices = manifest.voices.clone();
    if manifest.clones() {
        for id in list(home) {
            if !voices.iter().any(|v| v.id == id) {
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
    }
    voices
}

/// The model `say` asks for: `model` if given, else [`CLONE_MODEL`] if
/// `voice` is a cloned voice in `home`, else the daemon's `default`.
pub fn say_model<'a>(home: Option<&Path>, model: Option<&'a str>, voice: Option<&str>) -> &'a str {
    match (model, voice) {
        (Some(model), _) => model,
        (None, Some(voice)) if home.is_some_and(|h| find(h, voice).is_some()) => CLONE_MODEL,
        _ => "default",
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
/// transcript. Returns the clip's length in seconds, which must be within
/// `MIN_SECS..=MAX_SECS`. An existing voice is not replaced. Both `voice
/// add` and `POST /v1/audio/voices` come here.
pub fn add(home: &Path, name: &str, clip: &Path, text: &str) -> Result<f64, AddError> {
    check_name(name).map_err(AddError::Name)?;
    let text = text.trim();
    if text.is_empty() {
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

/// `afconvert`s `clip` to `wav` and returns its length in seconds.
fn convert(clip: &Path, wav: &Path) -> Result<f64, AddError> {
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
            })
        );
        // Outside voices/, even where both files exist.
        voice(&home.path().join("models"), "x", &[REF_WAV, REF_TXT]);
        for name in ["no-text", "no-wav", "nope", "../models/voices/x", ".tmp"] {
            assert_eq!(find(home.path(), name), None, "{name}");
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
    fn only_a_cloning_manifest_gets_the_cloned_voices() {
        let home = tempfile::tempdir().unwrap();
        voice(home.path(), "amy", &[REF_WAV, REF_TXT]);
        voice(home.path(), "ryan", &[REF_WAV, REF_TXT]);
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
        // The manifest's `ryan` shadows the cloned one.
        assert_eq!(
            ids(&manifest(true)),
            [ryan, ("amy".to_string(), None, false)]
        );
    }

    #[test]
    fn say_asks_for_the_cloning_model_only_for_a_cloned_voice() {
        let home = tempfile::tempdir().unwrap();
        voice(home.path(), "elise", &[REF_WAV, REF_TXT]);
        let h = Some(home.path());
        assert_eq!(say_model(h, None, Some("elise")), CLONE_MODEL);
        // A built-in or unknown voice, or none, is the daemon's default.
        assert_eq!(say_model(h, None, Some("af_heart")), "default");
        assert_eq!(say_model(h, None, None), "default");
        assert_eq!(say_model(None, None, Some("elise")), "default");
        // An explicit model wins.
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

        let secs = add(home.path(), "amy", &ok, "  Hello there.\n").unwrap();
        assert!((secs - 6.0).abs() < 0.05, "{secs}");
        let found = find(home.path(), "amy").unwrap();
        assert_eq!(found.text, "Hello there.");
        let spec = hound::WavReader::open(&found.wav).unwrap().spec();
        assert_eq!((spec.channels, spec.sample_rate), (1, SAMPLE_RATE));

        let err = add(home.path(), "amy", &ok, "again")
            .unwrap_err()
            .to_string();
        assert!(err.contains("already exists"), "{err}");
        for (name, path) in [("short", &short), ("long", &long)] {
            let err = add(home.path(), name, path, "text")
                .unwrap_err()
                .to_string();
            assert!(err.contains("it must be 3–30 s"), "{err}");
        }
        let err = add(home.path(), "junk", &src.path().join("nope.wav"), "t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("afconvert cannot read"), "{err}");
        assert!(add(home.path(), "../x", &ok, "t").is_err());
        assert!(add(home.path(), "blank", &ok, " \n").is_err());
        // Nothing half-made is left behind.
        assert_eq!(list(home.path()), ["amy"]);
        assert_eq!(
            std::fs::read_dir(home.path().join("tmp")).unwrap().count(),
            0
        );
    }
}
