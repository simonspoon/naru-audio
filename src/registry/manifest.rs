//! §3.2 manifests and the catalog: the built-in `catalog/*.toml` merged with
//! `$NARU_AUDIO_HOME/catalog.d/*.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::Deserialize;

use super::RegistryError;

/// Built-in manifests, `(file name, contents)`, compiled in from `catalog/`.
const BUILTIN: &[(&str, &str)] = &[
    (
        "kokoro-v1.0.toml",
        include_str!("../../catalog/kokoro-v1.0.toml"),
    ),
    (
        "parakeet-tdt-0.6b-v2-int8.toml",
        include_str!("../../catalog/parakeet-tdt-0.6b-v2-int8.toml"),
    ),
    (
        "parakeet-tdt-0.6b-v2-mlx.toml",
        include_str!("../../catalog/parakeet-tdt-0.6b-v2-mlx.toml"),
    ),
    (
        "pocket-tts-int8.toml",
        include_str!("../../catalog/pocket-tts-int8.toml"),
    ),
    (
        "qwen3-tts-0.6b-base-mlx.toml",
        include_str!("../../catalog/qwen3-tts-0.6b-base-mlx.toml"),
    ),
    (
        "qwen3-tts-0.6b-mlx.toml",
        include_str!("../../catalog/qwen3-tts-0.6b-mlx.toml"),
    ),
    (
        "qwen3-tts-1.7b-base-mlx.toml",
        include_str!("../../catalog/qwen3-tts-1.7b-base-mlx.toml"),
    ),
    (
        "qwen3-tts-1.7b-voicedesign-mlx.toml",
        include_str!("../../catalog/qwen3-tts-1.7b-voicedesign-mlx.toml"),
    ),
    (
        "silero-vad.toml",
        include_str!("../../catalog/silero-vad.toml"),
    ),
];

/// Derived files the registry knows how to generate (§3.2 `derive`).
pub const BPE_VOCAB: &str = "bpe.vocab";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Stt,
    Tts,
    Vad,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Stt => "stt",
            Kind::Tts => "tts",
            Kind::Vad => "vad",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub name: String,
    pub kind: Kind,
    pub backend: String,
    /// `<os>-<arch>` pairs such as `macos-arm64`; empty means every platform.
    #[serde(default)]
    pub platforms: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
}

/// A `[[file]]`. Without a `url` it comes out of an `[[archive]]` and is
/// checked after extraction.
#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub url: Option<String>,
    /// Optional only so the loader can reject its absence with a clear error.
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArchiveEntry {
    pub url: String,
    pub sha256: Option<String>,
    pub size: Option<u64>,
    /// Leading path components dropped on extraction, like `tar --strip-components`.
    #[serde(default)]
    pub strip: usize,
}

/// A TTS `[[voice]]`: the API's name for a speaker and the `sid` sherpa
/// addresses it by (§5.1).
#[derive(Debug, Clone, Deserialize)]
pub struct Voice {
    pub id: String,
    pub sid: i32,
    /// A voice-cloning model's reference recording (Pocket TTS), relative
    /// to the model directory; it must be a [[file]].
    pub reference: Option<String>,
    pub accent: Option<String>,
    pub gender: Option<String>,
    #[serde(default)]
    pub default: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: Model,
    #[serde(default)]
    pub backend: BTreeMap<String, toml::Table>,
    #[serde(default, rename = "file")]
    pub files: Vec<FileEntry>,
    #[serde(default, rename = "archive")]
    pub archives: Vec<ArchiveEntry>,
    #[serde(default, rename = "voice")]
    pub voices: Vec<Voice>,
    /// The whole document, `[[voice]]` and all; written out as `manifest.json`.
    #[serde(skip)]
    pub raw: toml::Table,
}

impl Manifest {
    /// Parses and validates one manifest. `origin` names it in errors.
    pub fn parse(text: &str, origin: &str) -> Result<Self, RegistryError> {
        let err = |message: String| RegistryError::Manifest {
            origin: origin.to_string(),
            message,
        };
        let raw: toml::Table = toml::from_str(text).map_err(|e| err(e.to_string()))?;
        Self::from_table(raw, origin)
    }

    /// Validates an already-parsed manifest, e.g. a pulled `manifest.json`.
    pub fn from_table(raw: toml::Table, origin: &str) -> Result<Self, RegistryError> {
        let err = |message: String| RegistryError::Manifest {
            origin: origin.to_string(),
            message,
        };
        let mut m = Manifest::deserialize(raw.clone()).map_err(|e| err(e.to_string()))?;
        m.raw = raw;
        m.validate().map_err(err)?;
        Ok(m)
    }

    /// §2.5 "can run on this machine": `platforms` names this OS and arch.
    pub fn runs_here(&self) -> bool {
        self.runs_on(std::env::consts::OS, std::env::consts::ARCH)
    }

    /// [`Manifest::runs_here`] on `os` and `arch`, as `std::env::consts`
    /// names them.
    pub fn runs_on(&self, os: &str, arch: &str) -> bool {
        let arch = match arch {
            "aarch64" => "arm64",
            other => other,
        };
        let here = format!("{os}-{arch}");
        self.model.platforms.is_empty() || self.model.platforms.contains(&here)
    }

    /// `backend.<model.backend>.derive`, e.g. `["bpe.vocab"]`.
    pub fn derive(&self) -> Vec<String> {
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("derive"))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `backend.<model.backend>.clone`: the model speaks in the cloned
    /// voices under `$NARU_AUDIO_HOME/voices/` (`crate::voices`).
    pub fn clones(&self) -> bool {
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("clone"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// `backend.<model.backend>.instruct`: the model takes a request's
    /// `instructions` (§2.3), which mlx-audio calls `instruct`.
    pub fn instructs(&self) -> bool {
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("instruct"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    fn validate(&mut self) -> Result<(), String> {
        let name = &self.model.name;
        if !is_relative_path(name) || name.contains('/') {
            return Err(format!("model name `{name}` is not a plain directory name"));
        }
        // It would collide with another model's `tmp/<name>.lock`.
        if name.ends_with(".lock") {
            return Err(format!("model name `{name}` must not end in `.lock`"));
        }

        // §3.2: every file and archive has a sha256, no exceptions.
        for a in &mut self.archives {
            a.sha256 = Some(check_sha(a.sha256.take(), &format!("archive `{}`", a.url))?);
            check_url(&a.url)?;
            if !(a.url.ends_with(".tar.bz2") || a.url.ends_with(".tar")) {
                return Err(format!("archive `{}` must be a .tar.bz2 or .tar", a.url));
            }
        }
        let mut paths = BTreeSet::new();
        for f in &mut self.files {
            f.sha256 = Some(check_sha(f.sha256.take(), &format!("file `{}`", f.path))?);
            if !is_relative_path(&f.path) {
                return Err(format!("file path `{}` must be relative", f.path));
            }
            if !paths.insert(f.path.clone()) {
                return Err(format!("file `{}` is listed twice", f.path));
            }
            match &f.url {
                Some(url) => check_url(url)?,
                None if self.archives.is_empty() => {
                    return Err(format!(
                        "file `{}` has no url and the manifest has no [[archive]]",
                        f.path
                    ));
                }
                None => {}
            }
        }

        for v in &self.voices {
            if let Some(r) = &v.reference
                && !paths.contains(r)
            {
                return Err(format!(
                    "voice `{}` reference `{r}` is not a [[file]]",
                    v.id
                ));
            }
        }

        let has = |p: &str| paths.contains(p);
        for d in self.derive() {
            if d != BPE_VOCAB {
                return Err(format!("unknown derived file `{d}`"));
            }
            if has(&d) {
                return Err(format!("`{d}` is derived, so it cannot also be a [[file]]"));
            }
            if !has("tokens.txt") {
                return Err(format!(
                    "`{d}` is derived from tokens.txt, which is not a [[file]]"
                ));
            }
        }
        Ok(())
    }
}

fn check_sha(sha: Option<String>, what: &str) -> Result<String, String> {
    let Some(sha) = sha else {
        return Err(format!(
            "{what} has no sha256; every file and archive must pin one"
        ));
    };
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{what} has a malformed sha256 `{sha}`"));
    }
    Ok(sha.to_ascii_lowercase())
}

fn check_url(url: &str) -> Result<(), String> {
    if ["https://", "http://", "file:///"]
        .iter()
        .any(|s| url.starts_with(s))
    {
        Ok(())
    } else {
        Err(format!("url `{url}` must be https://, http:// or file:///"))
    }
}

/// Non-empty and made only of normal components: no `/`, `..`, or `.`.
fn is_relative_path(p: &str) -> bool {
    !p.is_empty()
        && Path::new(p)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// Every known manifest by model name, `catalog.d` over the built-in catalog.
#[derive(Debug, Default)]
pub struct Catalog {
    pub models: BTreeMap<String, Manifest>,
}

impl Catalog {
    /// Loads the built-in catalog, then `catalog_d/*.toml` over it. A missing
    /// `catalog_d` is fine; an invalid manifest anywhere is an error.
    pub fn load(catalog_d: &Path) -> Result<Self, RegistryError> {
        let mut cat = Catalog::default();
        let add = |layer: &mut BTreeMap<String, Manifest>, text: &str, origin: String| {
            let m = Manifest::parse(text, &origin)?;
            if layer.contains_key(&m.model.name) {
                return Err(RegistryError::Manifest {
                    origin,
                    message: format!("model `{}` is defined twice", m.model.name),
                });
            }
            layer.insert(m.model.name.clone(), m);
            Ok(())
        };

        for (file, text) in BUILTIN {
            add(&mut cat.models, text, format!("catalog/{file}"))?;
        }

        let mut user = BTreeMap::new();
        let entries = match std::fs::read_dir(catalog_d) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(cat),
            Err(e) => {
                return Err(RegistryError::io(format!("read {}", catalog_d.display()))(
                    e,
                ));
            }
        };
        let mut paths = Vec::new();
        for entry in entries {
            let path = entry
                .map_err(RegistryError::io(format!("read {}", catalog_d.display())))?
                .path();
            if path.extension().is_some_and(|e| e == "toml") {
                paths.push(path);
            }
        }
        paths.sort();
        for path in paths {
            let text = std::fs::read_to_string(&path)
                .map_err(RegistryError::io(format!("read {}", path.display())))?;
            add(&mut user, &text, path.display().to_string())?;
        }
        cat.models.extend(user);
        Ok(cat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_pins_parakeet_and_its_vad() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        assert_eq!(
            cat.models.keys().collect::<Vec<_>>(),
            [
                "kokoro-v1.0",
                "parakeet-tdt-0.6b-v2-int8",
                "parakeet-tdt-0.6b-v2-mlx",
                "pocket-tts-int8",
                "qwen3-tts-0.6b-base-mlx",
                "qwen3-tts-0.6b-mlx",
                "qwen3-tts-1.7b-base-mlx",
                "qwen3-tts-1.7b-voicedesign-mlx",
                "silero-vad"
            ]
        );
        for m in cat.models.values() {
            assert!(!m.files.is_empty(), "{}", m.model.name);
            for f in &m.files {
                assert!(f.size.is_some_and(|s| s > 0), "{}", f.path);
                match &f.url {
                    Some(u) => assert!(u.starts_with("https://"), "{u}"),
                    None => assert!(!m.archives.is_empty(), "{}", f.path),
                }
            }
            for r in &m.model.requires {
                assert!(
                    cat.models.contains_key(r),
                    "{} requires unknown {r}",
                    m.model.name
                );
            }
        }

        let stt = &cat.models["parakeet-tdt-0.6b-v2-int8"];
        assert_eq!(stt.model.kind, Kind::Stt);
        assert_eq!(stt.model.requires, ["silero-vad"]);
        assert_eq!(stt.derive(), [BPE_VOCAB]);
        assert_eq!(stt.files.len(), 4);
        let mlx = &cat.models["parakeet-tdt-0.6b-v2-mlx"];
        assert_eq!(mlx.model.kind, Kind::Stt);
        assert_eq!(mlx.model.backend, "mlx");
        assert_eq!(mlx.model.requires, ["silero-vad"]);
        assert_eq!(mlx.files.len(), 2);
        assert_eq!(cat.models["silero-vad"].model.kind, Kind::Vad);
    }

    #[test]
    fn builtin_kokoro_pins_54_voices_with_af_heart_default() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["kokoro-v1.0"];
        assert_eq!(m.model.kind, Kind::Tts);
        for a in &m.archives {
            assert!(a.url.starts_with("https://"), "{}", a.url);
            assert!(a.size.is_some_and(|s| s > 0), "{}", a.url);
        }
        assert_eq!(
            m.raw["backend"]["sherpa-onnx"]["lang"].as_str(),
            Some("en-us")
        );
        for p in ["model.onnx", "voices.bin", "tokens.txt"] {
            assert!(m.files.iter().any(|f| f.path == p), "{p}");
        }
        assert!(
            m.files
                .iter()
                .any(|f| f.path.starts_with("espeak-ng-data/"))
        );

        let voices = m.raw["voice"].as_array().unwrap();
        let ids: BTreeSet<&str> = voices.iter().map(|v| v["id"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), 54);
        let mut sids: Vec<i64> = voices
            .iter()
            .map(|v| v["sid"].as_integer().unwrap())
            .collect();
        sids.sort();
        assert_eq!(sids, (0..54).collect::<Vec<_>>());
        let defaults: Vec<&str> = voices
            .iter()
            .filter(|v| v.get("default").and_then(|d| d.as_bool()) == Some(true))
            .map(|v| v["id"].as_str().unwrap())
            .collect();
        assert_eq!(defaults, ["af_heart"]);
        // The model's speaker2id order, not alphabetical: em_santa is last.
        let sid = |id: &str| {
            voices
                .iter()
                .find(|v| v["id"].as_str() == Some(id))
                .and_then(|v| v["sid"].as_integer())
        };
        assert_eq!(sid("af_heart"), Some(3));
        assert_eq!(sid("em_santa"), Some(53));
        // The typed table agrees with the raw one.
        assert_eq!(m.voices.len(), 54);
        let heart = m.voices.iter().find(|v| v.id == "af_heart").unwrap();
        assert_eq!((heart.sid, heart.default), (3, true));
    }

    #[test]
    fn builtin_pocket_clones_two_reference_voices() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["pocket-tts-int8"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(
            m.raw["backend"]["sherpa-onnx"]["family"].as_str(),
            Some("pocket")
        );
        let voices: Vec<(&str, i32, Option<&str>, bool)> = m
            .voices
            .iter()
            .map(|v| (v.id.as_str(), v.sid, v.reference.as_deref(), v.default))
            .collect();
        assert_eq!(
            voices,
            [
                ("bria", 0, Some("test_wavs/bria.wav"), true),
                ("loona", 0, Some("test_wavs/loona.wav"), false),
            ]
        );
        // Kokoro's voices have no reference.
        assert!(
            cat.models["kokoro-v1.0"]
                .voices
                .iter()
                .all(|v| v.reference.is_none())
        );
    }

    #[test]
    fn builtin_qwen3_tts_is_mlx_with_ryan_default() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["qwen3-tts-0.6b-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert!(m.runs_on("macos", "aarch64"));
        assert!(!m.runs_on("linux", "aarch64"));
        for p in ["model.safetensors", "speech_tokenizer/model.safetensors"] {
            assert!(m.files.iter().any(|f| f.path == p), "{p}");
        }
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(url.ends_with(&format!("/{}", f.path)), "{url}");
        }
        let ids: Vec<&str> = m.voices.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric",
                "dylan"
            ]
        );
        let defaults: Vec<&str> = m
            .voices
            .iter()
            .filter(|v| v.default)
            .map(|v| v.id.as_str())
            .collect();
        assert_eq!(defaults, ["ryan"]);
    }

    #[test]
    fn builtin_qwen3_tts_base_clones_and_has_no_preset_voices() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["qwen3-tts-0.6b-base-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert!(m.clones());
        assert!(m.voices.is_empty());
        // The same files as CustomVoice's, from the Base repo.
        let custom = &cat.models["qwen3-tts-0.6b-mlx"];
        assert!(!custom.clones());
        let paths = |m: &Manifest| m.files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(m), paths(custom));
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with(
                    "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit/resolve/"
                ) && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn builtin_qwen3_tts_1_7b_base_clones_and_has_no_preset_voices() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["qwen3-tts-1.7b-base-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert!(m.clones());
        assert!(!m.instructs());
        assert!(m.voices.is_empty());
        // The same files as the 0.6B Base's, from the 1.7B Base repo.
        let paths = |m: &Manifest| m.files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(m), paths(&cat.models["qwen3-tts-0.6b-base-mlx"]));
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with(
                    "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit/resolve/"
                ) && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn builtin_qwen3_tts_voicedesign_instructs_and_has_no_voices() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["qwen3-tts-1.7b-voicedesign-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert!(m.instructs());
        assert!(!m.clones());
        assert!(m.voices.is_empty());
        // Only VoiceDesign takes `instructions`.
        for other in [
            "qwen3-tts-0.6b-mlx",
            "qwen3-tts-0.6b-base-mlx",
            "qwen3-tts-1.7b-base-mlx",
            "kokoro-v1.0",
        ] {
            assert!(!cat.models[other].instructs(), "{other}");
        }
        // The same files as the 0.6B models', from the VoiceDesign repo.
        let paths = |m: &Manifest| m.files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths(m), paths(&cat.models["qwen3-tts-0.6b-mlx"]));
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with(
                    "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/resolve/"
                ) && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn a_voice_reference_must_be_a_pinned_file() {
        let text = include_str!("../../catalog/pocket-tts-int8.toml").replace(
            "test_wavs/loona.wav\"\ngender",
            "test_wavs/other.wav\"\ngender",
        );
        let err = Manifest::parse(&text, "pocket").unwrap_err().to_string();
        assert!(
            err.contains("reference `test_wavs/other.wav` is not a [[file]]"),
            "{err}"
        );
    }
}
