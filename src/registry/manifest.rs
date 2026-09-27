//! §3.2 manifests and the catalog: the built-in `catalog/*.toml` merged with
//! `$NARU_AUDIO_HOME/catalog.d/*.toml`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use super::RegistryError;

/// Built-in manifests, `(file name, contents)`, compiled in from `catalog/`.
const BUILTIN: &[(&str, &str)] = &[
    (
        "breeze-tts-2-mlx.toml",
        include_str!("../../catalog/breeze-tts-2-mlx.toml"),
    ),
    (
        "chatterbox-tts-8bit-mlx.toml",
        include_str!("../../catalog/chatterbox-tts-8bit-mlx.toml"),
    ),
    (
        "indextts-1.5-mlx.toml",
        include_str!("../../catalog/indextts-1.5-mlx.toml"),
    ),
    (
        "kokoro-v1.0.toml",
        include_str!("../../catalog/kokoro-v1.0.toml"),
    ),
    (
        "omnivoice-bf16-mlx.toml",
        include_str!("../../catalog/omnivoice-bf16-mlx.toml"),
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
    (
        "voxcpm2-8bit-mlx.toml",
        include_str!("../../catalog/voxcpm2-8bit-mlx.toml"),
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
    /// SPDX id or name, e.g. `Apache-2.0`. The built-in catalog always sets
    /// this; a `catalog.d` manifest may leave it unset.
    pub license: Option<String>,
    /// Where `license` can be read in full.
    pub license_url: Option<String>,
    /// The weights are restricted to non-commercial use (CC-BY-NC*,
    /// research-only, or otherwise), whatever `license` itself says.
    #[serde(default)]
    pub non_commercial: bool,
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

/// `backend.<model.backend>.prompt_format`: how a model reads style
/// guidance beyond the bare `instruct`/`exaggeration` booleans above —
/// hand-written documentation for a client, not something naru-audio
/// wires up itself. Every field is optional, so a manifest states only
/// what applies to that model.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptFormat {
    /// How the model reads `instructions` (§2.3): `"free_text"` (one
    /// prose string, e.g. Qwen3-TTS VoiceDesign), `"attributes"` (a
    /// comma-separated attribute list, e.g. OmniVoice), or
    /// `"inline_prefix"` (VoxCPM2's `(description)text`, where the
    /// description rides inside the spoken text rather than a separate
    /// field). Unset for a model that takes no `instructions` at all,
    /// even one with its own knobs (Chatterbox).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Tags placed inline in the text itself, distinct from `style`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<PromptFormatInline>,
    /// Generation knobs beyond `instructions`/`exaggeration`, e.g.
    /// `temperature` or `cfg_scale`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub knobs: Vec<PromptFormatKnob>,
    /// A short user-facing example, e.g. `"(cheerful, slightly
    /// faster)Hello there."`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// One inline tag syntax within [`PromptFormat`], e.g. OmniVoice's
/// `[laughter]` or Breeze's `(sigh)`/`[叹气]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptFormatInline {
    /// The tag's shape, e.g. `"[tag]"` or `"(tag)"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syntax: Option<String>,
    /// Known tags, without their brackets, e.g. `"laughter"`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// A short example using one of `tags`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
}

/// One generation knob within [`PromptFormat`]. `min`/`max` are set only
/// where the model or its docs actually give a range.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptFormatKnob {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
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

    /// `backend.<model.backend>.prompt_format`: how the model takes style
    /// guidance beyond `instruct`/`exaggeration` (§ [`PromptFormat`]), for
    /// a client to read next to `instructs()`. `None` when the manifest
    /// has no `prompt_format` table at all — every non-TTS model, since
    /// only a TTS manifest declares one. A TTS model with no style
    /// control at all (IndexTTS, Pocket TTS) still declares an *empty*
    /// table, so it comes back `Some(PromptFormat::default())` — a
    /// deliberate "nothing to declare" distinct from "not a TTS model".
    /// A malformed table (an unknown key, `deny_unknown_fields`, or the
    /// wrong shape) also comes back `None` here rather than failing the
    /// whole manifest, matching `instructs()` and friends above; the
    /// `builtin_prompt_format_is_declared_per_model_naru_1457` test below
    /// additionally asserts every catalog `prompt_format` table
    /// deserializes cleanly, so that silence never hides a typo.
    pub fn prompt_format(&self) -> Option<PromptFormat> {
        let table = self.backend.get(&self.model.backend)?;
        let value = table.get("prompt_format")?.clone();
        PromptFormat::deserialize(value).ok()
    }

    /// `backend.<model.backend>.exaggeration`: the model takes a request's
    /// `exaggeration` (§2.3 extra), which mlx-audio's Chatterbox calls
    /// `exaggeration`.
    pub fn exaggerates(&self) -> bool {
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("exaggeration"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// `backend.<model.backend>.clone_requires_transcript`: a cloning model
    /// (`clones()`) that will not clone a reference recording without its
    /// transcript too — Qwen3-TTS Base's in-context cloning needs both
    /// `ref_audio` and `ref_text` (mlx-audio's `qwen3_tts.py`: `use_icl =
    /// ref_audio is not None and ref_text is not None`), and Breeze's
    /// `generate` raises `"Breeze voice cloning requires ref_text with
    /// ref_audio."` outright (`breeze_tts.py`). Every other cloning model
    /// (Chatterbox, VoxCPM2, IndexTTS, OmniVoice) either has no `ref_text`
    /// parameter at all or reads it only if given, so this defaults to
    /// `false` for them; `naru-audio` still stores a transcript for every
    /// cloned voice regardless (`voices::add`), since one is cheap to keep
    /// and this flag is purely informational for a client choosing a model.
    pub fn clone_requires_transcript(&self) -> bool {
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("clone_requires_transcript"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// The model that a voice designed with this model (`instructs()`)
    /// ends up usable with: itself, for a model that both clones and
    /// instructs (VoxCPM2), so its own designed voice can be re-cloned;
    /// otherwise `backend.<backend>.design_voice_model`, the cloning model
    /// a design-only model (Qwen3-TTS VoiceDesign) saves its designed
    /// voices for; `None` if the model does not design at all, or design's
    /// not wired to any cloning model.
    pub fn design_voice_model(&self) -> Option<String> {
        if !self.instructs() {
            return None;
        }
        if self.clones() {
            return Some(self.model.name.clone());
        }
        self.backend
            .get(&self.model.backend)
            .and_then(|t| t.get("design_voice_model"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    }

    /// `[model] languages`, if the manifest declares any.
    pub fn languages(&self) -> Option<Vec<String>> {
        self.raw
            .get("model")?
            .get("languages")?
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
    }

    /// `[model] source`, if set explicitly, else the Hugging Face repo
    /// (`owner/repo`) derived from the first `huggingface.co` file or
    /// archive URL; `None` for a model with neither (e.g. sherpa-onnx's
    /// GitHub releases).
    pub fn source(&self) -> Option<String> {
        if let Some(s) = self
            .raw
            .get("model")
            .and_then(|v| v.get("source"))
            .and_then(|v| v.as_str())
        {
            return Some(s.to_string());
        }
        let url = self
            .archives
            .iter()
            .map(|a| a.url.as_str())
            .chain(self.files.iter().filter_map(|f| f.url.as_deref()))
            .next()?;
        let after = url.split_once("huggingface.co/")?.1;
        let mut parts = after.splitn(3, '/');
        let owner = parts.next()?;
        let repo = parts.next()?;
        Some(format!("{owner}/{repo}"))
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

        if let Some(url) = &self.model.license_url {
            check_url(url)?;
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
                "breeze-tts-2-mlx",
                "chatterbox-tts-8bit-mlx",
                "indextts-1.5-mlx",
                "kokoro-v1.0",
                "omnivoice-bf16-mlx",
                "parakeet-tdt-0.6b-v2-int8",
                "parakeet-tdt-0.6b-v2-mlx",
                "pocket-tts-int8",
                "qwen3-tts-0.6b-base-mlx",
                "qwen3-tts-0.6b-mlx",
                "qwen3-tts-1.7b-base-mlx",
                "qwen3-tts-1.7b-voicedesign-mlx",
                "silero-vad",
                "voxcpm2-8bit-mlx"
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
        assert!(m.clone_requires_transcript());
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
        assert!(m.clone_requires_transcript());
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
    fn builtin_chatterbox_clones_and_takes_exaggeration_but_not_instructions() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["chatterbox-tts-8bit-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert_eq!(m.model.license.as_deref(), Some("MIT"));
        assert!(!m.model.non_commercial);
        assert!(m.clones());
        assert!(!m.clone_requires_transcript());
        assert!(m.exaggerates());
        assert!(!m.instructs());
        assert!(m.voices.is_empty());
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with(
                    "https://huggingface.co/mlx-community/Chatterbox-TTS-8bit/resolve/"
                ) && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
        for other in [
            "qwen3-tts-0.6b-base-mlx",
            "qwen3-tts-0.6b-mlx",
            "qwen3-tts-1.7b-base-mlx",
            "qwen3-tts-1.7b-voicedesign-mlx",
            "kokoro-v1.0",
        ] {
            assert!(!cat.models[other].exaggerates(), "{other}");
        }
    }

    #[test]
    fn builtin_indextts_clones_and_has_no_extras() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["indextts-1.5-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert_eq!(m.model.license.as_deref(), Some("Apache-2.0"));
        assert!(!m.model.non_commercial);
        assert!(m.clones());
        assert!(!m.clone_requires_transcript());
        assert!(!m.exaggerates());
        assert!(!m.instructs());
        assert!(m.voices.is_empty());
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with("https://huggingface.co/mlx-community/IndexTTS-1.5/resolve/")
                    && url.ends_with(&format!("/{}", f.path)),
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
    fn builtin_voxcpm2_clones_and_instructs() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["voxcpm2-8bit-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert_eq!(m.model.license.as_deref(), Some("Apache-2.0"));
        assert!(!m.model.non_commercial);
        // The first model that both clones and designs a voice.
        assert!(m.clones());
        assert!(!m.clone_requires_transcript());
        assert!(m.instructs());
        assert!(!m.exaggerates());
        assert!(m.voices.is_empty());
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with("https://huggingface.co/mlx-community/VoxCPM2-8bit/resolve/")
                    && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn builtin_omnivoice_clones_and_instructs_and_is_non_commercial() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["omnivoice-bf16-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert_eq!(m.model.license.as_deref(), Some("CC-BY-NC-4.0"));
        assert_eq!(
            m.model.license_url.as_deref(),
            Some("https://creativecommons.org/licenses/by-nc/4.0/")
        );
        // Naru 1445: k2-fsa/OmniVoice's pre-trained weights are CC-BY-NC.
        assert!(m.model.non_commercial);
        assert!(m.clones());
        assert!(!m.clone_requires_transcript());
        assert!(m.instructs());
        assert!(!m.exaggerates());
        assert!(m.voices.is_empty());
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with("https://huggingface.co/mlx-community/OmniVoice-bf16/resolve/")
                    && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn builtin_breeze_clones_and_instructs_and_is_non_commercial() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();
        let m = &cat.models["breeze-tts-2-mlx"];
        assert_eq!(m.model.kind, Kind::Tts);
        assert_eq!(m.model.backend, "mlx");
        assert_eq!(
            m.model.license.as_deref(),
            Some("BreezeBlue Research and Non-Commercial License")
        );
        assert_eq!(
            m.model.license_url.as_deref(),
            Some(
                "https://huggingface.co/mlx-community/Breeze-TTS-2-mlx/blob/\
                 3c8829fb7fd335818f085cd2ef49b4100c0e46c8/LICENSE"
            )
        );
        // Naru 1446: BreezeBlue's own LICENSE grants research and
        // non-commercial use only.
        assert!(m.model.non_commercial);
        assert!(m.clones());
        assert!(m.clone_requires_transcript());
        assert!(m.instructs());
        assert!(!m.exaggerates());
        assert!(m.voices.is_empty());
        for f in &m.files {
            let url = f.url.as_deref().unwrap();
            assert!(
                url.starts_with("https://huggingface.co/mlx-community/Breeze-TTS-2-mlx/resolve/")
                    && url.ends_with(&format!("/{}", f.path)),
                "{url}"
            );
        }
    }

    #[test]
    fn builtin_prompt_format_is_declared_per_model_naru_1457() {
        let cat = Catalog::load(Path::new("/nonexistent/catalog.d")).unwrap();

        // Every TTS model declares a `prompt_format` table, even if empty;
        // only a non-TTS model has none at all.
        for name in [
            "parakeet-tdt-0.6b-v2-int8",
            "parakeet-tdt-0.6b-v2-mlx",
            "silero-vad",
        ] {
            assert!(
                cat.models[name].prompt_format().is_none(),
                "{name} should have no prompt_format"
            );
        }
        let tts_models: Vec<&str> = cat
            .models
            .values()
            .filter(|m| m.model.kind == Kind::Tts)
            .map(|m| m.model.name.as_str())
            .collect();
        assert!(!tts_models.is_empty());
        for name in &tts_models {
            let m = &cat.models[*name];
            // `prompt_format()` itself silently drops a table that fails
            // to deserialize (matching `instructs()` and friends, which
            // silently default rather than fail the whole manifest); this
            // asserts the raw table underneath actually deserializes
            // clean, so a typo'd key cannot hide behind that silence.
            let raw = m
                .backend
                .get(&m.model.backend)
                .and_then(|t| t.get("prompt_format"))
                .unwrap_or_else(|| panic!("{name} (TTS) has no [backend.*.prompt_format] table"))
                .clone();
            PromptFormat::deserialize(raw)
                .unwrap_or_else(|e| panic!("{name}'s prompt_format does not deserialize: {e}"));
            assert!(
                m.prompt_format().is_some(),
                "{name} (TTS) should have a prompt_format"
            );
        }

        let voicedesign = cat.models["qwen3-tts-1.7b-voicedesign-mlx"]
            .prompt_format()
            .unwrap();
        assert_eq!(voicedesign.style.as_deref(), Some("free_text"));
        assert!(voicedesign.inline.is_none());
        assert_eq!(
            voicedesign
                .knobs
                .iter()
                .map(|k| k.name.as_str())
                .collect::<Vec<_>>(),
            ["temperature", "top_p"]
        );

        // No `instruct`, but its own knobs are still declared.
        let chatterbox = cat.models["chatterbox-tts-8bit-mlx"]
            .prompt_format()
            .unwrap();
        assert!(chatterbox.style.is_none());
        assert_eq!(chatterbox.knobs[0].name, "exaggeration");
        assert_eq!(chatterbox.knobs[0].default, Some(0.1));

        // Style rides inline in the text, not a separate field.
        let voxcpm2 = cat.models["voxcpm2-8bit-mlx"].prompt_format().unwrap();
        assert_eq!(voxcpm2.style.as_deref(), Some("inline_prefix"));
        assert_eq!(
            voxcpm2.inline.unwrap().syntax.as_deref(),
            Some("(description)text")
        );
        // The `[[file]]` table right after `prompt_format` in the same
        // manifest is still intact: this regressed once (naru_1457),
        // deleting Breeze's LICENSE `[[file]]` when its prompt_format was
        // inserted above it.
        let breeze = &cat.models["breeze-tts-2-mlx"];
        assert!(
            breeze.files.iter().any(|f| f.path == "LICENSE"),
            "{:?}",
            breeze.files
        );

        let omnivoice = cat.models["omnivoice-bf16-mlx"].prompt_format().unwrap();
        assert_eq!(omnivoice.style.as_deref(), Some("attributes"));
        assert!(
            omnivoice
                .inline
                .unwrap()
                .tags
                .contains(&"laughter".to_string())
        );

        let breeze_pf = breeze.prompt_format().unwrap();
        assert_eq!(breeze_pf.style.as_deref(), Some("free_text"));
        assert!(breeze_pf.inline.unwrap().tags.contains(&"sigh".to_string()));

        let kokoro = cat.models["kokoro-v1.0"].prompt_format().unwrap();
        assert!(kokoro.style.is_none());
        assert_eq!(
            kokoro.knobs,
            [PromptFormatKnob {
                name: "speed".to_string(),
                default: Some(1.0),
                min: None,
                max: None,
            }]
        );

        // The Qwen3-TTS Base pair and CustomVoice take no `instruct`, but
        // still expose `Model.generate`'s own temperature/top_p.
        for name in [
            "qwen3-tts-0.6b-mlx",
            "qwen3-tts-0.6b-base-mlx",
            "qwen3-tts-1.7b-base-mlx",
        ] {
            let pf = cat.models[name].prompt_format().unwrap();
            assert!(pf.style.is_none(), "{name}");
            assert_eq!(
                pf.knobs.iter().map(|k| k.name.as_str()).collect::<Vec<_>>(),
                ["temperature", "top_p"],
                "{name}"
            );
        }

        // IndexTTS and Pocket TTS have no style control at all, but
        // declare that explicitly as an empty table — `Some(default())`,
        // not `None` — so a client can tell "TTS, nothing to declare"
        // apart from "not a TTS model".
        for name in ["indextts-1.5-mlx", "pocket-tts-int8"] {
            let pf = cat.models[name].prompt_format().unwrap();
            assert_eq!(pf.style, None, "{name}");
            assert!(pf.inline.is_none(), "{name}");
            assert!(pf.knobs.is_empty(), "{name}");
            assert!(pf.hint.is_none(), "{name}");
            // Serializes as `{}`, distinguishable from `null`.
            assert_eq!(
                serde_json::to_value(&pf).unwrap(),
                serde_json::json!({}),
                "{name}"
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
