//! The `mlx` [`TtsModel`] (§5.3), in-process through mlx-rs: 8-bit Breeze
//! (`crate::mlx::breeze`, naru tasks 2381-2384) and the two Qwen3-TTS Base
//! checkpoints (`crate::mlx::qwen3_tts`, naru tasks 2386-2389). Any other
//! `mlx` model has no engine and does not load.
//!
//! Breeze takes `instructions`, `seed` and the `cfg_scale`, `temperature`,
//! `top_p`, `top_k`, `repetition_penalty` and `max_tokens` knobs as
//! mlx-audio does; a voice is its `[[voice]]` id, a cloned voice (or a
//! preview's clip) goes as `ref_audio` and `ref_text` (naru task 2382). It
//! has no `speed`: anything but 1.0 is an error.
//!
//! Qwen3-TTS speaks only cloned voices: the clip (or a preview's) and its
//! transcript are the reference, sampled with `CLONE_SAMPLING` unless the
//! `temperature`, `top_p`, `top_k`, `repetition_penalty` or `max_tokens`
//! knobs say otherwise; `speed` is ignored, as mlx-audio's Qwen3-TTS
//! ignores it.
//!
//! Both stream 0.32 s chunks as they are made (naru task 2388). No gap and
//! no leveller, as for Pocket (`super::sherpa`): the chunks are fixed
//! lengths of audio cut mid-sentence, so a gap would be a pause mid-word,
//! and a gain per chunk would step in level from one to the next. `gap` is
//! still checked, so a request means the same whatever model serves it.

use std::path::{Path, PathBuf};

use super::sherpa::check;
use super::{Sink, SynthOptions, TtsError, TtsModel};
use crate::mlx::breeze::{self, Breeze};
use crate::mlx::qwen3_tts::{self, Qwen3Tts};
use crate::registry::manifest::{Manifest, Voice};
use crate::voices;

pub struct MlxTts {
    engine: Engine,
    name: String,
    dir: PathBuf,
    voices: Vec<Voice>,
    /// The home whose cloned voices the model speaks in, if it clones.
    cloned: Option<PathBuf>,
    /// Whether the model takes `instructions`.
    instructs: bool,
    /// `[mlx] breeze_depth_cache`, for Breeze.
    depth_cache: bool,
}

enum Engine {
    Breeze(Breeze),
    Qwen3(Qwen3Tts),
}

impl MlxTts {
    /// `dir` is `models/<name>/`, in the home whose voices it speaks (§3.1).
    pub fn load(manifest: &Manifest, dir: &Path) -> Result<Self, TtsError> {
        let home = dir.parent().and_then(Path::parent).unwrap_or(dir);
        let name = manifest.model.name.clone();
        // A failed load is 503 `backend_unavailable`, as for STT.
        let unavailable = |reason: String| TtsError::BackendUnavailable {
            backend: manifest.model.backend.clone(),
            reason,
        };
        let engine = if name.starts_with("breeze") {
            // `Breeze::load` refuses all but the 8-bit checkpoint.
            Engine::Breeze(Breeze::load(dir).map_err(unavailable)?)
        } else if matches!(
            name.as_str(),
            "qwen3-tts-0.6b-base-mlx" | "qwen3-tts-1.7b-base-mlx"
        ) {
            Engine::Qwen3(Qwen3Tts::load(dir).map_err(unavailable)?)
        } else {
            return Err(unavailable(format!("no native MLX engine for {name}")));
        };
        let depth_cache = match engine {
            Engine::Breeze(_) => crate::mlx::breeze_depth_cache(home).map_err(unavailable)?,
            Engine::Qwen3(_) => false,
        };
        Ok(MlxTts {
            engine,
            name,
            dir: dir.to_path_buf(),
            voices: manifest.voices.clone(),
            cloned: manifest.clones().then(|| home.to_path_buf()),
            instructs: manifest.instructs(),
            depth_cache,
        })
    }
}

impl TtsModel for MlxTts {
    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn sample_rate(&self) -> u32 {
        breeze::SAMPLE_RATE
    }

    fn synth(
        &self,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        sink: Sink,
    ) -> Result<(), TtsError> {
        check(text, options)?;
        match &self.engine {
            Engine::Breeze(model) => self.synth_breeze(model, text, voice, options, sink),
            Engine::Qwen3(model) => self.synth_qwen3(model, text, voice, options, sink),
        }
    }

    fn resident_bytes(&self) -> Option<u64> {
        Some(match &self.engine {
            Engine::Breeze(model) => model.resident_bytes(),
            Engine::Qwen3(model) => model.resident_bytes(),
        })
    }

    fn engine(&self) -> Option<(&'static str, Option<&str>)> {
        Some(("native", None))
    }
}

impl MlxTts {
    /// `synth` on Breeze: mlx-audio's `generate` arguments, as
    /// [`breeze::Params`].
    fn synth_breeze(
        &self,
        model: &Breeze,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        mut sink: Sink,
    ) -> Result<(), TtsError> {
        // Breeze has no `speed`; an error rather than normal-speed audio.
        if options.speed != 1.0 {
            return Err(TtsError::Native(format!(
                "{} does not support speed (got {}); only the default, 1.0, is accepted",
                self.name, options.speed
            )));
        }
        let mut params = breeze::Params::default();
        if let Some((wav, ref_text)) = &options.reference {
            params.reference = Some((wav.clone(), ref_text.clone()));
        } else if let Some(v) = self.voices.iter().find(|v| v.id == voice) {
            params.speaker = v.id.clone();
            if let Some(reference) = &v.reference {
                // No transcript: Breeze refuses this, as mlx-audio does.
                params.reference = Some((self.dir.join(reference), String::new()));
            }
        } else if let Some(c) = self
            .cloned
            .as_deref()
            .and_then(|home| voices::find(home, voice))
        {
            params.reference = Some((c.wav, c.text));
        } else if !(self.instructs && self.voices.is_empty()) {
            return Err(TtsError::UnknownVoice {
                model: self.name.clone(),
                voice: voice.to_string(),
            });
        }
        if self.instructs {
            params.instruct = options.instructions.clone();
        }
        params.seed = Some(options.seed);
        params.depth_cache = self.depth_cache;
        for (name, &value) in &options.knobs {
            match name.as_str() {
                "cfg_scale" => params.cfg_scale = Some(value),
                "temperature" => params.temperature = value,
                "top_p" => params.top_p = value,
                "top_k" => params.top_k = value as usize,
                "repetition_penalty" => params.repetition_penalty = value,
                "max_tokens" => params.max_tokens = value as usize,
                // mlx-audio's `generate` swallows any other kwarg.
                _ => {}
            }
        }
        model
            .synth(text, &params, &mut |samples| sink(samples))
            .map_err(TtsError::Native)
    }
}

impl MlxTts {
    /// `synth` on Qwen3-TTS: a cloned voice (or a preview's clip) as the
    /// reference, as [`qwen3_tts::Params`].
    fn synth_qwen3(
        &self,
        model: &Qwen3Tts,
        text: &str,
        voice: &str,
        options: &SynthOptions,
        mut sink: Sink,
    ) -> Result<(), TtsError> {
        let reference = if let Some((wav, ref_text)) = &options.reference {
            (wav.clone(), ref_text.clone())
        } else if let Some(c) = self
            .cloned
            .as_deref()
            .and_then(|home| voices::find(home, voice))
        {
            (c.wav, c.text)
        } else {
            return Err(TtsError::UnknownVoice {
                model: self.name.clone(),
                voice: voice.to_string(),
            });
        };
        let mut params = qwen3_tts::Params::new(reference);
        // `CLONE_SAMPLING`, tighter than mlx-audio's own defaults.
        params.temperature = 0.55;
        params.top_p = 0.8;
        params.top_k = 20;
        params.seed = Some(options.seed);
        for (name, &value) in &options.knobs {
            match name.as_str() {
                "temperature" => params.temperature = value,
                "top_p" => params.top_p = value,
                "top_k" => params.top_k = value as usize,
                "repetition_penalty" => params.repetition_penalty = value,
                "max_tokens" => params.max_tokens = value as usize,
                // mlx-audio's `generate` swallows any other kwarg.
                _ => {}
            }
        }
        model
            .synth(text, &params, &mut |samples| sink(samples))
            .map_err(TtsError::Native)
    }
}
