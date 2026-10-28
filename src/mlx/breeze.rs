//! Breeze TTS 2 (mlx-community's 8-bit, mxfp8) in-process through mlx-rs
//! (naru task 2381): a port
//! of mlx-audio 0.5.6's `tts/models/breeze_tts/breeze_tts.py` (T5Gemma2 text
//! encoder, Qwen3 backbone for codebook 0, Llama depth decoder for codebooks
//! 1-15, sampling, CFG) and the decoder half of
//! `qwen3_tts/speech_tokenizer.py` (streaming codec decoder), from the 2368
//! prototype. Its sampled codes and its audio are bit-identical to
//! mlx-audio's at the same seed, so the quirks below are deliberate:
//!
//! - GELU is compiled, in Python's exact form (`sqrt(2/pi)` folded in double);
//! - the codec's RMSNorm uses `power`, not `square`;
//! - RoPE inverse frequencies are computed on the device, like Python;
//! - Python `float` literals are weak, `mx.array(sqrt(d))` is strong f32, which
//!   makes the text encoder's (and so the prefill's) arithmetic f32;
//! - the 8-bit checkpoint is mxfp8, not MLX's affine 8-bit, and mlx-rs's
//!   `quantized_matmul`/`dequantize` hardcode affine, so those two go through
//!   `mlx_sys` directly.
//!
//! The depth decoder recomputes its whole sequence for each codebook, as
//! Python does. [`Params::depth_cache`] (naru task 2383, off by default)
//! caches its keys and values within a frame instead: ~30 % faster, but the
//! float order differs, so it samples different codes than mlx-audio.
//! Voice cloning (naru task 2382, from the 2369 prototype) adds the codec
//! *encoder* (Mimi SEANet, 8-layer transformer, stride-2 downsample,
//! split-RVQ encode, as `Qwen3TTSSpeechTokenizerEncoder` wires them), the
//! reference WAV load with mlx-audio's polyphase resampler ([`resample`]),
//! and the ref-text prompt segment. A reference's codes are a deterministic
//! function of its file, so they are cached per file content.
//!
//! MLX runs on one thread that owns the model: [`Breeze`] sends it work.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp, argmax_axis, argmin_axis};
use mlx_rs::ops::{self, argpartition_axis};
use mlx_rs::{Array, Dtype, nn, random};
use sha2::{Digest, Sha256};

mod resample;

pub(super) type R<T> = Result<T, Box<dyn std::error::Error>>;

/// MLX's buffer cache, capped (task 2368: ~5.0 GB peak instead of 8.4 GB, for
/// 2-7 % of speed). Process-wide, so raised only (`native::raise_cache_limit`).
pub(super) const CACHE_LIMIT: usize = 256 << 20;
/// `KVCache.step` in mlx-audio: caches grow in blocks of this many steps.
const KV_STEP: i32 = 256;

// Wrapper (config.json): 16 codebooks of 2051 ids, of which 2048 are codec
// entries; the backbone's lm_head has one extra class, EOS (= 2051).
const NCB: i32 = 16;
const VOCAB: i32 = 2051;
const CODEC_VOCAB: i32 = 2048;
/// `codebook_eos_token_id`: the frame that ends a reference's codes.
const EOS_CODE: i32 = 0;
// Backbone (backbone_config): Qwen3, 28 layers.
const BB_LAYERS: usize = 28;
const BB_H: i32 = 16;
const BB_KV: i32 = 8;
const BB_HD: i32 = 128;
const BB_EPS: f32 = 1e-6;
const BB_THETA: f32 = 1_000_000.0;
// Depth decoder (depth_decoder_config): Llama, 12 layers, llama3 rope.
const DD_LAYERS: usize = 12;
const DD_H: i32 = 8;
const DD_KV: i32 = 2;
const DD_HD: i32 = 128;
const DD_EPS: f32 = 1e-5;
// Text encoder (text_encoder_config): T5Gemma2, 26 layers.
const TE_LAYERS: usize = 26;
const TE_H: i32 = 4;
const TE_KV: i32 = 1;
const TE_HD: i32 = 256;
const TE_EPS: f32 = 1e-6;
const TE_EOI: i32 = 256000;
const TE_WINDOW: i32 = 512;
/// Full-attention layers' linear RoPE scaling, 1/8.
const TE_FULL_INV_SCALE: f32 = 1.0 / 8.0;
// Codec decoder (audio_tokenizer/config.json decoder_config).
const CD_LAYERS: usize = 8;
const CD_H: i32 = 16;
const CD_HD: i32 = 64;
const CD_EPS: f32 = 1e-5;
const CD_THETA: f32 = 10000.0;
const CD_RATES: [i32; 4] = [8, 5, 4, 3];
// Codec encoder (audio_tokenizer/config.json encoder_config).
const CE_LAYERS: usize = 8;
const CE_H: i32 = 8;
const CE_HD: i32 = 64;
const CE_RATIOS: [i32; 4] = [4, 5, 6, 8]; // reversed(upsampling_ratios)
const CE_NQ: usize = 16; // encoder_valid_num_quantizers
/// How many references' codes are kept (each is ~12 KB for a 15 s clip).
pub(super) const REF_CACHE: usize = 32;

/// Of every sample produced.
pub const SAMPLE_RATE: u32 = 24000;
/// mlx-audio's `streaming_interval` as naru-audio used it (0.32 s), in codec frames:
/// `int(0.32 * 24000 / 1920)`.
pub(super) const CHUNK_FRAMES: usize = 4;
/// `generate`'s own default.
pub const MAX_TOKENS: usize = 750;

/// What one synthesis takes, with `generate`'s own defaults in [`Params::default`].
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// A speaker tag, `S0`-`S9`, with or without the brackets.
    pub speaker: String,
    pub instruct: Option<String>,
    /// CFG needs `instruct` and a scale other than 1.
    pub cfg_scale: Option<f64>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    /// Seeds MLX's global random state before the first segment.
    pub seed: Option<u64>,
    /// A voice to clone: the reference WAV and its transcript (`ref_audio`,
    /// `ref_text`).
    pub reference: Option<(PathBuf, String)>,
    /// Cache the depth decoder's keys and values within each frame (not in
    /// Python): faster, but no longer bit-identical to mlx-audio.
    pub depth_cache: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            speaker: "S0".to_string(),
            instruct: None,
            cfg_scale: None,
            max_tokens: MAX_TOKENS,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            seed: None,
            reference: None,
            depth_cache: false,
        }
    }
}

impl Params {
    /// The checks `generate` makes before it does anything.
    fn check(&self) -> Result<(), String> {
        if self.max_tokens == 0 {
            return Err("max_tokens must be positive.".to_string());
        }
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err("temperature must be non-negative.".to_string());
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err("top_p must be between 0 and 1.".to_string());
        }
        if self.repetition_penalty.is_nan() || self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be positive.".to_string());
        }
        if self.cfg_scale.is_some_and(|c| !c.is_finite()) {
            return Err("cfg_scale must be finite.".to_string());
        }
        if self.reference.as_ref().is_some_and(|(_, t)| t.is_empty()) {
            return Err("Breeze voice cloning requires ref_text with ref_audio.".to_string());
        }
        Ok(())
    }

    /// `_speaker`: `[S0]` for no voice, else the tag in brackets.
    fn speaker_tag(&self) -> String {
        let s = self.speaker.as_str();
        let s = if s.is_empty() || s == "af_heart" {
            "S0"
        } else {
            s
        };
        if s.starts_with('[') {
            s.to_string()
        } else {
            format!("[{s}]")
        }
    }
}

enum Event {
    Chunk(Vec<f32>),
    Done(Result<(), String>),
}

type Job = (String, Params, mpsc::Sender<Event>);

/// A loaded model on its MLX thread; dropping it stops the thread and frees
/// the weights.
pub struct Breeze {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    /// What the load added to MLX's active memory.
    resident_bytes: u64,
}

impl Breeze {
    /// Loads `dir`: `model.safetensors`, `tokenizer.json`, `config.json` and
    /// `audio_tokenizer/model.safetensors`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        // Without a metallib MLX exits the process at the first GPU op.
        let metallib = super::native::metallib_path()?;
        if !metallib.is_file() {
            return Err(format!("{} missing", metallib.display()));
        }
        check_quantization(dir)?;
        let dir = dir.to_path_buf();
        let (loaded_tx, loaded_rx) = mpsc::channel();
        let (jobs, rx) = mpsc::channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("mlx-breeze".to_string())
            .spawn(move || {
                let model = (|| -> R<(Model, u64)> {
                    super::native::raise_cache_limit(CACHE_LIMIT)?;
                    let before = mlx_rs::memory::active_memory()?;
                    let model = Model::load(&dir)?;
                    let after = mlx_rs::memory::active_memory()?;
                    Ok((model, after.saturating_sub(before) as u64))
                })();
                let model = match model {
                    Ok((model, bytes)) => {
                        let _ = loaded_tx.send(Ok(bytes));
                        model
                    }
                    Err(e) => {
                        let _ = loaded_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                for (text, params, events) in rx {
                    // A panic in MLX (an `item()` on a failed graph) must not
                    // take the daemon's other models with it.
                    let done = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        model.generate(&text, &params, &mut |chunk| {
                            events.send(Event::Chunk(chunk)).is_ok()
                        })
                    }))
                    .unwrap_or_else(|_| Err("MLX panicked".into()));
                    let _ = events.send(Event::Done(done.map_err(|e| e.to_string())));
                }
                drop(model);
                let _ = mlx_rs::memory::clear_cache();
            })
            .map_err(|e| format!("cannot start the MLX thread: {e}"))?;
        match loaded_rx.recv() {
            Ok(Ok(resident_bytes)) => Ok(Breeze {
                jobs: Some(jobs),
                thread: Some(thread),
                resident_bytes,
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err("the MLX thread died loading the model".to_string())
            }
        }
    }

    /// Synthesises `text`, handing `on_chunk` each 0.32 s of 24 kHz mono
    /// samples as it is decoded; `on_chunk` returns `false` to stop (then
    /// this returns `Ok`).
    pub fn synth(
        &self,
        text: &str,
        params: &Params,
        on_chunk: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(), String> {
        let (events, answer) = mpsc::channel();
        let jobs = self.jobs.as_ref().expect("set until drop");
        jobs.send((text.to_string(), params.clone(), events))
            .map_err(|_| "the MLX thread is gone".to_string())?;
        loop {
            match answer.recv() {
                Ok(Event::Chunk(samples)) => {
                    // Dropping `answer` makes the thread's next send fail,
                    // which ends the synthesis.
                    if !on_chunk(&samples) {
                        return Ok(());
                    }
                }
                Ok(Event::Done(done)) => return done,
                Err(_) => return Err("the MLX thread died synthesising".to_string()),
            }
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl Drop for Breeze {
    fn drop(&mut self) {
        // Closing the queue ends the thread's loop.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Only the 8-bit mxfp8 checkpoint is ported: a bf16 or affine 4-bit one
/// would run its quantized ops wrong.
fn check_quantization(dir: &Path) -> Result<(), String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let q = &config["quantization"];
    if q["mode"] == "mxfp8" && q["bits"] == 8 && q["group_size"] == 32 {
        Ok(())
    } else {
        Err(format!(
            "only the 8-bit mxfp8 checkpoint has a native port (config.json quantization: {q})"
        ))
    }
}

// ---------------------------------------------------------------------------
// Segment splitting (`generate`'s `\n` split and `_split_long_segment`)
// ---------------------------------------------------------------------------

/// `generate`'s segments of `text`: its non-empty lines (stripped), each cut
/// to at most 600 characters; the whole text when there are none.
pub fn segments(text: &str) -> Vec<String> {
    let out: Vec<String> = text
        .split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .flat_map(|s| split_long_segment(s, 600))
        .collect();
    if out.is_empty() {
        vec![text.to_string()]
    } else {
        out
    }
}

/// `_split_long_segment`: non-empty chunks of at most `max_chars`
/// characters, cut after sentence terminators, else at a word boundary
/// from the second half of the limit, else hard.
fn split_long_segment(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }
    // `re.split(r"(?<=[.!?。！？])", text)`: each piece ends at a terminator.
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut piece: Vec<char> = Vec::new();
    for c in text.chars() {
        piece.push(c);
        if ".!?。！？".contains(c) {
            sentences.push(std::mem::take(&mut piece));
        }
    }
    if !piece.is_empty() {
        sentences.push(piece);
    }
    let trim = |s: &[char]| s.iter().collect::<String>().trim().to_string();
    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for sentence in sentences {
        let mut sentence = sentence;
        while !sentence.is_empty() {
            if !current.is_empty() && current.len() + sentence.len() > max_chars {
                chunks.push(trim(&current));
                current.clear();
                let skip = sentence.iter().take_while(|c| c.is_whitespace()).count();
                sentence.drain(..skip);
                continue;
            }
            let available = max_chars - current.len();
            if sentence.len() <= available {
                current.extend_from_slice(&sentence);
                break;
            }
            let mut split_at = available;
            // The last whitespace run in `sentence[: available + 1]`.
            let head = &sentence[..(available + 1).min(sentence.len())];
            let mut run_start = None;
            for (i, c) in head.iter().enumerate() {
                if c.is_whitespace() && (i == 0 || !head[i - 1].is_whitespace()) {
                    run_start = Some(i);
                }
            }
            if let Some(start) = run_start
                && start >= available / 2
            {
                split_at = start;
            }
            let mut chunk = current.clone();
            chunk.extend_from_slice(&sentence[..split_at]);
            let chunk = trim(&chunk);
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
            current.clear();
            sentence.drain(..split_at);
            let skip = sentence.iter().take_while(|c| c.is_whitespace()).count();
            sentence.drain(..skip);
        }
    }
    if !trim(&current).is_empty() {
        chunks.push(trim(&current));
    }
    chunks
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

/// A Python float literal: weakly typed, so it takes the other operand's dtype.
pub(super) fn s(x: f32, like: &Array) -> R<Array> {
    Ok(Array::from_f32(x).as_dtype(like.dtype())?)
}

/// mlx.nn.gelu_approx as Python writes it (constants folded in double),
/// compiled shapeless like Python's `@partial(mx.compile, shapeless=True)`.
fn gelu_approx(x: &Array) -> R<Array> {
    let f = |x: &Array| -> Result<Array, mlx_rs::error::Exception> {
        let c = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
        let inner = ops::add(x, ops::multiply(c(0.044715)?, ops::power(x, c(3.0)?)?)?)?;
        let t = ops::tanh(ops::multiply(
            c((2.0f64 / std::f64::consts::PI).sqrt() as f32)?,
            &inner,
        )?)?;
        ops::multiply(ops::multiply(c(0.5)?, x)?, ops::add(c(1.0)?, &t)?)
    };
    let mut cf = mlx_rs::transforms::compile::compile(f, true);
    Ok(cf(x)?)
}

/// mlx.nn.gelu: `x * (1 + erf(x / math.sqrt(2))) / 2`, compiled shapeless.
fn gelu(x: &Array) -> R<Array> {
    let f = |x: &Array| -> Result<Array, mlx_rs::error::Exception> {
        let c = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
        let e = ops::erf(ops::divide(x, c(std::f64::consts::SQRT_2 as f32)?)?)?;
        ops::divide(ops::multiply(x, ops::add(c(1.0)?, &e)?)?, c(2.0)?)
    };
    let mut cf = mlx_rs::transforms::compile::compile(f, true);
    Ok(cf(x)?)
}

/// `(2 * i) / div` for `i` in `from..to`, in f32 on the device: `mx.arange(0, dim, 2) / dim`.
fn arange_f32(from: i32, to: i32, div: f32) -> Result<Array, mlx_rs::error::Exception> {
    let v: Vec<f32> = (from..to).map(|i| (2 * i) as f32).collect();
    ops::divide(Array::from_slice(&v, &[to - from]), Array::from_f32(div))
}

/// mlx.nn.elu (alpha 1.0): `mx.where(x > 0, x, alpha * (mx.exp(x) - 1))`,
/// compiled shapeless.
fn elu(x: &Array) -> R<Array> {
    let f = |x: &Array| -> Result<Array, mlx_rs::error::Exception> {
        let c = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
        let neg = ops::multiply(c(1.0)?, ops::subtract(ops::exp(x)?, c(1.0)?)?)?;
        ops::select(&ops::gt(x, c(0.0)?)?, x, &neg)
    };
    let mut cf = mlx_rs::transforms::compile::compile(f, true);
    Ok(cf(x)?)
}

/// mimi `get_extra_padding_for_conv1d` (float division, as Python).
fn extra_padding(len: i32, k: i32, stride: i32, total: i32) -> i32 {
    let nframes = ((len + total - k).max(0)) as f64 / stride as f64 + 1.0;
    let ideal = (nframes.ceil() as i32 - 1) * stride + k - total;
    (ideal - len).max(0)
}

/// How a checkpoint's quantized layers are packed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Quant {
    /// Breeze's 8-bit: mxfp8, group 32, scales only.
    Mxfp8,
    /// MLX's affine mode (Qwen3-TTS 8-bit: group 64), scales and biases.
    Affine { group: i32, bits: i32 },
}

impl Quant {
    /// The mode, group size and bits mlx-c takes, and the biases, if any.
    fn args<'a>(
        self,
        w: &'a Weights,
        p: &str,
    ) -> R<(CString, i32, i32, Option<&'a Array>)> {
        Ok(match self {
            Quant::Mxfp8 => (CString::new("mxfp8")?, 32, 8, None),
            Quant::Affine { group, bits } => (
                CString::new("affine")?,
                group,
                bits,
                Some(w.g(&format!("{p}.biases"))?),
            ),
        })
    }
}

pub(super) struct Weights {
    map: HashMap<String, Array>,
    quant: Quant,
    /// One GPU stream for every quantized op, freed with the weights.
    stream: mlx_sys::mlx_stream,
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_stream_free(self.stream) };
    }
}

impl Weights {
    /// Weights whose quantized layers, if any, are mxfp8 (Breeze's).
    fn new(map: HashMap<String, Array>) -> Self {
        Self::quantized(map, Quant::Mxfp8)
    }
    pub(super) fn quantized(map: HashMap<String, Array>, quant: Quant) -> Self {
        Weights {
            map,
            quant,
            stream: unsafe { mlx_sys::mlx_default_gpu_stream_new() },
        }
    }
    pub(super) fn values(&self) -> impl Iterator<Item = &Array> {
        self.map.values()
    }
    pub(super) fn g(&self, k: &str) -> R<&Array> {
        self.map
            .get(k)
            .ok_or_else(|| format!("the checkpoint has no weight `{k}`").into())
    }
    fn opt(&self, k: &str) -> Option<&Array> {
        self.map.get(k)
    }
    /// nn.QuantizedLinear in this checkpoint's [`Quant`] mode. mlx-rs's
    /// quantized_matmul hardcodes affine mode, so call mlx-c.
    pub(super) fn qlin(&self, x: &Array, p: &str) -> R<Array> {
        let w = self.g(&format!("{p}.weight"))?;
        let sc = self.g(&format!("{p}.scales"))?;
        let (mode, group, bits, biases) = self.quant.args(self, p)?;
        let y = unsafe {
            let mut res = mlx_sys::mlx_array_new();
            let none = mlx_sys::mlx_array_new();
            let rc = mlx_sys::mlx_quantized_matmul(
                &mut res,
                x.as_ptr(),
                w.as_ptr(),
                sc.as_ptr(),
                biases.map_or(none, |b| b.as_ptr()),
                true,
                mlx_sys::mlx_optional_int {
                    value: group,
                    has_value: true,
                },
                mlx_sys::mlx_optional_int {
                    value: bits,
                    has_value: true,
                },
                mode.as_ptr(),
                self.stream,
            );
            mlx_sys::mlx_array_free(none);
            if rc != 0 {
                mlx_sys::mlx_array_free(res);
                return Err(format!("quantized_matmul failed for {p}").into());
            }
            Array::from_ptr(res)
        };
        Ok(match self.opt(&format!("{p}.bias")) {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }
    /// nn.QuantizedEmbedding in this checkpoint's [`Quant`] mode: dequantize
    /// the gathered rows.
    fn qemb(&self, ids: &Array, p: &str, wkey: Option<&str>) -> R<Array> {
        let w = self
            .g(wkey.unwrap_or(&format!("{p}.weight")))?
            .take_axis(ids, 0)?;
        let sc = self.g(&format!("{p}.scales"))?.take_axis(ids, 0)?;
        let (mode, group, bits, biases) = self.quant.args(self, p)?;
        let biases = biases.map(|b| b.take_axis(ids, 0)).transpose()?;
        unsafe {
            let mut res = mlx_sys::mlx_array_new();
            let none = mlx_sys::mlx_array_new();
            let none2 = mlx_sys::mlx_array_new();
            let rc = mlx_sys::mlx_dequantize(
                &mut res,
                w.as_ptr(),
                sc.as_ptr(),
                biases.as_ref().map_or(none, |b| b.as_ptr()),
                mlx_sys::mlx_optional_int {
                    value: group,
                    has_value: true,
                },
                mlx_sys::mlx_optional_int {
                    value: bits,
                    has_value: true,
                },
                mode.as_ptr(),
                none2,
                mlx_sys::mlx_optional_dtype {
                    value: mlx_sys::mlx_dtype__MLX_BFLOAT16,
                    has_value: false,
                },
                self.stream,
            );
            mlx_sys::mlx_array_free(none);
            mlx_sys::mlx_array_free(none2);
            if rc != 0 {
                mlx_sys::mlx_array_free(res);
                return Err(format!("dequantize failed for {p}").into());
            }
            Ok(Array::from_ptr(res))
        }
    }
    /// Float nn.Linear: x @ W.T + b.
    pub(super) fn lin(&self, x: &Array, p: &str) -> R<Array> {
        let y = ops::matmul(x, self.g(&format!("{p}.weight"))?.t())?;
        Ok(match self.opt(&format!("{p}.bias")) {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }
    pub(super) fn rms(&self, x: &Array, p: &str, eps: f32) -> R<Array> {
        Ok(fast::rms_norm(
            x,
            Some(self.g(&format!("{p}.weight"))?),
            eps,
        )?)
    }
}

/// mlx-audio's `KVCache`: keys and values preallocated in blocks of
/// [`KV_STEP`] steps, written in place, so a long generation does not leave
/// a buffer of every size in MLX's cache.
#[derive(Default, Clone)]
pub(super) struct KvCache {
    kv: Option<(Array, Array)>,
    pub(super) offset: i32,
}

impl KvCache {
    /// Appends `k` and `v` (`[B, H, n, D]`) and returns everything cached.
    pub(super) fn update(&mut self, k: Array, v: Array) -> R<(Array, Array)> {
        let prev = self.offset;
        let n = k.shape()[2];
        let full = self
            .kv
            .as_ref()
            .is_none_or(|(b, _)| prev + n > b.shape()[2]);
        if full {
            let [b, h, _, d] = k.shape()[..] else {
                return Err("a KV cache takes [B, H, n, D]".into());
            };
            let steps = (KV_STEP + n - 1) / KV_STEP;
            let shape = [b, h, steps * KV_STEP, d];
            let new_k = ops::zeros_dtype(&shape, k.dtype())?;
            let new_v = ops::zeros_dtype(&shape, v.dtype())?;
            self.kv = Some(match self.kv.take() {
                Some((mut ck, mut cv)) => {
                    if prev % KV_STEP != 0 {
                        ck = ck.index((.., .., ..prev, ..));
                        cv = cv.index((.., .., ..prev, ..));
                    }
                    (
                        ops::concatenate(&[ck, new_k], 2)?,
                        ops::concatenate(&[cv, new_v], 2)?,
                    )
                }
                None => (new_k, new_v),
            });
        }
        self.offset += n;
        let (ck, cv) = self.kv.as_mut().expect("set above");
        ck.try_index_mut((.., .., prev..self.offset, ..), &k)?;
        cv.try_index_mut((.., .., prev..self.offset, ..), &v)?;
        Ok((
            ck.index((.., .., ..self.offset, ..)),
            cv.index((.., .., ..self.offset, ..)),
        ))
    }
}

/// One depth decoder layer's keys and values for the frame being sampled
/// ([`Params::depth_cache`]).
type DepthKv = Option<(Array, Array)>;

pub(super) fn heads(x: &Array, b: i32, l: i32, h: i32, d: i32) -> R<Array> {
    Ok(x.reshape(&[b, l, h, d])?)
}

struct Model {
    w: Weights,
    tok: tokenizers::Tokenizer,
    dd_freqs: Array,
    // T5Gemma2 norms add 1 to a zero-initialised weight; precomputed in bf16
    // like Python's `1.0 + self.weight`.
    te_norms: HashMap<String, Array>,
    codec: Codec,
}

/// The Qwen3-TTS 12 Hz speech tokenizer (`speech_tokenizer.py`), which Breeze
/// ships as its `audio_tokenizer/` and Qwen3-TTS as its `speech_tokenizer/`
/// (the same weights): the streaming decoder, and the encoder for a
/// reference clip's codes.
pub(super) struct Codec {
    cw: Weights,
    cd_inv_freq: Array,
    /// Codec encoder weights, under the short names [`load_encoder`] gives.
    ew: Weights,
    /// Encoder RVQ codebooks (rvq_first's 1, then rvq_rest's first 15):
    /// `embedding_sum / max(usage, eps)` and `|e|^2 / 2`.
    ce_books: Vec<Codebook>,
    /// Reference codes by the SHA-256 of the reference file.
    ref_codes: RefCell<HashMap<[u8; 32], Array>>,
}

/// The conv buffers, overlap-add tails and transformer KV cache a stream of
/// codec frames carries (`reset_streaming_state` is a fresh one).
#[derive(Clone)]
pub(super) struct CodecState {
    bufs: HashMap<String, Array>,
    kv: Vec<KvCache>,
    offset: i32,
}

impl CodecState {
    pub(super) fn new() -> Self {
        CodecState {
            bufs: HashMap::new(),
            kv: (0..CD_LAYERS).map(|_| KvCache::default()).collect(),
            offset: 0,
        }
    }

    /// Copies every array carried into its own buffer, so a kept state holds
    /// only them: a conv buffer is a view of its whole input, which (even
    /// evaluated) keeps that input's memory alive.
    pub(super) fn detach(&mut self) -> R<()> {
        let own = |a: &Array| -> R<Array> {
            let f = a.as_dtype(Dtype::Float32)?;
            f.eval()?;
            Ok(Array::from_slice(f.as_slice::<f32>(), a.shape()).as_dtype(a.dtype())?)
        };
        for b in self.bufs.values_mut() {
            *b = own(b)?;
        }
        for c in &mut self.kv {
            if let Some((k, v)) = &mut c.kv {
                (*k, *v) = (own(k)?, own(v)?);
            }
        }
        Ok(())
    }
}

pub(super) fn caches(n: usize) -> Vec<KvCache> {
    (0..n).map(|_| KvCache::default()).collect()
}

impl Model {
    // ---------------- text encoder (T5Gemma2) ----------------

    fn t5rms(&self, x: &Array, p: &str) -> R<Array> {
        let w = self
            .te_norms
            .get(p)
            .ok_or_else(|| format!("the checkpoint has no norm `{p}`"))?;
        Ok(fast::rms_norm(x, Some(w), TE_EPS)?)
    }

    fn text_encode(&self, ids: &[u32]) -> R<Array> {
        let w = &self.w;
        let l = ids.len() as i32;
        let ids_a = Array::from_slice(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>(), &[1, l]);
        let tw = w.g("text_encoder.embed_tokens.weight")?;
        let emb = tw.take_axis(&ids_a, 0)?;
        // `mx.array(hidden ** 0.5)` is a float32 array, so this promotes to f32.
        let emb = ops::multiply(&emb, Array::from_f32((1152f64).sqrt() as f32))?;
        let is_eoi = ops::eq(&ids_a, Array::from_int(TE_EOI))?.expand_dims(-1)?;
        let mut h = ops::select(
            &is_eoi,
            w.g("text_encoder.embed_tokens.eoi_embedding")?,
            &emb,
        )?;
        // Sliding layers: (w+1)//2 left and w//2+1 right, both including self.
        let mut m = vec![false; (l * l) as usize];
        let (left, right) = ((TE_WINDOW + 1) / 2, TE_WINDOW / 2 + 1);
        for q in 0..l {
            for k in 0..l {
                let d = q - k;
                m[(q * l + k) as usize] = (d >= 0 && d < left) || (d < 0 && -d < right);
            }
        }
        let sliding_mask = Array::from_slice(&m, &[l, l]);
        for i in 0..TE_LAYERS {
            let p = format!("text_encoder.layers.{i}");
            let full = (i + 1) % 6 == 0;
            let x = self.t5rms(&h, &format!("{p}.pre_self_attn_layernorm"))?;
            let q = self
                .t5rms(
                    &heads(
                        &w.qlin(&x, &format!("{p}.self_attn.q_proj"))?,
                        1,
                        l,
                        TE_H,
                        TE_HD,
                    )?,
                    &format!("{p}.self_attn.q_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = self
                .t5rms(
                    &heads(
                        &w.qlin(&x, &format!("{p}.self_attn.k_proj"))?,
                        1,
                        l,
                        TE_KV,
                        TE_HD,
                    )?,
                    &format!("{p}.self_attn.k_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(
                &w.qlin(&x, &format!("{p}.self_attn.v_proj"))?,
                1,
                l,
                TE_KV,
                TE_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let (base, scale) = if full {
                (1_000_000.0, TE_FULL_INV_SCALE)
            } else {
                (10_000.0, 1.0)
            };
            let q = fast::rope(&q, TE_HD, false, base, scale, 0, None)?;
            let k = fast::rope(&k, TE_HD, false, base, scale, 0, None)?;
            let o = if full {
                fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, None, None)?
            } else {
                fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    1.0 / 16.0,
                    ScaledDotProductAttentionMask::Array(&sliding_mask),
                    None,
                )?
            };
            let o = w.qlin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, -1])?,
                &format!("{p}.self_attn.o_proj"),
            )?;
            let hid = ops::add(
                &h,
                self.t5rms(&o, &format!("{p}.post_self_attn_layernorm"))?,
            )?;
            let x = self.t5rms(&hid, &format!("{p}.pre_feedforward_layernorm"))?;
            let g = gelu_approx(&w.qlin(&x, &format!("{p}.mlp.gate_proj"))?)?;
            let u = w.qlin(&x, &format!("{p}.mlp.up_proj"))?;
            let f = w.qlin(&ops::multiply(&g, &u)?, &format!("{p}.mlp.down_proj"))?;
            h = ops::add(
                &hid,
                self.t5rms(&f, &format!("{p}.post_feedforward_layernorm"))?,
            )?;
        }
        let h = self.t5rms(&h, "text_encoder.norm")?;
        w.qlin(&h, "text_encoder_proj")
    }

    // ---------------- backbone (Qwen3) ----------------

    fn audio_embed(&self, frame: &[i32]) -> R<Array> {
        let ids: Vec<i32> = frame
            .iter()
            .enumerate()
            .map(|(i, &c)| c + i as i32 * VOCAB)
            .collect();
        let ids = Array::from_slice(&ids, &[1, 1, NCB]);
        let e = self.w.qemb(
            &ids,
            "backbone_model.embed_tokens.embed_audio_tokens",
            Some("depth_decoder.model.embed_tokens.weight"),
        )?;
        Ok(e.sum_axis(-2, None)?)
    }

    fn backbone(&self, x: &Array, cache: &mut [KvCache], offset: i32) -> R<Array> {
        let w = &self.w;
        let l = x.shape()[1];
        let mut h = x.clone();
        for (i, layer_cache) in cache.iter_mut().enumerate().take(BB_LAYERS) {
            let p = format!("backbone_model.layers.{i}");
            let xn = w.rms(&h, &format!("{p}.input_layernorm"), BB_EPS)?;
            let q = w
                .rms(
                    &heads(
                        &w.qlin(&xn, &format!("{p}.self_attn.q_proj"))?,
                        1,
                        l,
                        BB_H,
                        BB_HD,
                    )?,
                    &format!("{p}.self_attn.q_norm"),
                    BB_EPS,
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = w
                .rms(
                    &heads(
                        &w.qlin(&xn, &format!("{p}.self_attn.k_proj"))?,
                        1,
                        l,
                        BB_KV,
                        BB_HD,
                    )?,
                    &format!("{p}.self_attn.k_norm"),
                    BB_EPS,
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(
                &w.qlin(&xn, &format!("{p}.self_attn.v_proj"))?,
                1,
                l,
                BB_KV,
                BB_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let q = fast::rope(&q, BB_HD, false, BB_THETA, 1.0, offset, None)?;
            let k = fast::rope(&k, BB_HD, false, BB_THETA, 1.0, offset, None)?;
            let (k, v) = layer_cache.update(k, v)?;
            let scale = (BB_HD as f32).powf(-0.5);
            let o = if l > 1 {
                fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None,
                )?
            } else {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
            };
            let o = w.qlin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, -1])?,
                &format!("{p}.self_attn.o_proj"),
            )?;
            let hh = ops::add(&h, &o)?;
            let xn = w.rms(&hh, &format!("{p}.post_attention_layernorm"), BB_EPS)?;
            let g = nn::silu(w.qlin(&xn, &format!("{p}.mlp.gate_proj"))?)?;
            let u = w.qlin(&xn, &format!("{p}.mlp.up_proj"))?;
            h = ops::add(
                &hh,
                w.qlin(&ops::multiply(&g, &u)?, &format!("{p}.mlp.down_proj"))?,
            )?;
        }
        let h = w.rms(&h, "backbone_model.norm", BB_EPS)?;
        Ok(h.index((.., -1, ..)))
    }

    // ---------------- depth decoder (Llama) ----------------

    /// `_DepthDecoder.next_logits`: without `cache` the whole sequence is run
    /// again for each codebook, as in Python. With it (one frame's, started
    /// empty) only the tokens not yet cached are run.
    fn depth_logits(
        &self,
        tokens: &Array,
        bb_hidden: &Array,
        mut cache: Option<&mut [DepthKv]>,
    ) -> R<Array> {
        let w = &self.w;
        let t = tokens.shape()[1];
        let start = cache
            .as_ref()
            .and_then(|c| c[0].as_ref())
            .map_or(0, |(k, _)| k.shape()[2]);
        let n = t - start;
        // `token_ids + positions[None, :] * vocab_size`.
        let offsets: Vec<i32> = (start..t).map(|i| (i - 1).max(0) * VOCAB).collect();
        let tokens = if start == 0 {
            tokens.clone()
        } else {
            tokens.index((.., start..))
        };
        let ids = ops::add(&tokens, Array::from_slice(&offsets, &[1, n]))?;
        let emb = w.qemb(&ids, "depth_decoder.model.embed_tokens", None)?;
        let emb = if start == 0 {
            ops::concatenate(&[bb_hidden.expand_dims(1)?, emb.index((.., 1.., ..))], 1)?
        } else {
            emb
        };
        let mut h = w.qlin(&emb, "depth_decoder.model.inputs_embeds_projector")?;
        for i in 0..DD_LAYERS {
            let p = format!("depth_decoder.model.layers.{i}");
            let xn = w.rms(&h, &format!("{p}.input_layernorm"), DD_EPS)?;
            let q = heads(
                &w.qlin(&xn, &format!("{p}.self_attn.q_proj"))?,
                1,
                n,
                DD_H,
                DD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let k = heads(
                &w.qlin(&xn, &format!("{p}.self_attn.k_proj"))?,
                1,
                n,
                DD_KV,
                DD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(
                &w.qlin(&xn, &format!("{p}.self_attn.v_proj"))?,
                1,
                n,
                DD_KV,
                DD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let q = fast::rope(&q, DD_HD, false, None, 1.0, start, &self.dd_freqs)?;
            let k = fast::rope(&k, DD_HD, false, None, 1.0, start, &self.dd_freqs)?;
            let (k, v) = match cache.as_deref_mut() {
                Some(c) => {
                    // At most 16 steps, so growing by concatenation is cheap.
                    let (k, v) = match c[i].take() {
                        Some((ck, cv)) => (
                            ops::concatenate(&[ck, k], 2)?,
                            ops::concatenate(&[cv, v], 2)?,
                        ),
                        None => (k, v),
                    };
                    c[i] = Some((k.clone(), v.clone()));
                    (k, v)
                }
                None => (k, v),
            };
            let scale = (DD_HD as f32).powf(-0.5);
            let o = if n > 1 {
                fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None,
                )?
            } else {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
            };
            let o = w.qlin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, n, -1])?,
                &format!("{p}.self_attn.o_proj"),
            )?;
            let hh = ops::add(&h, &o)?;
            let xn = w.rms(&hh, &format!("{p}.post_attention_layernorm"), DD_EPS)?;
            let g = nn::silu(w.qlin(&xn, &format!("{p}.mlp.gate_proj"))?)?;
            let u = w.qlin(&xn, &format!("{p}.mlp.up_proj"))?;
            h = ops::add(
                &hh,
                w.qlin(&ops::multiply(&g, &u)?, &format!("{p}.mlp.down_proj"))?,
            )?;
        }
        let h = w.rms(&h, "depth_decoder.model.norm", DD_EPS)?;
        let last = h.index((.., -1, ..));
        let head = w.g("depth_decoder.codebooks_head.weight")?.index(t - 2);
        Ok(ops::matmul(&last, &head)?)
    }

    // ---------------- sampling ----------------

    /// `_mask_reserved_codec_logits`: ids CODEC_VOCAB..VOCAB get -inf.
    fn mask_reserved(logits: &Array) -> R<Array> {
        let n = logits.shape()[logits.ndim() - 1];
        if n < VOCAB {
            return Err(format!(
                "Breeze logits do not contain the wrapper vocabulary: {n} < {VOCAB}."
            )
            .into());
        }
        let m: Vec<f32> = (0..n)
            .map(|i| {
                if (CODEC_VOCAB..VOCAB).contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let m = Array::from_slice(&m, &[n]).as_dtype(logits.dtype())?;
        Ok(ops::add(logits, &m)?)
    }

    /// `_sample` with mlx-audio's `make_sampler(temp, top_p, top_k)`:
    /// log_softmax, then argmax at temperature 0, else top-p, top-k and
    /// categorical(logprobs / temp) on MLX's global key.
    fn sample(logits: &Array, p: &Params, allow_eos: bool) -> R<i32> {
        Ok(Self::sample_array(logits, p, allow_eos)?.try_item_exact::<u32>()? as i32)
    }

    /// [`Self::sample`]'s token as a lazy `[1]` array, not yet evaluated.
    fn sample_array(logits: &Array, p: &Params, allow_eos: bool) -> R<Array> {
        let valid = if allow_eos { VOCAB + 1 } else { VOCAB };
        if logits.shape()[logits.ndim() - 1] < valid {
            return Err(format!(
                "Breeze logits do not contain the expected codebook vocabulary: {} < {valid}.",
                logits.shape()[logits.ndim() - 1]
            )
            .into());
        }
        let logits = Self::mask_reserved(logits)?.index((.., ..valid));
        let mut top_k = if p.top_k == 0 {
            0
        } else {
            p.top_k.min(valid as usize)
        };
        if top_k == valid as usize {
            top_k = 0;
        }
        let mut lp = ops::subtract(&logits, logits.logsumexp_axis(-1, true)?)?;
        if p.temperature == 0.0 {
            return Ok(argmax_axis(&lp, -1, None)?);
        }
        if p.top_p > 0.0 && p.top_p < 1.0 {
            lp = Self::top_p(&lp, p.top_p)?;
        }
        if top_k > 0 {
            let k = top_k as i32;
            let neg = ops::negative(&lp)?;
            let idx = argpartition_axis(&neg, k - 1, -1)?.index((.., k..));
            let ninf = Array::from_f32(f32::NEG_INFINITY).as_dtype(lp.dtype())?;
            lp = lp.put_along_axis(&idx, &ninf, -1)?;
        }
        let tok = random::categorical(
            ops::multiply(&lp, s((1.0f64 / p.temperature) as f32, &lp)?)?,
            None,
            None,
            None,
        )?;
        Ok(tok)
    }

    /// `apply_top_p`: keep the tokens whose cumulative probability, summed
    /// from the least likely, exceeds `1 - top_p`.
    fn top_p(logprobs: &Array, top_p: f64) -> R<Array> {
        let probs = ops::exp(logprobs)?;
        let sorted_indices = ops::argsort_axis(logprobs, -1)?;
        let sorted_probs = probs.take_along_axis(&sorted_indices, -1)?;
        let cumulative = ops::cumsum(&sorted_probs, -1, None, None)?;
        let n = sorted_indices.shape()[sorted_indices.ndim() - 1];
        let ar = Array::arange::<_, i32>(0, n, None)?.as_dtype(sorted_indices.dtype())?;
        let inverse = ops::zeros_like(&sorted_indices)?.put_along_axis(&sorted_indices, &ar, -1)?;
        let cumulative = cumulative.take_along_axis(&inverse, -1)?;
        let ninf = Array::from_f32(f32::NEG_INFINITY).as_dtype(logprobs.dtype())?;
        let thresh = s((1.0 - top_p) as f32, &cumulative)?;
        Ok(ops::select(
            &ops::gt(&cumulative, &thresh)?,
            logprobs,
            &ninf,
        )?)
    }

    /// `_apply_repetition_penalty`: the sign-aware penalty on the first
    /// codebooks already generated.
    fn repetition_penalty(logits: &Array, generated: &[i32], penalty: f64) -> R<Array> {
        if penalty == 1.0 || generated.is_empty() {
            return Ok(logits.clone());
        }
        let n = logits.shape()[logits.ndim() - 1];
        let mut ids: Vec<i32> = generated
            .iter()
            .copied()
            .filter(|&t| 0 <= t && t < n)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            return Ok(logits.clone());
        }
        let indices = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let selected = logits.take_along_axis(&indices, -1)?;
        let pen = s(penalty as f32, &selected)?;
        let penalized = ops::select(
            &ops::lt(&selected, s(0.0, &selected)?)?,
            ops::multiply(&selected, &pen)?,
            ops::divide(&selected, &pen)?,
        )?;
        Ok(logits.put_along_axis(&indices, &penalized, -1)?)
    }

    /// `unconditional + scale * (conditional - unconditional)`.
    fn cfg(cond: Array, uncond: Option<Array>, scale: f64) -> R<Array> {
        Ok(match uncond {
            Some(u) => ops::add(
                &u,
                ops::multiply(ops::subtract(&cond, &u)?, s(scale as f32, &cond)?)?,
            )?,
            None => cond,
        })
    }

    /// `_prompt_embeddings`: with a reference (its codes and transcript),
    /// `{speaker}{ref_text}` encoded, then the reference codes and one EOS
    /// frame through the backbone's audio embedding; then the target text.
    fn prompt(
        &self,
        speaker: &str,
        text: &str,
        instruct: Option<&str>,
        reference: Option<(&Array, &str)>,
    ) -> R<Array> {
        let mut parts = Vec::new();
        if let Some((codes, ref_text)) = reference {
            let enc = self
                .tok
                .encode(format!("{speaker}{ref_text}"), true)
                .map_err(|e| format!("tokenize: {e}"))?;
            parts.push(self.text_encode(enc.get_ids())?);
            parts.push(self.audio_embed_codes(codes)?);
            parts.push(self.audio_embed(&[EOS_CODE; NCB as usize])?);
        }
        let target = match instruct {
            Some(i) if !i.is_empty() => format!("{speaker}<ins_bos>{i}<ins_eos>{text}"),
            _ => format!("{speaker}{text}"),
        };
        let enc = self
            .tok
            .encode(target, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        parts.push(self.text_encode(enc.get_ids())?);
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one part"));
        }
        Ok(ops::concatenate(&parts, 1)?)
    }

    /// `backbone_model.embed_tokens(codes)` for codes [1, T, 16].
    fn audio_embed_codes(&self, codes: &Array) -> R<Array> {
        let offs: Vec<i32> = (0..NCB).map(|i| i * VOCAB).collect();
        let offs = Array::from_slice(&offs, &[1, 1, NCB]).as_dtype(codes.dtype())?;
        let ids = ops::add(codes, &offs)?;
        let e = self.w.qemb(
            &ids,
            "backbone_model.embed_tokens.embed_audio_tokens",
            Some("depth_decoder.model.embed_tokens.weight"),
        )?;
        Ok(e.sum_axis(-2, None)?)
    }

}

impl Codec {
    // ---------------- codec encoder (Mimi, as Qwen3TTSSpeechTokenizerEncoder) ----------------

    /// mimi `StreamableConv1d.__call__` (causal) on NLC input.
    fn sconv(&self, x: &Array, p: &str, k: i32, stride: i32, edge: bool) -> R<Array> {
        let total = k - stride;
        let extra = extra_padding(x.shape()[1], k, stride, total);
        let mode = if edge {
            ops::PadMode::Edge
        } else {
            ops::PadMode::Constant
        };
        let x = ops::pad(x, &[(0, 0), (total, extra), (0, 0)][..], None, mode)?;
        let y = ops::conv1d(&x, self.ew.g(&format!("{p}.weight"))?, stride, 0, 1, 1)?;
        Ok(match self.ew.opt(&format!("{p}.bias")) {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }

    /// `Qwen3TTSSpeechTokenizerEncoder.encode`: 24 kHz audio (not empty) ->
    /// codes [1, T, 16] (uint32).
    fn encode_ref(&self, audio: &[f32]) -> R<Array> {
        let ew = &self.ew;
        let mut x = Array::from_slice(audio, &[1, audio.len() as i32, 1]);
        // SeanetEncoder
        x = self.sconv(&x, "seanet.init", 7, 1, false)?;
        for (li, &ratio) in CE_RATIOS.iter().enumerate() {
            let r = x.clone();
            let mut t = self.sconv(&elu(&x)?, &format!("seanet.{li}.res0"), 3, 1, false)?;
            t = self.sconv(&elu(&t)?, &format!("seanet.{li}.res1"), 1, 1, false)?;
            x = ops::add(&t, &r)?;
            x = self.sconv(
                &elu(&x)?,
                &format!("seanet.{li}.down"),
                ratio * 2,
                ratio,
                false,
            )?;
        }
        x = self.sconv(&elu(&x)?, "seanet.final", 3, 1, false)?;
        // ProjectedTransformer (8 layers, LayerNorm, RoPE, causal mask, gelu_approx MLP)
        let l = x.shape()[1];
        let mut m = vec![0f32; (l * l) as usize];
        for q in 0..l {
            for k in (q + 1)..l {
                m[(q * l + k) as usize] = f32::NEG_INFINITY;
            }
        }
        let mask = Array::from_slice(&m, &[1, 1, l, l]);
        let mut h = x;
        for i in 0..CE_LAYERS {
            let p = format!("tf.{i}");
            let n1 = fast::layer_norm(
                &h,
                ew.g(&format!("{p}.norm1.weight"))?,
                ew.g(&format!("{p}.norm1.bias"))?,
                1e-5,
            )?;
            let qkv = ew
                .lin(&n1, &format!("{p}.in_proj"))?
                .reshape(&[1, l, 3, CE_H, CE_HD])?;
            let q = qkv.index((.., .., 0)).transpose_axes(&[0, 2, 1, 3])?;
            let k = qkv.index((.., .., 1)).transpose_axes(&[0, 2, 1, 3])?;
            let v = qkv.index((.., .., 2)).transpose_axes(&[0, 2, 1, 3])?;
            let q = fast::rope(&q, CE_HD, false, 10000.0, 1.0, 0, None)?;
            let k = fast::rope(&k, CE_HD, false, 10000.0, 1.0, 0, None)?;
            let o = fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                0.125,
                ScaledDotProductAttentionMask::Array(&mask),
                None,
            )?;
            let o = ew.lin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, -1])?,
                &format!("{p}.out_proj"),
            )?;
            h = ops::add(&h, ops::multiply(&o, ew.g(&format!("{p}.ls1"))?)?)?;
            let n2 = fast::layer_norm(
                &h,
                ew.g(&format!("{p}.norm2.weight"))?,
                ew.g(&format!("{p}.norm2.bias"))?,
                1e-5,
            )?;
            let f = ew.lin(
                &gelu_approx(&ew.lin(&n2, &format!("{p}.fc1"))?)?,
                &format!("{p}.fc2"),
            )?;
            h = ops::add(&h, ops::multiply(&f, ew.g(&format!("{p}.ls2"))?)?)?;
        }
        // ConvDownsample1d (k 4, stride 2, edge padding, no bias)
        let x = self.sconv(&h, "down", 4, 2, true)?;
        // Split RVQ encode: rvq_first (1 codebook) and rvq_rest (its first 15
        // of 31; the residual is sequential, so stopping there is exact) both
        // start from x.
        let mut codes = Vec::new();
        for (name, books) in [("rvq_first", 0..1usize), ("rvq_rest", 1..CE_NQ)] {
            let w = ew.g(&format!("q.{name}.input_proj.weight"))?;
            let mut res = ops::conv1d(&x, w, 1, 0, 1, 1)?.reshape(&[-1, 256])?;
            for b in books {
                let (emb, c2) = &self.ce_books[b];
                let dot = ops::matmul(&res, emb.t())?;
                let idx = argmin_axis(ops::subtract(c2, &dot)?, -1, None)?;
                let qv = emb.take_axis(&idx, 0)?;
                res = ops::subtract(&res, &qv)?;
                codes.push(idx);
            }
        }
        Ok(ops::stack(&codes, -1)?.expand_dims(0)?)
    }

    /// The codes of the reference clip at `path`: from the cache if this
    /// file's content was encoded before, else loaded and encoded. Python
    /// encodes afresh for every prompt; the result is the same.
    pub(super) fn ref_codes(&self, path: &Path) -> R<Array> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let key: [u8; 32] = Sha256::digest(&bytes).into();
        if let Some(codes) = self.ref_codes.borrow().get(&key) {
            return Ok(codes.clone());
        }
        let audio = load_ref_audio(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let codes = self.encode_ref(&audio)?;
        codes.eval()?;
        let mut cache = self.ref_codes.borrow_mut();
        if cache.len() >= REF_CACHE {
            cache.clear();
        }
        cache.insert(key, codes.clone());
        Ok(codes)
    }

    // ---------------- codec decoder (Qwen3-TTS 12 Hz tokenizer) ----------------

    fn cconv(
        &self,
        st: &mut HashMap<String, Array>,
        x: &Array,
        p: &str,
        k: i32,
        dil: i32,
        groups: i32,
    ) -> R<Array> {
        // CausalConv1d.step: left context carried in a buffer.
        let pad = (k - 1) * dil;
        let x = if pad > 0 {
            let x = match st.remove(p) {
                Some(b) => ops::concatenate(&[b, x.clone()], 1)?,
                None => ops::pad(x, &[(0, 0), (pad, 0), (0, 0)][..], None, None)?,
            };
            let n = x.shape()[1];
            st.insert(p.to_string(), x.index((.., (n - pad).., ..)));
            x
        } else {
            x.clone()
        };
        let y = ops::conv1d(&x, self.cw.g(&format!("{p}.weight"))?, 1, 0, dil, groups)?;
        Ok(ops::add(&y, self.cw.g(&format!("{p}.bias"))?)?)
    }

    fn snake(&self, x: &Array, p: &str) -> R<Array> {
        let a = ops::exp(self.cw.g(&format!("{p}.alpha"))?)?;
        let b = ops::exp(self.cw.g(&format!("{p}.beta"))?)?;
        let inv = ops::divide(s(1.0, &b)?, ops::add(&b, s(1e-9, &b)?)?)?;
        let sn = ops::sin(ops::multiply(x, &a)?)?;
        Ok(ops::add(
            x,
            ops::multiply(&inv, ops::power(&sn, s(2.0, &sn)?)?)?,
        )?)
    }

    /// The codec's RMSNorm, with `x ** 2` (`power`) as in Python.
    fn crms(&self, x: &Array, p: &str) -> R<Array> {
        let var = ops::power(x, s(2.0, x)?)?.mean_axis(-1, true)?;
        let xn = ops::multiply(x, ops::rsqrt(ops::add(&var, s(CD_EPS, &var)?)?)?)?;
        Ok(ops::multiply(self.cw.g(&format!("{p}.weight"))?, &xn)?)
    }

    fn codec_transformer(&self, x: &Array, cache: &mut [KvCache], offset: i32) -> R<Array> {
        let cw = &self.cw;
        let l = x.shape()[1];
        let mut h = cw.lin(x, "decoder.pre_transformer.input_proj")?;
        let pos = Array::from_slice(
            &(offset..offset + l).map(|v| v as f32).collect::<Vec<_>>(),
            &[l, 1],
        );
        let freqs = ops::multiply(&pos, &self.cd_inv_freq)?; // [l, hd/2]
        let emb = ops::concatenate(&[freqs.clone(), freqs], -1)?;
        let (cos, sin) = (ops::cos(&emb)?, ops::sin(&emb)?);
        let rot = |t: &Array| -> R<Array> {
            let half = CD_HD / 2;
            let x1 = t.index((.., .., .., ..half));
            let x2 = t.index((.., .., .., half..));
            let rh = ops::concatenate(&[ops::negative(&x2)?, x1], -1)?;
            Ok(ops::add(
                ops::multiply(t, &cos)?,
                ops::multiply(&rh, &sin)?,
            )?)
        };
        let total = offset + l;
        let mask = if l > 1 {
            let mut m = vec![0f32; (l * total) as usize];
            for qi in 0..l {
                for ki in 0..total {
                    if offset + qi < ki {
                        m[(qi * total + ki) as usize] = f32::MIN;
                    }
                }
            }
            Some(Array::from_slice(&m, &[l, total]))
        } else {
            None
        };
        for (i, layer_cache) in cache.iter_mut().enumerate().take(CD_LAYERS) {
            let p = format!("decoder.pre_transformer.layers.{i}");
            let xn = self.crms(&h, &format!("{p}.input_layernorm"))?;
            let q = heads(
                &cw.lin(&xn, &format!("{p}.self_attn.q_proj"))?,
                1,
                l,
                CD_H,
                CD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let k = heads(
                &cw.lin(&xn, &format!("{p}.self_attn.k_proj"))?,
                1,
                l,
                CD_H,
                CD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(
                &cw.lin(&xn, &format!("{p}.self_attn.v_proj"))?,
                1,
                l,
                CD_H,
                CD_HD,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let (q, k) = (rot(&q)?, rot(&k)?);
            let (k, v) = layer_cache.update(k, v)?;
            let scale = (CD_HD as f32).powf(-0.5);
            let o = match &mask {
                Some(m) => fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Array(m),
                    None,
                )?,
                None => fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?,
            };
            let o = cw.lin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, -1])?,
                &format!("{p}.self_attn.o_proj"),
            )?;
            let hh = ops::add(
                &h,
                ops::multiply(cw.g(&format!("{p}.self_attn_layer_scale.scale"))?, &o)?,
            )?;
            let xn = self.crms(&hh, &format!("{p}.post_attention_layernorm"))?;
            let g = nn::silu(cw.lin(&xn, &format!("{p}.mlp.gate_proj"))?)?;
            let u = cw.lin(&xn, &format!("{p}.mlp.up_proj"))?;
            let f = cw.lin(&ops::multiply(&g, &u)?, &format!("{p}.mlp.down_proj"))?;
            h = ops::add(
                &hh,
                ops::multiply(cw.g(&format!("{p}.mlp_layer_scale.scale"))?, &f)?,
            )?;
        }
        let h = self.crms(&h, "decoder.pre_transformer.norm")?;
        cw.lin(&h, "decoder.pre_transformer.output_proj")
    }

    fn tconv(&self, x: &Array, p: &str, stride: i32) -> R<Array> {
        let y = ops::conv_transpose1d(x, self.cw.g(&format!("{p}.weight"))?, stride, 0, 1, 0, 1)?;
        Ok(ops::add(&y, self.cw.g(&format!("{p}.bias"))?)?)
    }

    /// `Qwen3TTSSpeechTokenizerDecoder.streaming_step`: decode only the new
    /// frames (codes: [n][16]), carrying conv buffers, overlap-add tails and
    /// the transformer KV cache in `st`.
    fn codec_step(&self, st: &mut CodecState, codes: &[Vec<i32>]) -> R<Array> {
        let cw = &self.cw;
        let n = codes.len() as i32;
        // Split RVQ decode: codebook 0 through rvq_first, 1-15 through rvq_rest.
        let mut quant: Vec<Array> = Vec::new();
        for (name, range) in [("rvq_first", 0..1usize), ("rvq_rest", 1..16usize)] {
            let mut acc = Array::zeros::<f32>(&[1, n, 256])?;
            for (li, q) in range.enumerate() {
                let ids =
                    Array::from_slice(&codes.iter().map(|f| f[q]).collect::<Vec<_>>(), &[1, n]);
                let e = cw
                    .g(&format!(
                        "decoder.quantizer.{name}.vq.layers.{li}.codebook.embed.weight"
                    ))?
                    .take_axis(&ids, 0)?;
                acc = ops::add(&acc, &e)?;
            }
            quant.push(ops::conv1d(
                &acc,
                cw.g(&format!("decoder.quantizer.{name}.output_proj.weight"))?,
                1,
                0,
                1,
                1,
            )?);
        }
        let mut h = ops::add(&quant[0], &quant[1])?; // [1, n, 512] NLC
        h = self.cconv(&mut st.bufs, &h, "decoder.pre_conv.conv", 3, 1, 1)?;
        h = self.codec_transformer(&h, &mut st.kv, st.offset)?;
        st.offset += n;
        for u in 0..2 {
            h = self.tconv(&h, &format!("decoder.upsample.{u}.0.conv"), 2)?;
            let p = format!("decoder.upsample.{u}.1");
            let r = h.clone();
            let mut x = self.cconv(&mut st.bufs, &h, &format!("{p}.dwconv.conv"), 7, 1, 1024)?;
            x = fast::layer_norm(
                &x,
                cw.g(&format!("{p}.norm.weight"))?,
                cw.g(&format!("{p}.norm.bias"))?,
                1e-6,
            )?;
            x = gelu(&cw.lin(&x, &format!("{p}.pwconv1"))?)?;
            x = cw.lin(&x, &format!("{p}.pwconv2"))?;
            h = ops::add(&r, ops::multiply(cw.g(&format!("{p}.gamma"))?, &x)?)?;
        }
        let mut x = self.cconv(&mut st.bufs, &h, "decoder.decoder.0.conv", 7, 1, 1)?;
        for (bi, &rate) in CD_RATES.iter().enumerate() {
            let p = format!("decoder.decoder.{}.block", bi + 1);
            x = self.snake(&x, &format!("{p}.0"))?;
            // DecoderBlockUpsample.step: overlap-add of the transposed conv's tail.
            let mut y = self.tconv(&x, &format!("{p}.1.conv"), rate)?;
            let key = format!("{p}.1.overflow");
            if let Some(ov) = st.bufs.remove(&key) {
                let ol = ov.shape()[1];
                y = ops::concatenate(
                    &[
                        ops::add(y.index((.., ..ol, ..)), &ov)?,
                        y.index((.., ol.., ..)),
                    ],
                    1,
                )?;
            }
            let yl = y.shape()[1];
            st.bufs.insert(key, y.index((.., (yl - rate).., ..)));
            x = y.index((.., ..(yl - rate), ..));
            for (ui, dil) in [(2, 1), (3, 3), (4, 9)] {
                let up = format!("{p}.{ui}");
                let r = x.clone();
                let mut t = self.snake(&x, &format!("{up}.act1"))?;
                t = self.cconv(&mut st.bufs, &t, &format!("{up}.conv1.conv"), 7, dil, 1)?;
                t = self.snake(&t, &format!("{up}.act2"))?;
                t = self.cconv(&mut st.bufs, &t, &format!("{up}.conv2.conv"), 1, 1, 1)?;
                x = ops::add(&t, &r)?;
            }
        }
        x = self.snake(&x, "decoder.decoder.5")?;
        x = self.cconv(&mut st.bufs, &x, "decoder.decoder.6.conv", 7, 1, 1)?;
        Ok(ops::clip(&x, (-1.0f32, 1.0f32))?.reshape(&[-1])?)
    }

    /// One codec step as samples; the step's graph is evaluated here.
    pub(super) fn decode_frames(&self, st: &mut CodecState, frames: &[Vec<i32>]) -> R<Vec<f32>> {
        let a = self.codec_step(st, frames)?;
        a.eval()?;
        Ok(a.as_slice::<f32>().to_vec())
    }

}

impl Model {
    // ---------------- generate ----------------

    /// mlx-audio's `generate(stream=True, streaming_interval=0.32)`.
    /// `emit` gets each chunk and returns `false` to stop.
    fn generate(&self, text: &str, p: &Params, emit: &mut dyn FnMut(Vec<f32>) -> bool) -> R<()> {
        p.check()?;
        if let Some(seed) = p.seed {
            random::seed(seed)?;
        }
        let speaker = p.speaker_tag();
        let instruct = p.instruct.as_deref().filter(|i| !i.is_empty());
        let use_cfg = instruct.is_some() && !matches!(p.cfg_scale, None | Some(1.0));
        let scale = p.cfg_scale.unwrap_or(1.0);
        let codes = match &p.reference {
            Some((path, _)) => Some(self.codec.ref_codes(path)?),
            None => None,
        };
        let reference = p
            .reference
            .as_ref()
            .zip(codes.as_ref())
            .map(|((_, ref_text), codes)| (codes, ref_text.as_str()));
        let segments = segments(text);
        for segment in &segments {
            // Under CFG the uncond prompt carries the reference too.
            let cond = self.prompt(&speaker, segment, instruct, reference)?;
            let uncond = if use_cfg {
                Some(self.prompt(&speaker, segment, None, reference)?)
            } else {
                None
            };
            let mut cc = caches(BB_LAYERS);
            let mut uc = caches(BB_LAYERS);
            let prompt_len = cond.shape()[1];
            let uprompt_len = uncond.as_ref().map(|u| u.shape()[1]).unwrap_or(0);
            let mut ch = self.backbone(&cond, &mut cc, 0)?;
            let mut uh = match &uncond {
                Some(u) => Some(self.backbone(u, &mut uc, 0)?),
                None => None,
            };
            let mut st = CodecState::new();
            let mut pending: Vec<Vec<i32>> = Vec::new();
            let mut firsts: Vec<i32> = Vec::new();
            let mut frames = 0usize;
            for _ in 0..p.max_tokens {
                let cl = self.w.qlin(&ch, "lm_head")?;
                let ul = match &uh {
                    Some(u) => Some(self.w.qlin(u, "lm_head")?),
                    None => None,
                };
                let logits = Self::cfg(cl, ul, scale)?;
                let logits = Self::repetition_penalty(&logits, &firsts, p.repetition_penalty)?;
                let logits = Self::mask_reserved(&logits)?;
                let first = Self::sample(&logits, p, true)?;
                if first == VOCAB {
                    break;
                }
                // The codebooks stay on the device between steps: each step's
                // graph is built while the GPU runs the one before, and the
                // frame is read back once. The same ops, so the same tokens
                // (with `depth_cache` off).
                let mut toks = Array::from_slice(&[0i32, first], &[1, 2]);
                let mut dc: Vec<DepthKv> = vec![None; DD_LAYERS];
                let mut udc: Vec<DepthKv> = vec![None; DD_LAYERS];
                for _ in 0..NCB - 1 {
                    let l = self.depth_logits(&toks, &ch, p.depth_cache.then_some(&mut dc[..]))?;
                    let ul = match &uh {
                        Some(u) => Some(self.depth_logits(
                            &toks,
                            u,
                            p.depth_cache.then_some(&mut udc[..]),
                        )?),
                        None => None,
                    };
                    let l = Self::mask_reserved(&Self::cfg(l, ul, scale)?)?;
                    let tok = Self::sample_array(&l, p, false)?.as_dtype(Dtype::Int32)?;
                    mlx_rs::transforms::async_eval([&tok])?;
                    toks = ops::concatenate(&[toks, tok.reshape(&[1, 1])?], 1)?;
                }
                let frame_a = toks.index((.., 1..));
                frame_a.eval()?;
                let frame: Vec<i32> = frame_a.as_slice::<i32>().to_vec();
                firsts.push(frame[0]);
                frames += 1;
                pending.push(frame.clone());
                if pending.len() >= CHUNK_FRAMES {
                    let chunk: Vec<Vec<i32>> = pending.drain(..CHUNK_FRAMES).collect();
                    if !emit(self.codec.decode_frames(&mut st, &chunk)?) {
                        return Ok(());
                    }
                }
                let e = self.audio_embed(&frame)?;
                // The frame just made is the `frames`th: it sits at this offset.
                ch = self.backbone(&e, &mut cc, prompt_len + frames as i32 - 1)?;
                if uh.is_some() {
                    uh = Some(self.backbone(&e, &mut uc, uprompt_len + frames as i32 - 1)?);
                }
            }
            if frames == 0 {
                // EOS before any frame: mlx-audio decodes one dummy frame of
                // ones (`_empty_audio`) rather than raising.
                let dummy = vec![vec![1i32; NCB as usize]];
                let audio = self.codec.decode_frames(&mut CodecState::new(), &dummy)?;
                if !emit(audio) {
                    return Ok(());
                }
            } else if !pending.is_empty() && !emit(self.codec.decode_frames(&mut st, &pending)?) {
                return Ok(());
            }
        }
        Ok(())
    }
}

impl Model {
    fn load(dir: &Path) -> R<Model> {
        let mut w = Array::load_safetensors(dir.join("model.safetensors"))?;
        w.retain(|k, _| {
            !k.starts_with("codec_model.")
                && !k.ends_with(".initialized")
                && !k.contains("rotary_emb.inv_freq")
        });
        // T5Gemma2 norm weights as `1.0 + w`, in the weight's own dtype (bf16).
        let mut te_norms = HashMap::new();
        for (k, v) in w.iter() {
            if k.starts_with("text_encoder.") && k.ends_with("norm.weight") {
                te_norms.insert(
                    k.trim_end_matches(".weight").to_string(),
                    ops::add(v, s(1.0, v)?)?,
                );
            }
        }
        // Llama3 rope frequencies for the depth decoder, computed as Python does.
        let ar = arange_f32(0, DD_HD / 2, DD_HD as f32)?;
        let freqs = ops::power(Array::from_f32(500000.0), &ar)?;
        let (factor, lo, hi, old) = (32.0f32, 0.001953125f32, 0.0078125f32, 16.0f32);
        let wl = ops::multiply(&freqs, Array::from_f32(2.0 * std::f32::consts::PI))?;
        let (lo_wl, hi_wl) = (old / lo, old / hi);
        let freqs = ops::select(
            &ops::gt(&wl, Array::from_f32(lo_wl))?,
            ops::multiply(&freqs, Array::from_f32(factor))?,
            &freqs,
        )?;
        let medium = ops::logical_and(
            &ops::gt(&wl, Array::from_f32(hi_wl))?,
            &ops::lt(&wl, Array::from_f32(lo_wl))?,
        )?;
        let smooth = ops::divide(
            ops::subtract(ops::divide(Array::from_f32(old), &wl)?, Array::from_f32(lo))?,
            Array::from_f32(hi - lo),
        )?;
        let one = Array::from_f32(1.0);
        let denom = ops::add(
            ops::divide(ops::subtract(&one, &smooth)?, Array::from_f32(factor))?,
            &smooth,
        )?;
        let dd_freqs = ops::select(&medium, ops::divide(&freqs, &denom)?, &freqs)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer: {e}"))?;
        let codec = Codec::load(&dir.join("audio_tokenizer/model.safetensors"))?;
        mlx_rs::transforms::eval(w.values().chain(te_norms.values()))?;
        Ok(Model {
            w: Weights::new(w),
            tok,
            dd_freqs,
            te_norms,
            codec,
        })
    }
}

impl Codec {
    /// Loads the speech tokenizer's `model.safetensors` at `path`, evaluated.
    pub(super) fn load(path: &Path) -> R<Codec> {
        // Codec decoder (the encoder goes to `load_encoder`): codebook embed =
        // embedding_sum / clip(usage);
        // conv weights from PyTorch layout to MLX's [out, kernel, in].
        let raw = Array::load_safetensors(path)?;
        let mut cw = HashMap::new();
        let mut usage: HashMap<String, Array> = HashMap::new();
        let mut sums: HashMap<String, Array> = HashMap::new();
        let mut enc_raw = HashMap::new();
        for (k, v) in raw {
            if k.starts_with("encoder.") {
                enc_raw.insert(k, v);
                continue;
            }
            if !k.starts_with("decoder.") {
                continue;
            }
            if let Some(b) = k.strip_suffix("._codebook.cluster_usage") {
                usage.insert(b.to_string(), v);
                continue;
            }
            if let Some(b) = k.strip_suffix("._codebook.embedding_sum") {
                sums.insert(b.to_string(), v);
                continue;
            }
            let is_t = (k.contains("upsample") && k.contains(".0.conv.weight"))
                || (k.contains("decoder.decoder") && k.contains("block.1.conv.weight"));
            let v = if v.ndim() == 3 && is_t {
                v.transpose_axes(&[1, 2, 0])?
            } else if v.ndim() == 3 && (k.contains("conv.weight") || k.contains("_proj.weight")) {
                v.transpose_axes(&[0, 2, 1])?
            } else {
                v
            };
            cw.insert(k, v);
        }
        for (b, u) in usage {
            let sum = sums
                .get(&b)
                .ok_or_else(|| format!("the codec has no embedding_sum for `{b}`"))?;
            let e = ops::divide(sum, ops::clip(u.expand_dims(-1)?, (1e-5f32, ()))?)?;
            cw.insert(format!("{b}.codebook.embed.weight"), e);
        }
        // `1.0 / (base ** (mx.arange(0, dim, 2, dtype=f32) / dim))`, on device.
        let ar = arange_f32(0, CD_HD / 2, CD_HD as f32)?;
        let cd_inv_freq = ops::divide(
            Array::from_f32(1.0),
            ops::power(Array::from_f32(CD_THETA), &ar)?,
        )?;
        let (ew, ce_books) = load_encoder(enc_raw)?;
        mlx_rs::transforms::eval(
            cw.values()
                .chain(ew.values())
                .chain(ce_books.iter().flat_map(|(e, c2)| [e, c2]))
                .chain([&cd_inv_freq]),
        )?;
        Ok(Codec {
            cw: Weights::new(cw),
            cd_inv_freq,
            ew: Weights::new(ew),
            ce_books,
            ref_codes: RefCell::new(HashMap::new()),
        })
    }
}

/// An encoder RVQ codebook's embeddings and their `|e|^2 / 2`.
type Codebook = (Array, Array);

/// The encoder half of `Qwen3TTSSpeechTokenizer.sanitize`, under short names:
/// conv weights [out, in, k] -> [out, k, in]; q/k/v concatenated into
/// `in_proj`; codebooks from `embedding_sum / max(cluster_usage, 1e-5)`.
fn load_encoder(raw: HashMap<String, Array>) -> R<(HashMap<String, Array>, Vec<Codebook>)> {
    let mut ew = HashMap::new();
    let conv = |v: &Array| -> R<Array> { Ok(v.swap_axes(-1, -2)?) };
    let seanet = |n: usize| -> Option<String> {
        Some(match n {
            0 => "seanet.init".into(),
            3 | 6 | 9 | 12 => format!("seanet.{}.down", n / 3 - 1),
            14 => "seanet.final".into(),
            _ => return None,
        })
    };
    let bad = |k: &str| format!("unexpected codec encoder weight `{k}`");
    let mut qkv: HashMap<usize, [Option<Array>; 3]> = HashMap::new();
    let mut books: HashMap<String, (Option<Array>, Option<Array>)> = HashMap::new();
    for (k, v) in raw {
        let parts: Vec<&str> = k.split('.').collect();
        let index = |i: usize| -> R<usize> {
            Ok(parts
                .get(i)
                .and_then(|p| p.parse().ok())
                .ok_or_else(|| bad(&k))?)
        };
        if let Some(rest) = k.strip_prefix("encoder.encoder.layers.") {
            let n = index(3)?;
            let leaf = parts.last().expect("split is never empty");
            let base = if rest.contains(".block.") {
                let li = [1, 4, 7, 10]
                    .iter()
                    .position(|&x| x == n)
                    .ok_or_else(|| bad(&k))?;
                let res = if parts.get(5) == Some(&"1") { 0 } else { 1 };
                format!("seanet.{li}.res{res}")
            } else {
                seanet(n).ok_or_else(|| bad(&k))?
            };
            let v = if v.ndim() == 3 { conv(&v)? } else { v };
            ew.insert(format!("{base}.{leaf}"), v);
        } else if k.starts_with("encoder.encoder_transformer.layers.") {
            let i = index(3)?;
            let rest = parts[4..].join(".");
            let name = match rest.as_str() {
                "self_attn.q_proj.weight" => {
                    qkv.entry(i).or_default()[0] = Some(v);
                    continue;
                }
                "self_attn.k_proj.weight" => {
                    qkv.entry(i).or_default()[1] = Some(v);
                    continue;
                }
                "self_attn.v_proj.weight" => {
                    qkv.entry(i).or_default()[2] = Some(v);
                    continue;
                }
                "self_attn.o_proj.weight" => "out_proj.weight",
                "mlp.fc1.weight" => "fc1.weight",
                "mlp.fc2.weight" => "fc2.weight",
                "input_layernorm.weight" => "norm1.weight",
                "input_layernorm.bias" => "norm1.bias",
                "post_attention_layernorm.weight" => "norm2.weight",
                "post_attention_layernorm.bias" => "norm2.bias",
                "self_attn_layer_scale.scale" => "ls1",
                "mlp_layer_scale.scale" => "ls2",
                _ => return Err(bad(&k).into()),
            };
            ew.insert(format!("tf.{i}.{name}"), v);
        } else if k == "encoder.downsample.conv.weight" {
            ew.insert("down.weight".into(), conv(&v)?);
        } else if let Some(rest) = k.strip_prefix("encoder.quantizer.") {
            let name = if rest.starts_with("semantic_") {
                "rvq_first"
            } else {
                "rvq_rest"
            };
            if rest.ends_with("input_proj.weight") {
                ew.insert(format!("q.{name}.input_proj.weight"), conv(&v)?);
            } else if rest.contains(".codebook.") && !rest.ends_with("initialized") {
                let e = books.entry(format!("{name}.{}", index(4)?)).or_default();
                if rest.ends_with("cluster_usage") {
                    e.0 = Some(v)
                } else {
                    e.1 = Some(v)
                }
            }
        }
    }
    for (i, [q, k, v]) in qkv {
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("codec encoder layer {i} lacks q, k or v").into());
        };
        ew.insert(
            format!("tf.{i}.in_proj.weight"),
            ops::concatenate(&[q, k, v], 0)?,
        );
    }
    let mut ce_books = Vec::new();
    for b in 0..CE_NQ {
        let key = if b == 0 {
            "rvq_first.0".to_string()
        } else {
            format!("rvq_rest.{}", b - 1)
        };
        let Some((Some(u), Some(sum))) = books.remove(&key) else {
            return Err(format!("the codec encoder has no codebook `{key}`").into());
        };
        let emb = ops::divide(
            &sum,
            ops::maximum(&u, Array::from_f32(1e-5))?.expand_dims(-1)?,
        )?;
        let c2 = ops::divide(ops::square(&emb)?.sum_axis(-1, None)?, Array::from_f32(2.0))?;
        ce_books.push((emb, c2));
    }
    Ok((ew, ce_books))
}

/// mlx-audio's `load_audio(path, sample_rate=24000)` for a mono 16-bit PCM
/// WAV (what naru-audio stores): s16 `/ 32768.0` as f32, and a higher rate
/// through the same kaiser_best polyphase FIR. Anything else is an error
/// rather than audio that differs from mlx-audio's.
pub(super) fn load_ref_audio(wav: &[u8]) -> R<Vec<f32>> {
    let mut r = hound::WavReader::new(std::io::Cursor::new(wav))?;
    let spec = r.spec();
    if (spec.sample_format, spec.bits_per_sample) != (hound::SampleFormat::Int, 16) {
        return Err("the native engine reads only 16-bit PCM WAV references".into());
    }
    if spec.channels != 1 {
        return Err("the native engine reads only mono references".into());
    }
    let x: Vec<f32> = r
        .samples::<i16>()
        .map(|v| v.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    if x.is_empty() {
        return Err("the reference has no audio".into());
    }
    let sr = spec.sample_rate as usize;
    let target = SAMPLE_RATE as usize;
    if sr == target {
        Ok(x)
    } else if sr > target {
        Ok(resample::resample_poly(&x, sr, target))
    } else {
        Err(format!("the native engine cannot upsample a {sr} Hz reference").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_lines_are_segments() {
        assert_eq!(segments("hello"), ["hello"]);
        assert_eq!(segments(" one \n\n two\n"), ["one", "two"]);
        // No non-empty line: the text as it came.
        assert_eq!(segments(" \n "), [" \n "]);
        assert_eq!(segments(""), [""]);
    }

    #[test]
    fn long_segments_split_at_sentences_then_words_then_hard() {
        // Sentences stay whole while they fit.
        assert_eq!(
            split_long_segment("Aaaa. Bbbb. Cccc.", 11),
            ["Aaaa. Bbbb.", "Cccc."]
        );
        // A sentence over the limit is cut at the last space from the second half.
        assert_eq!(
            split_long_segment("aaa bbb ccc ddd", 9),
            ["aaa bbb", "ccc ddd"]
        );
        // No space to use: a hard cut.
        assert_eq!(split_long_segment("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        // Full-width terminators end sentences with no space after them.
        assert_eq!(split_long_segment("你好。再见。", 3), ["你好。", "再见。"]);
        // Every chunk obeys the limit, whatever the text.
        let text = "word ".repeat(300);
        let chunks = split_long_segment(text.trim(), 600);
        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|c| c.chars().count() <= 600 && !c.is_empty())
        );
    }

    #[test]
    fn speaker_tags_are_bracketed() {
        let tag = |s: &str| {
            Params {
                speaker: s.to_string(),
                ..Params::default()
            }
            .speaker_tag()
        };
        assert_eq!(tag(""), "[S0]");
        assert_eq!(tag("af_heart"), "[S0]");
        assert_eq!(tag("S3"), "[S3]");
        assert_eq!(tag("[S9]"), "[S9]");
    }

    #[test]
    fn params_are_checked_like_generate() {
        assert!(Params::default().check().is_ok());
        let bad = |f: fn(&mut Params)| {
            let mut p = Params::default();
            f(&mut p);
            p.check().is_err()
        };
        assert!(bad(|p| p.max_tokens = 0));
        assert!(bad(|p| p.temperature = -1.0));
        assert!(bad(|p| p.top_p = 1.5));
        assert!(bad(|p| p.repetition_penalty = 0.0));
        assert!(bad(|p| p.cfg_scale = Some(f64::NAN)));
        assert!(bad(
            |p| p.reference = Some(("ref.wav".into(), String::new()))
        ));
        let p = Params {
            reference: Some(("ref.wav".into(), "Hello.".to_string())),
            ..Params::default()
        };
        assert!(p.check().is_ok());
    }

    fn wav(channels: u16, rate: u32, bits: u16, samples: &[i32]) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: bits,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
        for &v in samples {
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        out.into_inner()
    }

    #[test]
    fn references_load_as_mlx_audio_does_or_fail() {
        // 24 kHz s16 mono: s16 / 32768.
        let x = load_ref_audio(&wav(1, 24000, 16, &[0, 16384, -32768])).unwrap();
        assert_eq!(x, [0.0, 0.5, -1.0]);
        // A higher rate is resampled to 24 kHz.
        let x = load_ref_audio(&wav(1, 48000, 16, &[1000; 480])).unwrap();
        assert_eq!(x.len(), 240);
        // What would differ from mlx-audio's audio is an error.
        assert!(load_ref_audio(&wav(1, 24000, 16, &[])).is_err());
        assert!(load_ref_audio(&wav(2, 24000, 16, &[0, 0])).is_err());
        assert!(load_ref_audio(&wav(1, 24000, 24, &[0])).is_err());
        assert!(load_ref_audio(&wav(1, 16000, 16, &[0])).is_err());
        assert!(load_ref_audio(b"not a wav").is_err());
    }
}
