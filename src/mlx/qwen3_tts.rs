//! Qwen3-TTS Base (mlx-community's 8-bit, affine) in-process through mlx-rs
//! (naru task 2386), the native alternative to the sidecar's mlx-audio for
//! the two kept checkpoints, `qwen3-tts-0.6b-base-mlx` and
//! `qwen3-tts-1.7b-base-mlx`: a port of mlx-audio 0.5.6's
//! `tts/models/qwen3_tts` ICL path (`_prepare_icl_generation_inputs`,
//! `_generate_icl`, `_sample_token`, `talker.py`), as the sidecar drives it
//! for a cloned voice (`naru_audio_mlx/__main__.py`: `split_chunks`, a seed
//! per chunk, `primed`). The speech tokenizer (codec decoder and encoder) is
//! Breeze's ([`super::breeze::Codec`]), the same weights.
//!
//! As in [`super::breeze`], its sampled codes and audio are bit-identical to
//! the sidecar's at the same seed (naru task 2386, both sizes), so Python's
//! numerics are mirrored: the talker's RoPE angles from a matmul, cos/sin
//! cast to the activations' dtype; additive causal masks; `nn.log_softmax`'s
//! max shift under top-p; every projection run on the same shapes as Python
//! (the quantized kernels differ by row count).
//!
//! The speaker encoder (naru task 2387, `speaker_encoder.py`'s ECAPA-TDNN on
//! `mel_spectrogram`) turns the reference clip into the x-vector; it and the
//! clip's codec codes are cached by the clip's SHA-256, so a voice is encoded
//! once; so is the codec decoder primed with those codes. Audio streams as
//! the sidecar's does (naru task 2388): each 0.32 s of frames is decoded and
//! handed on as soon as it is generated.
//!
//! MLX runs on one thread that owns the model: [`Qwen3Tts`] sends it work.

use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::{self, LinspaceOptions, argpartition_axis};
use mlx_rs::random::{self, RandomState};
use mlx_rs::{Array, Dtype, nn};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::breeze::{
    CACHE_LIMIT, CHUNK_FRAMES, Codec, CodecState, KvCache, Quant, R, REF_CACHE, Weights, caches,
    heads, load_ref_audio, s,
};

pub use super::breeze::SAMPLE_RATE;

/// `generate`'s own default.
pub const MAX_TOKENS: usize = 4096;
/// `_sample_token`'s `repetition_context_size`.
const REPETITION_CONTEXT: usize = 64;
/// `generate` raises the ICL repetition penalty to at least this.
const ICL_REPETITION_PENALTY: f64 = 1.5;
const HEAD_DIM: i32 = 128;
const NCB: usize = 16;

/// What one synthesis takes, with `generate`'s own defaults in
/// [`Params::default`] (the sidecar's `CLONE_SAMPLING` is the caller's).
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// The voice to clone: the reference WAV and its transcript.
    pub reference: (PathBuf, String),
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    /// Seeds MLX's global random state before each chunk, as the sidecar does.
    pub seed: Option<u64>,
}

impl Params {
    pub fn new(reference: (PathBuf, String)) -> Self {
        Params {
            reference,
            max_tokens: MAX_TOKENS,
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
            seed: None,
        }
    }

    fn check(&self) -> Result<(), String> {
        if self.reference.1.trim().is_empty() {
            return Err("Qwen3-TTS voice cloning needs the reference's transcript".to_string());
        }
        Ok(())
    }
}

enum Event {
    Chunk(Vec<f32>),
    Done(Result<(), String>),
}

type Job = (String, Params, mpsc::Sender<Event>);

/// A loaded model on its MLX thread; dropping it stops the thread and frees
/// the weights.
pub struct Qwen3Tts {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    resident_bytes: u64,
}

impl Qwen3Tts {
    /// Loads `dir`: `config.json`, `model.safetensors`, `vocab.json`,
    /// `merges.txt`, `tokenizer_config.json` and `speech_tokenizer/`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let metallib = super::native::metallib_path()?;
        if !metallib.is_file() {
            return Err(format!("{} missing", metallib.display()));
        }
        let config = Config::read(dir)?;
        let dir = dir.to_path_buf();
        let (loaded_tx, loaded_rx) = mpsc::channel();
        let (jobs, rx) = mpsc::channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("mlx-qwen3-tts".to_string())
            .spawn(move || {
                let model = (|| -> R<(Model, u64)> {
                    super::native::raise_cache_limit(CACHE_LIMIT)?;
                    let before = mlx_rs::memory::active_memory()?;
                    let model = Model::load(&dir, config)?;
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
                    // A panic in MLX must not take the daemon's other models
                    // with it.
                    let done = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        model.synth(&text, &params, &mut |chunk| {
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
            Ok(Ok(resident_bytes)) => Ok(Qwen3Tts {
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

    /// Synthesises `text`, handing `on_chunk` 24 kHz mono samples; it
    /// returns `false` to stop (then this returns `Ok`).
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
                // An empty chunk only probes whether the caller is still
                // there (a send to a dropped receiver fails).
                Ok(Event::Chunk(samples)) if samples.is_empty() => {}
                Ok(Event::Chunk(samples)) => {
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

impl Drop for Qwen3Tts {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Text chunks (the sidecar's `split_sentences` and `split_chunks`)
// ---------------------------------------------------------------------------

/// A sentence shorter than this many characters joins its neighbour.
const MIN_SENTENCE: usize = 12;
/// A chunk is closed once it holds this many words.
const MIN_CHUNK_WORDS: usize = 15;

/// `_join`: a space between, but not after a CJK terminator.
fn join(a: &str, b: &str) -> String {
    if a.is_empty() || a.ends_with(['。', '！', '？']) {
        format!("{a}{b}")
    } else {
        format!("{a} {b}")
    }
}

/// `split_sentences`: split after `. ! ?` followed by whitespace (which is
/// dropped), after `。！？`, and at newlines; a piece under [`MIN_SENTENCE`]
/// characters joins the next (the last, the one before).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\n' {
            pieces.push(std::mem::take(&mut cur));
            while chars.peek() == Some(&'\n') {
                chars.next();
            }
            continue;
        }
        cur.push(c);
        match c {
            '。' | '！' | '？' => pieces.push(std::mem::take(&mut cur)),
            '.' | '!' | '?' if chars.peek().is_some_and(|n| n.is_whitespace()) => {
                pieces.push(std::mem::take(&mut cur));
                // `\s+` is consumed by the split (a newline included).
                while chars.peek().is_some_and(|n| n.is_whitespace()) {
                    chars.next();
                }
            }
            _ => {}
        }
    }
    pieces.push(cur);
    let mut out: Vec<String> = Vec::new();
    let mut carry = String::new();
    for piece in pieces.iter().map(|p| p.trim()) {
        if piece.is_empty() {
            continue;
        }
        let piece = join(&carry, piece);
        if piece.chars().count() < MIN_SENTENCE {
            carry = piece;
        } else {
            out.push(piece);
            carry.clear();
        }
    }
    if !carry.is_empty() {
        match out.last_mut() {
            Some(last) => *last = join(last, &carry),
            None => out.push(carry),
        }
    }
    if out.is_empty() {
        vec![text.to_string()]
    } else {
        out
    }
}

/// `_words`: whitespace words, plus half a word per Han, kana or hangul
/// character.
fn words(text: &str) -> usize {
    let cjk = text
        .chars()
        .filter(|&c| {
            matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{ac00}'..='\u{d7af}')
        })
        .count();
    text.split_whitespace().count() + cjk / 2
}

/// `split_chunks`: whole sentences joined until a chunk holds
/// [`MIN_CHUNK_WORDS`] words; a short last chunk joins the one before.
pub fn split_chunks(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for sentence in split_sentences(text) {
        current = join(&current, &sentence);
        if words(&current) >= MIN_CHUNK_WORDS {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        match chunks.last_mut() {
            Some(last) => *last = join(last, &current),
            None => chunks.push(current),
        }
    }
    chunks
}

// ---------------------------------------------------------------------------
// Config and tokenizer
// ---------------------------------------------------------------------------

/// What `config.json` says, for the parts that differ between sizes.
#[derive(Debug, Clone)]
struct Config {
    layers: usize,
    heads: i32,
    kv_heads: i32,
    eps: f32,
    theta: f32,
    vocab: i32,
    cp_layers: usize,
    cp_heads: i32,
    cp_kv_heads: i32,
    cp_eps: f32,
    cp_theta: f32,
    codec_eos: i32,
    codec_bos: i32,
    codec_pad: i32,
    codec_nothink: i32,
    codec_think_bos: i32,
    codec_think_eos: i32,
    tts_bos: i32,
    tts_eos: i32,
    tts_pad: i32,
    quant: Quant,
}

impl Config {
    fn read(dir: &Path) -> Result<Config, String> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let c: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let bad = |what: &str| format!("{}: no usable `{what}`", path.display());
        let int = |v: &Value, what: &str| -> Result<i32, String> {
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| bad(what))
        };
        let float = |v: &Value, what: &str| -> Result<f32, String> {
            v.as_f64().map(|n| n as f32).ok_or_else(|| bad(what))
        };
        if c["tts_model_type"] != "base" {
            return Err(format!(
                "only Qwen3-TTS Base has a native port ({}: tts_model_type {})",
                path.display(),
                c["tts_model_type"]
            ));
        }
        let q = &c["quantization"];
        if !(q["mode"] == "affine" && q["bits"] == 8) {
            return Err(format!(
                "only the 8-bit affine checkpoints have a native port (config.json quantization: {q})"
            ));
        }
        let t = &c["talker_config"];
        let cp = &t["code_predictor_config"];
        for (cfg, what) in [(t, "talker_config"), (cp, "code_predictor_config")] {
            if int(&cfg["head_dim"], "head_dim")? != HEAD_DIM {
                return Err(bad(&format!("{what}.head_dim")));
            }
        }
        if int(&t["num_code_groups"], "num_code_groups")? != NCB as i32 {
            return Err(bad("talker_config.num_code_groups"));
        }
        Ok(Config {
            layers: int(&t["num_hidden_layers"], "num_hidden_layers")? as usize,
            heads: int(&t["num_attention_heads"], "num_attention_heads")?,
            kv_heads: int(&t["num_key_value_heads"], "num_key_value_heads")?,
            eps: float(&t["rms_norm_eps"], "rms_norm_eps")?,
            theta: float(&t["rope_theta"], "rope_theta")?,
            vocab: int(&t["vocab_size"], "vocab_size")?,
            cp_layers: int(&cp["num_hidden_layers"], "num_hidden_layers")? as usize,
            cp_heads: int(&cp["num_attention_heads"], "num_attention_heads")?,
            cp_kv_heads: int(&cp["num_key_value_heads"], "num_key_value_heads")?,
            cp_eps: float(&cp["rms_norm_eps"], "rms_norm_eps")?,
            cp_theta: float(&cp["rope_theta"], "rope_theta")?,
            codec_eos: int(&t["codec_eos_token_id"], "codec_eos_token_id")?,
            codec_bos: int(&t["codec_bos_id"], "codec_bos_id")?,
            codec_pad: int(&t["codec_pad_id"], "codec_pad_id")?,
            codec_nothink: int(&t["codec_nothink_id"], "codec_nothink_id")?,
            codec_think_bos: int(&t["codec_think_bos_id"], "codec_think_bos_id")?,
            codec_think_eos: int(&t["codec_think_eos_id"], "codec_think_eos_id")?,
            tts_bos: int(&c["tts_bos_token_id"], "tts_bos_token_id")?,
            tts_eos: int(&c["tts_eos_token_id"], "tts_eos_token_id")?,
            tts_pad: int(&c["tts_pad_token_id"], "tts_pad_token_id")?,
            quant: Quant::Affine {
                group: int(&q["group_size"], "quantization.group_size")?,
                bits: 8,
            },
        })
    }
}

/// The Qwen2 fast tokenizer `transformers` converts the repo's slow one
/// (`vocab.json`, `merges.txt`, `tokenizer_config.json`) into: NFC, the
/// Qwen2 split regex, byte-level BPE, and the added tokens.
fn tokenizer(dir: &Path) -> R<tokenizers::Tokenizer> {
    let read = |name: &str| -> R<String> {
        let p = dir.join(name);
        Ok(std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)
    };
    let vocab: Value = serde_json::from_str(&read("vocab.json")?)?;
    let merges: Vec<Value> = read("merges.txt")?
        .lines()
        .filter(|l| !l.starts_with("#version") && !l.is_empty())
        .map(|l| match l.split_once(' ') {
            Some((a, b)) => Ok(serde_json::json!([a, b])),
            None => Err(format!("merges.txt: bad line `{l}`")),
        })
        .collect::<Result<_, _>>()?;
    let config: Value = serde_json::from_str(&read("tokenizer_config.json")?)?;
    let mut added: Vec<(u64, Value)> = config["added_tokens_decoder"]
        .as_object()
        .ok_or("tokenizer_config.json has no added_tokens_decoder")?
        .iter()
        .map(|(id, t)| {
            let id: u64 = id.parse()?;
            let mut t = t.clone();
            t["id"] = id.into();
            Ok((id, t))
        })
        .collect::<R<_>>()?;
    added.sort_by_key(|(id, _)| *id);
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added.into_iter().map(|(_, t)| t).collect::<Vec<_>>(),
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
             "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": merges},
    });
    let tok: tokenizers::Tokenizer = json
        .to_string()
        .parse()
        .map_err(|e| format!("tokenizer: {e}"))?;
    Ok(tok)
}

// ---------------------------------------------------------------------------
// Ops, in Python's exact form
// ---------------------------------------------------------------------------

/// `apply_rotary_pos_emb`: `q * cos + rotate_half(q) * sin`, same for `k`;
/// cos/sin are `[L, D]`, broadcast over batch and heads. Python compiles it;
/// fused or not, every op rounds to bf16 the same, so plain ops are used
/// (likewise for the other compiled helpers below).
fn rope(q: &Array, k: &Array, cos: &Array, sin: &Array) -> R<(Array, Array)> {
    let cos = cos.expand_dims_axes(&[0, 1])?;
    let sin = sin.expand_dims_axes(&[0, 1])?;
    let rot = |x: &Array| -> R<Array> {
        let half = x.shape()[3] / 2;
        let x1 = x.index((.., .., .., ..half));
        let x2 = x.index((.., .., .., half..));
        let rh = ops::concatenate(&[ops::negative(&x2)?, x1], -1)?;
        Ok(ops::add(ops::multiply(x, &cos)?, ops::multiply(&rh, &sin)?)?)
    };
    Ok((rot(q)?, rot(k)?))
}

/// RoPE's cos and sin for positions `offset..offset + l`, `[l, D]` in `dtype`:
/// `inv_freq * pos` in f32, doubled, then cos/sin cast down. The talker's
/// (`mrope`, `TalkerRotaryEmbedding`) takes the product as a matmul of
/// `[3, 1, D/2, 1] @ [3, 1, 1, l]`, which the GPU runs at lower precision
/// than a multiply: done the same way, it gives the same angles. Its three
/// position rows are equal, so the interleaving picks the same values.
fn rope_cos_sin(
    inv_freq: &Array,
    offset: i32,
    l: i32,
    dtype: Dtype,
    mrope: bool,
) -> R<(Array, Array)> {
    let pos: Vec<f32> = (offset..offset + l).map(|v| v as f32).collect();
    let freqs = if mrope {
        let half = inv_freq.shape()[0];
        let inv = ops::broadcast_to(inv_freq.reshape(&[1, 1, half, 1])?, &[3, 1, half, 1])?;
        let pos = ops::broadcast_to(Array::from_slice(&pos, &[1, 1, 1, l]), &[3, 1, 1, l])?;
        ops::matmul(&inv, &pos)?
            .swap_axes(2, 3)?
            .index((0, 0))
    } else {
        ops::multiply(Array::from_slice(&pos, &[l, 1]), inv_freq)?
    };
    let emb = ops::concatenate(&[freqs.clone(), freqs], -1)?;
    Ok((
        ops::cos(&emb)?.as_dtype(dtype)?,
        ops::sin(&emb)?.as_dtype(dtype)?,
    ))
}

/// `swiglu`: `silu(gate) * x`.
fn swiglu(gate: &Array, x: &Array) -> R<Array> {
    Ok(ops::multiply(nn::silu(gate)?, x)?)
}

/// `create_additive_causal_mask(l)` in f32, cast to `dtype`.
fn causal_mask(l: i32, dtype: Dtype) -> R<Array> {
    let mut m = vec![0f32; (l * l) as usize];
    for q in 0..l {
        for k in (q + 1)..l {
            m[(q * l + k) as usize] = f32::MIN;
        }
    }
    Ok(Array::from_slice(&m, &[l, l]).as_dtype(dtype)?)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Model {
    cfg: Config,
    /// `talker.*`.
    w: Weights,
    /// `speaker_encoder.*` (not quantized).
    spk: Weights,
    /// `mel_filters(24000, 1024, 128, 0, 12000, "slaney", "slaney")`,
    /// `[n_fft/2 + 1, n_mels]`.
    mel_fb: Array,
    /// `hanning(1024)` (symmetric), as Python builds it.
    window: Array,
    /// Reference x-vectors by the clip's SHA-256.
    xvectors: RefCell<HashMap<[u8; 32], Array>>,
    /// The codec decoder primed with a reference's codes, by the codes'
    /// SHA-256: priming takes longer than the first 0.32 s of speech.
    primed: RefCell<HashMap<[u8; 32], CodecState>>,
    tok: tokenizers::Tokenizer,
    codec: Codec,
    inv_freq: Array,
    cp_inv_freq: Array,
    /// Whether the code predictor projects the talker's hidden size down to
    /// its own (`small_to_mtp_projection`, the 1.7B).
    mtp_projection: bool,
}

/// A transformer stack's shape and weight prefix.
struct Stack<'a> {
    prefix: &'a str,
    layers: usize,
    heads: i32,
    kv_heads: i32,
    eps: f32,
    inv_freq: &'a Array,
    /// The talker's multimodal RoPE (see [`rope_cos_sin`]).
    mrope: bool,
}

impl Model {
    fn load(dir: &Path, cfg: Config) -> R<Model> {
        let mut map = Array::load_safetensors(dir.join("model.safetensors"))?;
        let spk = map
            .iter()
            .filter_map(|(k, v)| {
                let k = k.strip_prefix("speaker_encoder.")?;
                Some((k.to_string(), v.clone()))
            })
            .collect();
        let spk = Weights::quantized(spk, cfg.quant);
        map.retain(|k, _| k.starts_with("talker."));
        let w = Weights::quantized(map, cfg.quant);
        let mel_fb = mel_filters()?;
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (N_FFT - 1) as f64).cos()))
                    as f32
            })
            .collect();
        let window = Array::from_slice(&window, &[N_FFT as i32]);
        let codec = Codec::load(&dir.join("speech_tokenizer/model.safetensors"))?;
        let tok = tokenizer(dir)?;
        // `1.0 / (base ** (mx.arange(0, dim, 2, dtype=f32) / dim))`, on device.
        let inv = |theta: f32| -> R<Array> {
            let v: Vec<f32> = (0..HEAD_DIM / 2).map(|i| (2 * i) as f32).collect();
            let ar = ops::divide(
                Array::from_slice(&v, &[HEAD_DIM / 2]),
                Array::from_f32(HEAD_DIM as f32),
            )?;
            Ok(ops::divide(
                Array::from_f32(1.0),
                ops::power(Array::from_f32(theta), &ar)?,
            )?)
        };
        let inv_freq = inv(cfg.theta)?;
        let cp_inv_freq = inv(cfg.cp_theta)?;
        mlx_rs::transforms::eval(w.values().chain(spk.values()).chain([
            &inv_freq,
            &cp_inv_freq,
            &mel_fb,
            &window,
        ]))?;
        let mtp_projection = w
            .g("talker.code_predictor.small_to_mtp_projection.weight")
            .is_ok();
        Ok(Model {
            mtp_projection,
            cfg,
            w,
            spk,
            mel_fb,
            window,
            xvectors: RefCell::new(HashMap::new()),
            primed: RefCell::new(HashMap::new()),
            tok,
            codec,
            inv_freq,
            cp_inv_freq,
        })
    }

    fn talker_stack(&self) -> Stack<'_> {
        Stack {
            prefix: "talker.model",
            layers: self.cfg.layers,
            heads: self.cfg.heads,
            kv_heads: self.cfg.kv_heads,
            eps: self.cfg.eps,
            inv_freq: &self.inv_freq,
            mrope: true,
        }
    }

    fn cp_stack(&self) -> Stack<'_> {
        Stack {
            prefix: "talker.code_predictor.model",
            layers: self.cfg.cp_layers,
            heads: self.cfg.cp_heads,
            kv_heads: self.cfg.cp_kv_heads,
            eps: self.cfg.cp_eps,
            inv_freq: &self.cp_inv_freq,
            mrope: false,
        }
    }

    /// `Qwen3TTSTalkerModel` / `CodePredictorModel`: `x` `[1, L, H]` through
    /// `st`'s layers and final norm, every position.
    fn stack(&self, st: &Stack, x: &Array, cache: &mut [KvCache]) -> R<Array> {
        let w = &self.w;
        let l = x.shape()[1];
        let offset = cache.first().map_or(0, |c| c.offset);
        let (cos, sin) = rope_cos_sin(st.inv_freq, offset, l, x.dtype(), st.mrope)?;
        let mask = if l > 1 {
            Some(causal_mask(l, x.dtype())?)
        } else {
            None
        };
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let mut h = x.clone();
        for (i, layer_cache) in cache.iter_mut().enumerate().take(st.layers) {
            let p = format!("{}.layers.{i}", st.prefix);
            let xn = w.rms(&h, &format!("{p}.input_layernorm"), st.eps)?;
            let q = w
                .rms(
                    &heads(&w.qlin(&xn, &format!("{p}.self_attn.q_proj"))?, 1, l, st.heads, HEAD_DIM)?,
                    &format!("{p}.self_attn.q_norm"),
                    st.eps,
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = w
                .rms(
                    &heads(&w.qlin(&xn, &format!("{p}.self_attn.k_proj"))?, 1, l, st.kv_heads, HEAD_DIM)?,
                    &format!("{p}.self_attn.k_norm"),
                    st.eps,
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(&w.qlin(&xn, &format!("{p}.self_attn.v_proj"))?, 1, l, st.kv_heads, HEAD_DIM)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let (q, k) = rope(&q, &k, &cos, &sin)?;
            let (k, v) = layer_cache.update(k, v)?;
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
            let o = w.qlin(
                &o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, l, -1])?,
                &format!("{p}.self_attn.o_proj"),
            )?;
            let hh = ops::add(&h, &o)?;
            let xn = w.rms(&hh, &format!("{p}.post_attention_layernorm"), st.eps)?;
            let g = w.qlin(&xn, &format!("{p}.mlp.gate_proj"))?;
            let u = w.qlin(&xn, &format!("{p}.mlp.up_proj"))?;
            h = ops::add(&hh, w.qlin(&swiglu(&g, &u)?, &format!("{p}.mlp.down_proj"))?)?;
        }
        w.rms(&h, &format!("{}.norm", st.prefix), st.eps)
    }

    fn emb(&self, key: &str, ids: &Array) -> R<Array> {
        Ok(self.w.g(key)?.take_axis(ids, 0)?)
    }

    fn codec_emb(&self, ids: &Array) -> R<Array> {
        self.emb("talker.model.codec_embedding.weight", ids)
    }

    /// `code_predictor.codec_embedding[i]`.
    fn cp_emb(&self, i: usize, ids: &Array) -> R<Array> {
        self.emb(
            &format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
            ids,
        )
    }

    /// `text_projection(text_embedding(ids))`, `ids` `[1, n]`.
    fn text_proj(&self, ids: &[i32]) -> R<Array> {
        let ids = Array::from_slice(ids, &[1, ids.len() as i32]);
        let e = self.emb("talker.model.text_embedding.weight", &ids)?;
        let h = nn::silu(self.w.qlin(&e, "talker.text_projection.linear_fc1")?)?;
        self.w.qlin(&h, "talker.text_projection.linear_fc2")
    }

    fn encode(&self, text: &str) -> R<Vec<i32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| format!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i32).collect())
    }

    /// `_prepare_icl_generation_inputs` (language `auto`): the prefill
    /// embeddings. `codes` is the reference's `[1, T, 16]`.
    fn icl_prompt(&self, text: &str, ref_text: &str, codes: &Array, xvector: &Array) -> R<Array> {
        let c = &self.cfg;
        let ref_ids = self.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n"))?;
        let target_ids = self.encode(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        if ref_ids.len() < 5 || target_ids.len() < 8 {
            return Err("the chat template tokenized short".into());
        }
        let ref_text_ids = &ref_ids[3..ref_ids.len() - 2];
        let text_ids = &target_ids[3..target_ids.len() - 5];
        let tts = self.text_proj(&[c.tts_bos, c.tts_eos, c.tts_pad])?;
        let tts_bos = tts.index((.., 0..1, ..));
        let tts_eos = tts.index((.., 1..2, ..));
        let tts_pad = tts.index((.., 2..3, ..));
        let combined: Vec<i32> = ref_text_ids.iter().chain(text_ids).copied().collect();
        let text_embed = ops::concatenate(&[self.text_proj(&combined)?, tts_eos], 1)?;
        let text_lens = text_embed.shape()[1];
        // codec_bos, then the sum of every codebook's embedding of the ref codes.
        let mut ref_embed = self.codec_emb(&codes.index((.., .., 0)))?;
        for i in 0..NCB - 1 {
            ref_embed = ops::add(&ref_embed, self.cp_emb(i, &codes.index((.., .., (i + 1) as i32)))?)?;
        }
        let one = |id: i32| Array::from_slice(&[id], &[1, 1]);
        let codec_icl = ops::concatenate(&[self.codec_emb(&one(c.codec_bos))?, ref_embed], 1)?;
        let codec_lens = codec_icl.shape()[1];
        let hidden = codec_icl.shape()[2];
        let codec_pad = self.codec_emb(&one(c.codec_pad))?;
        let icl = ops::concatenate(
            &[
                ops::add(&text_embed, ops::broadcast_to(&codec_pad, &[1, text_lens, hidden])?)?,
                ops::add(&codec_icl, ops::broadcast_to(&tts_pad, &[1, codec_lens, hidden])?)?,
            ],
            1,
        )?;
        let prefill = [c.codec_nothink, c.codec_think_bos, c.codec_think_eos];
        let prefix = self.codec_emb(&Array::from_slice(&prefill, &[1, 3]))?;
        let suffix = self.codec_emb(&Array::from_slice(&[c.codec_pad, c.codec_bos], &[1, 2]))?;
        let spk = xvector.as_dtype(prefix.dtype())?.reshape(&[1, 1, -1])?;
        let prefix = ops::concatenate(&[prefix, spk, suffix], 1)?;
        let role = self.text_proj(&target_ids[..3])?;
        let pad_count = prefix.shape()[1] - 2;
        let pads = ops::broadcast_to(&tts_pad, &[1, pad_count, hidden])?;
        let combined_prefix = ops::add(
            ops::concatenate(&[pads, tts_bos], 1)?,
            prefix.index((.., ..-1, ..)),
        )?;
        Ok(ops::concatenate(&[role, combined_prefix, icl], 1)?)
    }

    /// `_sample_token` on the last position of `logits` `[1, L, V]`: a lazy
    /// `[1, 1]` token.
    fn sample(
        rng: &mut RandomState,
        logits: &Array,
        p: &Params,
        repetition_penalty: f64,
        recent: &[Array],
        suppress: Option<&Array>,
    ) -> R<Array> {
        let mut logits = logits.index((.., -1, ..));
        let vocab = logits.shape()[1];
        if let Some(idx) = suppress {
            let ninf = Array::from_f32(f32::NEG_INFINITY).as_dtype(logits.dtype())?;
            logits = logits.put_along_axis(idx, &ninf, -1)?;
        }
        // Python penalises each distinct recent token once; here the ids stay
        // on the device, repeats and all. A repeat reads and writes the same
        // penalised value, so the logits come out the same.
        if !recent.is_empty() && repetition_penalty != 1.0 {
            let idx = ops::concatenate(recent, 1)?;
            let sel = logits.take_along_axis(&idx, -1)?;
            let pen = s(repetition_penalty as f32, &sel)?;
            let penalized = ops::select(
                &ops::lt(&sel, s(0.0, &sel)?)?,
                ops::multiply(&sel, &pen)?,
                ops::divide(&sel, &pen)?,
            )?;
            logits = logits.put_along_axis(&idx, &penalized, -1)?;
        }
        if p.temperature <= 0.0 {
            return Ok(argmax_axis(&logits, -1, true)?);
        }
        if p.temperature != 1.0 {
            logits = ops::divide(&logits, s(p.temperature as f32, &logits)?)?;
        }
        if p.top_k > 0 && (p.top_k as i32) < vocab {
            let k = p.top_k as i32;
            let idx = argpartition_axis(&ops::negative(&logits)?, k - 1, -1)?.index((.., k..));
            let ninf = Array::from_f32(f32::NEG_INFINITY).as_dtype(logits.dtype())?;
            logits = logits.put_along_axis(&idx, &ninf, -1)?;
        }
        if 0.0 < p.top_p && p.top_p < 1.0 {
            let lp = top_p(&log_softmax(&logits)?, p.top_p)?;
            let ninf = s(f32::NEG_INFINITY, &logits)?;
            logits = ops::select(&ops::eq(&lp, s(f32::NEG_INFINITY, &lp)?)?, &ninf, &logits)?;
        }
        let tok = categorical(&logits, &rng.next_key()?)?;
        Ok(tok.reshape(&[1, 1])?)
    }

    /// One frame's 16 codes after the talker's `hidden` `[1, L, H]` and
    /// `logits`: codebook 0 sampled, then the code predictor's 15. All lazy.
    fn frame(
        &self,
        rng: &mut RandomState,
        hidden: &Array,
        logits: &Array,
        p: &Params,
        recent: &[Array],
        xs: &GenConsts,
    ) -> R<Vec<Array>> {
        let first = Self::sample(
            rng,
            logits,
            p,
            p.repetition_penalty.max(ICL_REPETITION_PENALTY),
            recent, Some(&xs.suppress))?;
        let mut codes = vec![first];
        let st = self.cp_stack();
        let mut cache = caches(st.layers);
        let last = hidden.index((.., -1.., ..));
        for step in 0..NCB - 1 {
            let input = if step == 0 {
                ops::concatenate(&[last.clone(), self.codec_emb(&codes[0])?], 1)?
            } else {
                self.cp_emb(step - 1, &codes[step])?
            };
            let input = if self.mtp_projection {
                self.w.qlin(&input, "talker.code_predictor.small_to_mtp_projection")?
            } else {
                input
            };
            let x = self.stack(&st, &input, &mut cache)?;
            let l = self
                .w
                .qlin(&x, &format!("talker.code_predictor.lm_head.{step}"))?;
            codes.push(Self::sample(rng, &l, p, 1.0, &[], None)?);
        }
        Ok(codes)
    }

    /// `_generate_icl(stream=True)` for one chunk: the frames until EOS or
    /// `max_tokens`, each [`CHUNK_FRAMES`] (0.32 s) decoded through `decoder` and
    /// emitted as soon as it is read back, the rest at the end. `false` if
    /// the synthesis was cancelled.
    /// Pipelined: each frame's graph is queued on the GPU before the frame
    /// before it is read back, so building one overlaps running the other,
    /// except for the frame that fills a chunk.
    /// The EOS check is a frame late, so one frame too many is sampled at
    /// the end and dropped (its random draws come after every kept one).
    fn generate(
        &self,
        rng: &mut RandomState,
        prompt: &Array,
        p: &Params,
        xs: &GenConsts,
        decoder: &mut CodecState,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> R<bool> {
        let st = self.talker_stack();
        let mut cache = caches(st.layers);
        let mut input = prompt.clone();
        // Frames read back and not yet decoded.
        let mut frames = Vec::new();
        // The last frames' codebook-0 tokens, for the repetition penalty.
        let mut recent: Vec<Array> = Vec::new();
        let mut queued: Option<Array> = None;
        // `false` once the frame read back is EOS.
        let keep = |all: Array, frames: &mut Vec<Vec<i32>>| -> R<bool> {
            let frame = all.as_slice::<i32>().to_vec();
            if frame[0] == self.cfg.codec_eos {
                return Ok(false);
            }
            frames.push(frame);
            Ok(true)
        };
        for _ in 0..p.max_tokens {
            // A cancelled synthesis stops here, not after the whole chunk.
            if !emit(Vec::new()) {
                return Ok(false);
            }
            // The frame that fills a chunk is read back and decoded before the
            // next frame is queued: queued after it, the decode would wait on
            // the GPU for that frame too.
            if frames.len() == CHUNK_FRAMES - 1
                && let Some(prev) = queued.take()
            {
                if !keep(prev, &mut frames)? {
                    break;
                }
                if !emit(self.codec.decode_frames(decoder, &std::mem::take(&mut frames))?) {
                    return Ok(false);
                }
            }
            let hidden = self.stack(&st, &input, &mut cache)?;
            let logits = self.w.qlin(&hidden, "talker.codec_head")?;
            let codes = self.frame(rng, &hidden, &logits, p, &recent, xs)?;
            let mut embed = self.codec_emb(&codes[0])?;
            for (i, c) in codes[1..].iter().enumerate() {
                embed = ops::add(&embed, self.cp_emb(i, c)?)?;
            }
            input = ops::add(&xs.tts_pad, &embed)?;
            let all = ops::concatenate(&codes, 1)?.as_dtype(Dtype::Int32)?;
            mlx_rs::transforms::async_eval([&input, &all])?;
            if let Some(prev) = queued.take()
                && !keep(prev, &mut frames)?
            {
                break;
            }
            recent.push(codes[0].clone());
            if recent.len() > REPETITION_CONTEXT {
                recent.remove(0);
            }
            queued = Some(all);
        }
        if let Some(prev) = queued {
            keep(prev, &mut frames)?;
        }
        Ok(frames.is_empty() || emit(self.codec.decode_frames(decoder, &frames)?))
    }

    /// The sidecar's `synth` for a cloned voice: per chunk of `text`, the
    /// seed, and ICL generation into the streaming decoder primed with the
    /// reference codes ([`Model::primed`]), fed 0.32 s at a time.
    fn synth(&self, text: &str, p: &Params, emit: &mut dyn FnMut(Vec<f32>) -> bool) -> R<()> {
        p.check()?;
        let (wav, ref_text) = &p.reference;
        let xvector = self.xvector(wav)?;
        let codes = self.codec.ref_codes(wav)?;
        let primed = self.primed(&codes)?;
        let xs = GenConsts::new(self)?;
        for chunk in split_chunks(text) {
            // The sidecar seeds MLX's global key before each chunk; sampling
            // splits a key off it per draw, as this does.
            let mut rng = match p.seed {
                Some(seed) => RandomState::with_seed(seed)?,
                None => RandomState::new()?,
            };
            let prompt = self.icl_prompt(&chunk, ref_text, &codes, &xvector)?;
            if !self.generate(&mut rng, &prompt, p, &xs, &mut primed.clone(), emit)? {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// What every step of a generation reuses.
struct GenConsts {
    /// `tts_pad_embed`: the text side once the text is used up (the ICL
    /// prompt carries all of it).
    tts_pad: Array,
    /// Codebook-0 ids sampling never picks: the top 1024, but EOS.
    suppress: Array,
}

impl GenConsts {
    fn new(m: &Model) -> R<Self> {
        let c = &m.cfg;
        let tts = m.text_proj(&[c.tts_bos, c.tts_eos, c.tts_pad])?;
        let ids: Vec<i32> = (c.vocab - 1024..c.vocab).filter(|&i| i != c.codec_eos).collect();
        Ok(GenConsts {
            tts_pad: tts.index((.., 2..3, ..)),
            suppress: Array::from_slice(&ids, &[1, ids.len() as i32]),
        })
    }
}

/// mlx.nn.log_softmax: shifted by the max first, then `x - logsumexp(x)`
/// (not mlx-audio's other samplers' bare `x - logsumexp(x)`, which keeps a
/// different set of tokens under top-p).
fn log_softmax(x: &Array) -> R<Array> {
    let x = ops::subtract(x, x.max_axis(-1, true)?)?;
    Ok(ops::subtract(&x, x.logsumexp_axis(-1, true)?)?)
}

/// `categorical_sampling(logits, 1.0)`: `categorical(logits * (1 / 1.0))`
/// with `key`, split off the random state as each Python draw does.
fn categorical(logits: &Array, key: &Array) -> R<Array> {
    let one = s(1.0, logits)?;
    Ok(random::categorical(ops::multiply(logits, &one)?, None, None, key)?)
}

/// `apply_top_p`: keep the tokens whose cumulative probability, summed from
/// the least likely, exceeds `1 - top_p`.
fn top_p(logprobs: &Array, top_p: f64) -> R<Array> {
    let probs = ops::exp(logprobs)?;
    let sorted = ops::argsort_axis(logprobs, -1)?;
    let sorted_probs = probs.take_along_axis(&sorted, -1)?;
    let cumulative = ops::cumsum(&sorted_probs, -1, None, None)?;
    let n = sorted.shape()[sorted.ndim() - 1];
    let ar = Array::arange::<_, i32>(0, n, None)?.as_dtype(sorted.dtype())?;
    let inverse = ops::zeros_like(&sorted)?.put_along_axis(&sorted, &ar, -1)?;
    let cumulative = cumulative.take_along_axis(&inverse, -1)?;
    let ninf = s(f32::NEG_INFINITY, logprobs)?;
    Ok(ops::select(
        &ops::gt(&cumulative, s((1.0 - top_p) as f32, &cumulative)?)?,
        logprobs,
        &ninf,
    )?)
}

// ---------------------------------------------------------------------------
// Speaker encoder (`speaker_encoder.py`, `mel_spectrogram`)
// ---------------------------------------------------------------------------

const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: i32 = 128;
const F_MAX: f64 = 12000.0;
/// `enc_dilations` of the three SE-Res2Net blocks (`enc_kernel_sizes` 3).
const SE_DILATIONS: [i32; 3] = [2, 3, 4];
const RES2NET_SCALE: i32 = 8;
const ASP_EPS: f32 = 1e-12;

/// mlx-audio's `mel_filters(24000, 1024, 128, 0, 12000, norm="slaney",
/// mel_scale="slaney")` in float32 on device, as Python runs it, but
/// returned `[n_fft/2 + 1, n_mels]` (the `.T` `mel_spectrogram` takes).
fn mel_filters() -> R<Array> {
    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    let m_max = min_log_mel + (F_MAX / 1000.0).ln() / logstep;
    let n_freqs = (N_FFT / 2 + 1) as i32;
    let all_freqs = ops::linspace::<f64, f32>(
        0.0,
        F_MAX,
        LinspaceOptions {
            count: n_freqs,
            endpoint: true,
        },
    )?;
    let m_pts = ops::linspace::<f64, f32>(
        0.0,
        m_max,
        LinspaceOptions {
            count: N_MELS + 2,
            endpoint: true,
        },
    )?;
    // mel_to_hz, slaney.
    let lin = ops::multiply(Array::from_f32(f_sp as f32), &m_pts)?;
    let log = ops::multiply(
        Array::from_f32(1000.0),
        ops::exp(ops::multiply(
            Array::from_f32(logstep as f32),
            ops::subtract(&m_pts, Array::from_f32(min_log_mel as f32))?,
        )?)?,
    )?;
    let ge = ops::ge(&m_pts, Array::from_f32(min_log_mel as f32))?;
    let f_pts = ops::select(&ge, &log, &lin)?;
    let n = N_MELS + 2;
    let f_diff = ops::subtract(f_pts.index(1..), f_pts.index(..n - 1))?;
    let slopes = ops::subtract(f_pts.reshape(&[1, n])?, all_freqs.reshape(&[n_freqs, 1])?)?;
    let down = ops::divide(
        ops::negative(slopes.index((.., ..n - 2)))?,
        f_diff.index(..n - 2),
    )?;
    let up = ops::divide(slopes.index((.., 2..)), f_diff.index(1..))?;
    let fb = ops::maximum(ops::zeros_like(&down)?, ops::minimum(&down, &up)?)?;
    let enorm = ops::divide(
        Array::from_f32(2.0),
        ops::subtract(f_pts.index(2..n), f_pts.index(..N_MELS))?,
    )?;
    Ok(ops::multiply(&fb, enorm.reshape(&[1, N_MELS])?)?)
}

/// `x[:, 1:pad+1][:, ::-1]`, x, `x[:, -(pad+1):-1][:, ::-1]` along axis 1.
fn reflect_pad(x: &Array, pad: i32) -> R<Array> {
    if pad <= 0 {
        return Ok(x.clone());
    }
    let t = x.shape()[1];
    if pad >= t {
        return Err("the reference is too short".into());
    }
    let left: Vec<i32> = (1..=pad).rev().collect();
    let right: Vec<i32> = (t - pad - 1..t - 1).rev().collect();
    let take = |idx: &[i32]| x.take_axis(Array::from_slice(idx, &[idx.len() as i32]), 1);
    Ok(ops::concatenate(
        &[take(&left)?, x.clone(), take(&right)?],
        1,
    )?)
}

impl Model {
    /// The reference's x-vector, cached by the clip's SHA-256.
    fn xvector(&self, path: &Path) -> R<Array> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let key: [u8; 32] = Sha256::digest(&bytes).into();
        if let Some(x) = self.xvectors.borrow().get(&key) {
            eprintln!(
                "naru-audio: qwen3-tts: reference cache hit, not re-encoding {}",
                path.display()
            );
            return Ok(x.clone());
        }
        let audio = load_ref_audio(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let mels = self.log_mel(&audio)?;
        mels.eval()?;
        let x = self.speaker_embedding(&mels)?;
        x.eval()?;
        let mut cache = self.xvectors.borrow_mut();
        if cache.len() >= REF_CACHE {
            cache.clear();
        }
        cache.insert(key, x.clone());
        Ok(x)
    }

    /// The sidecar's `primed`: the codec decoder after the reference's codes
    /// went through it (their audio dropped), cached by the codes' SHA-256.
    fn primed(&self, codes: &Array) -> R<CodecState> {
        let c = codes.as_dtype(Dtype::Int32)?;
        c.eval()?;
        let mut hash = Sha256::new();
        for code in c.as_slice::<i32>() {
            hash.update(code.to_le_bytes());
        }
        let key: [u8; 32] = hash.finalize().into();
        if let Some(st) = self.primed.borrow().get(&key) {
            return Ok(st.clone());
        }
        let frames: Vec<Vec<i32>> =
            c.as_slice::<i32>().chunks(NCB).map(<[i32]>::to_vec).collect();
        let mut st = CodecState::new();
        self.codec.decode_frames(&mut st, &frames)?;
        st.detach()?;
        let mut cache = self.primed.borrow_mut();
        if cache.len() >= REF_CACHE {
            cache.clear();
        }
        cache.insert(key, st.clone());
        Ok(st)
    }

    /// `mel_spectrogram`: reflect-padded by (n_fft - hop) / 2, an uncentred
    /// Hann STFT, `sqrt(|X|^2 + 1e-9)`, slaney mel, `log(clip(1e-5))`;
    /// `[1, frames, n_mels]`.
    fn log_mel(&self, audio: &[f32]) -> R<Array> {
        let pad = (N_FFT - HOP) / 2;
        if audio.len() <= pad + 1 {
            return Err("the reference is too short".into());
        }
        let n = audio.len();
        let mut x = Vec::with_capacity(n + 2 * pad);
        x.extend(audio[1..=pad].iter().rev());
        x.extend_from_slice(audio);
        x.extend(audio[n - pad - 1..n - 1].iter().rev());
        if x.len() < N_FFT {
            return Err("the reference is too short".into());
        }
        let t = 1 + (x.len() - N_FFT) / HOP;
        let mut frames = Vec::with_capacity(t * N_FFT);
        for f in 0..t {
            frames.extend_from_slice(&x[f * HOP..f * HOP + N_FFT]);
        }
        let frames = Array::from_slice(&frames, &[t as i32, N_FFT as i32]);
        let spec = mlx_rs::fft::rfft(ops::multiply(&frames, &self.window)?, None, -1)?;
        let mag = ops::abs(&spec)?;
        let mag = ops::sqrt(ops::add(ops::power(&mag, s(2.0, &mag)?)?, s(1e-9, &mag)?)?)?;
        let mel = ops::matmul(&mag, &self.mel_fb)?;
        let mel = ops::log(ops::maximum(&mel, s(1e-5, &mel)?)?)?;
        Ok(mel.reshape(&[1, t as i32, N_MELS])?)
    }

    /// nn.Conv1d on NLC.
    fn spk_conv(&self, x: &Array, p: &str, dil: i32) -> R<Array> {
        let y = ops::conv1d(x, self.spk.g(&format!("{p}.weight"))?, 1, 0, dil, 1)?;
        Ok(ops::add(&y, self.spk.g(&format!("{p}.bias"))?)?)
    }

    /// TimeDelayNetBlock on NCL: reflect "same" padding, conv, ReLU.
    fn tdnn(&self, x: &Array, p: &str, dil: i32) -> R<Array> {
        let k = self.spk.g(&format!("{p}.conv.weight"))?.shape()[1];
        let x = x.transpose_axes(&[0, 2, 1])?;
        let x = reflect_pad(&x, (k - 1) * dil / 2)?;
        let y = self.spk_conv(&x, &format!("{p}.conv"), dil)?;
        Ok(nn::relu(y.transpose_axes(&[0, 2, 1])?)?)
    }

    /// SqueezeExcitationRes2NetBlock.
    fn se_res2net(&self, x: &Array, p: &str, dil: i32) -> R<Array> {
        let h = self.tdnn(x, &format!("{p}.tdnn1"), 1)?;
        let chunks = h.split_equal(RES2NET_SCALE, 1)?;
        let mut outs: Vec<Array> = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let rb = |x: &Array| self.tdnn(x, &format!("{p}.res2net_block.blocks.{}", i - 1), dil);
            let out = match i {
                0 => chunk.clone(),
                1 => rb(chunk)?,
                _ => rb(&ops::add(chunk, outs.last().expect("i > 1"))?)?,
            };
            outs.push(out);
        }
        let h = ops::concatenate(&outs, 1)?;
        let h = self.tdnn(&h, &format!("{p}.tdnn2"), 1)?;
        // SqueezeExcitationBlock.
        let m = h.mean_axis(2, true)?.transpose_axes(&[0, 2, 1])?;
        let se = nn::relu(self.spk_conv(&m, &format!("{p}.se_block.conv1"), 1)?)?;
        let se = ops::sigmoid(self.spk_conv(&se, &format!("{p}.se_block.conv2"), 1)?)?;
        let h = ops::multiply(&h, se.transpose_axes(&[0, 2, 1])?)?;
        Ok(ops::add(&h, x)?)
    }

    /// AttentiveStatisticsPooling: `[1, 2C, 1]`.
    fn asp(&self, x: &Array) -> R<Array> {
        let sh = x.shape().to_vec();
        let eps = s(ASP_EPS, x)?;
        let mean = x.mean_axis(2, true)?;
        let std = ops::sqrt(ops::add(x.var_axis(2, true, 0)?, &eps)?)?;
        let a = ops::concatenate(
            &[
                x.clone(),
                ops::broadcast_to(&mean, &sh)?,
                ops::broadcast_to(&std, &sh)?,
            ],
            1,
        )?;
        let a = ops::tanh(self.tdnn(&a, "asp.tdnn", 1)?)?;
        let a = self.spk_conv(&a.transpose_axes(&[0, 2, 1])?, "asp.conv", 1)?;
        let a = ops::softmax_axis(a.transpose_axes(&[0, 2, 1])?, 2, None)?;
        let mean = ops::multiply(&a, x)?.sum_axis(2, true)?;
        let d = ops::subtract(x, &mean)?;
        let var = ops::multiply(&a, ops::power(&d, s(2.0, &d)?)?)?.sum_axis(2, true)?;
        let std = ops::sqrt(ops::maximum(&var, &eps)?)?;
        Ok(ops::concatenate(&[mean, std], 1)?)
    }

    /// Qwen3TTSSpeakerEncoder on `[1, frames, n_mels]`: `[1, enc_dim]`.
    fn speaker_embedding(&self, mels: &Array) -> R<Array> {
        let mut x = self.tdnn(&mels.transpose_axes(&[0, 2, 1])?, "blocks.0", 1)?;
        let mut hidden = Vec::with_capacity(SE_DILATIONS.len());
        for (i, dil) in SE_DILATIONS.into_iter().enumerate() {
            x = self.se_res2net(&x, &format!("blocks.{}", i + 1), dil)?;
            hidden.push(x.clone());
        }
        let x = self.tdnn(&ops::concatenate(&hidden, 1)?, "mfa", 1)?;
        let x = self.asp(&x)?;
        let x = self.spk_conv(&x.transpose_axes(&[0, 2, 1])?, "fc", 1)?;
        Ok(x.transpose_axes(&[0, 2, 1])?.squeeze_axes(&[-1])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_match_the_sidecar() {
        // (text, split_sentences, split_chunks) from the sidecar's own Python.
        let cases: &[(&str, &[&str], &[&str])] = &[
            ("Sure, will do.", &["Sure, will do."], &["Sure, will do."]),
            (
                "Hi. How are you today?\n\nFine.",
                &["Hi. How are you today? Fine."],
                &["Hi. How are you today? Fine."],
            ),
            (
                "One!  Two?\nThree... and four.  ",
                &["One! Two? Three... and four."],
                &["One! Two? Three... and four."],
            ),
            (
                "Wait.\n\n\nOkay then, see you at noon tomorrow.",
                &["Wait. Okay then, see you at noon tomorrow."],
                &["Wait. Okay then, see you at noon tomorrow."],
            ),
            (
                "你好。今天天气很好！我们去公园散步吧，好不好？",
                &["你好。今天天气很好！我们去公园散步吧，好不好？"],
                &["你好。今天天气很好！我们去公园散步吧，好不好？"],
            ),
            (
                "Dr. Smith said no. Really?! Yes.",
                &["Dr. Smith said no.", "Really?! Yes."],
                &["Dr. Smith said no. Really?! Yes."],
            ),
            ("", &[""], &[]),
            ("   ", &["   "], &["   "]),
            ("a.b.c. d", &["a.b.c. d"], &["a.b.c. d"]),
            ("End with newline\n", &["End with newline"], &["End with newline"]),
            (
                "Line one\nline two is a little longer than twelve.\nshort",
                &["Line one line two is a little longer than twelve. short"],
                &["Line one line two is a little longer than twelve. short"],
            ),
            (
                "I went through the build logs from last night, and the failure comes from a test that depends on the system clock. It passes in the morning and fails after six in the evening, which is why nobody could reproduce it during the day.",
                &[
                    "I went through the build logs from last night, and the failure comes from a test that depends on the system clock.",
                    "It passes in the morning and fails after six in the evening, which is why nobody could reproduce it during the day.",
                ],
                &[
                    "I went through the build logs from last night, and the failure comes from a test that depends on the system clock.",
                    "It passes in the morning and fails after six in the evening, which is why nobody could reproduce it during the day.",
                ],
            ),
        ];
        for (text, sentences, chunks) in cases {
            assert_eq!(split_sentences(text), *sentences, "{text:?}");
            assert_eq!(split_chunks(text), *chunks, "{text:?}");
        }
    }

    #[test]
    fn params_need_a_transcript() {
        let mut p = Params::new(("ref.wav".into(), " ".into()));
        assert!(p.check().is_err());
        p.reference.1 = "Hello there.".into();
        assert_eq!(p.check(), Ok(()));
    }
}
