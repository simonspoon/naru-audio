//! Parakeet TDT in-process through mlx-rs (naru task 2379): a port of parakeet_mlx 0.5.2's
//! inference path (log-mel, dw_striding subsampling, rel-pos conformer,
//! LSTM prediction net, joint, greedy TDT decode), from the 2367 prototype.
//!
//! The catalog's fp32 weights are cast to bf16 (as `from_pretrained` does)
//! and every Linear/Embedding is quantized to 8 bits, group 64, at load: the
//! same weights task 2366's `nn.quantize` wrote.
//!
//! MLX runs on one thread that owns the model: [`Parakeet`] sends it work.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;

use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::{self, dequantize, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype, fast, nn};
use serde_json::Value;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 8;
/// MLX's buffer cache, capped: uncapped it grows to ~2 GB (task 2367), for
/// ~10 % of latency.
const CACHE_LIMIT: usize = 64 << 20;

type Job = (Vec<f32>, mpsc::Sender<Result<String, String>>);

/// A loaded model on its MLX thread; dropping it stops the thread and frees
/// the weights.
pub struct Parakeet {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    /// What the load added to MLX's active memory.
    resident_bytes: u64,
}

impl Parakeet {
    /// Loads `dir`'s `model.safetensors` and `config.json`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        // Without a metallib MLX exits the process at the first GPU op.
        let metallib = super::native::metallib_path()?;
        if !metallib.is_file() {
            return Err(format!("{} missing", metallib.display()));
        }
        let dir = dir.to_path_buf();
        let (loaded_tx, loaded_rx) = mpsc::channel();
        let (jobs, rx) = mpsc::channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("mlx-parakeet".to_string())
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
                for (pcm, reply) in rx {
                    let _ = reply.send(model.transcribe(&pcm).map_err(|e| e.to_string()));
                }
                drop(model);
                let _ = mlx_rs::memory::clear_cache();
            })
            .map_err(|e| format!("cannot start the MLX thread: {e}"))?;
        match loaded_rx.recv() {
            Ok(Ok(resident_bytes)) => Ok(Parakeet {
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

    /// One utterance of 16 kHz mono samples to text.
    pub fn transcribe(&self, pcm16k: &[f32]) -> Result<String, String> {
        let (reply, answer) = mpsc::channel();
        let jobs = self.jobs.as_ref().expect("set until drop");
        jobs.send((pcm16k.to_vec(), reply))
            .map_err(|_| "the MLX thread is gone".to_string())?;
        answer
            .recv()
            .map_err(|_| "the MLX thread died decoding".to_string())?
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl Drop for Parakeet {
    fn drop(&mut self) {
        // Closing the queue ends the thread's loop.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Model {
    w: HashMap<String, Array>,
    vocab: Vec<String>,
    durations: Vec<usize>,
    max_symbols: usize,
    d_model: i32,
    heads: i32,
    layers: usize,
    conv_pad: i32,
    sub_channels: i32,
    pred_hidden: i32,
    pred_layers: usize,
    hop: usize,
    win: usize,
    n_fft: usize,
    /// `[n_mels, n_fft/2 + 1]` slaney mel filterbank.
    fb: Array,
    /// Hann over `win`, zero-padded to `n_fft`.
    window: Vec<f32>,
}

fn int(v: &Value, what: &str) -> R<i64> {
    v.as_i64().ok_or_else(|| format!("config.json: {what} is not an integer").into())
}

impl Model {
    fn load(dir: &Path) -> R<Self> {
        let cfg: Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
        let pre = &cfg["preprocessor"];
        let enc = &cfg["encoder"];
        let pred = &cfg["decoder"]["prednet"];
        let sr = int(&pre["sample_rate"], "preprocessor.sample_rate")? as f64;
        let n_fft = int(&pre["n_fft"], "preprocessor.n_fft")? as usize;
        let n_mels = int(&pre["features"], "preprocessor.features")? as usize;
        let win_s = pre["window_size"].as_f64().ok_or("config.json: preprocessor.window_size")?;
        let hop_s = pre["window_stride"].as_f64().ok_or("config.json: preprocessor.window_stride")?;
        let win = (win_s * sr).round() as usize;
        let hop = (hop_s * sr).round() as usize;
        let vocab: Vec<String> = cfg["joint"]["vocabulary"]
            .as_array()
            .ok_or("config.json: joint.vocabulary")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("config.json: joint.vocabulary"))
            .collect::<Result<_, _>>()?;
        let durations: Vec<usize> = cfg["decoding"]["durations"]
            .as_array()
            .ok_or("config.json: decoding.durations")?
            .iter()
            .map(|v| v.as_u64().map(|d| d as usize).ok_or("config.json: decoding.durations"))
            .collect::<Result<_, _>>()?;

        // fp32 file -> bf16, Linear/Embedding (every 2-D `.weight`) quantized.
        let mut w = HashMap::new();
        for (k, v) in Array::load_safetensors(dir.join("model.safetensors"))? {
            let v = v.as_dtype(Dtype::Bfloat16)?;
            match k.strip_suffix(".weight") {
                Some(prefix) if v.ndim() == 2 => {
                    let (q, scales, biases) = quantize(&v, GROUP_SIZE, BITS)?;
                    w.insert(k.clone(), q);
                    w.insert(format!("{prefix}.scales"), scales);
                    w.insert(format!("{prefix}.biases"), biases);
                }
                _ => {
                    w.insert(k, v);
                }
            }
        }
        mlx_rs::transforms::eval(w.values())?;

        let mut window: Vec<f32> = (0..win)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / win as f64).cos()) as f32)
            .collect();
        window.resize(n_fft, 0.0);
        let fb = Array::from_slice(
            &slaney_mel_fb(sr, n_fft, n_mels),
            &[n_mels as i32, (n_fft / 2 + 1) as i32],
        );
        Ok(Model {
            w,
            vocab,
            durations,
            max_symbols: int(&cfg["decoding"]["greedy"]["max_symbols"], "decoding.greedy.max_symbols")? as usize,
            d_model: int(&enc["d_model"], "encoder.d_model")? as i32,
            heads: int(&enc["n_heads"], "encoder.n_heads")? as i32,
            layers: int(&enc["n_layers"], "encoder.n_layers")? as usize,
            conv_pad: (int(&enc["conv_kernel_size"], "encoder.conv_kernel_size")? as i32 - 1) / 2,
            sub_channels: int(&enc["subsampling_conv_channels"], "encoder.subsampling_conv_channels")? as i32,
            pred_hidden: int(&pred["pred_hidden"], "decoder.prednet.pred_hidden")? as i32,
            pred_layers: int(&pred["pred_rnn_layers"], "decoder.prednet.pred_rnn_layers")? as usize,
            hop,
            win,
            n_fft,
            fb,
            window,
        })
    }

    fn g(&self, k: &str) -> R<&Array> {
        self.w.get(k).ok_or_else(|| format!("missing weight {k}").into())
    }

    /// nn.QuantizedLinear with an optional bias.
    fn linear(&self, x: &Array, p: &str) -> R<Array> {
        let y = quantized_matmul(
            x,
            self.g(&format!("{p}.weight"))?,
            self.g(&format!("{p}.scales"))?,
            self.w.get(&format!("{p}.biases")),
            true,
            GROUP_SIZE,
            BITS,
        )?;
        Ok(match self.w.get(&format!("{p}.bias")) {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }

    fn layer_norm(&self, x: &Array, p: &str) -> R<Array> {
        Ok(fast::layer_norm(
            x,
            self.g(&format!("{p}.weight"))?,
            self.g(&format!("{p}.bias"))?,
            1e-5,
        )?)
    }

    /// parakeet_mlx `get_logmel`: pre-emphasis, reflect-padded STFT,
    /// (|re| + |im|)^2, slaney mel, log, per-feature normalisation.
    fn logmel(&self, audio: &[f32]) -> R<Array> {
        let (n_fft, hop, win) = (self.n_fft, self.hop, self.win);
        let mut x = Vec::with_capacity(audio.len());
        x.push(audio[0]);
        for i in 1..audio.len() {
            x.push(audio[i] - 0.97 * audio[i - 1]);
        }
        let p = n_fft / 2;
        let n = x.len();
        let mut xp = Vec::with_capacity(n + 2 * p);
        // Reflect: index p - i clamps for clips under p samples.
        for i in 0..p {
            xp.push(x[(p - i).min(n - 1)]);
        }
        xp.extend_from_slice(&x);
        for i in 0..p {
            xp.push(x[n.saturating_sub(2 + i)]);
        }
        let t = (xp.len() - win + hop) / hop;
        let mut frames = vec![0f32; t * n_fft];
        for f in 0..t {
            for k in 0..n_fft {
                let s = f * hop + k;
                let v = if s < xp.len() { xp[s] } else { 0.0 };
                frames[f * n_fft + k] = v * self.window[k];
            }
        }
        let frames = Array::from_slice(&frames, &[t as i32, n_fft as i32]);
        let spec = mlx_rs::fft::rfft(&frames, None, -1)?;
        let mag = ops::add(ops::abs(spec.real()?)?, ops::abs(spec.imag()?)?)?;
        let mel = ops::matmul(&self.fb, ops::square(&mag)?.t())?;
        let mel = ops::log(ops::add(&mel, Array::from_f32(1e-5))?)?;
        let mean = mel.mean_axis(1, true)?;
        let std = ops::std_axis(&mel, 1, true, 0)?;
        let norm = ops::divide(
            ops::subtract(&mel, &mean)?,
            ops::add(&std, Array::from_f32(1e-5))?,
        )?;
        Ok(norm.t().expand_dims(0)?) // [1, t, n_mels]
    }

    fn subsample(&self, mel: &Array) -> R<Array> {
        // [1, T, mels] -> NHWC [1, T, mels, 1]
        let x = mel.expand_dims(-1)?;
        let conv = |x: &Array, i: usize, stride: i32, pad: i32, groups: i32| -> R<Array> {
            let y = ops::conv2d(
                x,
                self.g(&format!("encoder.pre_encode.conv.{i}.weight"))?,
                (stride, stride),
                (pad, pad),
                None,
                groups,
            )?;
            Ok(ops::add(&y, self.g(&format!("encoder.pre_encode.conv.{i}.bias"))?)?)
        };
        let c = self.sub_channels;
        let x = nn::relu(conv(&x, 0, 2, 1, 1)?)?;
        let x = conv(&x, 2, 2, 1, c)?;
        let x = nn::relu(conv(&x, 3, 1, 0, 1)?)?;
        let x = conv(&x, 5, 2, 1, c)?;
        let x = nn::relu(conv(&x, 6, 1, 0, 1)?)?;
        // [1, T', F', C] -> [1, T', C, F'] -> [1, T', C*F']
        let s = x.shape().to_vec();
        let x = x
            .transpose_axes(&[0, 1, 3, 2])?
            .reshape(&[s[0], s[1], s[3] * s[2]])?;
        self.linear(&x, "encoder.pre_encode.out")
    }

    /// Relative sinusoidal positions `t - 1 .. -(t - 1)`: `[1, 2t - 1, d]`.
    fn pos_emb(&self, t: i32) -> Array {
        let d = self.d_model as usize;
        let n = (2 * t - 1) as usize;
        let mut pe = vec![0f32; n * d];
        for j in 0..n {
            let pos = (t - 1 - j as i32) as f32;
            for i in 0..d / 2 {
                let div = ((2 * i) as f32 * -(10000f32.ln() / d as f32)).exp();
                pe[j * d + 2 * i] = (pos * div).sin();
                pe[j * d + 2 * i + 1] = (pos * div).cos();
            }
        }
        Array::from_slice(&pe, &[1, n as i32, self.d_model])
    }

    fn attention(&self, x: &Array, pos: &Array, p: &str) -> R<Array> {
        let (h, d) = (self.heads, self.d_model);
        let hd = d / h;
        let t = x.shape()[1];
        let heads = |y: Array, len: i32| -> R<Array> {
            Ok(y.reshape(&[1, len, h, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = self.linear(x, &format!("{p}.linear_q"))?.reshape(&[1, t, h, hd])?;
        let k = heads(self.linear(x, &format!("{p}.linear_k"))?, t)?;
        let v = heads(self.linear(x, &format!("{p}.linear_v"))?, t)?;
        let pl = pos.shape()[1];
        let pp = heads(self.linear(pos, &format!("{p}.linear_pos"))?, pl)?;
        let q_u = ops::add(&q, self.g(&format!("{p}.pos_bias_u"))?)?.transpose_axes(&[0, 2, 1, 3])?;
        let q_v = ops::add(&q, self.g(&format!("{p}.pos_bias_v"))?)?.transpose_axes(&[0, 2, 1, 3])?;
        let scale = (hd as f32).powf(-0.5);
        let bd = ops::matmul(&q_v, pp.swap_axes(-2, -1)?)?; // [1, h, t, 2t-1]
        // rel_shift
        let bd = ops::pad(&bd, &[(0, 0), (0, 0), (0, 0), (1, 0)][..], None, None)?;
        let bd = bd
            .reshape(&[1, h, pl + 1, t])?
            .index((.., .., 1.., ..))
            .reshape(&[1, h, t, pl])?;
        let bd = ops::multiply(bd.index((.., .., .., ..t)), Array::from_f32(scale))?;
        let o = fast::scaled_dot_product_attention(&q_u, &k, &v, scale, &bd, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, t, d])?;
        self.linear(&o, &format!("{p}.linear_out"))
    }

    fn ff(&self, x: &Array, p: &str) -> R<Array> {
        let h = nn::silu(self.linear(x, &format!("{p}.linear1"))?)?;
        self.linear(&h, &format!("{p}.linear2"))
    }

    fn conv_module(&self, x: &Array, p: &str) -> R<Array> {
        let pad = self.conv_pad;
        let x = ops::conv1d(x, self.g(&format!("{p}.pointwise_conv1.weight"))?, None, None, None, None)?;
        let x = nn::glu(&x, 2)?;
        let x = ops::pad(&x, &[(0, 0), (pad, pad), (0, 0)][..], None, None)?;
        let x = ops::conv1d(&x, self.g(&format!("{p}.depthwise_conv.weight"))?, None, None, None, self.d_model)?;
        // BatchNorm, eval mode.
        let bn = |n: &str| self.g(&format!("{p}.batch_norm.{n}"));
        let inv = ops::rsqrt(ops::add(bn("running_var")?, Array::from_f32(1e-5))?)?;
        let x = ops::multiply(ops::subtract(&x, bn("running_mean")?)?, &inv)?;
        let x = ops::add(ops::multiply(&x, bn("weight")?)?, bn("bias")?)?;
        let x = nn::silu(&x)?;
        Ok(ops::conv1d(&x, self.g(&format!("{p}.pointwise_conv2.weight"))?, None, None, None, None)?)
    }

    fn encode(&self, mel: &Array) -> R<Array> {
        let mut x = self.subsample(mel)?;
        let pos = self.pos_emb(x.shape()[1]);
        let half = Array::from_f32(0.5);
        for l in 0..self.layers {
            let p = format!("encoder.layers.{l}");
            let h = self.ff(&self.layer_norm(&x, &format!("{p}.norm_feed_forward1"))?, &format!("{p}.feed_forward1"))?;
            x = ops::add(&x, ops::multiply(&h, &half)?)?;
            let xn = self.layer_norm(&x, &format!("{p}.norm_self_att"))?;
            x = ops::add(&x, self.attention(&xn, &pos, &format!("{p}.self_attn"))?)?;
            x = ops::add(&x, self.conv_module(&self.layer_norm(&x, &format!("{p}.norm_conv"))?, &format!("{p}.conv"))?)?;
            let h = self.ff(&self.layer_norm(&x, &format!("{p}.norm_feed_forward2"))?, &format!("{p}.feed_forward2"))?;
            x = ops::add(&x, ops::multiply(&h, &half)?)?;
            x = self.layer_norm(&x, &format!("{p}.norm_out"))?;
        }
        Ok(x)
    }

    /// One prediction-net step: the token (`None` before the first) and
    /// the LSTM state in, `[1, pred_hidden]` and the new state out.
    fn predict(&self, token: Option<usize>, state: Option<&(Vec<Array>, Vec<Array>)>) -> R<(Array, (Vec<Array>, Vec<Array>))> {
        let mut x = match token {
            Some(tk) => {
                let idx = Array::from_slice(&[tk as i32], &[1]);
                let wq = self.g("decoder.prediction.embed.weight")?.take_axis(&idx, 0)?;
                let s = self.g("decoder.prediction.embed.scales")?.take_axis(&idx, 0)?;
                let b = self.g("decoder.prediction.embed.biases")?.take_axis(&idx, 0)?;
                dequantize(&wq, &s, &b, GROUP_SIZE, BITS)?
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[1, self.pred_hidden])?
            }
            None => ops::zeros::<f32>(&[1, self.pred_hidden])?,
        };
        let (mut hs, mut cs) = (vec![], vec![]);
        for l in 0..self.pred_layers {
            let p = format!("decoder.prediction.dec_rnn.lstm.{l}");
            let mut g = ops::addmm(self.g(&format!("{p}.bias"))?, &x, self.g(&format!("{p}.Wx"))?.t(), None, None)?;
            if let Some((h, _)) = state {
                g = ops::addmm(&g, &h[l], self.g(&format!("{p}.Wh"))?.t(), None, None)?;
            }
            let parts = g.split_equal(4, -1)?;
            let i = ops::sigmoid(&parts[0])?;
            let f = ops::sigmoid(&parts[1])?;
            let gg = ops::tanh(&parts[2])?;
            let o = ops::sigmoid(&parts[3])?;
            let c = match state {
                Some((_, c)) => ops::add(ops::multiply(&f, &c[l])?, ops::multiply(&i, &gg)?)?,
                None => ops::multiply(&i, &gg)?,
            };
            let h = ops::multiply(&o, ops::tanh(&c)?)?.as_dtype(Dtype::Float32)?;
            x = h.clone();
            hs.push(h);
            cs.push(c.as_dtype(Dtype::Float32)?);
        }
        Ok((x, (hs, cs)))
    }

    /// Greedy TDT: per frame, the best token (or blank) and how many frames
    /// to skip.
    fn decode(&self, enc: &Array) -> R<Vec<usize>> {
        let blank = self.vocab.len();
        let t = enc.shape()[1] as usize;
        let enc_proj = self.linear(enc, "joint.enc")?;
        enc_proj.eval()?;
        let mut tokens = vec![];
        let (mut dec_out, mut dec_state) = self.predict(None, None)?;
        let mut pred_proj = self.linear(&dec_out, "joint.pred")?;
        let mut step = 0usize;
        let mut new_symbols = 0usize;
        while step < t {
            let e = enc_proj.index((.., step as i32, ..));
            let j = nn::relu(ops::add(&e, &pred_proj)?)?;
            let logits = self.linear(&j, "joint.joint_net.2")?;
            let tok = argmax_axis(logits.index((.., ..(blank as i32 + 1))), -1, None)?;
            let dur = argmax_axis(logits.index((.., (blank as i32 + 1)..)), -1, None)?;
            mlx_rs::transforms::eval([&tok, &dur])?;
            let tok = tok.try_item_exact::<u32>()? as usize;
            let dur = self.durations[dur.try_item_exact::<u32>()? as usize];
            if tok != blank {
                tokens.push(tok);
                (dec_out, dec_state) = self.predict(Some(tok), Some(&dec_state))?;
                pred_proj = self.linear(&dec_out, "joint.pred")?;
            }
            step += dur;
            new_symbols += 1;
            if dur != 0 {
                new_symbols = 0;
            } else if self.max_symbols <= new_symbols {
                step += 1;
                new_symbols = 0;
            }
        }
        Ok(tokens)
    }

    fn transcribe(&self, audio: &[f32]) -> R<String> {
        // Shorter than one hop, there is no frame to decode.
        if audio.len() < self.hop {
            return Ok(String::new());
        }
        let enc = self.encode(&self.logmel(audio)?)?;
        enc.eval()?;
        let text: String = self
            .decode(&enc)?
            .iter()
            .map(|&t| self.vocab[t].replace('▁', " "))
            .collect();
        Ok(text.trim().to_string())
    }
}

/// librosa's slaney mel filterbank, `[n_mels, n_fft/2 + 1]` row-major.
fn slaney_mel_fb(sr: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * ((m - min_log_mel) * logstep).exp() } else { f_sp * m };
    let n_bins = n_fft / 2 + 1;
    let fft_f: Vec<f64> = (0..n_bins).map(|i| i as f64 * (sr / 2.0) / (n_bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut w = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for b in 0..n_bins {
            let lower = (fft_f[b] - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
            let upper = (mel_f[m + 2] - fft_f[b]) / (mel_f[m + 2] - mel_f[m + 1]);
            w[m * n_bins + b] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

