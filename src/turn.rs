//! End-of-turn detection (naru task 2355): Pipecat Smart Turn v3.2 says
//! whether the last seconds of a person's audio are a finished turn. The live
//! segmenter (`server/stream.rs`) asks it after each natural VAD close and
//! holds a span it calls unfinished. Every failure means "no answer", which
//! the segmenter treats as "finished": the model is optional.
//!
//! macOS only. ONNX Runtime comes from the copy sherpa-onnx-sys already links
//! (`ort-sys` has `disable-linking` in `Cargo.toml`); elsewhere [`shared`] is
//! always `None`.

use std::path::Path;
use std::sync::Arc;

/// The registry model.
pub const DEFAULT_MODEL: &str = "smart-turn-v3.2";
/// The ONNX file inside the model directory.
pub const MODEL_FILENAME: &str = "model.onnx";
/// Probability of "complete" at or above which a turn is finished.
pub const THRESHOLD: f32 = 0.5;

/// Something that says whether audio ends a turn.
pub trait EndOfTurn: Send + Sync {
    /// The probability the turn is complete, from the last [`WINDOW`]
    /// samples of 16 kHz mono `pcm` (zero-padded at the start when shorter);
    /// `None` when it cannot say.
    fn complete(&self, pcm: &[f32]) -> Option<f32>;
}

/// The model's input: 8 s at 16 kHz.
pub const WINDOW: usize = 8 * 16_000;

/// The process's detector, loaded on first use from the pulled model in
/// `dir`. `None` when the model is not pulled (asked again next time, it
/// costs a stat) or failed to load (remembered: never retried).
pub fn shared(dir: &Path) -> Option<Arc<dyn EndOfTurn>> {
    imp::shared(dir)
}

#[cfg(target_os = "macos")]
mod imp {
    use std::f64::consts::PI;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use ort::session::Session;
    use ort::value::Tensor;
    use realfft::{RealFftPlanner, RealToComplex};

    use super::{EndOfTurn, MODEL_FILENAME, WINDOW};

    const NFFT: usize = 400;
    const HOP: usize = 160;
    const MELS: usize = 80;
    const BINS: usize = NFFT / 2 + 1;
    const FRAMES: usize = WINDOW / HOP;

    static LOADED: Mutex<Option<Arc<SmartTurn>>> = Mutex::new(None);
    static FAILED: AtomicBool = AtomicBool::new(false);

    pub(super) fn shared(dir: &Path) -> Option<Arc<dyn EndOfTurn>> {
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = loaded.as_ref() {
            return Some(t.clone());
        }
        let file = dir.join(MODEL_FILENAME);
        if FAILED.load(Ordering::SeqCst) || !file.is_file() {
            return None;
        }
        match SmartTurn::load(&file) {
            Ok(t) => {
                let t = Arc::new(t);
                *loaded = Some(t.clone());
                Some(t)
            }
            Err(e) => {
                FAILED.store(true, Ordering::SeqCst);
                eprintln!("smart-turn: load failed, turns are not gated: {e}");
                None
            }
        }
    }

    struct SmartTurn {
        session: Mutex<Session>,
        /// `[BINS][MELS]`, Slaney scale and norm.
        filters: Vec<f32>,
        window: Vec<f32>,
        fft: Arc<dyn RealToComplex<f32>>,
    }

    impl SmartTurn {
        fn load(file: &Path) -> Result<Self, ort::Error> {
            let session = Session::builder()?
                .with_intra_threads(1)?
                .commit_from_file(file)?;
            Ok(SmartTurn {
                session: Mutex::new(session),
                filters: mel_filters(),
                window: (0..NFFT)
                    .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / NFFT as f64).cos()) as f32)
                    .collect(),
                fft: RealFftPlanner::<f32>::new().plan_fft_forward(NFFT),
            })
        }

        /// Whisper's log-mel features of the last 8 s of `pcm`: `[MELS][FRAMES]`.
        fn features(&self, pcm: &[f32]) -> Vec<f32> {
            let mut x = vec![0f32; WINDOW];
            if pcm.len() >= WINDOW {
                x.copy_from_slice(&pcm[pcm.len() - WINDOW..]);
            } else {
                x[WINDOW - pcm.len()..].copy_from_slice(pcm);
            }
            // Zero mean and unit variance over the whole padded window.
            let mean = x.iter().map(|&v| v as f64).sum::<f64>() / WINDOW as f64;
            let var = x
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / WINDOW as f64;
            let inv = 1.0 / (var + 1e-7).sqrt();
            for v in x.iter_mut() {
                *v = ((*v as f64 - mean) * inv) as f32;
            }
            // Centred frames: reflect-pad half a window each side.
            let half = NFFT / 2;
            let mut pad = Vec::with_capacity(WINDOW + NFFT);
            pad.extend((0..half).map(|i| x[half - i]));
            pad.extend_from_slice(&x);
            pad.extend((0..half).map(|i| x[WINDOW - 2 - i]));
            let mut logm = vec![0f32; MELS * FRAMES];
            let mut buf = self.fft.make_input_vec();
            let mut spec = self.fft.make_output_vec();
            let mut peak = f32::MIN;
            for t in 0..FRAMES {
                for i in 0..NFFT {
                    buf[i] = pad[t * HOP + i] * self.window[i];
                }
                if self.fft.process(&mut buf, &mut spec).is_err() {
                    return Vec::new();
                }
                let power: Vec<f32> = spec.iter().map(|c| c.re * c.re + c.im * c.im).collect();
                for m in 0..MELS {
                    let mut s = 0f32;
                    for (b, p) in power.iter().enumerate() {
                        s += p * self.filters[b * MELS + m];
                    }
                    let l = s.max(1e-10).log10();
                    logm[m * FRAMES + t] = l;
                    peak = peak.max(l);
                }
            }
            for v in logm.iter_mut() {
                *v = (v.max(peak - 8.0) + 4.0) / 4.0;
            }
            logm
        }
    }

    impl EndOfTurn for SmartTurn {
        fn complete(&self, pcm: &[f32]) -> Option<f32> {
            let features = self.features(pcm);
            if features.is_empty() {
                return None;
            }
            let input = Tensor::from_array(([1usize, MELS, FRAMES], features)).ok()?;
            let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
            let out = session.run(ort::inputs!["input_features" => input]).ok()?;
            let (_, p) = out[0].try_extract_tensor::<f32>().ok()?;
            p.first().copied().filter(|p| p.is_finite())
        }
    }

    fn hz_to_mel(f: f64) -> f64 {
        let (fsp, minlog, step) = (200.0 / 3.0, 1000.0, (6.4f64).ln() / 27.0);
        if f >= minlog {
            minlog / fsp + (f / minlog).ln() / step
        } else {
            f / fsp
        }
    }

    fn mel_to_hz(m: f64) -> f64 {
        let (fsp, minlog, step) = (200.0 / 3.0, 1000.0, (6.4f64).ln() / 27.0);
        let log_start = minlog / fsp;
        if m >= log_start {
            minlog * (step * (m - log_start)).exp()
        } else {
            m * fsp
        }
    }

    fn mel_filters() -> Vec<f32> {
        let bin_hz: Vec<f64> = (0..BINS)
            .map(|i| i as f64 * 8000.0 / (BINS - 1) as f64)
            .collect();
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let edges: Vec<f64> = (0..MELS + 2)
            .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
            .collect();
        let mut w = vec![0f32; BINS * MELS];
        for m in 0..MELS {
            let (l, c, r) = (edges[m], edges[m + 1], edges[m + 2]);
            for (b, &f) in bin_hz.iter().enumerate() {
                let v = ((f - l) / (c - l)).min((r - f) / (r - c)).max(0.0);
                w[b * MELS + m] = (v * 2.0 / (r - l)) as f32;
            }
        }
        w
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::path::Path;
    use std::sync::Arc;

    use super::EndOfTurn;

    pub(super) fn shared(_dir: &Path) -> Option<Arc<dyn EndOfTurn>> {
        None
    }
}
