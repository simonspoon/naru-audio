//! Pure-Rust DSP behind the voice-prep steps (`prep::pipeline`): biquad
//! filters (RBJ cookbook), BS.1770 loudness, a dynamics section (compressor,
//! de-esser, look-ahead limiter), WSOLA time-stretch with pitch shift, a
//! telephone effect and a Schroeder reverb. Everything works on 16 kHz mono
//! `f32` ([`FS`]), the pipeline's working rate; no ffmpeg, no extra crate.
//! Filter state is carried in `f64` so a low corner (80 Hz at 16 kHz) stays
//! numerically clean.

use std::f64::consts::PI;

use crate::stt::audio::TARGET_SAMPLE_RATE;

/// The working sample rate every function here assumes.
pub const FS: f64 = TARGET_SAMPLE_RATE as f64;

/// One second-order section, coefficients normalised so `a0 = 1`.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    fn norm(b: [f64; 3], a: [f64; 3]) -> Biquad {
        Biquad {
            b0: b[0] / a[0],
            b1: b[1] / a[0],
            b2: b[2] / a[0],
            a1: a[1] / a[0],
            a2: a[2] / a[0],
        }
    }

    fn trig(f0: f64, q: f64) -> (f64, f64) {
        let w0 = 2.0 * PI * f0 / FS;
        (w0.cos(), w0.sin() / (2.0 * q))
    }

    pub fn lowpass(f0: f64, q: f64) -> Biquad {
        let (c, al) = Self::trig(f0, q);
        Self::norm(
            [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0],
            [1.0 + al, -2.0 * c, 1.0 - al],
        )
    }

    pub fn highpass(f0: f64, q: f64) -> Biquad {
        let (c, al) = Self::trig(f0, q);
        Self::norm(
            [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0],
            [1.0 + al, -2.0 * c, 1.0 - al],
        )
    }

    pub fn peak(f0: f64, gain_db: f64, q: f64) -> Biquad {
        let (c, al) = Self::trig(f0, q);
        let a = 10f64.powf(gain_db / 40.0);
        Self::norm(
            [1.0 + al * a, -2.0 * c, 1.0 - al * a],
            [1.0 + al / a, -2.0 * c, 1.0 - al / a],
        )
    }

    pub fn low_shelf(f0: f64, gain_db: f64, q: f64) -> Biquad {
        let (c, al) = Self::trig(f0, q);
        let a = 10f64.powf(gain_db / 40.0);
        let t = 2.0 * a.sqrt() * al;
        Self::norm(
            [
                a * ((a + 1.0) - (a - 1.0) * c + t),
                2.0 * a * ((a - 1.0) - (a + 1.0) * c),
                a * ((a + 1.0) - (a - 1.0) * c - t),
            ],
            [
                (a + 1.0) + (a - 1.0) * c + t,
                -2.0 * ((a - 1.0) + (a + 1.0) * c),
                (a + 1.0) + (a - 1.0) * c - t,
            ],
        )
    }

    pub fn high_shelf(f0: f64, gain_db: f64, q: f64) -> Biquad {
        let (c, al) = Self::trig(f0, q);
        let a = 10f64.powf(gain_db / 40.0);
        let t = 2.0 * a.sqrt() * al;
        Self::norm(
            [
                a * ((a + 1.0) + (a - 1.0) * c + t),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
                a * ((a + 1.0) + (a - 1.0) * c - t),
            ],
            [
                (a + 1.0) - (a - 1.0) * c + t,
                2.0 * ((a - 1.0) - (a + 1.0) * c),
                (a + 1.0) - (a - 1.0) * c - t,
            ],
        )
    }

    /// Transposed direct form II step; `s` is the two-element state.
    fn tick(&self, s: &mut [f64; 2], x: f64) -> f64 {
        let y = self.b0 * x + s[0];
        s[0] = self.b1 * x - self.a1 * y + s[1];
        s[1] = self.b2 * x - self.a2 * y;
        y
    }

    pub fn process_f64(&self, x: &mut [f64]) {
        let mut s = [0.0; 2];
        for v in x.iter_mut() {
            *v = self.tick(&mut s, *v);
        }
    }

    pub fn process(&self, x: &mut [f32]) {
        let mut s = [0.0; 2];
        for v in x.iter_mut() {
            *v = self.tick(&mut s, *v as f64) as f32;
        }
    }
}

/// Q of the two sections of a 4th-order Butterworth.
const BUTTER4_Q: [f64; 2] = [0.541_196_1, 1.306_563];

/// 4th-order (24 dB/oct) Butterworth high-pass.
pub fn highpass4(x: &mut [f32], freq: f64) {
    for q in BUTTER4_Q {
        Biquad::highpass(freq, q).process(x);
    }
}

/// 4th-order (24 dB/oct) Butterworth low-pass.
pub fn lowpass4(x: &mut [f32], freq: f64) {
    for q in BUTTER4_Q {
        Biquad::lowpass(freq, q).process(x);
    }
}

fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn lin_to_db(lin: f64) -> f64 {
    20.0 * lin.max(1e-12).log10()
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

/// One-pole smoothing coefficient for a time constant in milliseconds.
fn coef(ms: f64) -> f64 {
    (-1.0 / (ms.max(0.01) * 1e-3 * FS)).exp()
}

// ---- loudness -------------------------------------------------------------

/// The two ITU-R BS.1770-4 K-weighting stages (a high shelf modelling the
/// head, then the RLB high-pass), derived for [`FS`] from the standard's
/// analogue prototype (f0/Q/gain below) via the bilinear transform, so the
/// 48 kHz tables are not reused at 16 kHz. The RLB stage is scaled to unity
/// gain at Nyquist.
fn k_filters() -> [Biquad; 2] {
    let (f0, g, q) = (
        1_681.974_450_955_533,
        3.999_843_853_973_347,
        0.707_175_236_955_419_6,
    );
    let k = (PI * f0 / FS).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad {
        b0: (vh + vb * k / q + k * k) / a0,
        b1: 2.0 * (k * k - vh) / a0,
        b2: (vh - vb * k / q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
    };
    let (f0, q) = (38.135_470_876_024_44, 0.500_327_037_323_877_3);
    let k = (PI * f0 / FS).tan();
    let a0 = 1.0 + k / q + k * k;
    let rlb = Biquad {
        b0: 1.0 / a0,
        b1: -2.0 / a0,
        b2: 1.0 / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
    };
    [shelf, rlb]
}

/// ITU-R BS.1770-4 integrated loudness (mono, channel weight 1) in LUFS:
/// K-weight, mean square per 400 ms block at 75 % overlap, drop blocks under
/// the absolute gate (-70 LUFS), then drop blocks more than 10 LU under the
/// mean of the rest (relative gate) and average what is left. A clip shorter
/// than one block is measured as a single block. `None` if nothing passes
/// the gates (silence).
pub fn integrated_lufs(pcm: &[f32]) -> Option<f64> {
    if pcm.is_empty() {
        return None;
    }
    let mut z: Vec<f64> = pcm.iter().map(|&s| s as f64).collect();
    for f in k_filters() {
        f.process_f64(&mut z);
    }
    let mut prefix = Vec::with_capacity(z.len() + 1);
    prefix.push(0.0);
    for v in &z {
        prefix.push(prefix.last().unwrap() + v * v);
    }
    let block = (0.4 * FS) as usize;
    let hop = (0.1 * FS) as usize;
    let mean_sq = |from: usize, to: usize| (prefix[to] - prefix[from]) / (to - from) as f64;
    let powers: Vec<f64> = if z.len() < block {
        vec![mean_sq(0, z.len())]
    } else {
        (0..=(z.len() - block) / hop)
            .map(|i| mean_sq(i * hop, i * hop + block))
            .collect()
    };
    let lufs = |p: f64| -0.691 + 10.0 * p.log10();
    let above = |gate: f64, v: &[f64]| -> Vec<f64> {
        v.iter()
            .copied()
            .filter(|&p| p > 0.0 && lufs(p) > gate)
            .collect()
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let kept = above(-70.0, &powers);
    if kept.is_empty() {
        return None;
    }
    let kept = above(lufs(mean(&kept)) - 10.0, &kept);
    if kept.is_empty() {
        return None;
    }
    Some(lufs(mean(&kept)))
}

/// Look-ahead peak limiter: after this, no sample exceeds `ceiling` (linear).
/// The gain needed at each sample is minimum-filtered over a 5 ms look-ahead
/// window, then averaged over the same length, which keeps every sample at or
/// under its own required gain with a smooth (click-free) gain curve.
pub fn limit(pcm: &mut [f32], ceiling: f32) {
    if ceiling <= 0.0 || pcm.iter().all(|s| s.abs() <= ceiling) {
        return;
    }
    let n = (0.005 * FS) as usize;
    let len = pcm.len();
    let need: Vec<f32> = pcm
        .iter()
        .map(|s| (ceiling / s.abs().max(1e-12)).min(1.0))
        .collect();
    let mins: Vec<f32> = (0..len)
        .map(|k| {
            need[k..(k + n).min(len)]
                .iter()
                .copied()
                .fold(1.0, f32::min)
        })
        .collect();
    for (i, s) in pcm.iter_mut().enumerate() {
        let g: f32 = (0..n).map(|j| mins[i.saturating_sub(j)]).sum::<f32>() / n as f32;
        *s = (*s * g).clamp(-ceiling, ceiling);
    }
}

/// Scales `pcm` to `target_lufs` integrated loudness, then limits so the
/// peak stays at or under `ceiling_db` (so a quiet-peaked loud take may end
/// up a little under target). A no-op on silence.
pub fn loudness(pcm: &mut [f32], target_lufs: f64, ceiling_db: f64) {
    let Some(measured) = integrated_lufs(pcm) else {
        return;
    };
    let gain = db_to_lin(target_lufs - measured) as f32;
    for s in pcm.iter_mut() {
        *s *= gain;
    }
    limit(pcm, db_to_lin(ceiling_db) as f32);
}

/// The 10th percentile of 50 ms non-overlapping frame RMS levels, in dBFS:
/// speech has pauses, so the quietest tenth of frames is the room/background
/// floor. `None` for clips under one frame or a floor of digital silence.
pub fn noise_floor_dbfs(pcm: &[f32]) -> Option<f64> {
    let frame = (0.05 * FS) as usize;
    let mut levels: Vec<f64> = pcm.chunks_exact(frame).map(rms).collect();
    if levels.is_empty() {
        return None;
    }
    levels.sort_by(|a, b| a.total_cmp(b));
    let r = levels[levels.len() / 10];
    (r > 0.0).then(|| lin_to_db(r))
}

// ---- dynamics -------------------------------------------------------------

/// Feed-forward compressor: a level detector smoothed in dB with separate
/// attack/release, gain reduction of `(level - threshold) * (1 - 1/ratio)`
/// above the threshold, plus `makeup_db`.
pub fn compress(
    pcm: &mut [f32],
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    makeup_db: f64,
) {
    let (atk, rel) = (coef(attack_ms), coef(release_ms));
    let mut env = -120.0;
    for s in pcm.iter_mut() {
        let level = lin_to_db((*s as f64).abs());
        let c = if level > env { atk } else { rel };
        env = c * env + (1.0 - c) * level;
        let reduction = (env - threshold_db).max(0.0) * (1.0 - 1.0 / ratio);
        *s *= db_to_lin(makeup_db - reduction) as f32;
    }
}

/// Split-band de-esser: a Linkwitz-Riley 4th-order crossover at `freq`
/// splits the signal into body and sibilance (the two bands are in phase, so
/// they sum to an all-pass: magnitude flat, only phase moves), and the
/// sibilance band alone is attenuated by `(level - threshold_db) * (1 -
/// 1/ratio)` dB whenever its peak-held level (instant attack, 20 ms release)
/// is over the threshold.
pub fn deess(pcm: &mut [f32], freq: f64, threshold_db: f64, ratio: f64) {
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let mut low = pcm.to_vec();
    let mut high = pcm.to_vec();
    for _ in 0..2 {
        Biquad::lowpass(freq, q).process(&mut low);
        Biquad::highpass(freq, q).process(&mut high);
    }
    let (rel, smooth) = (coef(20.0), coef(1.0));
    let (mut env, mut g) = (0.0f64, 1.0f64);
    for ((s, l), h) in pcm.iter_mut().zip(&low).zip(&high) {
        env = (*h as f64).abs().max(env * rel);
        let reduction = (lin_to_db(env) - threshold_db).max(0.0) * (1.0 - 1.0 / ratio);
        g = smooth * g + (1.0 - smooth) * db_to_lin(-reduction);
        *s = l + (*h as f64 * g) as f32;
    }
}

// ---- effects --------------------------------------------------------------

/// Soft saturation (tanh, unity small-signal gain), then a 4th-order
/// 300 Hz high-pass and 3.4 kHz low-pass: the narrowband "phone" sound.
pub fn telephone(pcm: &mut [f32]) {
    const DRIVE: f32 = 2.0;
    for s in pcm.iter_mut() {
        *s = (*s * DRIVE).tanh() / DRIVE;
    }
    highpass4(pcm, 300.0);
    lowpass4(pcm, 3400.0);
}

/// Schroeder reverb: four damped feedback combs in parallel into two
/// all-passes. `room_size` (0..1) sets the comb feedback (0.7..0.98), `wet`
/// (0..1) the dry/wet mix. The reverberant signal is level-matched to the
/// dry RMS before mixing, so `wet = 1` is about as loud as the input. The
/// output keeps the input's length: the tail is cut at the end of the clip.
pub fn reverb(pcm: &mut [f32], room_size: f64, wet: f64) {
    if pcm.is_empty() || wet <= 0.0 {
        return;
    }
    let feedback = 0.7 + 0.28 * room_size;
    let damp = 0.3;
    let ms = |t: f64| (t * 1e-3 * FS) as usize;
    let mut combs: Vec<(Vec<f64>, usize, f64)> = [29.7, 37.1, 41.1, 43.7]
        .iter()
        .map(|&t| (vec![0.0; ms(t)], 0, 0.0))
        .collect();
    let mut allpass: Vec<(Vec<f64>, usize)> =
        [5.0, 1.7].iter().map(|&t| (vec![0.0; ms(t)], 0)).collect();
    let mut rev = Vec::with_capacity(pcm.len());
    for &x in pcm.iter() {
        let mut acc = 0.0;
        for (buf, idx, store) in combs.iter_mut() {
            let out = buf[*idx];
            *store = out * (1.0 - damp) + *store * damp;
            buf[*idx] = x as f64 + *store * feedback;
            *idx = (*idx + 1) % buf.len();
            acc += out;
        }
        for (buf, idx) in allpass.iter_mut() {
            let delayed = buf[*idx];
            let out = delayed - acc;
            buf[*idx] = acc + delayed * 0.5;
            acc = out;
            *idx = (*idx + 1) % buf.len();
        }
        rev.push(acc as f32);
    }
    let scale = match rms(&rev) {
        r if r > 0.0 => (rms(pcm) / r) as f32,
        _ => 0.0,
    };
    let wet = wet as f32;
    for (s, r) in pcm.iter_mut().zip(&rev) {
        *s = *s * (1.0 - wet) + r * scale * wet;
    }
}

// ---- time and pitch -------------------------------------------------------

/// WSOLA (waveform-similarity overlap-add) time-stretch: output length is
/// `len / rate`, pitch unchanged. 32 ms Hann frames at 50 % synthesis
/// overlap; each frame is taken from within +-8 ms of its nominal position,
/// wherever it best continues the previous one (normalised cross-correlation).
pub fn time_stretch(x: &[f32], rate: f64) -> Vec<f32> {
    const N: usize = 512;
    const HOP: usize = N / 2;
    const TOL: isize = 128;
    if x.is_empty() || (rate - 1.0).abs() < 1e-9 {
        return x.to_vec();
    }
    let out_len = (x.len() as f64 / rate).round() as usize;
    let window: Vec<f32> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / N as f64).cos() as f32)
        .collect();
    let at = |i: isize| -> f32 {
        if i >= 0 && (i as usize) < x.len() {
            x[i as usize]
        } else {
            0.0
        }
    };
    let mut out = vec![0.0f32; out_len + N];
    let mut norm = vec![0.0f32; out_len + N];
    let mut prev = 0isize;
    let mut k = 0usize;
    while k * HOP < out_len {
        let nominal = (k as f64 * HOP as f64 * rate).round() as isize;
        let start = if k == 0 {
            nominal
        } else {
            let target = prev + HOP as isize;
            let mut best = (f64::NEG_INFINITY, nominal);
            for d in -TOL..=TOL {
                let cand = nominal + d;
                let (mut dot, mut e) = (0.0f64, 1e-9f64);
                for i in (0..N as isize).step_by(2) {
                    let c = at(cand + i) as f64;
                    dot += c * at(target + i) as f64;
                    e += c * c;
                }
                let score = dot / e.sqrt();
                if score > best.0 {
                    best = (score, cand);
                }
            }
            best.1
        };
        for (i, w) in window.iter().enumerate() {
            out[k * HOP + i] += w * at(start + i as isize);
            norm[k * HOP + i] += w;
        }
        prev = start;
        k += 1;
    }
    out.truncate(out_len);
    for (o, n) in out.iter_mut().zip(&norm) {
        if *n > 1e-6 {
            *o /= n;
        }
    }
    out
}

/// Reads `x` at `step` input samples per output sample (Catmull-Rom cubic),
/// low-passing first when `step > 1` so the faster read does not alias.
fn resample_step(x: &[f32], step: f64) -> Vec<f32> {
    let mut src = x.to_vec();
    if step > 1.0 {
        lowpass4(&mut src, FS / 2.0 * 0.9 / step);
    }
    let at = |i: isize| src[i.clamp(0, src.len() as isize - 1) as usize] as f64;
    (0..(x.len() as f64 / step).round() as usize)
        .map(|n| {
            let pos = n as f64 * step;
            let i = pos.floor() as isize;
            let t = pos - i as f64;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            (p1 + 0.5
                * t
                * (p2 - p0
                    + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0))))
                as f32
        })
        .collect()
}

/// Plays `x` `rate` times faster, pitch preserved (WSOLA).
pub fn speed(x: &[f32], rate: f64) -> Vec<f32> {
    time_stretch(x, rate)
}

/// Shifts pitch by `semitones` keeping duration: WSOLA-stretch by the pitch
/// ratio, then resample back to the original length. This moves formants
/// with the pitch (no formant preservation), so large shifts sound
/// chipmunk/giant rather than like a different speaker.
pub fn pitch_shift(x: &[f32], semitones: f64) -> Vec<f32> {
    if x.is_empty() || semitones.abs() < 1e-9 {
        return x.to_vec();
    }
    let ratio = 2f64.powf(semitones / 12.0);
    let mut y = resample_step(&time_stretch(x, 1.0 / ratio), ratio);
    y.resize(x.len(), 0.0);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, amp: f64, secs: f64) -> Vec<f32> {
        (0..(secs * FS) as usize)
            .map(|i| (amp * (2.0 * PI * freq * i as f64 / FS).sin()) as f32)
            .collect()
    }

    /// Amplitude of `freq` in `x` after a 0.25 s settle, by single-bin DFT.
    fn amp_at(x: &[f32], freq: f64) -> f64 {
        let x = &x[(0.25 * FS) as usize..];
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &s) in x.iter().enumerate() {
            let w = 2.0 * PI * freq * i as f64 / FS;
            re += s as f64 * w.cos();
            im += s as f64 * w.sin();
        }
        2.0 * (re * re + im * im).sqrt() / x.len() as f64
    }

    fn gain_db(before: &[f32], after: &[f32], freq: f64) -> f64 {
        lin_to_db(amp_at(after, freq)) - lin_to_db(amp_at(before, freq))
    }

    #[test]
    fn highpass_attenuates_30hz_and_passes_1khz() {
        for (freq, check) in [(30.0, true), (1000.0, false)] {
            let x = tone(freq, 0.5, 2.0);
            let mut y = x.clone();
            highpass4(&mut y, 80.0);
            let g = gain_db(&x, &y, freq);
            if check {
                assert!(g <= -20.0, "30 Hz gain {g} dB");
            } else {
                assert!(g.abs() < 1.0, "1 kHz gain {g} dB");
            }
        }
    }

    #[test]
    fn eq_biquads_hit_their_gain_at_the_corner() {
        let cases: [(Biquad, f64, f64); 4] = [
            (Biquad::peak(1000.0, 6.0, 1.0), 1000.0, 6.0),
            (Biquad::peak(1000.0, -6.0, 1.0), 1000.0, -6.0),
            (Biquad::low_shelf(300.0, 6.0, 0.707), 50.0, 6.0),
            (Biquad::high_shelf(3000.0, -6.0, 0.707), 7000.0, -6.0),
        ];
        for (bq, freq, want) in cases {
            let x = tone(freq, 0.3, 2.0);
            let mut y = x.clone();
            bq.process(&mut y);
            let g = gain_db(&x, &y, freq);
            assert!((g - want).abs() < 0.7, "{freq} Hz: {g} dB, want {want}");
        }
    }

    #[test]
    fn lufs_of_a_full_scale_sine_matches_bs1770() {
        let l = integrated_lufs(&tone(997.0, 1.0, 5.0)).unwrap();
        assert!((l + 3.01).abs() < 0.5, "{l} LUFS");
    }

    #[test]
    fn lufs_ignores_silence_via_gating() {
        let mut x = tone(997.0, 0.25, 3.0);
        let alone = integrated_lufs(&x).unwrap();
        x.extend(std::iter::repeat_n(0.0, 16_000 * 4));
        let padded = integrated_lufs(&x).unwrap();
        assert!((alone - padded).abs() < 0.5, "{alone} vs {padded}");
        assert!(integrated_lufs(&[0.0; 32_000]).is_none());
    }

    #[test]
    fn loudness_hits_target() {
        let mut x = tone(500.0, 0.03, 4.0);
        loudness(&mut x, -18.0, -1.0);
        let l = integrated_lufs(&x).unwrap();
        assert!((l + 18.0).abs() < 0.5, "{l} LUFS");
    }

    #[test]
    fn loudness_respects_the_peak_ceiling() {
        let mut x = tone(500.0, 0.5, 2.0);
        x[8000] = 0.95;
        loudness(&mut x, -10.0, -6.0);
        let peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 10f32.powf(-6.0 / 20.0) + 1e-6, "peak {peak}");
    }

    #[test]
    fn speed_scales_duration_and_keeps_pitch() {
        let x = tone(440.0, 0.5, 3.0);
        let y = speed(&x, 1.25);
        let want = x.len() as f64 / 1.25;
        assert!((y.len() as f64 - want).abs() / want < 0.02, "{}", y.len());
        assert!(amp_at(&y, 440.0) > 0.35, "pitch moved");
    }

    #[test]
    fn pitch_keeps_duration_and_moves_frequency() {
        let x = tone(440.0, 0.5, 3.0);
        let y = pitch_shift(&x, 4.0);
        assert!((y.len() as f64 - x.len() as f64).abs() / (x.len() as f64) < 0.02);
        let up = 440.0 * 2f64.powf(4.0 / 12.0);
        assert!(amp_at(&y, up) > 3.0 * amp_at(&y, 440.0), "no shift to {up}");
        assert_eq!(pitch_shift(&x, 0.0), x);
        for n in [1, 2, 3, 100, 1001] {
            for st in [-12.0, -5.0, 7.0, 12.0] {
                assert_eq!(pitch_shift(&vec![0.1; n], st).len(), n, "n={n} st={st}");
            }
        }
    }

    #[test]
    fn telephone_cuts_lows_and_highs() {
        let probe = |f| {
            let x = tone(f, 0.1, 2.0);
            let mut y = x.clone();
            telephone(&mut y);
            gain_db(&x, &y, f)
        };
        assert!(probe(1000.0).abs() < 3.0);
        assert!(probe(100.0) < -20.0, "100 Hz {}", probe(100.0));
        assert!(probe(6000.0) < -12.0, "6 kHz {}", probe(6000.0));
    }

    #[test]
    fn deess_reduces_sibilance_more_than_mid() {
        let probe = |f| {
            let x = tone(f, 0.5, 2.0);
            let mut y = x.clone();
            deess(&mut y, 6000.0, -30.0, 4.0);
            gain_db(&x, &y, f)
        };
        let (hi, mid) = (probe(7000.0), probe(1000.0));
        assert!(hi < -6.0, "7 kHz {hi}");
        assert!(mid > -1.0, "1 kHz {mid}");
    }

    #[test]
    fn compressor_reduces_a_loud_tone_and_leaves_a_quiet_one() {
        let probe = |amp| {
            let x = tone(500.0, amp, 2.0);
            let mut y = x.clone();
            compress(&mut y, -20.0, 4.0, 5.0, 50.0, 0.0);
            gain_db(&x, &y, 500.0)
        };
        assert!(probe(0.8) < -6.0);
        assert!(probe(0.01).abs() < 0.5);
    }

    #[test]
    fn reverb_adds_a_tail_and_wet_zero_is_a_no_op() {
        let mut x = tone(500.0, 0.3, 0.3);
        x.extend(std::iter::repeat_n(0.0, 8000));
        let dry = x.clone();
        reverb(&mut x, 0.5, 0.0);
        assert_eq!(x, dry);
        reverb(&mut x, 0.7, 0.5);
        assert!(rms(&x[6000..]) > 1e-3, "no tail");
    }

    #[test]
    fn limiter_caps_peaks() {
        let mut x = tone(300.0, 0.9, 1.0);
        limit(&mut x, 0.3);
        assert!(x.iter().all(|s| s.abs() <= 0.3));
    }

    #[test]
    fn noise_floor_is_the_quiet_frames() {
        let mut x = tone(300.0, 0.5, 1.0);
        x.extend(vec![0.001; 16_000]);
        let nf = noise_floor_dbfs(&x).unwrap();
        assert!((nf + 60.0).abs() < 1.0, "{nf}");
        assert!(noise_floor_dbfs(&[0.0; 16_000]).is_none());
    }
}
