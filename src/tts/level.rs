//! Levelling synthesised chunks against each other. Ported unchanged from
//! kokoro-rs `src/level.rs` (3f722cf); a chunk here is one sherpa callback
//! piece, i.e. one sentence.
//!
//! Kokoro synthesises each chunk independently, so a short emphatic opener
//! ("Right.") can land noticeably louder than the sentences that follow it —
//! nothing ties the loudness of one chunk to the next. This module tracks a
//! running voiced-RMS target across chunks and nudges each chunk's gain
//! toward it.
//!
//! A running target rather than per-chunk peak normalisation: peak
//! normalisation would flatten real emphasis in the text and make quiet
//! sentences loud, which is worse than the problem it fixes. Levelling
//! against a slowly-adapting target lets loud and quiet lines keep their
//! relative emphasis while pulling outliers back toward the pack.
//!
//! It must be single-pass and streaming-safe: `-o -` streams WAV to stdout as
//! synthesis happens, and `tail -f log | kokoro-rs` never ends, so nothing
//! may depend on seeing the whole render. `Leveller` only ever looks at the
//! chunk in hand plus its own accumulated state — no lookahead, no
//! buffering, no two-pass.
//!
//! One consequence of being single-pass: the first chunk has no history to
//! level against. It is levelled against the fixed `TARGET_RMS` seed rather
//! than passed through — passing it through is exactly the bug being fixed,
//! since the first chunk is the one most often too loud.

/// Samples quieter than this don't count toward voiced RMS, so silence
/// between words doesn't drag the level down. Mirrors the |s| > 300 (of
/// 32768) threshold the loudness bug was originally measured with.
const VOICED_FLOOR: f32 = 300.0 / 32768.0;
/// Seed for the running target, calibrated from the settled voiced RMS of
/// real renders.
const TARGET_RMS: f32 = 0.094;
/// How far any one chunk's gain may move, in decibels.
const MAX_GAIN_DB: f32 = 3.0;
/// Samples over which gain slews from the previous chunk's gain to this
/// chunk's — 30 ms at 24 kHz. Long enough that the correction isn't audible
/// as a step at the chunk boundary.
const RAMP_SAMPLES: usize = 720;
/// How fast the running target adapts toward a chunk's raw RMS.
const EMA_ALPHA: f32 = 0.3;
/// Voiced samples corresponding to a full EMA step (0.5 s at 24 kHz) — a
/// chunk with less voiced audio than this moves the target proportionally
/// less, so a short opener can't dominate it.
const EMA_REF_VOICED: f32 = 12_000.0;
/// Voiced samples below this (10 ms) give no usable level measurement.
const MIN_VOICED: usize = 240;

/// Levels chunks against a running voiced-RMS target, one chunk at a time.
pub struct Leveller {
    target: f32,
    prev_gain: f32,
}

impl Default for Leveller {
    fn default() -> Self {
        Self::new()
    }
}

impl Leveller {
    pub fn new() -> Self {
        Self {
            target: TARGET_RMS,
            prev_gain: 1.0,
        }
    }

    /// Level one chunk in place.
    pub fn apply(&mut self, samples: &mut [f32]) {
        let min_gain = 10f32.powf(-MAX_GAIN_DB / 20.0);
        let max_gain = 10f32.powf(MAX_GAIN_DB / 20.0);

        let mut sum_sq = 0.0f32;
        let mut voiced = 0usize;
        let mut peak = 0.0f32;
        for &s in samples.iter() {
            let a = s.abs();
            peak = peak.max(a);
            if a > VOICED_FLOOR {
                sum_sq += s * s;
                voiced += 1;
            }
        }

        if voiced < MIN_VOICED || sum_sq == 0.0 {
            // No usable level measurement — ramp to the previous gain (a
            // no-op ramp) and leave the target untouched.
            ramp(samples, self.prev_gain, self.prev_gain);
            return;
        }

        let rms = (sum_sq / voiced as f32).sqrt();
        let gain = (self.target / rms).clamp(min_gain, max_gain);
        // Boosting must never push a chunk into clipping once it's converted
        // to i16. The ramp's multiplier sweeps from prev_gain to gain, so
        // BOTH ends have to clear this chunk's peak — capping only `gain`
        // would still let a loud onset early in the ramp clip. Where
        // prev_gain itself would clip this chunk, the ramp starts lower and
        // the boundary carries a small level step; a bounded step is the
        // better artefact of the two.
        let ceiling = 0.99 / peak; // peak > 0 is guaranteed by the voiced check above
        let gain = gain.min(ceiling);
        let from = self.prev_gain.min(ceiling);

        ramp(samples, from, gain);
        self.prev_gain = gain;

        // Update the running target from the chunk's raw (pre-gain) RMS,
        // weighted by how much voiced audio the chunk actually had so a
        // short opener doesn't dominate it.
        let a = EMA_ALPHA * (voiced as f32 / EMA_REF_VOICED).min(1.0);
        self.target = (1.0 - a) * self.target + a * rms;
    }
}

/// Scale `samples` by a gain that slews linearly from `from` to `to` over
/// the first `RAMP_SAMPLES` (or the whole slice if shorter), then holds at
/// `to`.
fn ramp(samples: &mut [f32], from: f32, to: f32) {
    let ramp_len = RAMP_SAMPLES.min(samples.len());
    for (i, s) in samples.iter_mut().enumerate().take(ramp_len) {
        let t = i as f32 / ramp_len as f32;
        *s *= from + (to - from) * t;
    }
    for s in samples.iter_mut().skip(ramp_len) {
        *s *= to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced_rms(samples: &[f32]) -> f32 {
        let mut sum_sq = 0.0f32;
        let mut voiced = 0usize;
        for &s in samples {
            if s.abs() > VOICED_FLOOR {
                sum_sq += s * s;
                voiced += 1;
            }
        }
        (sum_sq / voiced as f32).sqrt()
    }

    #[test]
    fn a_loud_first_chunk_is_pulled_toward_the_target() {
        let mut leveller = Leveller::new();
        // ~2.1 dB above TARGET_RMS — loud, but inside the 3 dB cap so the
        // gain isn't clamped and the settled RMS should land on the target.
        let mut samples = vec![0.12f32; 4_000];
        leveller.apply(&mut samples);
        // Skip the ramp region, which is still transitioning.
        let settled = &samples[RAMP_SAMPLES..];
        let rms = voiced_rms(settled);
        assert!(
            (rms - TARGET_RMS).abs() < TARGET_RMS * 0.05,
            "settled rms {rms} not within 5% of target {TARGET_RMS}"
        );
    }

    #[test]
    fn a_gain_beyond_the_cap_is_moved_by_exactly_the_cap() {
        let mut leveller = Leveller::new();
        // Far quieter than the target — well beyond the 3 dB cap.
        let mut samples = vec![0.01f32; 4_000];
        leveller.apply(&mut samples);
        let max_gain = 10f32.powf(MAX_GAIN_DB / 20.0);
        assert!((leveller.prev_gain - max_gain).abs() < 1e-4);
    }

    #[test]
    fn gain_ramps_continuously_across_a_boundary() {
        let mut leveller = Leveller::new();
        let mut first = vec![0.3f32; 4_000];
        leveller.apply(&mut first);
        let g = leveller.prev_gain;

        let mut second = vec![0.05f32; 4_000];
        let original_first_sample = second[0];
        leveller.apply(&mut second);
        // The ramp starts from the previous chunk's gain, so the first
        // sample of the next chunk is scaled by it, not by the new gain.
        assert!((second[0] - original_first_sample * g).abs() < 1e-6);
    }

    #[test]
    fn a_near_silent_chunk_leaves_the_target_unchanged_and_is_stable() {
        let mut leveller = Leveller::new();
        let target_before = leveller.target;
        let mut samples = vec![0.0f32; 4_000];
        leveller.apply(&mut samples);
        assert_eq!(leveller.target, target_before);
        assert!(samples.iter().all(|s| s.is_finite()));
        assert!(leveller.prev_gain.is_finite());
    }

    #[test]
    fn output_never_clips_when_input_peak_is_near_one() {
        let mut leveller = Leveller::new();
        // A very quiet chunk (large boost needed) but one loud peak sample.
        let mut samples = vec![0.01f32; 4_000];
        samples[100] = 0.999;
        leveller.apply(&mut samples);
        for &s in &samples {
            assert!(s.abs() <= 1.0, "sample {s} exceeds 1.0");
        }
    }

    #[test]
    fn a_loud_onset_early_in_the_ramp_does_not_clip_when_prev_gain_is_high() {
        let mut leveller = Leveller::new();
        // Drive prev_gain up to MAX_GAIN.
        let mut first = vec![0.03f32; 4_000];
        leveller.apply(&mut first);

        // A quiet chunk with a loud onset right at the start of the ramp —
        // the clip guard must cap prev_gain's end of the ramp too, not just
        // the new chunk's gain.
        let mut second = vec![0.02f32; 4_000];
        second[5] = 0.99;
        leveller.apply(&mut second);
        for &s in &second {
            assert!(s.abs() <= 1.0, "sample {s} exceeds 1.0");
        }
    }
}
