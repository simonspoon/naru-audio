//! Echo reference (naru task 2030): the client sends what it is playing
//! (`docs/live-protocol.md`, typed frames) and this module tells whether the
//! microphone is hearing that audio back. It is the second line behind the
//! client's own voice processing, never the first.
//!
//! The test is normalised cross-correlation of the recent microphone audio
//! against the reference over a window of lags, on a 4x decimated copy. Energy
//! alone never gates: a short "yes" from the person does not correlate with
//! the agent's voice, and with no reference, a stale one, too little audio or
//! a score under the threshold the microphone passes (fail open).

use std::collections::VecDeque;
use std::time::Duration;

const RATE: usize = 16_000;
/// Decimation before correlating (boxcar average): 16 kHz to 4 kHz.
const DECIM: usize = 4;
/// Reference audio kept.
const REF_KEEP: usize = 6 * RATE;
/// Microphone audio compared per frame.
const MIC_WINDOW: usize = RATE / 5;
/// Less microphone audio than this is not judged.
const MIC_MIN: usize = RATE / 10;
/// How far behind the reference the microphone may hear it.
const MAX_LAG: usize = RATE * 2 / 5;
/// A reference silent for longer than this is over.
const STALE: Duration = Duration::from_millis(600);
/// Correlation at or above this is an echo.
pub const THRESHOLD: f32 = 0.5;
/// Either side quieter than this (rms) is never an echo.
const ENERGY_FLOOR: f32 = 0.003;

/// A ring of the reference and the recent microphone audio.
pub struct EchoCanceller {
    reference: VecDeque<f32>,
    /// Absolute index one past the newest reference sample.
    end: usize,
    last_push: Option<Duration>,
    mic: VecDeque<f32>,
}

impl Default for EchoCanceller {
    fn default() -> Self {
        Self::new()
    }
}

impl EchoCanceller {
    pub fn new() -> Self {
        Self {
            reference: VecDeque::new(),
            end: 0,
            last_push: None,
            mic: VecDeque::new(),
        }
    }

    /// Whether any reference has been received.
    pub fn has_reference(&self) -> bool {
        self.last_push.is_some()
    }

    /// Adds reference audio received `at` (time since the session began).
    /// Frames land back to back; after a gap the silence between is filled in
    /// so the ring stays on the real-time axis.
    pub fn push_reference(&mut self, samples: &[f32], at: Duration) {
        let now = (at.as_secs_f64() * RATE as f64) as usize;
        if now > self.end + samples.len() {
            let gap = (now - samples.len() - self.end).min(REF_KEEP);
            self.reference.extend(std::iter::repeat_n(0.0, gap));
            self.end = now - samples.len();
        }
        self.reference.extend(samples);
        self.end += samples.len();
        if self.reference.len() > REF_KEEP {
            let drop = self.reference.len() - REF_KEEP;
            self.reference.drain(..drop);
        }
        self.last_push = Some(at);
    }

    /// Records a microphone frame received `at`; whether it is the reference
    /// heard back.
    pub fn is_echo(&mut self, mic: &[f32], at: Duration) -> bool {
        self.mic.extend(mic);
        if self.mic.len() > MIC_WINDOW {
            let drop = self.mic.len() - MIC_WINDOW;
            self.mic.drain(..drop);
        }
        let Some(last) = self.last_push else {
            return false;
        };
        if at.saturating_sub(last) > STALE || self.mic.len() < MIC_MIN {
            return false;
        }
        let window: Vec<f32> = self.mic.iter().copied().collect();
        self.score(&window, 0, MAX_LAG)
            .is_some_and(|s| s >= THRESHOLD)
    }

    /// Whether a finished segment's audio is the reference heard back.
    /// `back` is the samples between the end of `clip` and now.
    pub fn clip_is_echo(&self, clip: &[f32], back: usize) -> bool {
        if self.last_push.is_none() {
            return false;
        }
        let tail = &clip[clip.len().saturating_sub(RATE)..];
        if tail.len() < MIC_MIN {
            return false;
        }
        self.score(tail, back, MAX_LAG)
            .is_some_and(|s| s >= THRESHOLD)
    }

    /// The best correlation of `mic` against the reference ending `back`
    /// samples ago, trying up to `span` further lags back. `None` when the
    /// ring does not reach, or either side is quiet.
    fn score(&self, mic: &[f32], back: usize, span: usize) -> Option<f32> {
        let n = mic.len();
        let held = self.reference.len();
        if back + n > held {
            return None;
        }
        let span = span.min(held - back - n);
        let from = held - back - span - n;
        let region: Vec<f32> = self.reference.iter().skip(from).copied().collect();
        let region = decimate(&region);
        let mic = decimate(mic);
        let nd = mic.len();
        if nd == 0 || region.len() < nd {
            return None;
        }
        let mic_energy: f32 = mic.iter().map(|s| s * s).sum();
        if (mic_energy / nd as f32).sqrt() < ENERGY_FLOOR {
            return None;
        }
        let floor = ENERGY_FLOOR * ENERGY_FLOOR * nd as f32;
        let mut best: Option<f32> = None;
        for start in 0..=(region.len() - nd) {
            let seg = &region[start..start + nd];
            let energy: f32 = seg.iter().map(|s| s * s).sum();
            if energy < floor {
                continue;
            }
            let dot: f32 = seg.iter().zip(&mic).map(|(a, b)| a * b).sum();
            let score = dot / (energy * mic_energy).sqrt();
            best = Some(best.map_or(score, |b| b.max(score)));
        }
        best
    }
}

fn decimate(x: &[f32]) -> Vec<f32> {
    x.as_chunks::<DECIM>()
        .0
        .iter()
        .map(|c| c.iter().sum::<f32>() / DECIM as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 640;

    /// A deterministic speech-like signal: seeded noise, low-passed and
    /// shaped by a slow envelope. Two seeds are unrelated.
    fn voice(len: usize, seed: f32) -> Vec<f32> {
        let mut state = (seed * 1000.0) as u64 + 12345;
        let mut low = 0.0f32;
        (0..len)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let white = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                low += 0.35 * (white - low);
                let t = i as f32 / RATE as f32;
                let env = 0.6 + 0.4 * (t * 7.0 + seed).sin();
                1.2 * env * low
            })
            .collect()
    }

    fn at(frame: usize) -> Duration {
        Duration::from_secs_f64((frame * FRAME) as f64 / RATE as f64)
    }

    /// Plays `reference` in real time and hears `heard(i)` back; whether any
    /// frame after the first 0.5 s was judged an echo, and how many were.
    fn run(reference: &[f32], heard: &[f32]) -> usize {
        let mut echo = EchoCanceller::new();
        let mut hits = 0;
        for (k, (r, m)) in reference.chunks(FRAME).zip(heard.chunks(FRAME)).enumerate() {
            echo.push_reference(r, at(k));
            if k > 25 && echo.is_echo(m, at(k)) {
                hits += 1;
            }
        }
        hits
    }

    #[test]
    fn a_delayed_copy_is_an_echo() {
        let reference = voice(3 * RATE, 0.3);
        for delay in [0usize, 800, 2371, 5000] {
            let mut heard = vec![0.0; delay];
            heard.extend(reference.iter().map(|s| s * 0.4));
            heard.truncate(reference.len());
            let hits = run(&reference, &heard);
            assert!(hits > 30, "delay {delay}: only {hits} frames flagged");
        }
    }

    #[test]
    fn an_unrelated_voice_is_not() {
        let reference = voice(3 * RATE, 0.3);
        let other = voice(3 * RATE, 4.1);
        assert_eq!(run(&reference, &other), 0);
    }

    #[test]
    fn a_short_burst_over_the_reference_passes() {
        // The reference plays; the person says a short word, no echo at all.
        let reference = voice(3 * RATE, 0.3);
        let mut heard = vec![0.0; 3 * RATE];
        let word = voice(RATE / 4, 9.0);
        heard[RATE + 1000..RATE + 1000 + word.len()].copy_from_slice(&word);
        assert_eq!(run(&reference, &heard), 0);
    }

    #[test]
    fn no_reference_or_a_stale_one_passes() {
        let mic = voice(RATE, 0.3);
        let mut none = EchoCanceller::new();
        assert!(!none.is_echo(&mic, at(40)));
        let mut stale = EchoCanceller::new();
        stale.push_reference(&mic, Duration::from_secs(1));
        assert!(!stale.is_echo(&mic, Duration::from_secs(5)));
    }

    #[test]
    fn a_quiet_reference_is_never_an_echo() {
        let quiet: Vec<f32> = voice(RATE, 0.3).iter().map(|s| s * 0.001).collect();
        assert_eq!(run(&quiet, &quiet), 0);
    }

    #[test]
    fn a_clip_is_judged_against_the_reference_it_follows() {
        let reference = voice(3 * RATE, 0.3);
        let mut echo = EchoCanceller::new();
        for (k, r) in reference.chunks(FRAME).enumerate() {
            echo.push_reference(r, at(k));
        }
        // The clip is the reference's middle second, ended 0.5 s ago.
        let clip: Vec<f32> = reference[RATE..2 * RATE].iter().map(|s| s * 0.5).collect();
        let back = reference.len() - 2 * RATE;
        assert!(echo.clip_is_echo(&clip, back));
        assert!(!echo.clip_is_echo(&voice(RATE, 4.1), back));
    }
}
