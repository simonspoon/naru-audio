//! Best-take picking (naru task 1576): cuts a speaker's usable speech into
//! 1.5-8 s stretches ("takes"), scores each on noise, clipping, crosstalk,
//! word confidence and pace, and greedily picks the best until the total
//! reaches the target. Pure: the caller loads the clip and its transcript.

use serde::Serialize;

use super::dsp;
use super::edit::{self, Segment};
use super::{SpeakerSpan, TranscriptWord};
use crate::stt::audio::TARGET_SAMPLE_RATE;

/// Takes shorter than this are dropped.
pub const MIN_TAKE_SECS: f64 = 1.0;
/// A take is closed at a word gap once it is at least this long.
const SOFT_MIN_SECS: f64 = 1.5;
/// No take is longer than this.
const MAX_TAKE_SECS: f64 = 8.0;
/// A silence between words at least this long may end a take.
const GAP_SECS: f64 = 0.3;
/// A silence this long always ends a take.
const LONG_GAP_SECS: f64 = 0.8;
/// Context kept either side of a take's first and last word.
const PAD_SECS: f64 = 0.05;
/// `|x|` at or above this counts as clipped.
const CLIP_LEVEL: f32 = 0.99;

#[derive(Debug, Clone, Copy)]
pub struct TakeOptions<'a> {
    pub speaker: Option<i32>,
    /// The currently kept region, sorted and merged; `None` = the whole clip.
    pub kept: Option<&'a [Segment]>,
    pub target_min: f64,
    pub target_max: f64,
    pub exclude_overlaps: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TakeMetrics {
    pub noise_floor_dbfs: Option<f64>,
    pub clipped_ratio: f64,
    pub overlap_ratio: f64,
    pub confidence: Option<f64>,
    pub words_per_sec: Option<f64>,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Take {
    pub start: f64,
    pub end: f64,
    pub score: f64,
    pub picked: bool,
    pub metrics: TakeMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct Takes {
    pub takes: Vec<Take>,
    pub picked_secs: f64,
}

/// The takes of `pcm` (16 kHz mono, the whole clip) and which ones to keep.
pub fn plan_takes(
    pcm: &[f32],
    words: &[TranscriptWord],
    speakers: &[SpeakerSpan],
    overlaps: &[Segment],
    opts: &TakeOptions,
) -> Takes {
    let mut regions = speech_regions(pcm, words, speakers, opts.speaker);
    if let Some(kept) = opts.kept {
        regions = edit::intersect(&regions, kept);
    }
    if opts.exclude_overlaps {
        regions = edit::subtract(&regions, overlaps);
    }

    let mut takes: Vec<Take> = regions
        .iter()
        .flat_map(|r| stretches(r, words))
        .map(|s| score_take(pcm, words, overlaps, s))
        .collect();
    takes.sort_by(|a, b| a.start.total_cmp(&b.start));

    // Best score first; the earlier take wins a tie.
    let mut order: Vec<usize> = (0..takes.len()).collect();
    order.sort_by(|&a, &b| {
        takes[b]
            .score
            .total_cmp(&takes[a].score)
            .then(takes[a].start.total_cmp(&takes[b].start))
    });
    let mut picked_secs = 0.0;
    for i in order {
        if picked_secs >= opts.target_min {
            break;
        }
        let len = takes[i].end - takes[i].start;
        if picked_secs + len <= opts.target_max {
            takes[i].picked = true;
            picked_secs += len;
        }
    }
    Takes { takes, picked_secs }
}

/// Where `speaker` (or anyone) speaks: their diarized spans, else, with no
/// diarization at all, the transcript's word extents, else (no words either)
/// the whole clip.
fn speech_regions(
    pcm: &[f32],
    words: &[TranscriptWord],
    speakers: &[SpeakerSpan],
    speaker: Option<i32>,
) -> Vec<Segment> {
    let spans: Vec<Segment> = speakers
        .iter()
        .filter(|s| speaker.is_none_or(|sp| s.speaker == sp))
        .map(|s| Segment {
            start: s.start,
            end: s.end,
        })
        .filter(|s| s.end > s.start)
        .collect();
    if !spans.is_empty() || !speakers.is_empty() {
        return edit::merge(spans);
    }
    let from_words: Vec<Segment> = words
        .iter()
        .map(|w| Segment {
            start: w.start,
            end: w.end,
        })
        .filter(|s| s.end > s.start)
        .collect();
    if !from_words.is_empty() {
        // Words in one breath are one region; a long pause splits it.
        let mut merged: Vec<Segment> = Vec::new();
        for w in edit::merge(from_words) {
            match merged.last_mut() {
                Some(last) if w.start - last.end < LONG_GAP_SECS => last.end = w.end,
                _ => merged.push(w),
            }
        }
        return merged;
    }
    vec![Segment {
        start: 0.0,
        end: pcm.len() as f64 / TARGET_SAMPLE_RATE as f64,
    }]
}

/// `region` cut into takes at word gaps, each [`MIN_TAKE_SECS`] to
/// [`MAX_TAKE_SECS`] long. A region with no words is cut evenly.
fn stretches(region: &Segment, words: &[TranscriptWord]) -> Vec<Segment> {
    let inside: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| {
            let mid = (w.start + w.end) / 2.0;
            mid >= region.start && mid < region.end
        })
        .collect();
    if inside.is_empty() {
        let n = (region.len() / MAX_TAKE_SECS).ceil().max(1.0) as usize;
        let step = region.len() / n as f64;
        return (0..n)
            .map(|i| Segment {
                start: region.start + i as f64 * step,
                end: region.start + (i + 1) as f64 * step,
            })
            .filter(|s| s.len() >= MIN_TAKE_SECS)
            .collect();
    }

    let close = |first: &TranscriptWord, last: &TranscriptWord| Segment {
        start: (first.start - PAD_SECS).max(region.start),
        end: (last.end + PAD_SECS).min(region.end),
    };
    let mut out = Vec::new();
    let mut first = inside[0];
    let mut last = inside[0];
    for w in &inside[1..] {
        let gap = w.start - last.end;
        let len = last.end - first.start;
        let end_here = gap >= LONG_GAP_SECS
            || (gap >= GAP_SECS && len >= SOFT_MIN_SECS)
            || w.end - first.start > MAX_TAKE_SECS - 2.0 * PAD_SECS;
        if end_here {
            out.push(close(first, last));
            first = w;
        }
        last = w;
    }
    out.push(close(first, last));
    out.retain(|s| s.len() >= MIN_TAKE_SECS);
    out
}

fn score_take(pcm: &[f32], words: &[TranscriptWord], overlaps: &[Segment], s: Segment) -> Take {
    let rate = TARGET_SAMPLE_RATE as f64;
    let from = ((s.start * rate).round() as usize).min(pcm.len());
    let to = ((s.end * rate).round() as usize).clamp(from, pcm.len());
    let slice = &pcm[from..to];
    let duration = s.len();

    let noise_floor_dbfs = dsp::noise_floor_dbfs(slice).filter(|v| v.is_finite());
    let clipped_ratio = if slice.is_empty() {
        0.0
    } else {
        slice.iter().filter(|v| v.abs() >= CLIP_LEVEL).count() as f64 / slice.len() as f64
    };
    let overlap_ratio = (edit::overlap_secs(&s, overlaps) / duration).clamp(0.0, 1.0);
    let inside: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| {
            let mid = (w.start + w.end) / 2.0;
            mid >= s.start && mid < s.end
        })
        .collect();
    let confs: Vec<f64> = inside
        .iter()
        .filter_map(|w| w.confidence.map(f64::from))
        .collect();
    let confidence = (!confs.is_empty()).then(|| confs.iter().sum::<f64>() / confs.len() as f64);
    let words_per_sec = (!inside.is_empty()).then(|| inside.len() as f64 / duration);

    let metrics = TakeMetrics {
        noise_floor_dbfs,
        clipped_ratio,
        overlap_ratio,
        confidence,
        words_per_sec,
        duration,
    };
    Take {
        start: s.start,
        end: s.end,
        score: score(&metrics),
        picked: false,
        metrics,
    }
}

/// Weighted mean of per-metric scores in [0, 1], over the metrics present:
/// noise floor (-60 dBFS or lower is 1, -30 or higher is 0), clipping
/// (0.5% of samples clipped is 0), overlap, word confidence, and pace
/// (2-3.5 words/s is 1, falling to 0 two words/s outside it).
pub fn score(m: &TakeMetrics) -> f64 {
    let mut terms: Vec<(f64, f64)> = vec![
        (0.2, (1.0 - m.clipped_ratio * 200.0).clamp(0.0, 1.0)),
        (0.2, (1.0 - m.overlap_ratio).clamp(0.0, 1.0)),
    ];
    if let Some(nf) = m.noise_floor_dbfs {
        terms.push((0.3, ((-30.0 - nf) / 30.0).clamp(0.0, 1.0)));
    }
    if let Some(c) = m.confidence {
        terms.push((0.2, c.clamp(0.0, 1.0)));
    }
    if let Some(p) = m.words_per_sec {
        let off = if p < 2.0 { 2.0 - p } else { (p - 3.5).max(0.0) };
        terms.push((0.1, (1.0 - off / 2.0).clamp(0.0, 1.0)));
    }
    let weight: f64 = terms.iter().map(|t| t.0).sum();
    terms.iter().map(|t| t.0 * t.1).sum::<f64>() / weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(start: f64, end: f64, confidence: Option<f32>) -> TranscriptWord {
        TranscriptWord {
            start,
            end,
            text: "w".into(),
            confidence,
        }
    }

    /// Words every 0.4 s from `from` to `to`, 0.3 s each.
    fn words(from: f64, to: f64, confidence: Option<f32>) -> Vec<TranscriptWord> {
        let mut out = Vec::new();
        let mut t = from;
        while t + 0.3 <= to {
            out.push(word(t, t + 0.3, confidence));
            t += 0.4;
        }
        out
    }

    fn span(start: f64, end: f64, speaker: i32) -> SpeakerSpan {
        SpeakerSpan {
            start,
            end,
            speaker,
        }
    }

    fn opts<'a>() -> TakeOptions<'a> {
        TakeOptions {
            speaker: Some(0),
            kept: None,
            target_min: 10.0,
            target_max: 20.0,
            exclude_overlaps: true,
        }
    }

    fn tone(secs: f64, amp: f32) -> Vec<f32> {
        (0..(secs * 16_000.0) as usize)
            .map(|i| amp * (i as f32 * 0.05).sin())
            .collect()
    }

    #[test]
    fn score_ranks_clean_above_noisy_clipped_and_crosstalk() {
        let clean = TakeMetrics {
            noise_floor_dbfs: Some(-60.0),
            clipped_ratio: 0.0,
            overlap_ratio: 0.0,
            confidence: Some(0.95),
            words_per_sec: Some(2.7),
            duration: 5.0,
        };
        assert!(score(&clean) > 0.95);
        let noisy = TakeMetrics {
            noise_floor_dbfs: Some(-35.0),
            ..clean.clone()
        };
        let clipped = TakeMetrics {
            clipped_ratio: 0.01,
            ..clean.clone()
        };
        let crosstalk = TakeMetrics {
            overlap_ratio: 0.5,
            ..clean.clone()
        };
        let fast = TakeMetrics {
            words_per_sec: Some(6.0),
            ..clean.clone()
        };
        for worse in [noisy, clipped, crosstalk, fast] {
            assert!(score(&worse) < score(&clean));
        }
        let no_conf = TakeMetrics {
            confidence: None,
            words_per_sec: None,
            ..clean
        };
        let s = score(&no_conf);
        assert!((0.0..=1.0).contains(&s) && s > 0.9);
    }

    #[test]
    fn takes_are_cut_at_gaps_and_within_bounds() {
        // One speaker talking for 30 s, with a 1 s pause at 10 s.
        let mut w = words(0.0, 9.6, None);
        w.extend(words(10.6, 30.0, None));
        let pcm = tone(30.0, 0.1);
        let out = plan_takes(&pcm, &w, &[span(0.0, 30.0, 0)], &[], &opts());
        assert!(out.takes.len() >= 4);
        for t in &out.takes {
            let len = t.end - t.start;
            assert!((MIN_TAKE_SECS..=MAX_TAKE_SECS).contains(&len), "{len}");
            assert!((0.0..=1.0).contains(&t.score));
        }
        assert!(out.takes.windows(2).all(|p| p[0].start <= p[1].start));
        // No take straddles the 1 s pause.
        assert!(out.takes.iter().all(|t| !(t.start < 9.8 && t.end > 10.5)));
    }

    #[test]
    fn picking_lands_in_target_and_never_exceeds_max() {
        let w = words(0.0, 60.0, None);
        let pcm = tone(60.0, 0.1);
        let out = plan_takes(&pcm, &w, &[span(0.0, 60.0, 0)], &[], &opts());
        assert!(
            (10.0..=20.0).contains(&out.picked_secs),
            "{}",
            out.picked_secs
        );
        let sum: f64 = out
            .takes
            .iter()
            .filter(|t| t.picked)
            .map(|t| t.end - t.start)
            .sum();
        assert!((sum - out.picked_secs).abs() < 1e-9);
    }

    #[test]
    fn overlaps_are_excluded_or_penalised() {
        let w = words(0.0, 20.0, None);
        let pcm = tone(20.0, 0.1);
        let speakers = [span(0.0, 20.0, 0), span(8.0, 12.0, 1)];
        let overlaps = edit::overlaps_of(&speakers);
        let out = plan_takes(&pcm, &w, &speakers, &overlaps, &opts());
        assert!(out.takes.iter().all(|t| t.end <= 8.0 || t.start >= 12.0));
        let keep = TakeOptions {
            exclude_overlaps: false,
            ..opts()
        };
        let out = plan_takes(&pcm, &w, &speakers, &overlaps, &keep);
        assert!(out.takes.iter().any(|t| t.metrics.overlap_ratio > 0.0));
    }

    #[test]
    fn kept_region_and_speaker_limit_the_candidates() {
        let w = words(0.0, 40.0, None);
        let pcm = tone(40.0, 0.1);
        let speakers = [span(0.0, 20.0, 0), span(20.0, 40.0, 1)];
        let kept = [Segment {
            start: 5.0,
            end: 15.0,
        }];
        let o = TakeOptions {
            kept: Some(&kept),
            ..opts()
        };
        let out = plan_takes(&pcm, &w, &speakers, &[], &o);
        assert!(!out.takes.is_empty());
        assert!(out.takes.iter().all(|t| t.start >= 5.0 && t.end <= 15.0));
        let o = TakeOptions {
            speaker: Some(1),
            ..opts()
        };
        let out = plan_takes(&pcm, &w, &speakers, &[], &o);
        assert!(out.takes.iter().all(|t| t.start >= 20.0));
    }

    #[test]
    fn clipping_and_confidence_show_in_metrics() {
        let w = words(0.0, 4.0, Some(0.5));
        let mut pcm = tone(4.0, 0.1);
        pcm[1000..1100].fill(1.0);
        let o = TakeOptions {
            speaker: None,
            ..opts()
        };
        let out = plan_takes(&pcm, &w, &[span(0.0, 4.0, 0)], &[], &o);
        let m = &out.takes[0].metrics;
        assert!(m.clipped_ratio > 0.0);
        assert!((m.confidence.unwrap() - 0.5).abs() < 1e-6);
        assert!(m.words_per_sec.unwrap() > 2.0);
    }

    #[test]
    fn no_diarization_falls_back_to_words() {
        let w = words(0.0, 12.0, None);
        let pcm = tone(12.0, 0.1);
        let o = TakeOptions {
            speaker: None,
            ..opts()
        };
        let out = plan_takes(&pcm, &w, &[], &[], &o);
        assert!(!out.takes.is_empty());
    }
}
