//! Pure helpers for editing a clip in the Sample Studio (naru task 1576):
//! multi-segment edits (validate, merge, splice with a crossfade), interval
//! arithmetic, crosstalk detection from diarized spans, and waveform peaks
//! read straight from `working.wav` (so a long clip is never decoded whole
//! just to be drawn).

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{PrepError, SpeakerSpan};
use crate::stt::audio::TARGET_SAMPLE_RATE;

/// One time range on a clip's `working.wav`, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
}

impl Segment {
    pub fn len(&self) -> f64 {
        self.end - self.start
    }
}

/// The most segments one edit may carry.
pub const MAX_SEGMENTS: usize = 2048;

/// Crossfade between spliced segments: 10 ms.
pub const CROSSFADE_SECS: f64 = 0.010;

/// Segments closer than this are one segment.
const MERGE_GAP_SECS: f64 = 0.001;

/// Slack past the clip's end that is clamped rather than refused (the
/// client's float arithmetic on `duration_secs`).
const END_SLACK_SECS: f64 = 0.05;

/// `raw` validated against a clip `duration` seconds long, then sorted and
/// merged: every `start`/`end` finite, `start >= 0`, `end > start`, `end` at
/// most `duration` (a hair over is clamped), at most [`MAX_SEGMENTS`].
/// `Err` names the offending index.
pub fn normalize_segments(raw: &[Segment], duration: f64) -> Result<Vec<Segment>, String> {
    if raw.len() > MAX_SEGMENTS {
        return Err(format!(
            "segments has {} entries, at most {MAX_SEGMENTS}",
            raw.len()
        ));
    }
    let mut segs = Vec::with_capacity(raw.len());
    for (i, s) in raw.iter().enumerate() {
        if !s.start.is_finite() || !s.end.is_finite() {
            return Err(format!("segments[{i}] has a non-finite time"));
        }
        if s.start < 0.0 {
            return Err(format!("segments[{i}].start must be >= 0"));
        }
        if s.end <= s.start {
            return Err(format!("segments[{i}] is empty: end must exceed start"));
        }
        if s.end > duration + END_SLACK_SECS {
            return Err(format!(
                "segments[{i}].end {} is past the clip's {duration:.3} s",
                s.end
            ));
        }
        segs.push(Segment {
            start: s.start,
            end: s.end.min(duration),
        });
        if s.start >= duration {
            return Err(format!("segments[{i}].start is past the clip's end"));
        }
    }
    Ok(merge(segs))
}

/// `segs` sorted by start, with overlapping or touching ones merged.
pub fn merge(mut segs: Vec<Segment>) -> Vec<Segment> {
    segs.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<Segment> = Vec::with_capacity(segs.len());
    for s in segs {
        match out.last_mut() {
            Some(last) if s.start <= last.end + MERGE_GAP_SECS => last.end = last.end.max(s.end),
            _ => out.push(s),
        }
    }
    out
}

/// The parts of `a` inside `b`; both sorted and merged, as is the result.
pub fn intersect(a: &[Segment], b: &[Segment]) -> Vec<Segment> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let start = a[i].start.max(b[j].start);
        let end = a[i].end.min(b[j].end);
        if end > start {
            out.push(Segment { start, end });
        }
        if a[i].end < b[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// The parts of `a` outside `b`; both sorted and merged, as is the result.
pub fn subtract(a: &[Segment], b: &[Segment]) -> Vec<Segment> {
    let mut out = Vec::new();
    for seg in a {
        let mut cursor = seg.start;
        for cut in b.iter().filter(|c| c.end > seg.start && c.start < seg.end) {
            if cut.start > cursor {
                out.push(Segment {
                    start: cursor,
                    end: cut.start,
                });
            }
            cursor = cursor.max(cut.end);
        }
        if cursor < seg.end {
            out.push(Segment {
                start: cursor,
                end: seg.end,
            });
        }
    }
    out
}

/// Total seconds of `segs` that fall inside `of`'s overlap with `with`.
pub fn overlap_secs(seg: &Segment, with: &[Segment]) -> f64 {
    intersect(std::slice::from_ref(seg), with)
        .iter()
        .map(Segment::len)
        .sum()
}

/// `pcm` cut to `segs` (sorted, merged, seconds) and spliced in order, each
/// join crossfaded over [`CROSSFADE_SECS`] (shortened for a piece too short
/// to give that much). One segment is a plain crop.
pub fn splice(pcm: &[f32], segs: &[Segment]) -> Vec<f32> {
    let rate = TARGET_SAMPLE_RATE as f64;
    let fade = (CROSSFADE_SECS * rate).round() as usize;
    let mut out: Vec<f32> = Vec::new();
    for seg in segs {
        let from = ((seg.start.max(0.0) * rate).round() as usize).min(pcm.len());
        let to = ((seg.end.max(0.0) * rate).round() as usize).clamp(from, pcm.len());
        let piece = &pcm[from..to];
        if piece.is_empty() {
            continue;
        }
        let n = fade.min(out.len() / 2).min(piece.len() / 2);
        if n == 0 {
            out.extend_from_slice(piece);
            continue;
        }
        let tail = out.len() - n;
        for k in 0..n {
            let t = (k as f32 + 0.5) / n as f32;
            out[tail + k] = out[tail + k] * (1.0 - t) + piece[k] * t;
        }
        out.extend_from_slice(&piece[n..]);
    }
    out
}

/// Time ranges where spans of two or more different speakers overlap,
/// sorted and merged.
pub fn overlaps_of(spans: &[SpeakerSpan]) -> Vec<Segment> {
    // (time, +1 start / -1 end, speaker); ends sort before starts at the
    // same instant so touching spans are not an overlap.
    let mut events: Vec<(f64, i32, i32)> = spans
        .iter()
        .filter(|s| s.end > s.start)
        .flat_map(|s| [(s.start, 1, s.speaker), (s.end, -1, s.speaker)])
        .collect();
    events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut active: HashMap<i32, i32> = HashMap::new();
    let mut open: Option<f64> = None;
    let mut out = Vec::new();
    for (t, delta, speaker) in events {
        let count = active.entry(speaker).or_insert(0);
        *count += delta;
        if *count <= 0 {
            active.remove(&speaker);
        }
        match (active.len() >= 2, open) {
            (true, None) => open = Some(t),
            (false, Some(start)) => {
                if t > start {
                    out.push(Segment { start, end: t });
                }
                open = None;
            }
            _ => {}
        }
    }
    merge(out)
}

/// Min and max per bucket over a time range of a clip.
#[derive(Debug, Serialize)]
pub struct Peaks {
    pub sample_rate: u32,
    pub duration: f64,
    pub start: f64,
    pub end: f64,
    pub buckets: usize,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

/// Peaks of `start..end` seconds (`None` = clip start / end) of the WAV at
/// `path`, in `buckets` buckets. Reads only that range, one pass, never the
/// whole file into memory. `Err(PrepError::BadAudio)` for an empty range.
pub fn peaks(
    path: &Path,
    start: Option<f64>,
    end: Option<f64>,
    buckets: usize,
) -> Result<Peaks, PrepError> {
    let file = std::fs::File::open(path).map_err(|e| PrepError::Io(format!("open wav: {e}")))?;
    let mut wav = hound::WavReader::new(BufReader::with_capacity(1 << 20, file))
        .map_err(|e| PrepError::BadAudio(e.to_string()))?;
    let spec = wav.spec();
    let channels = spec.channels.max(1) as u32;
    let total = wav.duration();
    let rate = spec.sample_rate as f64;
    let duration = total as f64 / rate;
    let start = start.unwrap_or(0.0).clamp(0.0, duration);
    let end = end.unwrap_or(duration).clamp(start, duration);
    let first = ((start * rate).round() as u32).min(total);
    let last = ((end * rate).round() as u32).clamp(first, total);
    let frames = (last - first) as u64;
    if frames == 0 {
        return Err(PrepError::BadAudio("the range is empty".to_string()));
    }
    let buckets = buckets.max(1).min(frames as usize);
    let mut min = vec![f32::INFINITY; buckets];
    let mut max = vec![f32::NEG_INFINITY; buckets];
    wav.seek(first)
        .map_err(|e| PrepError::Io(format!("seek wav: {e}")))?;

    // One value per sample (the channel average is not needed for a
    // waveform sketch: channel 0 stands in for a multi-channel file).
    let mut visit = |frame: u64, channel: u32, v: f32| {
        if channel == 0 {
            let b = (frame * buckets as u64 / frames) as usize;
            if v < min[b] {
                min[b] = v;
            }
            if v > max[b] {
                max[b] = v;
            }
        }
    };
    let take = frames * channels as u64;
    let bad = |e: hound::Error| PrepError::BadAudio(e.to_string());
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            for (i, s) in wav.samples::<i32>().take(take as usize).enumerate() {
                let i = i as u64;
                visit(
                    i / channels as u64,
                    (i % channels as u64) as u32,
                    s.map_err(bad)? as f32 / scale,
                );
            }
        }
        hound::SampleFormat::Float => {
            for (i, s) in wav.samples::<f32>().take(take as usize).enumerate() {
                let i = i as u64;
                visit(
                    i / channels as u64,
                    (i % channels as u64) as u32,
                    s.map_err(bad)?,
                );
            }
        }
    }
    for b in 0..buckets {
        if min[b] > max[b] {
            // A short read left this bucket empty.
            (min[b], max[b]) = (0.0, 0.0);
        }
    }
    Ok(Peaks {
        sample_rate: spec.sample_rate,
        duration,
        start,
        end,
        buckets,
        min,
        max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64) -> Segment {
        Segment { start, end }
    }

    fn span(start: f64, end: f64, speaker: i32) -> SpeakerSpan {
        SpeakerSpan {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn normalize_sorts_merges_and_clamps() {
        let out = normalize_segments(&[seg(5.0, 6.0), seg(1.0, 3.0), seg(2.5, 4.0)], 10.0).unwrap();
        assert_eq!(out, vec![seg(1.0, 4.0), seg(5.0, 6.0)]);
        let out = normalize_segments(&[seg(1.0, 10.02)], 10.0).unwrap();
        assert_eq!(out, vec![seg(1.0, 10.0)]);
    }

    #[test]
    fn normalize_rejects_bad_segments() {
        assert!(normalize_segments(&[seg(2.0, 2.0)], 10.0).is_err());
        assert!(normalize_segments(&[seg(-1.0, 2.0)], 10.0).is_err());
        assert!(normalize_segments(&[seg(1.0, 11.0)], 10.0).is_err());
        assert!(normalize_segments(&[seg(f64::NAN, 2.0)], 10.0).is_err());
        let many = vec![seg(0.0, 0.5); MAX_SEGMENTS + 1];
        assert!(normalize_segments(&many, 10.0).is_err());
    }

    #[test]
    fn interval_arithmetic() {
        let a = [seg(0.0, 4.0), seg(6.0, 9.0)];
        let b = [seg(3.0, 7.0)];
        assert_eq!(intersect(&a, &b), vec![seg(3.0, 4.0), seg(6.0, 7.0)]);
        assert_eq!(subtract(&a, &b), vec![seg(0.0, 3.0), seg(7.0, 9.0)]);
        assert_eq!(subtract(&a, &[]), a.to_vec());
        assert!((overlap_secs(&seg(2.0, 8.0), &b) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn splice_crossfades_the_join() {
        let rate = TARGET_SAMPLE_RATE as usize;
        let pcm = vec![1.0f32; rate * 4];
        let out = splice(&pcm, &[seg(0.0, 1.0), seg(2.0, 3.0)]);
        let fade = (CROSSFADE_SECS * rate as f64).round() as usize;
        assert_eq!(out.len(), 2 * rate - fade);
        // A constant signal stays constant through an equal-gain fade.
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        // One segment is a plain crop.
        assert_eq!(splice(&pcm, &[seg(1.0, 2.0)]).len(), rate);
    }

    #[test]
    fn splice_crossfade_blends_different_levels() {
        let rate = TARGET_SAMPLE_RATE as usize;
        let mut pcm = vec![0.0f32; rate * 2];
        pcm[rate..].iter_mut().for_each(|v| *v = 1.0);
        let out = splice(&pcm, &[seg(0.0, 1.0), seg(1.0, 2.0)]);
        // Touching segments would have merged; spliced directly the join
        // ramps from 0 to 1 instead of stepping.
        let fade = (CROSSFADE_SECS * rate as f64).round() as usize;
        let join = rate - fade;
        assert!(out[join] < 0.1 && out[join + fade - 1] > 0.9);
        assert!(out[join..join + fade].windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn overlaps_need_two_different_speakers() {
        let spans = [
            span(0.0, 5.0, 0),
            span(4.0, 8.0, 1),
            span(8.0, 9.0, 0),
            span(1.0, 2.0, 0), // same speaker nested: not crosstalk
        ];
        assert_eq!(overlaps_of(&spans), vec![seg(4.0, 5.0)]);
        assert!(overlaps_of(&[span(0.0, 1.0, 0), span(1.0, 2.0, 1)]).is_empty());
        assert!(overlaps_of(&[]).is_empty());
    }

    #[test]
    fn peaks_cover_the_requested_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.wav");
        // 1 s of 0.5 then 1 s of -0.25.
        let mut pcm = vec![0.5f32; 16_000];
        pcm.extend(vec![-0.25f32; 16_000]);
        super::super::write_wav(&path, &pcm).unwrap();
        let p = peaks(&path, None, None, 4).unwrap();
        assert_eq!((p.buckets, p.min.len(), p.max.len()), (4, 4, 4));
        assert!((p.duration - 2.0).abs() < 1e-9);
        assert!(p.max[0] > 0.49 && p.min[3] < -0.24 && p.max[3] < -0.24);
        let p = peaks(&path, Some(1.0), Some(2.0), 8).unwrap();
        assert!(p.max.iter().all(|&v| v < -0.24));
        assert_eq!((p.start, p.end), (1.0, 2.0));
        assert!(peaks(&path, Some(1.0), Some(1.0), 8).is_err());
    }
}
