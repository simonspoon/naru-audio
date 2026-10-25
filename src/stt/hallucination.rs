//! Whisper's stock outputs on noise (naru task 2254): "Thank you.",
//! "Спасибо", looping "Wow. Wow. Wow.", "0000 0000 …". [`is_hallucination`]
//! judges a segment's final text against its PCM; it is pure and fails open,
//! so anything it does not recognise passes.

use std::collections::HashSet;

use super::audio::TARGET_SAMPLE_RATE;
use super::vad::MIN_SILENCE_SECONDS;

/// Real short replies ("yes", "okay", "seven", "thank you") measured 0.6-0.7 s
/// at about -18 dBFS; stock phrases came from room tone (2.4 s, -36 dBFS) and
/// speech-shaped noise (~20 s segments).
/// Length and level are measured over the speech span, not the padded segment.
const STOCK_MIN_SECS: f32 = 2.0;
/// A stock phrase whose speech is `STOCK_MIN_SECS` or longer is dropped only
/// if it is quieter than this.
const STOCK_LOUD_DBFS: f32 = -30.0;
/// A stock phrase quieter than this is dropped whatever its length.
const STOCK_MAX_DBFS: f32 = -40.0;
/// Real 1-4 word replies never fill 6 s of continuous VAD speech; the noise
/// segments held 1-4 word stock phrases over ~20 s.
const SPARSE_MAX_WORDS: usize = 4;
const SPARSE_MIN_SECS: f32 = 6.0;
/// Past `SPARSE_MIN_SECS`, fewer words a second than this is sparse too
/// (naru task 2345): "Oh, yeah. Oh, yeah. Oh, yeah." held a ~20 s noise
/// segment, 0.3 a second. Real speech runs about 2.5 a second and slow
/// dictation over 1; a VAD span has no pause of `min_silence` (0.5 s) or more.
/// A client that widens `min_silence` lets thinking pauses into a span, so
/// the floor holds only at the default or tighter.
const SPARSE_MIN_WPS: f32 = 0.5;
/// Loops ("Wow. Wow. Wow.") compress far past whisper's own 2.4 zlib
/// threshold. No compression crate here, so it is the same idea on words:
/// at least this many, of which at most `LOOP_MAX_UNIQUE` (a share) differ.
const LOOP_MIN_CHARS: usize = 40;
const LOOP_MIN_WORDS: usize = 12;
const LOOP_MAX_UNIQUE: f32 = 0.4;

const STOCK: &[&str] = &[
    "thank you",
    "thanks",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "please subscribe",
    "you",
    "bye",
    "okay",
    "ok",
    "alright",
    "uh-huh",
    "so",
    "спасибо",
    "obrigado",
    "gracias",
    "danke",
    "merci",
    "谢谢",
    "ご視聴ありがとうございました",
];

/// Why `text` over `samples` (the unpadded speech, 16 kHz mono, split at
/// `min_silence` seconds of quiet) should be dropped, if it should.
pub fn is_hallucination(text: &str, samples: &[f32], min_silence: f32) -> Option<&'static str> {
    let secs = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
    let words: Vec<&str> = text.split_whitespace().collect();

    let norm = normalize(text);
    let level = dbfs(samples);
    if STOCK.contains(&norm.as_str())
        && ((secs >= STOCK_MIN_SECS && level <= STOCK_LOUD_DBFS) || level <= STOCK_MAX_DBFS)
    {
        return Some("stock phrase");
    }
    // Unspaced scripts have no word count to speak of.
    if !words.is_empty()
        && (words.len() <= SPARSE_MAX_WORDS
            || (min_silence <= MIN_SILENCE_SECONDS && (words.len() as f32) < SPARSE_MIN_WPS * secs))
        && secs >= SPARSE_MIN_SECS
        && !text.chars().any(unspaced)
    {
        return Some("sparse text");
    }
    if text.chars().count() >= LOOP_MIN_CHARS && words.len() >= LOOP_MIN_WORDS {
        let unique: HashSet<String> = words.iter().map(|w| normalize(w)).collect();
        if (unique.len() as f32) / (words.len() as f32) <= LOOP_MAX_UNIQUE {
            return Some("repetition loop");
        }
    }
    None
}

/// CJK ideographs, kana, Hangul, Thai, Lao, Khmer and Myanmar: scripts written without spaces.
fn unspaced(c: char) -> bool {
    matches!(c as u32,
        0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1100..=0x11FF | 0x1780..=0x17FF | 0x3040..=0x30FF | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF)
}

/// Lowercase, letters/digits and inner hyphens and spaces only.
fn normalize(text: &str) -> String {
    let kept: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-')
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    10.0 * mean.max(1e-12).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the default `min_silence`.
    fn is_hallucination(text: &str, samples: &[f32]) -> Option<&'static str> {
        super::is_hallucination(text, samples, MIN_SILENCE_SECONDS)
    }

    /// A square-ish tone `secs` long at `db` dBFS RMS.
    fn clip(secs: f32, db: f32) -> Vec<f32> {
        let amp = 10f32.powf(db / 20.0);
        (0..(secs * TARGET_SAMPLE_RATE as f32) as usize)
            .map(|i| if i % 80 < 40 { amp } else { -amp })
            .collect()
    }

    #[test]
    fn real_short_replies_pass() {
        let pcm = clip(0.7, -18.0);
        for t in [
            "Yes.",
            "No",
            "No.",
            "7.",
            "Okay.",
            "Thank you.",
            "Thanks.",
            "Спасибо.",
        ] {
            assert_eq!(is_hallucination(t, &pcm), None, "{t}");
        }
    }

    #[test]
    fn stock_phrase_in_room_tone_drops() {
        assert_eq!(
            is_hallucination("Thank you.", &clip(2.4, -36.0)),
            Some("stock phrase")
        );
        assert_eq!(
            is_hallucination("Спасибо", &clip(0.7, -45.0)),
            Some("stock phrase")
        );
    }

    #[test]
    fn long_sparse_text_drops() {
        assert_eq!(
            is_hallucination("Have a great day.", &clip(20.0, -30.0)),
            Some("sparse text")
        );
    }

    #[test]
    fn short_loops_in_long_noise_drop() {
        let pcm = clip(19.5, -28.0);
        for t in [
            "Oh, yeah. Oh, yeah. Oh, yeah.",
            "Oh, God. Oh, God. Oh, God.",
        ] {
            assert_eq!(is_hallucination(t, &pcm), Some("sparse text"), "{t}");
        }
    }

    #[test]
    fn slow_real_speech_passes() {
        // Slow dictation, about 1 word a second, and a slow sentence.
        assert_eq!(
            is_hallucination(
                "Remind me to call the dentist tomorrow morning at nine.",
                &clip(10.0, -20.0)
            ),
            None
        );
        assert_eq!(
            is_hallucination("Turn off the lights.", &clip(5.5, -20.0)),
            None
        );
        assert_eq!(
            is_hallucination("Please turn off all the lights.", &clip(9.0, -20.0)),
            None
        );
    }

    #[test]
    fn wide_min_silence_keeps_slow_speech() {
        // Thinking pauses let in by a 2 s `min_silence`: 10 words over 22 s.
        let text = "I think we should, maybe, move the meeting to Friday.";
        let pcm = clip(22.0, -20.0);
        assert_eq!(super::is_hallucination(text, &pcm, 2.0), None);
        assert_eq!(is_hallucination(text, &pcm), Some("sparse text"));
    }

    #[test]
    fn loops_drop() {
        let pcm = clip(3.0, -20.0);
        assert_eq!(
            is_hallucination(&"Wow. ".repeat(14), &pcm),
            Some("repetition loop")
        );
        let zeros = "0000 ".repeat(40);
        assert_eq!(
            is_hallucination(zeros.trim(), &pcm),
            Some("repetition loop")
        );
    }

    #[test]
    fn real_repeated_replies_pass() {
        let pcm = clip(3.0, -20.0);
        assert_eq!(is_hallucination(&"no ".repeat(8), &pcm), None);
        assert_eq!(is_hallucination(&"yes ".repeat(7), &pcm), None);
    }

    #[test]
    fn unspaced_script_is_not_sparse() {
        let pcm = clip(6.5, -20.0);
        assert_eq!(
            is_hallucination("请把客厅的灯关掉再把卧室的空调打开", &pcm),
            None
        );
    }

    #[test]
    fn padding_is_not_speech() {
        // 0.7 s of speech at -18 dBFS between 1.0 s of silence each side:
        // judged on the speech span only, it passes.
        let mut padded = vec![0.0; 16_000];
        padded.extend(clip(0.7, -18.0));
        padded.extend(vec![0.0; 16_000]);
        let speech = &padded[16_000..16_000 + 11_200];
        assert_eq!(is_hallucination("Thank you.", speech), None);
    }

    #[test]
    fn real_sentences_pass() {
        let pcm = clip(5.0, -20.0);
        assert_eq!(
            is_hallucination("Please turn the lights off in the kitchen.", &pcm),
            None
        );
        assert_eq!(is_hallucination("", &pcm), None);
    }
}
