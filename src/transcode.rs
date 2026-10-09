//! Non-macOS replacement for `afconvert`: decodes a clip (WAV, MP3, AAC/M4A,
//! ALAC) with symphonia, averages it to mono, resamples with rubato and
//! writes a 16-bit mono WAV. macOS keeps shelling out to `afconvert`.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Converts `src` to a mono 16-bit WAV at `rate` Hz at `dst`. Errors once the
/// decoded audio passes `max_secs`, so the clip is never held whole beyond it.
pub fn to_wav(src: &Path, dst: &Path, rate: u32, max_secs: u32) -> Result<(), String> {
    let (mono, src_rate) = decode_mono(src, max_secs)?;
    if mono.is_empty() {
        return Err("no audio in the file".to_string());
    }
    let samples = resample(mono, src_rate, rate)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(dst, spec).map_err(|e| e.to_string())?;
    for s in samples {
        let i = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(i).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())
}

/// Decodes the first audio track to mono f32 (channels averaged); returns
/// the samples and their rate.
fn decode_mono(src: &Path, max_secs: u32) -> Result<(Vec<f32>, u32), String> {
    let file = File::open(src).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = src.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut mono = Vec::new();
    let mut rate = 0;
    let mut channels = 0;
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            // Lenient on purpose: a truncated file keeps what decoded, as
            // afconvert and the other decoders here do. A stream that needs
            // a decoder reset (new codec parameters) is not supported.
            Err(SymError::ResetRequired) => return Err("stream changes format mid-file".into()),
            Err(e) => return Err(e.to_string()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt packet is skipped, as other decoders do.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };
        let spec = *decoded.spec();
        if rate != 0 && (spec.rate != rate || spec.channels.count() != channels) {
            return Err("audio format changes mid-file".into());
        }
        rate = spec.rate;
        channels = spec.channels.count().max(1);
        // Packets can grow (e.g. MP3 frames); reallocate when this one won't fit.
        if buf
            .as_ref()
            .is_none_or(|b| b.capacity() < decoded.capacity() * channels)
        {
            buf = Some(SampleBuffer::new(decoded.capacity() as u64, spec));
        }
        let sb = buf.as_mut().expect("just set");
        sb.copy_interleaved_ref(decoded);
        mono.extend(
            sb.samples()
                .chunks(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32),
        );
        if mono.len() as u64 > u64::from(max_secs) * u64::from(rate) {
            return Err(format!("audio is longer than {max_secs} s"));
        }
    }
    Ok((mono, rate))
}

/// Resamples mono `input` from `from` to `to` Hz; a no-op when equal.
fn resample(input: Vec<f32>, from: u32, to: u32) -> Result<Vec<f32>, String> {
    if from == to || from == 0 {
        return Ok(input);
    }
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    const CHANNELS: usize = 1;
    const CHUNK_SIZE: usize = 1024;
    let mut resampler = Fft::<f32>::new(
        from as usize,
        to as usize,
        CHUNK_SIZE,
        CHANNELS,
        FixedSync::Both,
    )
    .map_err(|e| e.to_string())?;
    let adapter =
        InterleavedSlice::new(&input[..], CHANNELS, input.len()).map_err(|e| e.to_string())?;
    let out = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| e.to_string())?;
    Ok(out.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `secs` of a 440 Hz tone, stereo 16-bit, at `rate`.
    fn tone_wav(path: &Path, rate: u32, secs: f64) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..(secs * f64::from(rate)) as u32 {
            let t = f64::from(i) / f64::from(rate);
            let s = ((t * 440.0 * std::f64::consts::TAU).sin() * 8000.0) as i16;
            w.write_sample(s).unwrap();
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    /// Converts `src` and checks: mono, `rate`, `secs` long within `tol`, not silent.
    fn check(src: &Path, rate: u32, secs: f64, tol: f64) {
        let dst = src.with_file_name("out.wav");
        to_wav(src, &dst, rate, 60).unwrap();
        let mut r = hound::WavReader::open(&dst).unwrap();
        let spec = r.spec();
        assert_eq!((spec.channels, spec.sample_rate), (1, rate));
        let samples: Vec<i16> = r.samples::<i16>().map(Result::unwrap).collect();
        let got = samples.len() as f64 / f64::from(rate);
        assert!((got - secs).abs() <= secs * tol, "{got} s vs {secs} s");
        let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!(peak > 1000, "silent output, peak {peak}");
    }

    #[test]
    fn stereo_44k_wav_becomes_mono_at_the_target_rate() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("in.wav");
        tone_wav(&src, 44_100, 3.0);
        check(&src, 16_000, 3.0, 0.01);
        check(&src, 24_000, 3.0, 0.01);
    }

    #[cfg(target_os = "macos")]
    fn afconvert(src: &Path, dst: &Path, args: &[&str]) -> bool {
        std::process::Command::new("afconvert")
            .args(args)
            .arg(src)
            .arg(dst)
            .output()
            .is_ok_and(|o| o.status.success())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn afconvert_made_m4a_decodes() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("in.wav");
        tone_wav(&wav, 44_100, 3.0);
        let m4a = dir.path().join("in.m4a");
        assert!(afconvert(&wav, &m4a, &["-f", "m4af", "-d", "aac"]));
        // AAC adds encoder delay/padding; allow 5%.
        check(&m4a, 16_000, 3.0, 0.05);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn afconvert_made_mp3_decodes_if_afconvert_can_write_one() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("in.wav");
        tone_wav(&wav, 44_100, 3.0);
        let mp3 = dir.path().join("in.mp3");
        if !afconvert(&wav, &mp3, &["-f", "MPG3", "-d", ".mp3"]) {
            eprintln!("skipped: afconvert cannot write MP3 here");
            return;
        }
        check(&mp3, 16_000, 3.0, 0.05);
    }

    #[test]
    fn clips_past_the_cap_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("in.wav");
        tone_wav(&src, 8_000, 3.0);
        assert!(to_wav(&src, &dir.path().join("o.wav"), 16_000, 2).is_err());
        assert!(to_wav(&src, &dir.path().join("o.wav"), 16_000, 3).is_ok());
    }

    #[test]
    fn garbage_and_empty_files_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let junk = dir.path().join("junk.mp3");
        std::fs::write(&junk, b"definitely not audio, just some text bytes").unwrap();
        let empty = dir.path().join("empty.wav");
        std::fs::write(&empty, b"").unwrap();
        for p in [junk, empty, dir.path().join("missing.wav")] {
            assert!(to_wav(&p, &dir.path().join("o.wav"), 16_000, 60).is_err());
        }
    }
}
