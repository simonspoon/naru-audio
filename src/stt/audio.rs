//! WAV bytes to 16 kHz mono `f32`, ported from auris `src/audio.rs`
//! (d8644d6), plus the `is_silent` energy gate.
//!
//! Accepts 8/16/24/32-bit PCM and 32-bit float WAV, any channel count
//! (averaged to mono), at 1 kHz–384 kHz (resampled with rubato). Input is capped at
//! 256 MiB and 10 minutes.

use std::io::{Cursor, ErrorKind, Read};

/// Cap on bytes read from the source.
pub const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;

/// Cap on decoded audio, in seconds.
pub const MAX_DECODED_SECONDS: u32 = 10 * 60;

/// What the recognizer and the VAD are built for.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// −60 dBFS. auris measured the quietest speech fixture at 1.1e-1 max
/// window RMS and digital silence at 0; Parakeet hallucinates "Okay." on
/// digital silence, so this gate runs before the recognizer. It catches "no
/// signal", not room tone — that is the Silero gate's job.
const SILENCE_RMS_THRESHOLD: f32 = 1e-3;

/// [`is_silent`]'s window: 30 ms, so a short burst of speech is not averaged
/// away.
pub const SILENCE_WINDOW_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize * 30) / 1000;

/// True when the maximum RMS over any [`SILENCE_WINDOW_SAMPLES`] window is
/// below −60 dBFS. A slice shorter than one window is one window.
pub fn is_silent(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let window = SILENCE_WINDOW_SAMPLES.min(samples.len()).max(1);

    // Sliding sum of squares, O(n).
    let mut sum_sq: f64 = samples[..window]
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum();
    let mut max_sum_sq = sum_sq;
    for i in window..samples.len() {
        let entering = samples[i] as f64;
        let leaving = samples[i - window] as f64;
        sum_sq += entering * entering - leaving * leaving;
        if sum_sq > max_sum_sq {
            max_sum_sq = sum_sq;
        }
    }
    let max_rms = (max_sum_sq / window as f64).sqrt() as f32;
    max_rms < SILENCE_RMS_THRESHOLD
}

/// Source rates accepted. rubato's FFT resampler sizes its buffers by
/// `rate_in / gcd(rate_in, 16000)`, so an unchecked rate can allocate
/// gigabytes; nothing reaches it without passing this range.
const MIN_SAMPLE_RATE: u32 = 1_000;
const MAX_SAMPLE_RATE: u32 = 384_000;

#[derive(Debug)]
pub enum AudioError {
    /// The reader failed (not a size-cap trip).
    Io(std::io::Error),
    /// Zero bytes: distinct from silence and from garbage.
    NoInput,
    /// Not RIFF. Names the container it looked like, when recognisable.
    NotWav(String),
    /// `hound` rejected the stream: malformed, truncated, or an unsupported
    /// variant.
    Wav(hound::Error),
    Resample(String),
    /// More than [`MAX_INPUT_BYTES`].
    TooLarge,
    /// Longer than [`MAX_DECODED_SECONDS`].
    TooLong,
    /// A valid WAV with no samples.
    Empty,
    /// The header's sample rate is outside 1 kHz–384 kHz.
    UnsupportedRate(u32),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Io(e) => write!(f, "failed to read audio: {e}"),
            AudioError::NoInput => write!(f, "input was empty; expected a wav stream"),
            AudioError::NotWav(detected) => write!(
                f,
                "not a wav file ({detected}); expected a wav stream (RIFF/WAVE)"
            ),
            AudioError::Wav(e) => write!(f, "failed to decode wav: {e}"),
            AudioError::Resample(msg) => write!(f, "failed to resample audio: {msg}"),
            AudioError::TooLarge => write!(
                f,
                "audio exceeds the {} MiB input size cap",
                MAX_INPUT_BYTES / 1024 / 1024
            ),
            AudioError::TooLong => write!(
                f,
                "decoded audio exceeds the {}-minute duration cap",
                MAX_DECODED_SECONDS / 60
            ),
            AudioError::Empty => write!(f, "no audio samples decoded"),
            AudioError::UnsupportedRate(rate) => write!(
                f,
                "wav declares an unsupported sample rate ({rate} hz, must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE})"
            ),
        }
    }
}

impl std::error::Error for AudioError {}

/// Fails with `ErrorKind::OutOfMemory` once more than `cap` bytes have come
/// through, so the caller can tell "too big" from an I/O error.
struct LimitedReader<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> LimitedReader<R> {
    fn new(inner: R, cap: u64) -> Self {
        LimitedReader {
            inner,
            remaining: cap,
        }
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            // At the cap: one probe byte tells "ends exactly here" from "more".
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(std::io::Error::new(
                    ErrorKind::OutOfMemory,
                    "input exceeds size cap",
                )),
            };
        }
        let n = self.inner.read(buf)?;
        self.remaining = self.remaining.saturating_sub(n as u64);
        Ok(n)
    }
}

fn is_size_cap_error(e: &std::io::Error) -> bool {
    e.kind() == ErrorKind::OutOfMemory
}

fn map_hound_error(e: hound::Error) -> AudioError {
    if let hound::Error::IoError(io_err) = &e
        && is_size_cap_error(io_err)
    {
        return AudioError::TooLarge;
    }
    AudioError::Wav(e)
}

fn map_io_error(e: std::io::Error) -> AudioError {
    if is_size_cap_error(&e) {
        AudioError::TooLarge
    } else {
        AudioError::Io(e)
    }
}

/// Averages interleaved channels to mono.
fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let channels = channels as usize;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Checked at the native rate, before paying for a resample.
fn check_duration(mono: &[f32], sample_rate: u32) -> Result<(), AudioError> {
    if mono.is_empty() {
        return Err(AudioError::Empty);
    }
    let seconds = mono.len() as f64 / sample_rate as f64;
    if seconds > MAX_DECODED_SECONDS as f64 {
        return Err(AudioError::TooLong);
    }
    Ok(())
}

/// Resamples mono audio to [`TARGET_SAMPLE_RATE`]; a no-op at 16 kHz.
fn resample_to_target(input: &[f32], input_rate: u32) -> Result<Vec<f32>, AudioError> {
    if input_rate == TARGET_SAMPLE_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }

    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    const CHANNELS: usize = 1;
    const CHUNK_SIZE: usize = 1024;

    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        CHUNK_SIZE,
        CHANNELS,
        FixedSync::Both,
    )
    .map_err(|e| AudioError::Resample(e.to_string()))?;

    let input_adapter = InterleavedSlice::new(input, CHANNELS, input.len())
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    let output = resampler
        .process_all(&input_adapter, input.len(), None)
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    Ok(output.take_data())
}

/// The sniffed magic bytes replayed ahead of the size-capped remainder.
type SniffedReader<R> = std::io::Chain<Cursor<Vec<u8>>, LimitedReader<R>>;

/// Frames pulled per read: 256 ms at 16 kHz. Also the chunk size the
/// pipeline feeds the VAD in (auris's streaming loop).
pub const READ_CHUNK_FRAMES: usize = 4096;

/// A WAV whose header has been parsed and validated; the body is read in
/// chunks. auris streams from this; here only [`decode`] uses it.
struct WavStream<R: Read> {
    wav: hound::WavReader<SniffedReader<R>>,
    channels: u16,
    sample_rate: u32,
    format: hound::SampleFormat,
    bits_per_sample: u16,
    frames_read: u64,
}

impl<R: Read> WavStream<R> {
    fn open(reader: R) -> Result<Self, AudioError> {
        let mut limited = LimitedReader::new(reader, MAX_INPUT_BYTES);
        let mut magic = [0u8; 8];
        let n = fill_as_much_as_possible(&mut limited, &mut magic).map_err(map_io_error)?;

        if n == 0 {
            return Err(AudioError::NoInput);
        }
        if n < 4 || &magic[0..4] != b"RIFF" {
            return Err(AudioError::NotWav(describe_non_wav(&magic[..n])));
        }
        let chained = Cursor::new(magic[..n].to_vec()).chain(limited);

        let wav = hound::WavReader::new(chained).map_err(map_hound_error)?;
        let spec = wav.spec();
        if spec.channels == 0 {
            return Err(AudioError::Wav(hound::Error::FormatError(
                "wav header declares zero channels",
            )));
        }
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&spec.sample_rate) {
            return Err(AudioError::UnsupportedRate(spec.sample_rate));
        }
        // §2.2: 8/16/24/32-bit int or 32-bit float. hound reads 8-bit WAV,
        // which is unsigned, as signed around zero.
        let format = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 8 | 16 | 24 | 32) | (hound::SampleFormat::Float, 32) => {
                spec.sample_format
            }
            _ => return Err(AudioError::Wav(hound::Error::Unsupported)),
        };

        Ok(WavStream {
            wav,
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            format,
            bits_per_sample: spec.bits_per_sample,
            frames_read: 0,
        })
    }

    /// Up to `frames` more mono samples at the source rate; empty at EOF.
    fn next_chunk(&mut self, frames: usize) -> Result<Vec<f32>, AudioError> {
        let wanted = frames * self.channels as usize;
        let interleaved: Vec<f32> = match self.format {
            hound::SampleFormat::Int => {
                // Full scale is 2^(bits-1): 32768.0 at 16 bits.
                let scale = (1u64 << (self.bits_per_sample - 1)) as f32;
                self.wav
                    .samples::<i32>()
                    .take(wanted)
                    .map(|s| s.map(|v| v as f32 / scale).map_err(map_hound_error))
                    .collect::<Result<_, _>>()?
            }
            hound::SampleFormat::Float => self
                .wav
                .samples::<f32>()
                .take(wanted)
                .map(|s| s.map_err(map_hound_error))
                .collect::<Result<_, _>>()?,
        };

        let mono = downmix_to_mono(&interleaved, self.channels);
        self.frames_read += mono.len() as u64;
        if self.frames_read > MAX_DECODED_SECONDS as u64 * self.sample_rate as u64 {
            return Err(AudioError::TooLong);
        }
        Ok(mono)
    }

    /// The rest of the stream, mono at [`TARGET_SAMPLE_RATE`].
    fn rest_resampled(&mut self) -> Result<Vec<f32>, AudioError> {
        let mut mono = Vec::new();
        loop {
            let chunk = self.next_chunk(READ_CHUNK_FRAMES)?;
            if chunk.is_empty() {
                break;
            }
            mono.extend_from_slice(&chunk);
        }
        check_duration(&mono, self.sample_rate)?;
        resample_to_target(&mono, self.sample_rate)
    }
}

/// Names a recognisable wrong-container magic in `head`.
fn describe_non_wav(head: &[u8]) -> String {
    const EBML: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
    if head.starts_with(&EBML) {
        return "looks like webm/matroska".to_string();
    }
    if head.starts_with(b"OggS") {
        return "looks like ogg".to_string();
    }
    if head.starts_with(b"fLaC") {
        return "looks like flac".to_string();
    }
    if head.starts_with(b"ID3") || (head.len() >= 2 && head[0] == 0xFF && (head[1] & 0xE0) == 0xE0)
    {
        return "looks like mp3".to_string();
    }
    if head.len() >= 8 && &head[4..8] == b"ftyp" {
        return "looks like mp4/m4a".to_string();
    }
    "unrecognised header".to_string()
}

/// Reads up to `buf.len()` bytes, stopping early only at EOF.
fn fill_as_much_as_possible(reader: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Decodes a whole WAV stream to 16 kHz mono.
pub fn decode(reader: impl Read) -> Result<Vec<f32>, AudioError> {
    WavStream::open(reader)?.rest_resampled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_wav_i16(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn write_wav_f32(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn mono_16k_16bit_is_a_no_op() {
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768, 5];
        let wav = write_wav_i16(16_000, 1, &samples);
        let out = decode(Cursor::new(wav)).unwrap();
        assert_eq!(out.len(), samples.len());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1000.0 / 32768.0).abs() < 1e-6);
        assert!((out[3] - 32767.0 / 32768.0).abs() < 1e-6);
        assert!((out[4] - (-1.0)).abs() < 1e-6);
    }

    fn write_wav_int(bits_per_sample: u16, samples: &[i32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    /// Minimum, a negative half, zero, a positive half and maximum at each
    /// int depth §2.2 promises, scaled to [-1, 1).
    #[test]
    fn every_int_depth_scales_to_unit_range() {
        for bits in [8u16, 16, 24, 32] {
            let full = 1i64 << (bits - 1);
            let samples = [-full, -full / 2, 0, full / 2, full - 1].map(|v| v as i32);
            let out = decode(Cursor::new(write_wav_int(bits, &samples))).unwrap();
            let expected = [-1.0, -0.5, 0.0, 0.5, (full - 1) as f32 / full as f32];
            assert_eq!(out.len(), expected.len(), "{bits}-bit");
            for (a, b) in out.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6, "{bits}-bit: {a} != {b}");
            }
        }
    }

    /// 8-bit WAV is unsigned: the byte 0x80 is zero, 0x00 is -1.0.
    #[test]
    fn eight_bit_is_unsigned_on_disk() {
        let wav = write_wav_int(8, &[-128, 0, 127]);
        assert_eq!(&wav[wav.len() - 3..], &[0x00, 0x80, 0xFF]);
        let out = decode(Cursor::new(wav)).unwrap();
        assert_eq!(out, vec![-1.0, 0.0, 127.0 / 128.0]);
    }

    #[test]
    fn stereo_44100_resamples_and_downmixes() {
        let in_rate = 44_100;
        let n = 4410; // 0.1s
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / in_rate as f32;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let s = (v * 16000.0) as i16;
            samples.push(s);
            samples.push(s / 2); // distinct but non-cancelling right channel
        }
        let wav = write_wav_i16(in_rate as u32, 2, &samples);
        let out = decode(Cursor::new(wav)).unwrap();

        let expected_len = n * TARGET_SAMPLE_RATE as usize / in_rate;
        let tolerance = expected_len / 10 + 8;
        assert!(
            (out.len() as i64 - expected_len as i64).unsigned_abs() as usize <= tolerance,
            "out.len()={} expected~={}",
            out.len(),
            expected_len
        );

        assert!(out.iter().any(|&v| v.abs() > 0.01), "output looks silent");
        assert!(out.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn float32_round_trips() {
        let samples = vec![0.0f32, 0.5, -0.5, 0.999, -1.0];
        let wav = write_wav_f32(16_000, 1, &samples);
        let out = decode(Cursor::new(wav)).unwrap();
        assert_eq!(out.len(), samples.len());
        for (a, b) in out.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn truncated_wav_errors_not_panics() {
        let wav = write_wav_i16(16_000, 1, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let cut = &wav[..wav.len() - 4]; // chop off part of the data chunk
        let result = decode(Cursor::new(cut.to_vec()));
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_is_no_input_not_not_wav() {
        let err = decode(Cursor::new(Vec::<u8>::new())).unwrap_err();
        assert!(matches!(err, AudioError::NoInput));
    }

    #[test]
    fn empty_data_chunk_is_empty() {
        let wav = write_wav_i16(16_000, 1, &[]);
        let err = decode(Cursor::new(wav)).unwrap_err();
        assert!(matches!(err, AudioError::Empty));
    }

    #[test]
    fn plain_text_is_rejected() {
        let err = decode(Cursor::new(b"this is not audio at all".to_vec())).unwrap_err();
        assert!(matches!(err, AudioError::NotWav(_)));
    }

    #[test]
    fn webm_magic_is_named_in_the_error() {
        let mut blob = vec![0x1A, 0x45, 0xDF, 0xA3];
        blob.extend_from_slice(&[0u8; 16]);
        let err = decode(Cursor::new(blob)).unwrap_err();
        match err {
            AudioError::NotWav(msg) => assert!(msg.contains("webm")),
            other => panic!("expected NotWav, got {other:?}"),
        }
    }

    /// A minimal PCM WAV with an arbitrary `sample_rate`; `hound::WavWriter`
    /// panics on a zero or absurd rate, so it is built by hand.
    fn hand_crafted_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let bits_per_sample: u16 = 16;
        let block_align = channels as u32 * (bits_per_sample as u32 / 8);
        let byte_rate = sample_rate.wrapping_mul(block_align);
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&(block_align as u16).to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data);
        buf
    }

    #[test]
    fn wav_zero_rate_is_rejected() {
        let wav = hand_crafted_wav(0, 1, &[1, 2, 3, 4]);
        let err = decode(Cursor::new(wav)).unwrap_err();
        assert!(matches!(err, AudioError::UnsupportedRate(0)));
    }

    #[test]
    fn wav_absurd_rate_is_rejected() {
        // Above the cap, but with a consistent byte_rate so hound's own
        // fmt-chunk check doesn't intercept it first.
        let wav = hand_crafted_wav(1_000_000, 1, &[1, 2, 3, 4]);
        let err = decode(Cursor::new(wav)).unwrap_err();
        assert!(matches!(err, AudioError::UnsupportedRate(_)));
    }

    #[test]
    fn rate_bounds_are_inclusive() {
        for rate in [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE] {
            let wav = hand_crafted_wav(rate, 1, &[1000; 64]);
            assert!(decode(Cursor::new(wav)).is_ok(), "{rate} hz rejected");
        }
        for rate in [MIN_SAMPLE_RATE - 1, MAX_SAMPLE_RATE + 1] {
            let wav = hand_crafted_wav(rate, 1, &[1000; 64]);
            let err = decode(Cursor::new(wav)).unwrap_err();
            assert!(matches!(err, AudioError::UnsupportedRate(r) if r == rate));
        }
    }

    /// The byte cap, at a cap small enough to test: a WAV read through a
    /// `LimitedReader` shorter than it is `TooLarge`, not a hound error.
    #[test]
    fn oversized_input_is_rejected() {
        let wav = write_wav_i16(16_000, 1, &[1000; 256]);
        let cap = wav.len() as u64 / 2;
        let reader = hound::WavReader::new(LimitedReader::new(Cursor::new(wav), cap)).unwrap();
        let err = reader
            .into_samples::<i16>()
            .map(|s| s.map_err(map_hound_error))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_err();
        assert!(matches!(err, AudioError::TooLarge), "{err:?}");
    }

    #[test]
    fn input_exactly_at_the_cap_is_accepted() {
        let wav = write_wav_i16(16_000, 1, &[1000; 256]);
        let cap = wav.len() as u64;
        let mut bytes = Vec::new();
        LimitedReader::new(Cursor::new(wav), cap)
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes.len() as u64, cap);
    }

    fn sine_at(amplitude: f32) -> Vec<f32> {
        (0..TARGET_SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / TARGET_SAMPLE_RATE as f32;
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn all_zero_samples_are_silent() {
        assert!(is_silent(&vec![0.0f32; TARGET_SAMPLE_RATE as usize]));
    }

    #[test]
    fn ordinary_speech_level_is_not_silent() {
        assert!(!is_silent(&sine_at(0.1)));
    }

    #[test]
    fn very_low_level_signal_is_silent() {
        assert!(is_silent(&sine_at(1e-4)));
    }

    /// −45 dBFS has an RMS of ~3.5e-3, above the gate.
    #[test]
    fn low_but_real_signal_is_not_silent() {
        assert!(!is_silent(&sine_at(5e-3)));
    }

    /// Why the gate is a windowed max, not an overall RMS.
    #[test]
    fn a_short_burst_amid_silence_is_not_silent() {
        let mut samples = vec![0.0f32; TARGET_SAMPLE_RATE as usize];
        let burst = sine_at(0.1);
        let start = samples.len() / 2;
        samples[start..start + burst.len().min(1600)]
            .copy_from_slice(&burst[..burst.len().min(1600)]);
        assert!(!is_silent(&samples));
    }

    #[test]
    fn empty_slice_is_silent() {
        assert!(is_silent(&[]));
    }
}
