//! Offline source separation (naru task 1461 §8, follow-up): isolates
//! vocals from music/noise, the `isolate` pipeline step `create_sample`
//! previously refused outright. The `sherpa-onnx` crate (1.13.6) does not
//! wrap this C API yet, but the prebuilt static lib it already links
//! (`target/sherpa-onnx-prebuilt/.../lib/libsherpa-onnx-c-api.a`, checked
//! with `nm -g` on both the macOS arm64 and Linux x64 v1.13.6 static libs —
//! same symbol set on both, so this FFI needs no platform `cfg` gate)
//! exports the C symbols directly. [`ffi`] is a hand-written binding
//! against `sherpa-onnx/c-api/c-api.h` at tag v1.13.6 (quoted inline where
//! relied on); [`Isolator`] is the safe wrapper, shaped like
//! [`super::diarize::Diarizer`]/[`super::denoise::Denoiser`].
//!
//! **Stereo, not mono, even for our mono pipeline.** `SherpaOnnxOffline
//! SourceSeparationProcess`'s own C++ (`offline-source-separation-spleeter-
//! impl.h`'s `ComputeStft(input, 1)`) does `if (ch >= input.samples.data.
//! size()) { SHERPA_ONNX_LOGE(...); SHERPA_ONNX_EXIT(-1); }` — a hard
//! `exit(-1)` of the whole process, not a recoverable error, if
//! `num_channels` is 1. [`Isolator::vocals`] always passes the mono buffer
//! as two identical channels to avoid it; checked against a real pulled
//! `sherpa-onnx-spleeter-2stems-int8` model on a real clip (this session's
//! smoke test) rather than left as a theoretical read of the C++.

use std::ffi::CString;
use std::path::{Path, PathBuf};

/// `sherpa-onnx-spleeter-2stems-int8`'s two files.
pub const VOCALS_FILENAME: &str = "vocals.int8.onnx";
pub const ACCOMPANIMENT_FILENAME: &str = "accompaniment.int8.onnx";

/// A hand-written binding to `sherpa-onnx/c-api/c-api.h` (tag v1.13.6,
/// lines 4329-4448 as fetched from
/// `raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.6/sherpa-onnx/c-api/c-api.h`
/// during this session), the "Source separation" section:
///
/// ```c
/// typedef struct SherpaOnnxOfflineSourceSeparationSpleeterModelConfig {
///   const char *vocals;
///   const char *accompaniment;
/// } SherpaOnnxOfflineSourceSeparationSpleeterModelConfig;
///
/// typedef struct SherpaOnnxOfflineSourceSeparationUvrModelConfig {
///   const char *model;
/// } SherpaOnnxOfflineSourceSeparationUvrModelConfig;
///
/// typedef struct SherpaOnnxOfflineSourceSeparationModelConfig {
///   SherpaOnnxOfflineSourceSeparationSpleeterModelConfig spleeter;
///   SherpaOnnxOfflineSourceSeparationUvrModelConfig uvr;
///   int32_t num_threads;
///   int32_t debug;
///   const char *provider;
/// } SherpaOnnxOfflineSourceSeparationModelConfig;
///
/// typedef struct SherpaOnnxOfflineSourceSeparationConfig {
///   SherpaOnnxOfflineSourceSeparationModelConfig model;
/// } SherpaOnnxOfflineSourceSeparationConfig;
///
/// typedef struct SherpaOnnxOfflineSourceSeparation
///     SherpaOnnxOfflineSourceSeparation;
///
/// SHERPA_ONNX_API const SherpaOnnxOfflineSourceSeparation *
/// SherpaOnnxCreateOfflineSourceSeparation(
///     const SherpaOnnxOfflineSourceSeparationConfig *config);
/// SHERPA_ONNX_API void SherpaOnnxDestroyOfflineSourceSeparation(
///     const SherpaOnnxOfflineSourceSeparation *ss);
/// SHERPA_ONNX_API int32_t SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(
///     const SherpaOnnxOfflineSourceSeparation *ss);
/// SHERPA_ONNX_API int32_t SherpaOnnxOfflineSourceSeparationGetNumberOfStems(
///     const SherpaOnnxOfflineSourceSeparation *ss);
///
/// typedef struct SherpaOnnxSourceSeparationStem {
///   float **samples;      // samples[c] is channel c's heap array
///   int32_t num_channels;
///   int32_t n;             // samples per channel
/// } SherpaOnnxSourceSeparationStem;
///
/// typedef struct SherpaOnnxSourceSeparationOutput {
///   const SherpaOnnxSourceSeparationStem *stems;
///   int32_t num_stems;
///   int32_t sample_rate;
/// } SherpaOnnxSourceSeparationOutput;
///
/// SHERPA_ONNX_API const SherpaOnnxSourceSeparationOutput *
/// SherpaOnnxOfflineSourceSeparationProcess(
///     const SherpaOnnxOfflineSourceSeparation *ss, const float *const *samples,
///     int32_t num_channels, int32_t num_samples, int32_t sample_rate);
/// SHERPA_ONNX_API void SherpaOnnxDestroySourceSeparationOutput(
///     const SherpaOnnxSourceSeparationOutput *p);
/// ```
///
/// `offline-source-separation-spleeter-impl.h`'s `Process` (same tag)
/// pushes vocals as `stems[0]` and accompaniment as `stems[1]` — checked
/// directly, not assumed, since nothing in the header states the order.
mod ffi {
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct SpleeterModelConfig {
        pub vocals: *const c_char,
        pub accompaniment: *const c_char,
    }

    #[repr(C)]
    pub struct UvrModelConfig {
        pub model: *const c_char,
    }

    #[repr(C)]
    pub struct ModelConfig {
        pub spleeter: SpleeterModelConfig,
        pub uvr: UvrModelConfig,
        pub num_threads: c_int,
        pub debug: c_int,
        pub provider: *const c_char,
    }

    #[repr(C)]
    pub struct Config {
        pub model: ModelConfig,
    }

    /// Opaque; never constructed on the Rust side.
    #[repr(C)]
    pub struct OfflineSourceSeparation {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Stem {
        pub samples: *mut *mut f32,
        pub num_channels: c_int,
        pub n: c_int,
    }

    #[repr(C)]
    pub struct Output {
        pub stems: *const Stem,
        pub num_stems: c_int,
        pub sample_rate: c_int,
    }

    // No `#[link(...)]`: `sherpa-onnx-sys`'s build script already emits
    // the `cargo:rustc-link-lib`/`-search` directives that pull
    // `libsherpa-onnx-c-api` into any binary in this workspace, and Cargo
    // applies every dependency's link directives to the final link
    // regardless of which crate's `extern` block names the symbol.
    unsafe extern "C" {
        pub fn SherpaOnnxCreateOfflineSourceSeparation(
            config: *const Config,
        ) -> *const OfflineSourceSeparation;
        pub fn SherpaOnnxDestroyOfflineSourceSeparation(ss: *const OfflineSourceSeparation);
        pub fn SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(
            ss: *const OfflineSourceSeparation,
        ) -> c_int;
        pub fn SherpaOnnxOfflineSourceSeparationGetNumberOfStems(
            ss: *const OfflineSourceSeparation,
        ) -> c_int;
        pub fn SherpaOnnxOfflineSourceSeparationProcess(
            ss: *const OfflineSourceSeparation,
            samples: *const *const f32,
            num_channels: c_int,
            num_samples: c_int,
            sample_rate: c_int,
        ) -> *const Output;
        pub fn SherpaOnnxDestroySourceSeparationOutput(p: *const Output);
    }
}

/// Silence padded onto each end of the input to [`Isolator::vocals`].
const EDGE_PAD_SECS: f64 = 0.5;

/// `pad` samples at `in_rate`, as samples at `out_rate`.
fn trim_len(pad: usize, in_rate: i32, out_rate: i32) -> usize {
    (pad as f64 * out_rate as f64 / in_rate.max(1) as f64).round() as usize
}

#[derive(Debug)]
pub enum IsolateError {
    MissingModel(PathBuf),
    /// The model path was not valid UTF-8/had an interior NUL, or
    /// `SherpaOnnxCreateOfflineSourceSeparation` returned `NULL`.
    CreateFailed,
    /// `SherpaOnnxOfflineSourceSeparationProcess` returned `NULL`, or an
    /// empty/malformed output.
    ProcessFailed,
}

impl std::fmt::Display for IsolateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsolateError::MissingModel(path) => {
                write!(f, "source separation model not found: {}", path.display())
            }
            IsolateError::CreateFailed => write!(f, "failed to create the source separator"),
            IsolateError::ProcessFailed => write!(f, "source separation produced no output"),
        }
    }
}

impl std::error::Error for IsolateError {}

fn cstring(path: &Path) -> Result<CString, IsolateError> {
    CString::new(path.to_string_lossy().into_owned()).map_err(|_| IsolateError::CreateFailed)
}

pub struct Isolator {
    ptr: *const ffi::OfflineSourceSeparation,
}

// SAFETY: mirrors `Diarizer`/`Denoiser`: the underlying sherpa-onnx C++
// object has no thread-affinity of its own beyond what `Process` already
// serialises through the FFI call.
unsafe impl Send for Isolator {}
unsafe impl Sync for Isolator {}

impl Isolator {
    /// `dir` is the pulled `source-separation-spleeter-2stems-int8` model
    /// directory.
    pub fn load(dir: &Path) -> Result<Self, IsolateError> {
        let vocals = dir.join(VOCALS_FILENAME);
        if !vocals.is_file() {
            return Err(IsolateError::MissingModel(vocals));
        }
        let accompaniment = dir.join(ACCOMPANIMENT_FILENAME);
        if !accompaniment.is_file() {
            return Err(IsolateError::MissingModel(accompaniment));
        }
        let vocals_c = cstring(&vocals)?;
        let accompaniment_c = cstring(&accompaniment)?;
        let provider_c = CString::new("cpu").expect("no interior NUL");
        let config = ffi::Config {
            model: ffi::ModelConfig {
                spleeter: ffi::SpleeterModelConfig {
                    vocals: vocals_c.as_ptr(),
                    accompaniment: accompaniment_c.as_ptr(),
                },
                uvr: ffi::UvrModelConfig {
                    model: std::ptr::null(),
                },
                num_threads: 1,
                debug: 0,
                provider: provider_c.as_ptr(),
            },
        };
        // SAFETY: `config` and the `CString`s it borrows from all outlive
        // this call; the C side copies whatever it needs before returning.
        let ptr = unsafe { ffi::SherpaOnnxCreateOfflineSourceSeparation(&config) };
        if ptr.is_null() {
            return Err(IsolateError::CreateFailed);
        }
        let isolator = Isolator { ptr };
        // SAFETY: `ptr` is non-null. Spleeter reports 2 stems (vocals,
        // accompaniment); anything less means `Self::vocals` cannot find
        // `stems[0]` and should fail now, at load, not on first use.
        let stems = unsafe { ffi::SherpaOnnxOfflineSourceSeparationGetNumberOfStems(ptr) };
        if stems < 1 {
            return Err(IsolateError::CreateFailed);
        }
        Ok(isolator)
    }

    /// The sample rate [`Self::vocals`]'s output is at (Spleeter: 44100 Hz,
    /// regardless of the input rate given to `vocals` — the C++ resamples
    /// internally, matching [`super::denoise::Denoiser::run`]'s contract).
    pub fn output_sample_rate(&self) -> i32 {
        // SAFETY: `self.ptr` is non-null for the object's whole lifetime.
        unsafe { ffi::SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(self.ptr) }
    }

    /// Separates mono `pcm` (at `input_sample_rate`) and returns the
    /// vocals stem, mono, at [`Self::output_sample_rate`]. `pcm` is
    /// duplicated into two identical channels before the call: the C++
    /// aborts the whole process (`exit(-1)`, not a recoverable error) on
    /// `num_channels: 1` (see this module's doc comment).
    ///
    /// The model's STFT blows up at both edges of its input (measured on a
    /// real 10 s clip: samples near 0 and the end reach +40 dBFS, 100x the
    /// input), so `pcm` is padded with [`EDGE_PAD_SECS`] of silence each side
    /// and the same span is cut from the output, putting the artefacts
    /// outside the returned audio.
    pub fn vocals(&self, pcm: &[f32], input_sample_rate: i32) -> Result<Vec<f32>, IsolateError> {
        let pad = (input_sample_rate.max(0) as f64 * EDGE_PAD_SECS) as usize;
        let mut padded = vec![0.0; pad];
        padded.extend_from_slice(pcm);
        padded.resize(pad + pcm.len() + pad, 0.0);
        let out = self.separate(&padded, input_sample_rate)?;
        let cut = trim_len(pad, input_sample_rate, self.output_sample_rate());
        let end = out.len().saturating_sub(cut);
        Ok(out[cut.min(end)..end].to_vec())
    }

    fn separate(&self, pcm: &[f32], input_sample_rate: i32) -> Result<Vec<f32>, IsolateError> {
        let channels: [*const f32; 2] = [pcm.as_ptr(), pcm.as_ptr()];
        // SAFETY: `channels` points at `pcm`, which outlives this call;
        // `pcm.len()` matches `num_samples` for both (identical) channels.
        let output = unsafe {
            ffi::SherpaOnnxOfflineSourceSeparationProcess(
                self.ptr,
                channels.as_ptr(),
                2,
                pcm.len() as i32,
                input_sample_rate,
            )
        };
        if output.is_null() {
            return Err(IsolateError::ProcessFailed);
        }
        // SAFETY: `output` is non-null and, per the C++ above, points at a
        // fully populated `Output` with `num_stems` `Stem`s, freed exactly
        // once below regardless of which branch returns.
        let result = unsafe {
            let out = &*output;
            if out.num_stems < 1 || out.stems.is_null() {
                Err(IsolateError::ProcessFailed)
            } else {
                let stems = std::slice::from_raw_parts(out.stems, out.num_stems as usize);
                let vocals_stem = &stems[0]; // stems[0] is vocals; checked above.
                if vocals_stem.num_channels < 1 || vocals_stem.samples.is_null() {
                    Err(IsolateError::ProcessFailed)
                } else {
                    let channel_ptrs = std::slice::from_raw_parts(
                        vocals_stem.samples,
                        vocals_stem.num_channels as usize,
                    );
                    let n = vocals_stem.n.max(0) as usize;
                    Ok(std::slice::from_raw_parts(channel_ptrs[0], n).to_vec())
                }
            }
        };
        // SAFETY: `output` came from `Process` above and is destroyed
        // exactly once, after every read of it.
        unsafe { ffi::SherpaOnnxDestroySourceSeparationOutput(output) };
        result
    }
}

impl Drop for Isolator {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is non-null and was created by `Self::load`.
        unsafe { ffi::SherpaOnnxDestroyOfflineSourceSeparation(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_is_an_error_not_a_panic() {
        let err = Isolator::load(Path::new("/nonexistent/naru-audio-isolate-test-model"))
            .err()
            .expect("expected an error");
        assert!(matches!(err, IsolateError::MissingModel(_)), "{err}");
    }

    #[test]
    fn partial_model_dir_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(VOCALS_FILENAME), b"").unwrap();
        let err = Isolator::load(tmp.path()).err().expect("expected an error");
        assert!(matches!(err, IsolateError::MissingModel(_)), "{err}");
    }

    #[test]
    fn padding_trims_to_the_output_rate() {
        assert_eq!(trim_len(8000, 16_000, 44_100), 22_050);
        assert_eq!(trim_len(0, 16_000, 44_100), 0);
    }

    #[test]
    fn isolator_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Isolator>();
    }
}
