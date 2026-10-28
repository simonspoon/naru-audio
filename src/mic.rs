//! Headless mic capture (naru task 2371): the daemon listens on a local input
//! device with no client app, delivering 16 kHz mono f32 frames on a channel,
//! the format `/v1/audio/live` takes from sockets.
//!
//! One supervisor thread owns at most one cpal stream. It opens the wanted
//! device: the configured one by exact name, else the built-in mic (not the
//! system default, which may be AirPods that drop to headset quality while
//! open), else the system default. It reopens when the stream errors, stalls,
//! changes sample rate, or a better device than the open one appears, with
//! backoff after failures, and logs each reconnect.
//!
//! Opening blocks while the macOS mic prompt is up (spike 2370), which is why
//! it happens on this thread and never on server startup. A plain binary
//! cannot ask whether it was denied; a denied mic delivers exact zeros, so
//! ~3 s of them reads as `probably_denied`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::mpsc;

use crate::log::Logger;

/// Frames go out at this rate, mono.
pub const RATE: u32 = 16_000;
/// Exact zeros for this long mean the mic is probably denied.
pub const DENIED_AFTER: Duration = Duration::from_secs(3);

/// What `/health`'s `mic` reports.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MicStatus {
    /// The open device, or the one being opened.
    pub device: Option<String>,
    /// A stream is open and delivering audio.
    pub running: bool,
    /// RMS of the latest callback's audio, dBFS (floored at -120).
    pub level_dbfs: Option<f32>,
    pub probably_denied: bool,
    /// Streams reopened since start.
    pub reconnects: u64,
    pub last_error: Option<String>,
}

/// The device to capture from: `NARU_AUDIO_MIC_DEVICE`, else `config.toml`'s
/// `[mic] device`; `None` means the built-in mic.
pub fn configured_device(home: &Path) -> Result<Option<String>, String> {
    if let Some(name) = std::env::var("NARU_AUDIO_MIC_DEVICE")
        .ok()
        .filter(|n| !n.is_empty())
    {
        return Ok(Some(name));
    }
    let config = crate::registry::read_config(home)?;
    match config.get("mic").and_then(|m| m.get("device")) {
        None => Ok(None),
        Some(toml::Value::String(name)) if !name.is_empty() => Ok(Some(name.clone())),
        Some(_) => Err(format!(
            "{}: mic.device must be a device name",
            crate::registry::config_path(home).display()
        )),
    }
}

/// A running capture; dropping it stops the stream and joins its thread.
pub struct Mic {
    status: Arc<Mutex<MicStatus>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Mic {
    pub fn status(&self) -> MicStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Starts capturing from `want` (by exact name; `None` for the built-in
    /// mic) on its own thread and returns at once. Frames that `frames` has
    /// no room for are dropped.
    #[cfg(target_os = "macos")]
    pub fn start(
        want: Option<String>,
        log: Arc<Logger>,
        frames: mpsc::Sender<Vec<f32>>,
    ) -> std::io::Result<Self> {
        let status = Arc::new(Mutex::new(MicStatus::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new()
            .name("naru-audio-mic".into())
            .spawn({
                let status = status.clone();
                let stop = stop.clone();
                move || capture::supervise(want, log, frames, status, stop)
            })?;
        Ok(Self {
            status,
            stop,
            thread: Some(thread),
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn start(
        _want: Option<String>,
        _log: Arc<Logger>,
        _frames: mpsc::Sender<Vec<f32>>,
    ) -> std::io::Result<Self> {
        Err(std::io::Error::other("mic capture is macOS-only"))
    }
}

impl Drop for Mic {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The input devices' names.
#[cfg(target_os = "macos")]
pub fn input_names() -> Vec<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    cpal::default_host()
        .input_devices()
        .map(|ds| ds.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

#[cfg(not(target_os = "macos"))]
pub fn input_names() -> Vec<String> {
    Vec::new()
}

/// RMS of `samples` in dBFS, floored at -120.
pub fn dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -120.0;
    }
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    (20.0 * rms.max(1e-6).log10()).max(-120.0)
}

/// Device-rate mono to [`RATE`], in a stream of arbitrary-length pushes.
#[cfg(any(target_os = "macos", test))]
struct To16k {
    resampler: Option<rubato::Fft<f32>>,
    pending: Vec<f32>,
}

#[cfg(any(target_os = "macos", test))]
impl To16k {
    fn new(rate: u32) -> Result<Self, String> {
        let resampler = if rate == RATE {
            None
        } else {
            Some(
                rubato::Fft::<f32>::new(
                    rate as usize,
                    RATE as usize,
                    1024,
                    1,
                    rubato::FixedSync::Input,
                )
                .map_err(|e| e.to_string())?,
            )
        };
        Ok(Self {
            resampler,
            pending: Vec::new(),
        })
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        use rubato::Resampler;
        use rubato::audioadapter_buffers::direct::InterleavedSlice;
        let Some(r) = &mut self.resampler else {
            return input.to_vec();
        };
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();
        loop {
            let need = r.input_frames_next();
            if self.pending.len() < need {
                return out;
            }
            if let Ok(chunk) = InterleavedSlice::new(&self.pending[..need], 1, need)
                && let Ok(o) = r.process(&chunk, None)
            {
                out.extend(o.take_data());
            }
            self.pending.drain(..need);
        }
    }
}

/// When to reopen a stream, as pure functions of what the supervisor saw.
#[cfg(any(target_os = "macos", test))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod policy {
    use std::time::Duration;

    pub const POLL: Duration = Duration::from_millis(500);
    /// No callback for this long: the stream is dead.
    pub const STALL: Duration = Duration::from_secs(2);
    pub const MIN_BACKOFF: Duration = Duration::from_millis(500);
    pub const MAX_BACKOFF: Duration = Duration::from_secs(5);
    /// A stream that ran this long resets the backoff.
    pub const HEALTHY: Duration = Duration::from_secs(5);
    /// A device that failed to open is passed over for this long, so a
    /// broken preferred device does not leave capture deaf.
    pub const SKIP_FAILED: Duration = Duration::from_secs(30);

    /// The device to open from the inputs present: `want` by exact name,
    /// else the built-in mic, else the system default, else the first.
    pub fn pick<'a>(
        inputs: &'a [String],
        want: Option<&str>,
        built_in: Option<&str>,
        default: Option<&str>,
    ) -> Option<&'a str> {
        let find = |name: Option<&str>| {
            name.and_then(|n| inputs.iter().find(|i| i.as_str() == n))
                .map(String::as_str)
        };
        find(want)
            .or_else(|| find(built_in))
            .or_else(|| find(default))
            .or_else(|| inputs.first().map(String::as_str))
    }

    pub fn next_backoff(current: Duration) -> Duration {
        (current * 2).min(MAX_BACKOFF)
    }
}

#[cfg(target_os = "macos")]
mod capture {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc as std_mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use tokio::sync::mpsc;

    use super::policy::{self, HEALTHY, MAX_BACKOFF, MIN_BACKOFF, POLL, SKIP_FAILED, STALL};
    use super::{DENIED_AFTER, MicStatus, To16k, dbfs};
    use crate::log::Logger;

    /// One open stream; dropping it stops the callback.
    struct Session {
        _stream: cpal::Stream,
        device: cpal::Device,
        name: String,
        rate: u32,
        channels: u16,
        audio: std_mpsc::Receiver<Vec<f32>>,
        error: Arc<Mutex<Option<String>>>,
        to16k: To16k,
        opened: Instant,
        last_audio: Instant,
        zeros_since: Option<Instant>,
    }

    pub(super) fn supervise(
        want: Option<String>,
        log: Arc<Logger>,
        frames: mpsc::Sender<Vec<f32>>,
        status: Arc<Mutex<MicStatus>>,
        stop: Arc<AtomicBool>,
    ) {
        let set = |f: &dyn Fn(&mut MicStatus)| update(&status, f);
        let host = cpal::default_host();
        let mut session: Option<Session> = None;
        let mut backoff = MIN_BACKOFF;
        let mut retry_at = Instant::now();
        // Since when, and why, capture has been down (for the recovery time).
        let mut down: Option<(Instant, String)> = Some((Instant::now(), "start".into()));
        let mut next_poll = Instant::now();
        let mut opened_once = false;
        // Devices that failed to open, and when.
        let mut failed: HashMap<String, Instant> = HashMap::new();

        while !stop.load(Ordering::Relaxed) {
            if let Some(s) = &mut session {
                pump(s, &frames, &log, &status);
            } else {
                std::thread::sleep(Duration::from_millis(50));
            }
            let now = Instant::now();
            if now < next_poll {
                continue;
            }
            next_poll = now + POLL;

            let inputs = inputs(&host);
            let names: Vec<String> = inputs.iter().map(|(n, _)| n.clone()).collect();
            let built_in = super::built_in_input_name();
            let default = host.default_input_device().and_then(|d| d.name().ok());
            failed.retain(|_, at| now.duration_since(*at) < SKIP_FAILED);
            let usable: Vec<String> = names
                .iter()
                .filter(|n| !failed.contains_key(*n))
                .cloned()
                .collect();
            let target = policy::pick(
                &usable,
                want.as_deref(),
                built_in.as_deref(),
                default.as_deref(),
            )
            .map(str::to_string);

            if let Some(s) = &session {
                let reason =
                    if let Some(e) = s.error.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        Some(format!("error: {e}"))
                    } else if now.duration_since(s.last_audio) >= STALL {
                        Some("stalled".to_string())
                    } else if s
                        .device
                        .default_input_config()
                        .is_ok_and(|c| c.sample_rate().0 != s.rate || c.channels() != s.channels)
                    {
                        Some("format_changed".to_string())
                    } else if target.as_deref() != Some(s.name.as_str()) {
                        Some(match &target {
                            Some(t) if names.contains(&s.name) => format!("switch_to={t:?}"),
                            _ => "device_gone".to_string(),
                        })
                    } else {
                        None
                    };
                if let Some(reason) = reason {
                    let s = session.take().expect("checked above");
                    let ran = now.duration_since(s.opened);
                    let lost = s.name.clone();
                    drop(s);
                    log.line(
                        "warn",
                        None,
                        &format!(
                            "mic_lost device={lost:?} reason={reason} ran_ms={}",
                            ran.as_millis()
                        ),
                    );
                    set(&|st| {
                        st.running = false;
                        st.level_dbfs = None;
                        if reason.starts_with("error") || reason == "stalled" {
                            st.last_error = Some(reason.clone());
                        }
                    });
                    // A device switch is not a failure: reopen at once.
                    if reason.starts_with("switch_to") || ran >= HEALTHY {
                        backoff = MIN_BACKOFF;
                        retry_at = now;
                    } else {
                        retry_at = now + backoff;
                        backoff = policy::next_backoff(backoff);
                    }
                    down = Some((now, reason));
                }
            }

            if session.is_none() && now >= retry_at {
                let Some(name) = target else {
                    set(&|st| {
                        st.device = None;
                        st.last_error = Some("no input device".into());
                    });
                    retry_at = now + MAX_BACKOFF;
                    continue;
                };
                let device = inputs
                    .into_iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, d)| d)
                    .expect("picked from inputs");
                set(&|st| st.device = Some(name.clone()));
                match open(device, name.clone()) {
                    Ok(s) => {
                        let (since, why) = down.take().unwrap_or((now, "start".into()));
                        log.info(
                            None,
                            &format!(
                                "mic_open device={name:?} rate={} reason={why} recovered_ms={}",
                                s.rate,
                                since.elapsed().as_millis()
                            ),
                        );
                        set(&|st| {
                            st.running = true;
                            st.probably_denied = false;
                            if opened_once {
                                st.reconnects += 1;
                            }
                        });
                        opened_once = true;
                        failed.remove(&name);
                        session = Some(s);
                    }
                    Err(e) => {
                        log.line(
                            "warn",
                            None,
                            &format!(
                                "mic_open_failed device={name:?} error={e:?} retry_ms={}",
                                backoff.as_millis()
                            ),
                        );
                        set(&|st| st.last_error = Some(e.clone()));
                        failed.insert(name, now);
                        retry_at = now + backoff;
                        backoff = policy::next_backoff(backoff);
                    }
                }
            }
        }
        drop(session);
        set(&|st| st.running = false);
        log.info(None, "mic_stop");
    }

    /// Moves the callback's audio out: resample, meter, forward.
    fn pump(
        s: &mut Session,
        frames: &mpsc::Sender<Vec<f32>>,
        log: &Logger,
        status: &Mutex<MicStatus>,
    ) {
        let Ok(first) = s.audio.recv_timeout(Duration::from_millis(50)) else {
            return;
        };
        let mut raw = first;
        while let Ok(more) = s.audio.try_recv() {
            raw.extend(more);
        }
        let now = Instant::now();
        s.last_audio = now;
        let denied = if raw.iter().all(|x| *x == 0.0) {
            let since = *s.zeros_since.get_or_insert(now);
            now.duration_since(since) >= DENIED_AFTER
        } else {
            s.zeros_since = None;
            false
        };
        let level = dbfs(&raw);
        let newly_denied = {
            let mut st = status.lock().unwrap_or_else(|e| e.into_inner());
            st.level_dbfs = Some((level * 10.0).round() / 10.0);
            let newly = denied && !st.probably_denied;
            st.probably_denied = denied;
            newly
        };
        if newly_denied {
            log.line(
                "warn",
                None,
                &format!(
                    "mic_probably_denied device={:?} zeros_s={}",
                    s.name,
                    DENIED_AFTER.as_secs()
                ),
            );
        }
        let out = s.to16k.push(&raw);
        if !out.is_empty() {
            let _ = frames.try_send(out);
        }
    }

    fn update(status: &Mutex<MicStatus>, f: &dyn Fn(&mut MicStatus)) {
        f(&mut status.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn inputs(host: &cpal::Host) -> Vec<(String, cpal::Device)> {
        // Enumerated devices, not `default_input_device()`: cpal only stops
        // and reports a stream whose device disappears for these.
        host.input_devices()
            .map(|ds| ds.filter_map(|d| Some((d.name().ok()?, d))).collect())
            .unwrap_or_default()
    }

    fn open(device: cpal::Device, name: String) -> Result<Session, String> {
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let channels = config.channels() as usize;
        let channel_count = config.channels();
        let rate = config.sample_rate().0;
        let (tx, audio) = std_mpsc::sync_channel::<Vec<f32>>(64);
        let error = Arc::new(Mutex::new(None));
        let on_error = {
            let error = error.clone();
            move |e: cpal::StreamError| {
                *error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e.to_string());
            }
        };
        let stream_config: cpal::StreamConfig = config.clone().into();
        macro_rules! build {
            ($t:ty) => {
                device.build_input_stream(
                    &stream_config,
                    move |data: &[$t], _: &_| {
                        let mono = data
                            .chunks(channels)
                            .map(|f| {
                                f.iter()
                                    .map(|&x| <f32 as cpal::FromSample<$t>>::from_sample_(x))
                                    .sum::<f32>()
                                    / channels as f32
                            })
                            .collect();
                        let _ = tx.try_send(mono);
                    },
                    on_error,
                    None,
                )
            };
        }
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => build!(f32),
            cpal::SampleFormat::I16 => build!(i16),
            cpal::SampleFormat::I32 => build!(i32),
            f => return Err(format!("unsupported sample format {f:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        let now = Instant::now();
        Ok(Session {
            _stream: stream,
            device,
            name,
            rate,
            channels: channel_count,
            audio,
            error,
            to16k: To16k::new(rate)?,
            opened: now,
            last_audio: now,
            zeros_since: None,
        })
    }
}

/// The built-in input device's name (transport type built-in, with input
/// streams), from CoreAudio: cpal does not expose the transport type.
#[cfg(target_os = "macos")]
fn built_in_input_name() -> Option<String> {
    use core_foundation_sys::base::{CFRelease, kCFAllocatorDefault};
    use core_foundation_sys::data::{CFDataGetBytePtr, CFDataGetLength};
    use core_foundation_sys::string::{
        CFStringCreateExternalRepresentation, CFStringRef, kCFStringEncodingUTF8,
    };
    use coreaudio_sys::*;
    use std::mem::size_of;
    use std::ptr::null;

    let address = |selector, scope| AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMaster,
    };
    // SAFETY: each call passes a property address and a buffer of the size
    // CoreAudio reported or the property's documented fixed size.
    unsafe {
        let devices_addr = address(
            kAudioHardwarePropertyDevices,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut size = 0u32;
        if AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &devices_addr,
            0,
            null(),
            &mut size,
        ) != 0
        {
            return None;
        }
        let mut ids = vec![0 as AudioObjectID; size as usize / size_of::<AudioObjectID>()];
        if AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &devices_addr,
            0,
            null(),
            &mut size,
            ids.as_mut_ptr().cast(),
        ) != 0
        {
            return None;
        }
        for id in ids {
            let mut transport = 0u32;
            let mut tsize = size_of::<u32>() as u32;
            let taddr = address(
                kAudioDevicePropertyTransportType,
                kAudioObjectPropertyScopeGlobal,
            );
            if AudioObjectGetPropertyData(
                id,
                &taddr,
                0,
                null(),
                &mut tsize,
                (&mut transport as *mut u32).cast(),
            ) != 0
                || transport != kAudioDeviceTransportTypeBuiltIn
            {
                continue;
            }
            let saddr = address(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeInput);
            let mut ssize = 0u32;
            if AudioObjectGetPropertyDataSize(id, &saddr, 0, null(), &mut ssize) != 0 || ssize == 0
            {
                continue;
            }
            let naddr = address(kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal);
            let mut name: CFStringRef = std::ptr::null();
            let mut nsize = size_of::<CFStringRef>() as u32;
            if AudioObjectGetPropertyData(
                id,
                &naddr,
                0,
                null(),
                &mut nsize,
                (&mut name as *mut CFStringRef).cast(),
            ) != 0
                || name.is_null()
            {
                continue;
            }
            let data = CFStringCreateExternalRepresentation(
                kCFAllocatorDefault,
                name,
                kCFStringEncodingUTF8,
                0,
            );
            CFRelease(name.cast());
            if data.is_null() {
                continue;
            }
            let bytes =
                std::slice::from_raw_parts(CFDataGetBytePtr(data), CFDataGetLength(data) as usize);
            let s = String::from_utf8_lossy(bytes).into_owned();
            CFRelease(data.cast());
            return Some(s);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::policy::*;
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn pick_prefers_configured_then_built_in_then_default() {
        let inputs = names(&["AirPods Pro", "BlackHole 2ch", "MacBook Pro Microphone"]);
        let built_in = Some("MacBook Pro Microphone");
        let default = Some("AirPods Pro");
        assert_eq!(
            pick(&inputs, Some("BlackHole 2ch"), built_in, default),
            Some("BlackHole 2ch")
        );
        // Not the system default (AirPods would drop to headset quality).
        assert_eq!(
            pick(&inputs, None, built_in, default),
            Some("MacBook Pro Microphone")
        );
        // A configured device that is gone falls back to the built-in mic.
        assert_eq!(
            pick(&inputs, Some("Gone"), built_in, default),
            Some("MacBook Pro Microphone")
        );
        // No built-in mic (Mac mini): the system default.
        assert_eq!(pick(&inputs, None, None, default), Some("AirPods Pro"));
        // Names match exactly, not by substring.
        assert_eq!(
            pick(&inputs, Some("BlackHole"), None, default),
            Some("AirPods Pro")
        );
        assert_eq!(pick(&[], None, built_in, default), None);
    }

    #[test]
    fn backoff_doubles_to_a_cap() {
        let mut b = MIN_BACKOFF;
        for _ in 0..10 {
            b = next_backoff(b);
        }
        assert_eq!(b, MAX_BACKOFF);
        assert_eq!(next_backoff(MIN_BACKOFF), MIN_BACKOFF * 2);
    }

    #[test]
    fn resamples_48k_stream_to_16k() {
        let mut r = To16k::new(48_000).unwrap();
        let tone: Vec<f32> = (0..48_000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut out = Vec::new();
        // Odd push sizes, as callbacks deliver them.
        for chunk in tone.chunks(471) {
            out.extend(r.push(chunk));
        }
        // One second in, about one second out (less what is still buffered).
        assert!((15_000..=16_000).contains(&out.len()), "{}", out.len());
        // A 0.5 sine is -9 dBFS; skip the resampler's ramp-in.
        assert!(
            (dbfs(&out[4000..]) + 9.0).abs() < 0.5,
            "{}",
            dbfs(&out[4000..])
        );
    }

    /// Reads devices without opening any. CI runners have no built-in mic.
    #[cfg(target_os = "macos")]
    #[test]
    fn built_in_mic_is_an_input_device() {
        if let Some(name) = built_in_input_name() {
            eprintln!("built-in mic: {name:?}");
            assert!(
                input_names().contains(&name),
                "{name:?} not in {:?}",
                input_names()
            );
        }
    }

    #[test]
    fn dbfs_of_silence_is_floored() {
        assert_eq!(dbfs(&[0.0; 160]), -120.0);
        assert_eq!(dbfs(&[]), -120.0);
        assert!((dbfs(&[1.0; 160])).abs() < 1e-4);
    }
}
