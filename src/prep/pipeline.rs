//! The voice-prep step model (`docs/design.md` §8.1): an ordered list of
//! [`Step`]s, each `{"type": "...", "enabled": true, ...params}` with serde
//! defaults for every param, run over the cropped span at 16 kHz mono. The
//! model-backed steps (isolate, denoise, trim_silence's VAD) go through
//! [`ModelSteps`] so this module stays free of the HTTP and registry layers;
//! everything else is [`super::dsp`]. Param ranges live in one table
//! ([`specs`]) that both validation and the `GET /v1/audio/prep/steps`
//! catalogue read, so a UI never offers a value the server rejects.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{PrepError, dsp, normalize_peak, peak_dbfs, trim_to_spans};
use crate::stt::audio::TARGET_SAMPLE_RATE;
use crate::stt::vad::VadConfig;

/// Most steps a chain may have.
pub const MAX_STEPS: usize = 32;
/// Longest an output may be relative to its input, [`run`]'s backstop (plus
/// one second) and [`validate`]'s bound on chained `speed` steps.
pub const MAX_GROWTH: usize = 4;
const MIN_SPEED_PRODUCT: f64 = 1.0 / MAX_GROWTH as f64;
/// Most bands one `eq` step may have.
pub const MAX_EQ_BANDS: usize = 16;

fn on() -> bool {
    true
}

macro_rules! step_struct {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident : $t:ty = $d:expr),* $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        pub struct $name {
            pub enabled: bool,
            $($(#[$fm])* pub $f: $t,)*
        }
        impl Default for $name {
            fn default() -> Self {
                Self { enabled: on(), $($f: $d,)* }
            }
        }
    };
}

step_struct!(
    /// Vocal isolation (Spleeter). `bleed` mixes that fraction of the
    /// un-isolated input back in.
    Isolate { model: Option<String> = None, bleed: f32 = 0.0 }
);
step_struct!(
    /// Neural denoise (GTCRN). `amount` is the wet/dry mix.
    Denoise { model: Option<String> = None, amount: f32 = 1.0 }
);
step_struct!(
    /// Trims to the first/last Silero VAD speech span, keeping `pad_ms`
    /// either side (clamped to the buffer).
    TrimSilence { threshold: f32 = VadConfig::default().threshold, pad_ms: f32 = 0.0 }
);
step_struct!(
    /// Peak-normalises to `peak_db` dBFS.
    Normalize { peak_db: f32 = -1.0 }
);
step_struct!(
    /// 4th-order Butterworth high-pass.
    Highpass { freq_hz: f32 = 80.0 }
);
step_struct!(
    /// Parametric EQ: biquad bands applied in order.
    Eq { bands: Vec<EqBand> = Vec::new() }
);
step_struct!(
    /// Split-band de-esser.
    Deess { freq_hz: f32 = 6000.0, threshold_db: f32 = -30.0, ratio: f32 = 4.0 }
);
step_struct!(
    /// BS.1770 integrated-loudness gain, then a limiter at the peak ceiling.
    Loudness { target_lufs: f32 = -18.0, peak_ceiling_db: f32 = -1.0 }
);
step_struct!(
    /// Feed-forward compressor.
    Compressor {
        threshold_db: f32 = -20.0,
        ratio: f32 = 3.0,
        attack_ms: f32 = 10.0,
        release_ms: f32 = 100.0,
        makeup_db: f32 = 0.0,
    }
);
step_struct!(
    /// Pitch shift at constant duration (formants move with it).
    Pitch { semitones: f32 = 0.0 }
);
step_struct!(
    /// Speed change, pitch preserved (WSOLA).
    Speed { rate: f32 = 1.0 }
);
step_struct!(
    /// Narrowband telephone: saturation then 300-3400 Hz band-pass.
    Telephone {}
);
step_struct!(
    /// Schroeder reverb.
    Reverb { room_size: f32 = 0.5, wet: f32 = 0.2 }
);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandKind {
    Peak,
    LowShelf,
    HighShelf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EqBand {
    pub kind: BandKind,
    pub freq_hz: f32,
    pub gain_db: f32,
    pub q: f32,
}

impl Default for EqBand {
    fn default() -> Self {
        EqBand {
            kind: BandKind::Peak,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 0.707,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Step {
    Isolate(Isolate),
    Denoise(Denoise),
    TrimSilence(TrimSilence),
    Normalize(Normalize),
    Highpass(Highpass),
    Eq(Eq),
    Deess(Deess),
    Loudness(Loudness),
    Compressor(Compressor),
    Pitch(Pitch),
    Speed(Speed),
    Telephone(Telephone),
    Reverb(Reverb),
}

impl Step {
    pub fn enabled(&self) -> bool {
        match self {
            Step::Isolate(s) => s.enabled,
            Step::Denoise(s) => s.enabled,
            Step::TrimSilence(s) => s.enabled,
            Step::Normalize(s) => s.enabled,
            Step::Highpass(s) => s.enabled,
            Step::Eq(s) => s.enabled,
            Step::Deess(s) => s.enabled,
            Step::Loudness(s) => s.enabled,
            Step::Compressor(s) => s.enabled,
            Step::Pitch(s) => s.enabled,
            Step::Speed(s) => s.enabled,
            Step::Telephone(s) => s.enabled,
            Step::Reverb(s) => s.enabled,
        }
    }
}

/// The legacy four-flag chain of `POST /v1/audio/samples`: isolate →
/// denoise → trim_silence → normalize(-1), each carrying its flag as
/// `enabled`.
pub fn default_chain(
    isolate: bool,
    denoise: bool,
    trim_silence: bool,
    normalize: bool,
    isolation_model: Option<String>,
    denoise_model: Option<String>,
) -> Vec<Step> {
    vec![
        Step::Isolate(Isolate {
            enabled: isolate,
            model: isolation_model,
            ..Default::default()
        }),
        Step::Denoise(Denoise {
            enabled: denoise,
            model: denoise_model,
            ..Default::default()
        }),
        Step::TrimSilence(TrimSilence {
            enabled: trim_silence,
            ..Default::default()
        }),
        Step::Normalize(Normalize {
            enabled: normalize,
            ..Default::default()
        }),
    ]
}

// ---- param table ----------------------------------------------------------

/// One numeric param: its name, inclusive range and unit.
pub struct Spec {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub unit: &'static str,
}

const fn sp(name: &'static str, min: f64, max: f64, unit: &'static str) -> Spec {
    Spec {
        name,
        min,
        max,
        unit,
    }
}

const NYQUIST_GUARD: f64 = 7900.0;

/// The numeric params of step type `ty`.
pub fn specs(ty: &str) -> Vec<Spec> {
    match ty {
        "isolate" => vec![sp("bleed", 0.0, 1.0, "ratio")],
        "denoise" => vec![sp("amount", 0.0, 1.0, "ratio")],
        "trim_silence" => vec![
            sp("threshold", 0.01, 0.99, "probability"),
            sp("pad_ms", 0.0, 2000.0, "ms"),
        ],
        "normalize" => vec![sp("peak_db", -60.0, 0.0, "dBFS")],
        "highpass" => vec![sp("freq_hz", 20.0, 1000.0, "Hz")],
        "deess" => vec![
            sp("freq_hz", 2000.0, NYQUIST_GUARD, "Hz"),
            sp("threshold_db", -80.0, 0.0, "dBFS"),
            sp("ratio", 1.0, 20.0, "ratio"),
        ],
        "loudness" => vec![
            sp("target_lufs", -60.0, 0.0, "LUFS"),
            sp("peak_ceiling_db", -30.0, 0.0, "dBFS"),
        ],
        "compressor" => vec![
            sp("threshold_db", -80.0, 0.0, "dBFS"),
            sp("ratio", 1.0, 20.0, "ratio"),
            sp("attack_ms", 0.1, 500.0, "ms"),
            sp("release_ms", 1.0, 5000.0, "ms"),
            sp("makeup_db", -24.0, 24.0, "dB"),
        ],
        "pitch" => vec![sp("semitones", -12.0, 12.0, "semitones")],
        "speed" => vec![sp("rate", 0.5, 2.0, "ratio")],
        "reverb" => vec![
            sp("room_size", 0.0, 1.0, "ratio"),
            sp("wet", 0.0, 1.0, "ratio"),
        ],
        _ => Vec::new(),
    }
}

const BAND_SPECS: &[Spec] = &[
    sp("freq_hz", 20.0, NYQUIST_GUARD, "Hz"),
    sp("gain_db", -24.0, 24.0, "dB"),
    sp("q", 0.1, 10.0, "Q"),
];

/// `(type, description)` for every step type, in catalogue order.
const TYPES: &[(&str, &str)] = &[
    (
        "isolate",
        "Vocal isolation (Spleeter); bleed mixes the original back in",
    ),
    (
        "denoise",
        "Neural denoise (GTCRN); amount is the wet/dry mix",
    ),
    (
        "trim_silence",
        "Trim leading/trailing silence by Silero VAD, keeping pad_ms",
    ),
    ("normalize", "Peak-normalise to peak_db"),
    ("highpass", "4th-order Butterworth high-pass (rumble)"),
    ("eq", "Parametric EQ: peak / low_shelf / high_shelf bands"),
    (
        "deess",
        "Split-band de-esser: attenuates sibilance above freq_hz",
    ),
    (
        "loudness",
        "BS.1770 integrated-loudness gain, then limit to the peak ceiling",
    ),
    ("compressor", "Feed-forward compressor"),
    (
        "pitch",
        "Pitch shift at constant duration (formants move with pitch)",
    ),
    ("speed", "Speed change with pitch preserved (WSOLA)"),
    (
        "telephone",
        "Narrowband telephone: saturation plus 300-3400 Hz band-pass",
    ),
    ("reverb", "Schroeder reverb; output keeps the input length"),
];

fn default_step(ty: &str) -> Step {
    match ty {
        "isolate" => Step::Isolate(Default::default()),
        "denoise" => Step::Denoise(Default::default()),
        "trim_silence" => Step::TrimSilence(Default::default()),
        "normalize" => Step::Normalize(Default::default()),
        "highpass" => Step::Highpass(Default::default()),
        "eq" => Step::Eq(Default::default()),
        "deess" => Step::Deess(Default::default()),
        "loudness" => Step::Loudness(Default::default()),
        "compressor" => Step::Compressor(Default::default()),
        "pitch" => Step::Pitch(Default::default()),
        "speed" => Step::Speed(Default::default()),
        "telephone" => Step::Telephone(Default::default()),
        _ => Step::Reverb(Default::default()),
    }
}

fn param_json(spec: &Spec, default: &Value) -> Value {
    json!({
        "name": spec.name, "type": "number", "default": default,
        "min": spec.min, "max": spec.max, "unit": spec.unit,
    })
}

/// The `GET /v1/audio/prep/steps` body: every step type with its params'
/// default/min/max/unit, plus the default chain.
pub fn catalogue() -> Value {
    let steps: Vec<Value> = TYPES
        .iter()
        .map(|&(ty, description)| {
            let defaults = serde_json::to_value(default_step(ty)).unwrap_or_default();
            let mut params = vec![json!({
                "name": "enabled", "type": "boolean", "default": true,
            })];
            if matches!(ty, "isolate" | "denoise") {
                params.push(json!({"name": "model", "type": "string", "default": null}));
            }
            params.extend(specs(ty).iter().map(|s| param_json(s, &defaults[s.name])));
            if ty == "eq" {
                let band = serde_json::to_value(EqBand::default()).unwrap_or_default();
                let mut item = vec![json!({
                    "name": "kind", "type": "string", "default": "peak",
                    "values": ["peak", "low_shelf", "high_shelf"],
                })];
                item.extend(BAND_SPECS.iter().map(|s| param_json(s, &band[s.name])));
                params.push(json!({
                    "name": "bands", "type": "array", "default": [],
                    "max_items": MAX_EQ_BANDS, "items": item,
                }));
            }
            json!({"type": ty, "description": description, "params": params})
        })
        .collect();
    json!({
        "steps": steps,
        "default_chain": default_chain(true, true, true, true, None, None),
    })
}

fn check(path: &str, spec: &Spec, v: &Value) -> Result<(), String> {
    let x = v.get(spec.name).and_then(Value::as_f64);
    // Params are `f32`, widened to `f64` here, so the advertised bounds are
    // compared as the `f32` they round to (0.01 widened is under 0.01).
    let (min, max) = (spec.min as f32 as f64, spec.max as f32 as f64);
    match x {
        Some(x) if x.is_finite() && x >= min && x <= max => Ok(()),
        _ => Err(format!(
            "{path}.{} must be a number in {}..={} {}",
            spec.name, spec.min, spec.max, spec.unit
        )),
    }
}

/// Validates a chain: at most [`MAX_STEPS`] steps, every numeric param in its
/// [`specs`] range. The error names the offending param, `steps[2].amount`.
pub fn validate(steps: &[Step]) -> Result<(), String> {
    if steps.len() > MAX_STEPS {
        return Err(format!(
            "steps has {} entries, at most {MAX_STEPS}",
            steps.len()
        ));
    }
    let slowdown: f64 = steps
        .iter()
        .filter_map(|s| match s {
            Step::Speed(s) if s.enabled && s.rate > 0.0 => Some(s.rate as f64),
            _ => None,
        })
        .product();
    if slowdown < MIN_SPEED_PRODUCT {
        return Err(format!(
            "the speed steps multiply to {slowdown}, under {MIN_SPEED_PRODUCT} (output over {MAX_GROWTH}x the input)"
        ));
    }
    for (i, step) in steps.iter().enumerate() {
        let v = serde_json::to_value(step).map_err(|e| e.to_string())?;
        let ty = v["type"].as_str().unwrap_or_default();
        let path = format!("steps[{i}]");
        for spec in &specs(ty) {
            check(&path, spec, &v)?;
        }
        if let Step::Eq(eq) = step {
            if eq.bands.len() > MAX_EQ_BANDS {
                return Err(format!(
                    "{path}.bands has {} entries, at most {MAX_EQ_BANDS}",
                    eq.bands.len()
                ));
            }
            for (j, band) in eq.bands.iter().enumerate() {
                let bv = serde_json::to_value(band).map_err(|e| e.to_string())?;
                for spec in BAND_SPECS {
                    check(&format!("{path}.bands[{j}]"), spec, &bv)?;
                }
            }
        }
    }
    Ok(())
}

// ---- running --------------------------------------------------------------

/// What the model-backed steps need from their caller (which resolves and
/// loads the engines, and records which were used).
pub trait ModelSteps {
    /// Vocals of `pcm`, at 16 kHz.
    fn isolate(&mut self, model: Option<&str>, pcm: &[f32]) -> Result<Vec<f32>, PrepError>;
    fn denoise(&mut self, model: Option<&str>, pcm: &[f32]) -> Result<Vec<f32>, PrepError>;
    /// Speech spans of `pcm` as `(start, end)` sample indices.
    fn speech_spans(
        &mut self,
        pcm: &[f32],
        cfg: &VadConfig,
    ) -> Result<Vec<(usize, usize)>, PrepError>;
}

/// Which part of a chain to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Select {
    /// Every enabled step.
    All,
    /// Enabled steps `0..=n`.
    Until(usize),
    /// Only step `n`, applied even if it is disabled.
    Solo(usize),
}

impl Select {
    /// The indices of `steps` this selection runs, in order.
    pub fn indices(self, steps: &[Step]) -> Vec<usize> {
        match self {
            Select::All => (0..steps.len()).filter(|&i| steps[i].enabled()).collect(),
            Select::Until(n) => (0..steps.len().min(n.saturating_add(1)))
                .filter(|&i| steps[i].enabled())
                .collect(),
            Select::Solo(n) => vec![n],
        }
    }
}

fn mix(dry: &[f32], wet: &[f32], amount: f32) -> Vec<f32> {
    wet.iter()
        .enumerate()
        .map(|(i, &w)| dry.get(i).copied().unwrap_or(0.0) * (1.0 - amount) + w * amount)
        .collect()
}

/// Applies one step (enabled or not: the caller decides) to `pcm`.
pub fn run_step(
    step: &Step,
    pcm: Vec<f32>,
    models: &mut dyn ModelSteps,
) -> Result<Vec<f32>, PrepError> {
    let mut x = pcm;
    match step {
        Step::Isolate(s) => {
            let wet = models.isolate(s.model.as_deref(), &x)?;
            x = if s.bleed > 0.0 {
                mix(&x, &wet, 1.0 - s.bleed)
            } else {
                wet
            };
        }
        Step::Denoise(s) => {
            let wet = models.denoise(s.model.as_deref(), &x)?;
            x = if s.amount < 1.0 {
                mix(&x, &wet, s.amount)
            } else {
                wet
            };
        }
        Step::TrimSilence(s) => {
            let cfg = VadConfig {
                threshold: s.threshold,
                ..VadConfig::default()
            };
            let spans = models.speech_spans(&x, &cfg)?;
            let pad = (s.pad_ms as f64 * 1e-3 * TARGET_SAMPLE_RATE as f64).round() as usize;
            x = trim_to_spans(&x, &spans, pad);
        }
        Step::Normalize(s) => normalize_peak(&mut x, s.peak_db),
        Step::Highpass(s) => dsp::highpass4(&mut x, s.freq_hz as f64),
        Step::Eq(s) => {
            for b in &s.bands {
                let (f, g, q) = (b.freq_hz as f64, b.gain_db as f64, b.q as f64);
                match b.kind {
                    BandKind::Peak => dsp::Biquad::peak(f, g, q),
                    BandKind::LowShelf => dsp::Biquad::low_shelf(f, g, q),
                    BandKind::HighShelf => dsp::Biquad::high_shelf(f, g, q),
                }
                .process(&mut x);
            }
        }
        Step::Deess(s) => dsp::deess(
            &mut x,
            s.freq_hz as f64,
            s.threshold_db as f64,
            s.ratio as f64,
        ),
        Step::Loudness(s) => dsp::loudness(&mut x, s.target_lufs as f64, s.peak_ceiling_db as f64),
        Step::Compressor(s) => dsp::compress(
            &mut x,
            s.threshold_db as f64,
            s.ratio as f64,
            s.attack_ms as f64,
            s.release_ms as f64,
            s.makeup_db as f64,
        ),
        Step::Pitch(s) => x = dsp::pitch_shift(&x, s.semitones as f64),
        Step::Speed(s) => x = dsp::speed(&x, s.rate as f64),
        Step::Telephone(_) => dsp::telephone(&mut x),
        Step::Reverb(s) => dsp::reverb(&mut x, s.room_size as f64, s.wet as f64),
    }
    Ok(x)
}

/// Runs the `select`ed part of `steps` over `pcm`, in order.
pub fn run(
    steps: &[Step],
    select: Select,
    mut pcm: Vec<f32>,
    models: &mut dyn ModelSteps,
) -> Result<Vec<f32>, PrepError> {
    let cap = pcm.len() * MAX_GROWTH + TARGET_SAMPLE_RATE as usize;
    for i in select.indices(steps) {
        pcm = run_step(&steps[i], pcm, models)?;
        if pcm.len() > cap {
            return Err(PrepError::BadAudio(format!(
                "step {i} grew the audio past {MAX_GROWTH}x the input plus 1 s"
            )));
        }
    }
    Ok(pcm)
}

// ---- analysis -------------------------------------------------------------

/// Measurements of a prepared clip. Every level is `None` (JSON `null`) for
/// digital silence rather than `-inf`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    /// ITU-R BS.1770-4 integrated loudness (see [`dsp::integrated_lufs`]).
    pub integrated_lufs: Option<f64>,
    /// Sample peak.
    pub peak_dbfs: Option<f64>,
    /// 10th-percentile 50 ms frame RMS (see [`dsp::noise_floor_dbfs`]).
    pub noise_floor_dbfs: Option<f64>,
    /// Total length of Silero VAD speech spans; `None` when the VAD model
    /// is not pulled.
    pub speech_secs: Option<f64>,
    pub duration_secs: f64,
}

pub fn analyze(pcm: &[f32], speech_secs: Option<f64>) -> Analysis {
    let finite = |v: f64| v.is_finite().then_some(v);
    Analysis {
        integrated_lufs: dsp::integrated_lufs(pcm).and_then(finite),
        peak_dbfs: finite(peak_dbfs(pcm) as f64),
        noise_floor_dbfs: dsp::noise_floor_dbfs(pcm).and_then(finite),
        speech_secs,
        duration_secs: pcm.len() as f64 / TARGET_SAMPLE_RATE as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoModels;
    impl ModelSteps for NoModels {
        fn isolate(&mut self, _: Option<&str>, _: &[f32]) -> Result<Vec<f32>, PrepError> {
            Err(PrepError::Io("no model".into()))
        }
        fn denoise(&mut self, _: Option<&str>, pcm: &[f32]) -> Result<Vec<f32>, PrepError> {
            Ok(pcm.iter().map(|s| s * 0.5).collect())
        }
        fn speech_spans(
            &mut self,
            _: &[f32],
            _: &VadConfig,
        ) -> Result<Vec<(usize, usize)>, PrepError> {
            Ok(vec![(100, 200)])
        }
    }

    fn parse(v: Value) -> Vec<Step> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn defaults_deserialize() {
        let steps = parse(json!([
            {"type": "highpass"}, {"type": "loudness"}, {"type": "trim_silence"},
            {"type": "isolate"}, {"type": "telephone"}, {"type": "eq"},
        ]));
        assert_eq!(
            steps[0],
            Step::Highpass(Highpass {
                enabled: true,
                freq_hz: 80.0
            })
        );
        let Step::Loudness(l) = &steps[1] else {
            panic!()
        };
        assert_eq!((l.target_lufs, l.peak_ceiling_db), (-18.0, -1.0));
        let Step::TrimSilence(t) = &steps[2] else {
            panic!()
        };
        assert_eq!(
            (t.threshold, t.pad_ms),
            (VadConfig::default().threshold, 0.0)
        );
        assert!(steps.iter().all(Step::enabled));
        assert!(validate(&steps).is_ok());
    }

    #[test]
    fn unknown_type_and_param_are_rejected() {
        assert!(serde_json::from_value::<Vec<Step>>(json!([{"type": "bogus"}])).is_err());
        assert!(
            serde_json::from_value::<Vec<Step>>(json!([{"type": "denoise", "amout": 1}])).is_err()
        );
    }

    #[test]
    fn validation_names_the_param() {
        let bad = |v: Value| validate(&parse(json!([{"type": "highpass"}, v]))).unwrap_err();
        assert!(bad(json!({"type": "denoise", "amount": 1.5})).starts_with("steps[1].amount"));
        assert!(bad(json!({"type": "highpass", "freq_hz": 5})).contains("steps[1].freq_hz"));
        assert!(bad(json!({"type": "speed", "rate": 0})).contains("steps[1].rate"));
        assert!(bad(json!({"type": "pitch", "semitones": 40})).contains("steps[1].semitones"));
        assert!(
            bad(json!({"type": "eq", "bands": [{"freq_hz": 9000}]}))
                .contains("steps[1].bands[0].freq_hz")
        );
        let many = vec![Step::Telephone(Default::default()); MAX_STEPS + 1];
        assert!(validate(&many).is_err());
    }

    #[test]
    fn every_advertised_bound_is_accepted() {
        for (ty, _) in TYPES {
            for spec in specs(ty) {
                for bound in [spec.min, spec.max] {
                    let steps = parse(json!([{"type": ty, spec.name: bound}]));
                    assert_eq!(validate(&steps), Ok(()), "{ty}.{} = {bound}", spec.name);
                }
            }
        }
        let band = |f: f64| json!([{"type": "eq", "bands": [{"freq_hz": f, "q": 0.1}]}]);
        for f in [20.0, NYQUIST_GUARD] {
            assert_eq!(validate(&parse(band(f))), Ok(()));
        }
    }

    #[test]
    fn chained_slow_speeds_are_rejected_and_run_caps_growth() {
        let slow = |n| parse(json!(vec![json!({"type": "speed", "rate": 0.5}); n]));
        assert!(validate(&slow(2)).is_ok());
        assert!(validate(&slow(6)).unwrap_err().contains("speed"));
        let mut off = slow(6);
        for s in &mut off {
            let Step::Speed(s) = s else { panic!() };
            s.enabled = false;
        }
        assert!(validate(&off).is_ok(), "disabled steps do not count");
        // Bypassing validation, the run backstop still stops it.
        let err = run(&slow(3), Select::All, vec![0.1; 16_000], &mut NoModels).unwrap_err();
        assert!(err.to_string().contains("grew"), "{err}");
    }

    #[test]
    fn catalogue_defaults_match_the_step_defaults() {
        let cat = catalogue();
        for entry in cat["steps"].as_array().unwrap() {
            let ty = entry["type"].as_str().unwrap();
            let defaults = serde_json::to_value(default_step(ty)).unwrap();
            for p in entry["params"].as_array().unwrap() {
                let name = p["name"].as_str().unwrap();
                assert_eq!(p["default"], defaults[name], "{ty}.{name}");
                if p["type"] == "number" {
                    assert!(p["min"].as_f64() <= p["default"].as_f64(), "{ty}.{name}");
                    assert!(p["default"].as_f64() <= p["max"].as_f64(), "{ty}.{name}");
                }
            }
        }
        assert_eq!(cat["steps"].as_array().unwrap().len(), TYPES.len());
        assert_eq!(cat["default_chain"].as_array().unwrap().len(), 4);
        assert!(validate(&default_chain(true, true, true, true, None, None)).is_ok());
    }

    /// The default chain's model-free step is the old fixed chain's final
    /// stage exactly: `normalize_peak(-1)`.
    #[test]
    fn default_chain_normalize_matches_the_old_code_path() {
        let pcm: Vec<f32> = (0..4000).map(|i| ((i as f32) * 0.05).sin() * 0.2).collect();
        let mut old = pcm.clone();
        normalize_peak(&mut old, -1.0);
        let chain = default_chain(false, false, false, true, None, None);
        let new = run(&chain, Select::All, pcm, &mut NoModels).unwrap();
        assert_eq!(new, old);
    }

    #[test]
    fn model_steps_with_flags_off_never_touch_the_models() {
        // isolate would error in NoModels; a disabled one must be skipped.
        let chain = default_chain(false, false, false, false, None, None);
        assert_eq!(
            run(&chain, Select::All, vec![0.1; 10], &mut NoModels).unwrap(),
            vec![0.1; 10]
        );
    }

    #[test]
    fn denoise_amount_mixes_wet_and_dry() {
        let chain = parse(json!([{"type": "denoise", "amount": 0.5}]));
        let out = run(&chain, Select::All, vec![1.0; 4], &mut NoModels).unwrap();
        assert_eq!(out, vec![0.75; 4]);
    }

    #[test]
    fn select_until_solo_and_disabled() {
        let chain = parse(json!([
            {"type": "normalize", "peak_db": -6},
            {"type": "normalize", "peak_db": -20, "enabled": false},
            {"type": "normalize", "peak_db": -12},
        ]));
        assert_eq!(Select::All.indices(&chain), vec![0, 2]);
        assert_eq!(Select::Until(1).indices(&chain), vec![0]);
        assert_eq!(Select::Solo(1).indices(&chain), vec![1]);
        let peak = |sel| {
            let out = run(&chain, sel, vec![0.5, -0.25], &mut NoModels).unwrap();
            peak_dbfs(&out)
        };
        assert!((peak(Select::Until(0)) + 6.0).abs() < 0.01);
        assert!(
            (peak(Select::Solo(1)) + 20.0).abs() < 0.01,
            "solo applies a disabled step"
        );
        assert!((peak(Select::All) + 12.0).abs() < 0.01);
    }

    #[test]
    fn trim_silence_pad_is_clamped_to_the_buffer() {
        let chain = parse(json!([{"type": "trim_silence", "pad_ms": 1000}]));
        let out = run(&chain, Select::All, vec![0.1; 1000], &mut NoModels).unwrap();
        assert_eq!(out.len(), 1000);
        let chain = parse(json!([{"type": "trim_silence", "pad_ms": 0.25}]));
        let out = run(&chain, Select::All, vec![0.1; 1000], &mut NoModels).unwrap();
        assert_eq!(out.len(), 100 + 4 + 4);
    }

    #[test]
    fn analysis_serialises_silence_as_null() {
        let a = analyze(&vec![0.0; 16_000], None);
        let v = serde_json::to_value(&a).unwrap();
        assert!(v["integrated_lufs"].is_null() && v["peak_dbfs"].is_null());
        assert!(v["noise_floor_dbfs"].is_null() && v["speech_secs"].is_null());
        assert_eq!(v["duration_secs"], 1.0);
    }
}
