//! Task 9 spike: Kokoro v1.0 through sherpa-onnx against kokoro-rs.
//! Results and the rerun command are in docs/tts-spike.md.
//!
//! Subcommands, each printing one JSON line per run:
//!   probe  DIR                      which lang/lexicon/dict_dir configs load and speak
//!   sherpa DIR VOICE MODE TRIALS    MODE is `whole` (one generate, max_num_sentences=1,
//!                                   callback) or `chunked` (the §5.1 character chunker)
//!   kokoro VOICE                    one kokoro-rs process streaming to a pipe
//!   silence WAV...                  leading/trailing silence of finished files
//!   ab DIR VOICE OUTDIR N           blind A/B pair for one voice
//!
//! Time to first audio (TTFA) is measured from the request: for sherpa from the
//! generate call (and, cold, also from process start); for kokoro-rs from spawn
//! to the first PCM bytes after the WAV header.

#[cfg(all(windows, target_env = "msvc"))]
use naru_audio as _; // links the STL shim (build.rs)
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig,
};

const RATE: f32 = 24_000.0;
const CORPUS: &str = include_str!("../docs/tts-spike-corpus.txt");

/// The 5 voices and their sherpa sids (model metadata `speaker2id`).
const VOICES: [(&str, i32); 5] = [
    ("af_heart", 3),
    ("af_bella", 2),
    ("am_michael", 16),
    ("bf_emma", 21),
    ("bm_george", 26),
];

fn sid(voice: &str) -> i32 {
    VOICES
        .iter()
        .find(|(n, _)| *n == voice)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("voice {voice} is not in the spike set"))
}

fn corpus_text() -> String {
    CORPUS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn threads() -> i32 {
    std::env::var("SPIKE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() as i32))
}

#[cfg(unix)]
use libc::{RUSAGE_CHILDREN, RUSAGE_SELF};
#[cfg(not(unix))]
const RUSAGE_SELF: i32 = 0;
#[cfg(not(unix))]
const RUSAGE_CHILDREN: i32 = -1;

#[cfg(not(unix))]
fn max_rss_bytes(_who: i32) -> u64 {
    0
}

#[cfg(unix)]
fn max_rss_bytes(who: i32) -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(who, &mut ru) };
    // Bytes on macOS, kilobytes on Linux.
    let v = ru.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        v
    } else {
        v * 1024
    }
}

fn load(
    dir: &str,
    lang: Option<&str>,
    lexicon: Option<String>,
    dict: bool,
    mns: i32,
) -> Option<OfflineTts> {
    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            kokoro: OfflineTtsKokoroModelConfig {
                model: Some(format!("{dir}/model.onnx")),
                voices: Some(format!("{dir}/voices.bin")),
                tokens: Some(format!("{dir}/tokens.txt")),
                data_dir: Some(format!("{dir}/espeak-ng-data")),
                dict_dir: dict.then(|| format!("{dir}/dict")),
                lexicon,
                lang: lang.map(str::to_string),
                ..Default::default()
            },
            num_threads: threads(),
            provider: Some("cpu".into()),
            ..Default::default()
        },
        max_num_sentences: mns,
        ..Default::default()
    };
    OfflineTts::create(&config)
}

fn load_en(dir: &str, mns: i32) -> OfflineTts {
    load(dir, Some("en-us"), None, false, mns).expect("sherpa kokoro failed to load")
}

/// librosa.effects.trim bounds (frame 2048, hop 512, top_db 60, ref = max),
/// as kokoro-rs trim.rs computes them.
fn trim_range(audio: &[f32]) -> (usize, usize) {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    const AMIN: f32 = 1e-10;
    let pad = FRAME / 2;
    let padded = audio.len() + 2 * pad;
    if padded < FRAME {
        return (0, audio.len());
    }
    let at = |i: usize| {
        if i < pad || i >= pad + audio.len() {
            0.0
        } else {
            audio[i - pad]
        }
    };
    let power: Vec<f32> = (0..1 + (padded - FRAME) / HOP)
        .map(|f| {
            (f * HOP..f * HOP + FRAME)
                .map(|i| at(i) * at(i))
                .sum::<f32>()
                / FRAME as f32
        })
        .collect();
    let peak = power.iter().copied().fold(0.0f32, f32::max);
    let ref_db = 10.0 * AMIN.max(peak).log10();
    let loud = |p: &f32| 10.0 * AMIN.max(*p).log10() - ref_db > -60.0;
    let Some(first) = power.iter().position(loud) else {
        return (0, 0);
    };
    let last = power.iter().rposition(loud).unwrap();
    (first * HOP, audio.len().min((last + 1) * HOP))
}

/// Leading and trailing silence in seconds.
fn silence(audio: &[f32]) -> (f32, f32) {
    let (s, e) = trim_range(audio);
    (s as f32 / RATE, (audio.len() - e) as f32 / RATE)
}

/// The §5.1 schedule: the first chunk is at most 100 chars, later ones double
/// up to 400; a break is made at a sentence end, else a clause end, else a space.
fn chunk(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    let mut budget = 100;
    while !rest.is_empty() {
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        if chars.len() <= budget {
            out.push(rest.to_string());
            break;
        }
        // Byte offset just after the last good break within the budget.
        let window = &chars[..=budget];
        let after = |pred: &dyn Fn(char) -> bool| {
            window
                .windows(2)
                .rev()
                .find(|w| pred(w[0].1) && w[1].1 == ' ')
                .map(|w| w[1].0)
        };
        let cut = after(&|c| matches!(c, '.' | '!' | '?' | '…' | '"'))
            .or_else(|| after(&|c| matches!(c, ',' | ';' | ':' | '—')))
            .or_else(|| {
                window
                    .iter()
                    .rev()
                    .find(|(_, c)| *c == ' ')
                    .map(|(i, _)| *i)
            })
            .unwrap_or(window[budget].0);
        out.push(rest[..cut].trim().to_string());
        rest = rest[cut..].trim_start();
        budget = (budget * 2).min(400);
    }
    out
}

struct Run {
    ttfa: f64,
    wall: f64,
    audio_s: f64,
    pieces: Vec<(usize, f32, f32)>,
    samples: Vec<f32>,
}

fn run_whole(tts: &OfflineTts, text: &str, sid: i32) -> Run {
    let gen_cfg = GenerationConfig {
        sid,
        ..Default::default()
    };
    let first: Arc<Mutex<Option<f64>>> = Arc::default();
    let pieces: Arc<Mutex<Vec<(usize, f32, f32)>>> = Arc::default();
    let t0 = Instant::now();
    let (f, p) = (first.clone(), pieces.clone());
    let cb = move |s: &[f32], _progress: f32| {
        let mut f = f.lock().unwrap();
        if f.is_none() && !s.is_empty() {
            *f = Some(t0.elapsed().as_secs_f64());
        }
        let (lead, trail) = silence(s);
        p.lock().unwrap().push((s.len(), lead, trail));
        true
    };
    let audio = tts
        .generate_with_config(text, &gen_cfg, Some(cb))
        .expect("generate failed");
    let wall = t0.elapsed().as_secs_f64();
    let samples = audio.samples().to_vec();
    let ttfa = first.lock().unwrap().unwrap_or(wall);
    let pieces = pieces.lock().unwrap().clone();
    Run {
        ttfa,
        wall,
        audio_s: samples.len() as f64 / RATE as f64,
        pieces,
        samples,
    }
}

fn run_chunked(tts: &OfflineTts, text: &str, sid: i32) -> Run {
    let gen_cfg = GenerationConfig {
        sid,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut ttfa = None;
    let mut samples = Vec::new();
    let mut pieces = Vec::new();
    for c in chunk(text) {
        let audio = tts
            .generate_with_config(&c, &gen_cfg, None::<fn(&[f32], f32) -> bool>)
            .expect("generate failed");
        ttfa.get_or_insert(t0.elapsed().as_secs_f64());
        let s = audio.samples();
        let (lead, trail) = silence(s);
        pieces.push((s.len(), lead, trail));
        samples.extend_from_slice(s);
    }
    let wall = t0.elapsed().as_secs_f64();
    Run {
        ttfa: ttfa.unwrap(),
        wall,
        audio_s: samples.len() as f64 / RATE as f64,
        pieces,
        samples,
    }
}

fn write_wav(path: &str, samples: &[f32], normalise: bool) {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let gain = if normalise && peak > 0.0 {
        0.99 / peak
    } else {
        1.0
    };
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    for s in samples {
        w.write_sample(((s * gain).clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn sherpa(dir: &str, voice: &str, mode: &str, trials: usize) {
    let text = corpus_text();
    let t0 = Instant::now();
    let tts = load_en(dir, if mode == "whole" { 1 } else { 100 });
    let load_s = t0.elapsed().as_secs_f64();
    for trial in 0..=trials {
        let r = match mode {
            "whole" => run_whole(&tts, &text, sid(voice)),
            "chunked" => run_chunked(&tts, &text, sid(voice)),
            _ => panic!("mode is whole or chunked"),
        };
        if let Ok(dir) = std::env::var("SPIKE_WAV_DIR") {
            write_wav(
                &format!("{dir}/sherpa-{voice}-{mode}-{trial}.wav"),
                &r.samples,
                false,
            );
        }
        let (lead, trail) = silence(&r.samples);
        println!(
            "{}",
            json!({
                "engine": "sherpa", "mode": mode, "voice": voice, "threads": threads(),
                // Trial 0 is the first request after load; the rest are warm.
                "trial": trial, "load_s": load_s,
                "ttfa_s": r.ttfa,
                "ttfa_from_start_s": if trial == 0 { Some(load_s + r.ttfa) } else { None },
                "wall_s": r.wall, "audio_s": r.audio_s, "rtf": r.wall / r.audio_s,
                "max_rss_mb": max_rss_bytes(RUSAGE_SELF) as f64 / 1048576.0,
                "lead_s": lead, "trail_s": trail,
                "pieces": r.pieces.iter().map(|(n, l, t)| json!([*n as f64 / RATE as f64, l, t])).collect::<Vec<_>>(),
            })
        );
    }
}

fn kokoro(voice: &str) {
    let text = corpus_text();
    let t0 = Instant::now();
    let mut child = Command::new("kokoro-rs")
        .args(["-q", "--no-download", "-v", voice, "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn kokoro-rs");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let mut out = child.stdout.take().unwrap();
    let mut bytes = Vec::new();
    let mut buf = [0u8; 65536];
    let mut ttfa = None;
    loop {
        let n = out.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        if ttfa.is_none() && bytes.len() > 44 {
            ttfa = Some(t0.elapsed().as_secs_f64());
        }
    }
    child.wait().unwrap();
    let wall = t0.elapsed().as_secs_f64();
    let samples: Vec<f32> = bytes[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect();
    if let Ok(dir) = std::env::var("SPIKE_WAV_DIR") {
        write_wav(&format!("{dir}/kokoro-{voice}.wav"), &samples, false);
    }
    let audio_s = samples.len() as f64 / RATE as f64;
    let (lead, trail) = silence(&samples);
    println!(
        "{}",
        json!({
            "engine": "kokoro-rs", "voice": voice,
            "ttfa_from_start_s": ttfa, "wall_s": wall, "audio_s": audio_s, "rtf": wall / audio_s,
            "max_rss_mb": max_rss_bytes(RUSAGE_CHILDREN) as f64 / 1048576.0,
            "lead_s": lead, "trail_s": trail,
        })
    );
}

fn probe(dir: &str) {
    let variants: [(&str, Option<&str>, Option<String>, bool); 4] = [
        ("lang=en-us", Some("en-us"), None, false),
        ("lang=en-us+dict", Some("en-us"), None, true),
        (
            "lexicon-us-en+dict, no lang",
            None,
            Some(format!("{dir}/lexicon-us-en.txt")),
            true,
        ),
        // Last: sherpa exits the process on this one.
        ("no lang", None, None, false),
    ];
    for (name, lang, lexicon, dict) in variants {
        let t0 = Instant::now();
        let tts = load(dir, lang, lexicon, dict, 1);
        let load_s = t0.elapsed().as_secs_f64();
        let result = tts.map(|t| {
            let r = run_whole(&t, "Hello there, this is a quick test.", 3);
            (t.num_speakers(), t.sample_rate(), r.audio_s)
        });
        println!(
            "{}",
            json!({"variant": name, "loaded": result.is_some(), "load_s": load_s,
                   "speakers": result.map(|r| r.0), "rate": result.map(|r| r.1),
                   "audio_s": result.map(|r| r.2)})
        );
    }
}

fn ab(dir: &str, voice: &str, outdir: &str, n: usize) {
    // Sentences 3 and 7 of the corpus.
    let lines: Vec<&str> = CORPUS.lines().filter(|l| !l.trim().is_empty()).collect();
    let text = format!("{} {}", lines[2], lines[6]);
    let tts = load_en(dir, 1);
    let sherpa = run_whole(&tts, &text, sid(voice)).samples;
    let tmp = format!("{outdir}/.kokoro-{voice}.wav");
    let ok = Command::new("kokoro-rs")
        .args(["-q", "--no-download", "-v", voice, "-o", &tmp, &text])
        .status()
        .expect("spawn kokoro-rs")
        .success();
    assert!(ok, "kokoro-rs failed");
    let kokoro = read_wav(&tmp);
    std::fs::remove_file(&tmp).unwrap();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let (a, b, key) = if nanos.is_multiple_of(2) {
        (&sherpa, &kokoro, "A=sherpa B=kokoro-rs")
    } else {
        (&kokoro, &sherpa, "A=kokoro-rs B=sherpa")
    };
    write_wav(&format!("{outdir}/voice{n}-A.wav"), a, true);
    write_wav(&format!("{outdir}/voice{n}-B.wav"), b, true);
    println!("voice{n} ({voice}): {key}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["probe", dir] => probe(dir),
        ["sherpa", dir, voice, mode, trials] => sherpa(dir, voice, mode, trials.parse().unwrap()),
        ["kokoro", voice] => kokoro(voice),
        ["chunks"] => {
            for c in chunk(&corpus_text()) {
                println!("{} {c}", c.chars().count());
            }
        }
        ["silence", files @ ..] => {
            for f in files {
                let (lead, trail) = silence(&read_wav(f));
                println!("{}", json!({"file": f, "lead_s": lead, "trail_s": trail}));
            }
        }
        ["ab", dir, voice, outdir, n] => ab(dir, voice, outdir, n.parse().unwrap()),
        _ => {
            eprintln!("usage: see the header of examples/tts_spike.rs");
            std::process::exit(2);
        }
    }
}
