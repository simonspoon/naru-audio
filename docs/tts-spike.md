# naru-audio — TTS spike (task 9, design §6.3 item 9)

Status: arm64 measured, i9 pending, A/B sign-off pending. Task 1369. Date: 2026-09-24.

Question: can Kokoro v1.0 through `sherpa-onnx` replace kokoro-rs (§5.1), and
do we still need the §5.1 character chunker and the trim.rs port?

**Answer on arm64: yes, drop the chunker, drop the trim port.** Latency and
RTF are at parity; sherpa's first audio is ~0.25 s later from a cold process
but ~0.05 s earlier from a loaded model, which is the daemon's case.

## Setup

| Item | Value |
|---|---|
| Machine | Apple M5 Max, 18 cores (arm64). i9 (Alien) offline. |
| sherpa | `sherpa-onnx =1.13.6`, `static`, `provider=cpu`, `num_threads=18` (all cores; `SPIKE_THREADS` overrides). `GenerationConfig::default()` (silence_scale 0.2, speed 1.0). |
| sherpa package | `https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2`, 349,906,910 B, sha256 `c5f7e2d2caf082bc1d20fb70334a61d99d20b484500aad32e7cf84c128ea3298`. |
| sherpa config | `model=model.onnx`, `voices=voices.bin`, `tokens=tokens.txt`, `data_dir=espeak-ng-data`, `lang=en-us`, no lexicon, no `dict_dir`. |
| kokoro-rs | 0.1.2 (`~/.local/bin/kokoro-rs`), defaults (`-l en-us`, `--chunk-phonemes 500`, `--gap 0.12`), `-q --no-download -o -`, text on stdin. |
| Harness | `examples/tts_spike.rs` (no src/ changes, no new deps). |
| Runs | One engine at a time, never in parallel. OS file cache warm for both. |

## Corpus

`docs/tts-spike-corpus.txt`, 10 sentences, 639 chars, synthesised as one
request (joined with spaces). Sentence 1 is 119 chars with a clause comma, so
the character chunker's first cut (99 chars) lands mid-sentence.

Voices: af_heart (US F), af_bella (US F), am_michael (US M), bf_emma (UK F),
bm_george (UK M). All speak with `lang=en-us`, as kokoro-rs does by default.

## Numbers

Median (min–max) of 3. **Cold** = a new process: load + first request
(kokoro-rs has no other mode; every call is a process). **Warm** = sherpa,
model loaded, requests 2–3 in the same process (6 values). TTFA = request to
first samples: kokoro-rs spawn → first PCM bytes on stdout; sherpa cold =
process load + generate → first callback; warm = generate → first callback.
RTF = synthesis wall / audio duration. RSS = peak `ru_maxrss`.

| Voice | Metric | kokoro-rs arm64 | sherpa arm64 | kokoro-rs i9 | sherpa i9 |
|---|---|---|---|---|---|
| af_heart | TTFA cold s | 0.67 (0.63–0.67) | 0.92 (0.88–0.97) | pending | pending |
| | TTFA warm s | n/a | 0.73 (0.64–0.87) | pending | pending |
| | RTF cold / warm | 0.13 (0.12–0.13) / n/a | 0.12 (0.12–0.13) / 0.13 (0.13–0.13) | pending | pending |
| | peak RSS MB | 755 (755–757) | 935 (933–935) | pending | pending |
| af_bella | TTFA cold s | 0.73 (0.72–0.73) | 0.98 (0.93–1.05) | pending | pending |
| | TTFA warm s | n/a | 0.74 (0.66–0.97) | pending | pending |
| | RTF cold / warm | 0.13 (0.12–0.13) / n/a | 0.12 (0.12–0.13) / 0.13 (0.13–0.14) | pending | pending |
| | peak RSS MB | 768 (768–769) | 945 (945–946) | pending | pending |
| am_michael | TTFA cold s | 0.70 (0.70–0.70) | 1.00 (0.97–1.00) | pending | pending |
| | TTFA warm s | n/a | 0.73 (0.66–0.76) | pending | pending |
| | RTF cold / warm | 0.12 (0.12–0.12) / n/a | 0.11 (0.11–0.12) / 0.12 (0.11–0.13) | pending | pending |
| | peak RSS MB | 778 (777–779) | 944 (944–945) | pending | pending |
| bf_emma | TTFA cold s | 0.66 (0.66–0.67) | 0.91 (0.86–0.95) | pending | pending |
| | TTFA warm s | n/a | 0.70 (0.63–0.77) | pending | pending |
| | RTF cold / warm | 0.13 (0.13–0.13) / n/a | 0.11 (0.11–0.12) / 0.13 (0.12–0.13) | pending | pending |
| | peak RSS MB | 756 (756–757) | 929 (929–933) | pending | pending |
| bm_george | TTFA cold s | 0.71 (0.70–0.71) | 0.97 (0.92–1.16) | pending | pending |
| | TTFA warm s | n/a | 0.81 (0.70–0.89) | pending | pending |
| | RTF cold / warm | 0.12 (0.12–0.12) / n/a | 0.11 (0.11–0.12) / 0.12 (0.12–0.13) | pending | pending |
| | peak RSS MB | 772 (772–773) | 936 (935–937) | pending | pending |

- sherpa load is 0.26 s (0.25–0.28). Its cold TTFA is that load plus the
  first generate's 0.64 s (af_heart); the ~0.25 s gap to kokoro-rs is the
  load. Warm TTFA (0.70–0.81 s) is no better than the first request.
- sherpa peak RSS after one request is 724–788 MB, the same as kokoro-rs; it
  grows to ~935 MB by the 2nd request and was flat at the 3rd. Budget the
  loaded model at ~0.95 GB.
- Audio is 4–9% shorter from sherpa (af_heart 37.8 s vs 42.0 s): kokoro-rs
  reinserts a 0.12 s gap per sentence and sherpa's `silence_scale` 0.2 shrinks
  pauses. Not a speed difference; RTF is per audio second.

## (a) Chunker: sherpa sentence split + callback vs §5.1 character chunker

af_heart, 3 processes each, same corpus. (i) one `generate_with_config`,
`max_num_sentences=1`, callback. The callback fires once per sentence with
that sentence's samples only (10 calls; their lengths sum to the output).
(ii) §5.1 chunker, `max_num_sentences=100`, one generate per chunk; chunks
are 99 / 133 / 381 / 23 chars.

| af_heart, TTFA s | (i) sherpa split + callback | (ii) character chunker |
|---|---|---|
| first request after load | 0.64 (0.62–0.68) | 0.64 (0.58–0.74) |
| warm | 0.73 (0.64–0.87) | 0.67 (0.60–0.85) |
| RTF warm | 0.13 (0.13–0.13) | 0.13 (0.13–0.14) |

**Decision: drop the chunker.** TTFA is equal on the first request and
0.06 s apart warm, inside both spreads, although the chunker's first piece is
cut mid-sentence (99 chars) and sherpa's is the whole 119-char sentence. The
callback also delivers audio per sentence, which is a finer stream than the
chunker's 4 pieces. Residual risk: a single first sentence of several hundred
chars would delay sherpa's first callback in proportion; if that shows up, a
clause split of the *first sentence only* is the cheap fix.

## (b) Trim

Leading/trailing silence with kokoro-rs trim.rs's rule (librosa: frame 2048,
hop 512, top_db 60, ref = max; resolution one hop, 21 ms).

| | lead s | trail s |
|---|---|---|
| sherpa, per callback piece (450 pieces, 5 voices) | median 0.021–0.043 per voice, max 0.064 | median 0–0.049, max 0.076 |
| sherpa, per chunker piece (36 pieces) | max 0.043 | max 0.014 |
| kokoro-rs output (after its trim) | max 0.043 | max 0.011 |

**Decision: sherpa trims.** No piece has more than 76 ms at either end,
against Kokoro's ~2 s raw padding (§5.1). Do not port trim.rs.

## lang, lexicon, dict

`probe` subcommand, "Hello there, this is a quick test.", sid 3:

| Config | Result |
|---|---|
| `lang=en-us` | loads 0.27 s, 54 speakers, 24 kHz, 1.83 s audio |
| `lang=en-us` + `dict_dir=dict` | same audio length |
| `lexicon=lexicon-us-en.txt` + `dict_dir`, no `lang` | same audio length |
| `lang=en-gb` | loads; generate fails: espeak-ng "Failed to set eSpeak-ng voice" |
| no `lang`, no lexicon | sherpa prints "please pass --kokoro-lexicon or --kokoro-lang" and **exits the process** |

**Use `lang=en-us`, no lexicon, no `dict_dir`** for English, UK voices
included. `dict/` is jieba (Chinese) and the lexicons are `us-en`, `gb-en`,
`zh`; none is needed for English. The daemon must always set `lang`: the
missing-lang failure is `exit()`, not an error return.

## Weight identity

- **Voices: identical.** `voices.bin` (28,200,960 B) is the 54 `(510,1,256)`
  f32 arrays of kokoro-rs's `voices-v1.0.bin` npz concatenated in sid order;
  every block is byte-equal to the npz array of the same name.
- **Model: same weights, different export.** sha256 differs
  (`b40f62b1…` 325,560,556 B vs kokoro-rs `7d5df8ec…` 325,532,387 B). sherpa's
  graph is opset 14 with 4805 nodes, kokoro-rs's opset 20 with 2464; same
  inputs (`tokens`, `style`, `speed`). Large tensors: 81,084,064 vs 81,079,836
  params; 137 of 225 byte-identical, 84 equal as value multisets within
  3e-7 (float rounding of export folding), 4 small norm tensors (1024/1090
  elements) without a counterpart. Metadata: `model_url` is the same
  thewh1teagle/kokoro-onnx release kokoro-rs uses, `n_speakers=54`,
  `version=2`.
- The design's assumption holds. Audible parity is Simon's A/B call.

## name → sid

From the model's `speaker2id` metadata, identical to the voices.bin block
order proven above. Note em_santa is 53, not in alphabetical position.

| sid | name | sid | name | sid | name |
|---|---|---|---|---|---|
| 0 | af_alloy | 18 | am_puck | 36 | im_nicola |
| 1 | af_aoede | 19 | am_santa | 37 | jf_alpha |
| 2 | af_bella | 20 | bf_alice | 38 | jf_gongitsune |
| 3 | af_heart | 21 | bf_emma | 39 | jf_nezumi |
| 4 | af_jessica | 22 | bf_isabella | 40 | jf_tebukuro |
| 5 | af_kore | 23 | bf_lily | 41 | jm_kumo |
| 6 | af_nicole | 24 | bm_daniel | 42 | pf_dora |
| 7 | af_nova | 25 | bm_fable | 43 | pm_alex |
| 8 | af_river | 26 | bm_george | 44 | pm_santa |
| 9 | af_sarah | 27 | bm_lewis | 45 | zf_xiaobei |
| 10 | af_sky | 28 | ef_dora | 46 | zf_xiaoni |
| 11 | am_adam | 29 | em_alex | 47 | zf_xiaoxiao |
| 12 | am_echo | 30 | ff_siwis | 48 | zf_xiaoyi |
| 13 | am_eric | 31 | hf_alpha | 49 | zm_yunjian |
| 14 | am_fenrir | 32 | hf_beta | 50 | zm_yunxi |
| 15 | am_liam | 33 | hm_omega | 51 | zm_yunxia |
| 16 | am_michael | 34 | hm_psi | 52 | zm_yunyang |
| 17 | am_onyx | 35 | if_sara | 53 | em_santa |

## A/B pack

`/Users/simonspoon/naru-audio-spike-ab/`: `voice1..5-{A,B}.wav`, corpus
sentences 3 and 7 per voice (order af_heart, af_bella, am_michael, bf_emma,
bm_george), both engines, peak-normalised, 24 kHz, random letter per pair.
The key is in that directory only. sherpa output is raw (no leveller);
kokoro-rs output is its normal output (trimmed, levelled).

## i9 rerun

On the i9, with kokoro-rs installed and the package extracted to `$D`:

```sh
cargo build --example tts_spike --release
B=target/release/examples/tts_spike D=/path/to/kokoro-multi-lang-v1_0
for v in af_heart af_bella am_michael bf_emma bm_george; do for t in 1 2 3; do
  $B kokoro $v; $B sherpa $D $v whole 2 2>/dev/null; done; done > i9.jsonl
for t in 1 2 3; do $B sherpa $D af_heart chunked 2 2>/dev/null; done >> i9.jsonl
```

Each line is one run (JSON); medians as above. `SPIKE_THREADS` sets sherpa's
thread count (default: all logical cores). arm64 wall time: ~4 min.

## Open

- i9 run (Alien offline). The arm64 RTF of 0.12 says nothing about the i9;
  that column decides whether §5.1's "[assumption] fast enough" holds.
- Simon's A/B sign-off: "sherpa no worse than kokoro-rs: yes/no" per voice.
