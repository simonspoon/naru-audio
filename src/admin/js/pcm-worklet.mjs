// AudioWorkletProcessor for the Playground Listen panel's mic path: takes
// the mic's native sample rate and downsamples to 16 kHz mono f32le
// frames, posted to the main thread for the WS `/v1/audio/transcriptions/
// stream` handshake (`format: "f32le", sample_rate: 16000` — src/server/
// stream.rs). Same-origin module file (CSP `default-src 'self'`); no
// inline script, so this has to be its own asset (registered in
// src/server/admin.rs's ASSETS table).
//
// Nearest-neighbor decimation, not a windowed-sinc resampler: good enough
// for speech transcription and simple enough to read at a glance.

const TARGET_RATE = 16000;

class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global in worklet scope: the AudioContext's rate.
    this.ratio = sampleRate / TARGET_RATE;
    this.pos = 0; // fractional read position into the running input stream
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input || input.length === 0) return true;
    const out = [];
    while (this.pos < input.length) {
      out.push(input[Math.floor(this.pos)]);
      this.pos += this.ratio;
    }
    this.pos -= input.length;
    if (out.length) this.port.postMessage(new Float32Array(out));
    return true;
  }
}

registerProcessor('pcm-worklet', PcmWorklet);
