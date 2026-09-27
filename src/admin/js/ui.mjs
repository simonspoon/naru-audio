// Small DOM/audio helpers shared by every tab module: element creation,
// toasts, WAV decode/encode, waveform peaks -> SVG bars, and a play button
// wired to an <audio> element. No framework, no build step.

/** `el('div', {class: 'x'}, ['text', el('span', {}, ['y'])])` */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on') && typeof v === 'function') {
      node.addEventListener(k.slice(2), v);
    } else if (v === false || v == null) {
      // skip
    } else if (v === true) {
      node.setAttribute(k, '');
    } else {
      node.setAttribute(k, v);
    }
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

let toastId = 0;
export function toast(message, isError = false) {
  const host = document.getElementById('toasts');
  if (!host) return;
  const id = ++toastId;
  const node = el('div', { class: `toast${isError ? ' err' : ''}` }, [message]);
  node.dataset.toastId = id;
  host.append(node);
  setTimeout(() => node.remove(), isError ? 6000 : 3000);
}

/** Decodes a WAV `ArrayBuffer` into `{sampleRate, channels, samples}`
 * where `samples` is a `Float32Array` of interleaved samples in [-1, 1].
 * Supports 16-bit PCM (`hound`'s default output) and 32-bit float. */
export function decodeWav(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== 0x52494646 /* "RIFF" */) {
    throw new Error('not a WAV file');
  }
  let pos = 12;
  let fmt = null;
  let data = null;
  while (pos + 8 <= view.byteLength) {
    const id = String.fromCharCode(
      view.getUint8(pos),
      view.getUint8(pos + 1),
      view.getUint8(pos + 2),
      view.getUint8(pos + 3),
    );
    const size = view.getUint32(pos + 4, true);
    const body = pos + 8;
    if (id === 'fmt ') {
      fmt = {
        audioFormat: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bitsPerSample: view.getUint16(body + 14, true),
      };
    } else if (id === 'data') {
      data = { offset: body, length: size };
    }
    pos = body + size + (size % 2);
  }
  if (!fmt || !data) throw new Error('malformed WAV: missing fmt/data chunk');
  const count = data.length / (fmt.bitsPerSample / 8);
  const samples = new Float32Array(count);
  if (fmt.bitsPerSample === 16) {
    for (let i = 0; i < count; i++) {
      samples[i] = view.getInt16(data.offset + i * 2, true) / 32768;
    }
  } else if (fmt.bitsPerSample === 32 && fmt.audioFormat === 3) {
    for (let i = 0; i < count; i++) {
      samples[i] = view.getFloat32(data.offset + i * 4, true);
    }
  } else if (fmt.bitsPerSample === 32) {
    for (let i = 0; i < count; i++) {
      samples[i] = view.getInt32(data.offset + i * 4, true) / 2147483648;
    }
  } else {
    throw new Error(`unsupported WAV bit depth ${fmt.bitsPerSample}`);
  }
  return { sampleRate: fmt.sampleRate, channels: fmt.channels, samples };
}

/** Encodes mono/interleaved `Float32Array` samples in [-1, 1] as a 16-bit
 * PCM WAV `Blob`, the inverse of `decodeWav`. Used by the design/playground
 * tabs to send recorded or synthesized audio back to the API. */
export function encodeWav({ sampleRate, channels = 1, samples }) {
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/** Reduces `samples` to `bars` peak amplitudes in [0, 1], one per bar,
 * for a compact waveform preview (mockups: 70 bars over a 280px SVG). */
export function peaks(samples, bars = 70) {
  const out = new Array(bars).fill(0);
  const step = Math.max(1, Math.floor(samples.length / bars));
  for (let b = 0; b < bars; b++) {
    let max = 0;
    const start = b * step;
    const end = Math.min(samples.length, start + step);
    for (let i = start; i < end; i++) max = Math.max(max, Math.abs(samples[i]));
    out[b] = max;
  }
  return out;
}

/** Renders `peaks()` output as the mockup's `<svg class="wv">` bar chart.
 * `el()` can't build this: `document.createElement('svg'/'rect')` makes
 * HTML-namespace elements that never paint, so the root is created with
 * `createElementNS` instead — that gives `innerHTML` the right namespace
 * context to parse the `<rect>` markup as real SVG shapes. */
export function waveformSvg(peakValues) {
  const w = peakValues.length * 4;
  const h = 28;
  const bars = peakValues
    .map((p, i) => {
      const barH = Math.max(3, p * (h - 4));
      const y = (h - barH) / 2;
      return `<rect x="${i * 4}" y="${y.toFixed(2)}" width="2" height="${barH.toFixed(2)}" rx="1"/>`;
    })
    .join('');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'wv');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.innerHTML = bars;
  return svg;
}

/** Fetches a WAV `Response`, decodes it, and returns a waveform `<svg>`
 * plus the underlying `ArrayBuffer` (so the caller can build an object URL
 * for playback without a second fetch). */
export async function waveformFromResponse(res) {
  const buffer = await res.arrayBuffer();
  const { samples } = decodeWav(buffer);
  return { svg: waveformSvg(peaks(samples)), buffer };
}

/** A play/pause icon button over an audio `src` (object URL or API path).
 * The `<audio>` element is created lazily on first play and torn down on
 * pause, so pausing frees the decode buffer. */
export function playButton(getSrc) {
  let audio = null;
  const btn = el('button', { class: 'play', type: 'button' }, ['▶']);
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (audio) {
      audio.pause();
      audio = null;
      btn.textContent = '▶';
      return;
    }
    const src = await getSrc();
    if (!src) return;
    audio = new Audio(src);
    audio.addEventListener('ended', () => {
      audio = null;
      btn.textContent = '▶';
    });
    audio.play();
    btn.textContent = '■';
  });
  return btn;
}

/** A `<select>` of `{id, label}` options, defaulting to `selected` if it
 * appears in `items`. Used for the model/voice pickers on Clone/Design/
 * Playground. */
export function picker(items, selected, onChange) {
  const select = el(
    'select',
    { onchange: (e) => onChange(e.target.value) },
    items.map((item) =>
      el('option', { value: item.id, selected: item.id === selected || undefined }, [
        item.label ?? item.id,
      ]),
    ),
  );
  return select;
}

export function fmtBytes(n) {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export function fmtUptime(startIso) {
  const start = new Date(startIso).getTime();
  if (Number.isNaN(start)) return '—';
  let s = Math.max(0, Math.floor((Date.now() - start) / 1000));
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}
