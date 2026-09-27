// Playground tab (mockup: na-admin-play2): Speak (text -> audio) and
// Listen (audio -> text) panels driven entirely by each model's `/v1/
// models` fields — no hardcoded per-model behavior. `x_prompt_format`
// documents how a model reads style guidance (fmt.md, manifest.rs
// `PromptFormat`): `style` is `"free_text"`/`"attributes"` (a separate
// `instructions` field) or `"inline_prefix"` (the description rides
// inside the spoken text itself, e.g. VoxCPM2's `(cheerful)Hello`); a
// model may additionally take inline tags placed within the text
// (`inline.tags`, e.g. OmniVoice's `[laughter]`), independent of `style`.
// Declared `knobs` (including `speed`, when a model declares it) go under
// the request's `knobs` object — the top-level `speed`/`exaggeration`
// fields are for the models that predate the knob API.
//
// Exposes `window.naruAdmin.playground.{speak, transcribeFile}` so a
// headless test without a mic/speaker can drive both panels directly.

import { getJson, request } from './api.mjs';
import { el, toast, picker, decodeWav, encodeWav, waveformSvg, peaks, playButton } from './ui.mjs';

const TEST_LINE = 'Listen. The build finished at fourteen thirty and nothing broke.';
const DEFAULT_SPEED = { name: 'speed', default: 1, min: 0.5, max: 2, step: 0.1 };

function concatBytes(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/** `bytes` is the assembled body of a §2.3 streamed `response_format:
 * "wav"` response: a 44-byte header (`wav_header`, src/server/speech.rs)
 * whose declared sizes are a placeholder followed by raw s16le PCM. Patches
 * the two size fields in place so `decodeWav` (ui.mjs) reads the real
 * sample count instead of the placeholder. */
function fixWavSizes(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataSize = bytes.byteLength - 44;
  view.setUint32(4, dataSize + 36, true);
  view.setUint32(40, dataSize, true);
  return bytes;
}

/** `response_format: "pcm"` has no WAV container at all (§2.3 `headers`):
 * raw s16le mono at `x-audio-sample-rate`. Wraps it as a WAV so the rest of
 * the pipeline (playback, waveform, download) doesn't need a second path. */
function pcmToWav(bytes, sampleRate) {
  const samples = new Float32Array(bytes.length / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
  return { sampleRate, samples };
}

function wordCount(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length;
}

export function mount(view, params) {
  let disposed = false;
  let ttsModels = [];
  let sttModels = [];
  let speakModel = params.model ?? null;
  let voices = [];
  let speakVoice = null;
  let speakFormat = 'wav';
  let knobValues = {}; // name -> current value, only for touched/rendered knobs
  let lastBlob = null;

  let sttModel = null;
  let ws = null;
  let audioCtx = null;
  let mediaStream = null;
  let recordStarted = 0;
  let firstResultAt = null;
  let finals = [];
  let partialText = '';

  // --- Speak panel -------------------------------------------------------

  const speakModelBar = el('div', { class: 'filt' });
  const instructWrap = el('div', {});
  const prefixWrap = el('div', {});
  const textBox = el('textarea', { class: 'inp', rows: 3, value: TEST_LINE });
  const tagStrip = el('div', {});
  const voicePickerWrap = el('div', {});
  const formatPickerWrap = el('div', {});
  const speedWrap = el('div', {});
  const knobsWrap = el('div', {});
  const speakBtn = el('button', { class: 'big', style: 'margin:0 0 0 auto;max-width:140px' }, ['▶ SPEAK']);
  const waveWrap = el('div', { style: 'margin-top:10px' });
  const statRow = el('div', { class: 'stat' });
  const downloadLink = el('a', { class: 'btn', style: 'display:none' }, ['⇩ download']);
  const curlBox = el('div', { class: 'code' });
  const copyCurlBtn = el('span', { class: 'btn', style: 'margin-left:auto' }, ['⧉ copy']);

  const speakCard = el('div', { class: 'card' }, [
    el('h3', {}, ['SPEAK · TEXT → AUDIO']),
    el('div', { class: 'lab' }, ['MODEL']),
    speakModelBar,
    instructWrap,
    prefixWrap,
    el('div', { class: 'lab', style: 'margin-top:8px' }, ['TEXT TO SPEAK']),
    textBox,
    tagStrip,
    el('div', { style: 'display:flex;gap:14px;margin-top:8px;align-items:flex-end;flex-wrap:wrap' }, [
      voicePickerWrap,
      speedWrap,
      formatPickerWrap,
      speakBtn,
    ]),
    knobsWrap,
    waveWrap,
    statRow,
    el('div', { style: 'display:flex;gap:8px;align-items:center' }, [downloadLink]),
    el('div', { style: 'display:flex' }, [curlBox]),
  ]);

  // --- Listen panel --------------------------------------------------------

  const sttModelBar = el('div', { class: 'filt' });
  const recordBtn = el('button', { class: 'rec' }, ['● RECORD']);
  const dropZone = el('div', { class: 'drop' });
  const liveBox = el('div', { class: 'live' }, [el('span', {}, ['record or drop a file to transcribe']) ]);
  const listenStat = el('div', { class: 'stat' });
  const copyTranscriptBtn = el('span', { class: 'btn', style: 'margin-left:auto' }, ['⧉ copy text']);

  const listenCard = el('div', { class: 'card' }, [
    el('h3', {}, ['LISTEN · AUDIO → TEXT']),
    el('div', { class: 'lab' }, ['MODEL']),
    sttModelBar,
    el('div', { style: 'display:flex;gap:8px;align-items:center;margin:8px 0' }, [
      recordBtn,
      el('span', { style: 'color:var(--muted);font-size:11px' }, ['or drop a file below']),
    ]),
    dropZone,
    el('div', { class: 'lab', style: 'margin-top:10px' }, ['TRANSCRIPT']),
    liveBox,
    listenStat,
  ]);

  view.append(el('div', { class: 'two' }, [speakCard, listenCard]));

  // --- Speak: model/voice/format/knobs -----------------------------------

  function currentSpeakModel() {
    return ttsModels.find((m) => m.id === speakModel) ?? null;
  }

  function declaredKnobs(model) {
    const pf = model?.x_prompt_format;
    return pf?.knobs ?? [];
  }

  function speedKnob(model) {
    return declaredKnobs(model).find((k) => k.name === 'speed') ?? null;
  }

  function renderSpeakModelBar() {
    speakModelBar.replaceChildren(
      ...ttsModels.map((m) =>
        el(
          'span',
          {
            class: `chip${m.id === speakModel ? ' on' : ''}`,
            onclick: () => {
              speakModel = m.id;
              speakVoice = null;
              onSpeakModelChanged();
            },
          },
          [m.id],
        ),
      ),
    );
  }

  function renderInstructions() {
    const model = currentSpeakModel();
    instructWrap.replaceChildren();
    prefixWrap.replaceChildren();
    if (!model) return;
    const style = model.x_prompt_format?.style;
    if (model.x_instruct && style !== 'inline_prefix') {
      instructWrap.append(
        el('div', { style: 'display:flex;align-items:center;gap:8px' }, [
          el('div', { class: 'lab', style: 'margin:0' }, ['OVERALL INSTRUCTION']),
          el('span', { class: 'cap y' }, ['this model takes one']),
        ]),
        el('textarea', { class: 'inp', rows: 1, id: 'pg-instruction' }),
      );
    } else if (style === 'inline_prefix') {
      prefixWrap.append(
        el('div', { class: 'lab' }, ['DESCRIPTION · PREFIXED TO THE TEXT']),
        el('input', { class: 'inp', id: 'pg-prefix' }),
        el('div', { style: 'color:var(--muted);font-size:11px;margin-top:2px' }, [
          model.x_prompt_format?.hint ?? 'e.g. (cheerful, slightly faster)',
        ]),
      );
    }
  }

  function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? textarea.value.length;
    textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);
    const pos = start + text.length;
    textarea.setSelectionRange(pos, pos);
    textarea.focus();
  }

  function renderTagStrip() {
    const model = currentSpeakModel();
    const inline = model?.x_prompt_format?.inline;
    tagStrip.replaceChildren();
    if (!inline?.tags?.length) return;
    const syntax = inline.syntax ?? '[tag]';
    const wrap = (tag) => syntax.replace(/tag/, tag);
    tagStrip.append(
      el('div', { class: 'cmp', style: 'border-left:2px solid var(--violet);margin-top:8px' }, [
        el('span', { style: 'color:var(--violet)' }, ['INLINE FOR THIS MODEL']),
        ' ',
        ...inline.tags.map((tag) =>
          el(
            'span',
            { class: 'chip', style: 'margin-right:4px', onclick: () => insertAtCursor(textBox, wrap(tag)) },
            [wrap(tag)],
          ),
        ),
        el('span', { style: 'color:var(--muted);font-size:11px;display:block;margin-top:4px' }, [
          inline.example ?? model.x_prompt_format?.hint ?? 'click a tag to insert it at the cursor',
        ]),
      ]),
    );
  }

  function renderVoicePicker() {
    voicePickerWrap.replaceChildren();
    if (!voices.length) return;
    const items = voices.map((v) => ({ id: v.id, label: v.id }));
    const select = picker(items, speakVoice, (id) => (speakVoice = id));
    select.className = 'sel';
    voicePickerWrap.append(el('div', { class: 'lab' }, ['VOICE']), select);
  }

  function renderFormatPicker() {
    const select = picker(
      [{ id: 'wav', label: 'wav' }, { id: 'pcm', label: 'pcm' }],
      speakFormat,
      (id) => (speakFormat = id),
    );
    select.className = 'sel';
    formatPickerWrap.replaceChildren(el('div', { class: 'lab' }, ['FORMAT']), select);
  }

  function knobSlider(knob, onDefaultRow) {
    const hasRange = knob.min != null && knob.max != null;
    const step = knob.step ?? (hasRange ? (knob.max - knob.min) / 100 : 0.1);
    const initial = knobValues[knob.name] ?? knob.default ?? 0;
    knobValues[knob.name] = initial;
    const valueLabel = el('span', { style: 'color:var(--muted);font-size:11px' }, [
      `${initial}${knob.name === 'speed' ? '×' : ''}`,
    ]);
    const input = hasRange
      ? el('input', {
          type: 'range',
          min: knob.min,
          max: knob.max,
          step,
          value: initial,
        })
      : el('input', { type: 'number', class: 'inp', step, value: initial });
    input.addEventListener('input', () => {
      const v = Number(input.value);
      knobValues[knob.name] = v;
      valueLabel.textContent = `${v}${knob.name === 'speed' ? '×' : ''}`;
    });
    if (onDefaultRow) {
      return el('div', {}, [el('div', { class: 'lab' }, [knob.name.toUpperCase()]), input, valueLabel]);
    }
    return el('div', { class: 'row' }, [
      el('b', {}, [knob.name]),
      input,
      valueLabel,
    ]);
  }

  function renderSpeedControl() {
    const model = currentSpeakModel();
    speedWrap.replaceChildren();
    const knob = speedKnob(model);
    if (!knob) return;
    speedWrap.append(knobSlider(knob, true));
  }

  function renderKnobs() {
    const model = currentSpeakModel();
    knobsWrap.replaceChildren();
    const knobs = declaredKnobs(model).filter((k) => k.name !== 'speed');
    if (!knobs.length) return;
    knobsWrap.append(
      el('div', { class: 'lab', style: 'margin-top:8px' }, ['KNOBS']),
      ...knobs.map((k) => knobSlider(k, false)),
    );
  }

  async function refreshVoices() {
    if (!speakModel) {
      voices = [];
      renderVoicePicker();
      return;
    }
    try {
      voices = (await getJson(`/v1/audio/voices?model=${encodeURIComponent(speakModel)}`, { silent: true }))
        .voices;
    } catch {
      voices = [];
    }
    if (!speakVoice && voices.length) speakVoice = voices.find((v) => v.default)?.id ?? voices[0].id;
    renderVoicePicker();
  }

  function onSpeakModelChanged() {
    knobValues = {};
    renderSpeakModelBar();
    renderInstructions();
    renderTagStrip();
    renderFormatPicker();
    renderSpeedControl();
    renderKnobs();
    refreshVoices();
  }

  // --- Speak: request/measure/play ----------------------------------------

  function currentInstructions(model) {
    if (model?.x_instruct && model.x_prompt_format?.style !== 'inline_prefix') {
      return document.getElementById('pg-instruction')?.value.trim() || undefined;
    }
    return undefined;
  }

  function currentInput(model) {
    const raw = textBox.value;
    if (model?.x_prompt_format?.style === 'inline_prefix') {
      const desc = document.getElementById('pg-prefix')?.value.trim();
      if (desc) return `(${desc})${raw}`;
    }
    return raw;
  }

  function currentKnobs(model) {
    const names = new Set(declaredKnobs(model).map((k) => k.name));
    const out = {};
    for (const [name, value] of Object.entries(knobValues)) {
      if (names.has(name)) out[name] = value;
    }
    return out;
  }

  function buildCurl(body) {
    return `curl ${location.origin}/v1/audio/speech \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body)}'`;
  }

  function renderStats({ firstAudioMs, totalMs, durationSec }) {
    const realtime = durationSec > 0 ? durationSec / (totalMs / 1000) : 0;
    statRow.replaceChildren(
      el('span', {}, ['first audio ', el('b', {}, [`${(firstAudioMs / 1000).toFixed(2)} s`])]),
      el('span', {}, ['total ', el('b', {}, [`${(totalMs / 1000).toFixed(2)} s`])]),
      el('span', {}, ['audio ', el('b', {}, [`${durationSec.toFixed(2)} s`])]),
      el('span', {}, ['speed ', el('b', {}, [`${realtime.toFixed(1)}× realtime`])]),
    );
  }

  /** Runs the full Speak flow: builds the request from the current UI (or
   * `overrides`), streams the response measuring first-chunk and total
   * time, decodes/wraps it as a playable WAV, plays it, and updates the
   * waveform/stats/download/curl. Returns the measurement + blob so a
   * headless caller (`window.naruAdmin.playground.speak`) can assert on
   * it without a speaker. */
  async function doSpeak(overrides = {}) {
    const model = ttsModels.find((m) => m.id === (overrides.model ?? speakModel));
    if (!model) {
      toast('pick a model first', true);
      return null;
    }
    const format = overrides.format ?? speakFormat;
    const body = {
      model: model.id,
      input: overrides.input ?? currentInput(model),
      response_format: format,
      stream: true,
    };
    const voice = overrides.voice ?? speakVoice;
    if (voice) body.voice = voice;
    const instructions = overrides.instructions ?? currentInstructions(model);
    if (instructions) body.instructions = instructions;
    const knobs = overrides.knobs ?? currentKnobs(model);
    if (Object.keys(knobs).length) body.knobs = knobs;

    curlBox.replaceChildren(el('i', {}, ['curl']), ` ${buildCurl(body).replace(/^curl /, '')}`, copyCurlBtn);

    const started = performance.now();
    let firstAudioMs = null;
    let res;
    try {
      res = await request('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch {
      return null;
    }
    const reader = res.body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (firstAudioMs == null) firstAudioMs = performance.now() - started;
      chunks.push(value);
    }
    const totalMs = performance.now() - started;
    const bytes = concatBytes(chunks);
    const decoded =
      format === 'pcm'
        ? pcmToWav(bytes, Number(res.headers.get('x-audio-sample-rate')) || 24000)
        : decodeWav(fixWavSizes(bytes).buffer);
    const blob = encodeWav(decoded);
    lastBlob = blob;
    const durationSec = decoded.samples.length / decoded.sampleRate;
    renderStats({ firstAudioMs, totalMs, durationSec });
    const url = URL.createObjectURL(blob);
    waveWrap.replaceChildren(playButton(() => url), waveformSvg(peaks(decoded.samples)));
    downloadLink.href = url;
    downloadLink.download = `speech.${format === 'pcm' ? 'wav' : format}`;
    downloadLink.style.display = '';
    new Audio(url).play().catch(() => {});
    return { firstAudioMs, totalMs, durationSec, blob };
  }

  speakBtn.addEventListener('click', () => doSpeak());
  copyCurlBtn.addEventListener('click', () => {
    navigator.clipboard?.writeText(curlBox.textContent.replace(/⧉ copy$/, '').trim());
    toast('curl command copied');
  });

  // --- Listen: mic + WS streaming ------------------------------------------

  function renderSttModelBar() {
    sttModelBar.replaceChildren(
      ...sttModels.map((m) =>
        el(
          'span',
          {
            class: `chip${m.id === sttModel ? ' on' : ''}`,
            onclick: () => {
              sttModel = m.id;
              renderSttModelBar();
            },
          },
          [m.id],
        ),
      ),
    );
  }

  function renderLive() {
    liveBox.replaceChildren(
      finals.join(' '),
      partialText ? el('span', {}, [` ${partialText}`]) : null,
    );
  }

  function renderListenStat(latencyMs) {
    const text = finals.join(' ');
    listenStat.replaceChildren(
      el('span', {}, ['latency ', el('b', {}, [latencyMs != null ? `${Math.round(latencyMs)} ms` : '—'])]),
      el('span', {}, ['words ', el('b', {}, [String(wordCount(text))])]),
      copyTranscriptBtn,
    );
  }

  function noteResult() {
    if (firstResultAt == null) firstResultAt = performance.now();
    renderListenStat(firstResultAt - recordStarted);
  }

  async function stopRecording() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
    }
    mediaStream?.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    if (audioCtx) {
      await audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    recordBtn.classList.remove('on');
    recordBtn.textContent = '● RECORD';
  }

  async function startRecording() {
    if (!sttModel) {
      toast('pick a model first', true);
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      toast('microphone recording is not available in this browser', true);
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      toast('microphone permission denied', true);
      return;
    }
    finals = [];
    partialText = '';
    firstResultAt = null;
    renderLive();
    renderListenStat(null);
    recordStarted = performance.now();

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/v1/audio/transcriptions/stream`);
    ws.binaryType = 'arraybuffer';
    ws.addEventListener('open', () => {
      ws.send(JSON.stringify({ type: 'start', model: sttModel, sample_rate: 16000, format: 'f32le', partials: true }));
    });
    ws.addEventListener('message', (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'partial') {
        partialText = msg.text;
        noteResult();
        renderLive();
      } else if (msg.type === 'final') {
        if (msg.text) finals.push(msg.text);
        partialText = '';
        noteResult();
        renderLive();
      } else if (msg.type === 'error') {
        toast(msg.message, true);
        stopRecording();
      }
    });
    ws.addEventListener('close', () => stopRecording());

    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule(new URL('./pcm-worklet.mjs', import.meta.url));
    const source = audioCtx.createMediaStreamSource(mediaStream);
    const node = new AudioWorkletNode(audioCtx, 'pcm-worklet');
    node.port.onmessage = (e) => {
      if (ws?.readyState === WebSocket.OPEN) ws.send(e.data.buffer);
    };
    source.connect(node);

    recordBtn.classList.add('on');
    recordBtn.textContent = '■ STOP · recording';
  }

  recordBtn.addEventListener('click', () => {
    if (mediaStream) stopRecording();
    else startRecording();
  });
  copyTranscriptBtn.addEventListener('click', () => {
    navigator.clipboard?.writeText(finals.join(' '));
    toast('transcript copied');
  });

  // --- Listen: drop a file (no mic needed) --------------------------------

  /** Decodes `file`, resamples to the 16 kHz mono `POST /v1/audio/
   * transcriptions` expects, and posts it as WAV. No WS session needed, so
   * this also serves as `window.naruAdmin.playground.transcribeFile` for a
   * headless test without a file picker. Returns the transcript text. */
  async function transcribeFile(file) {
    const buffer = await file.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    let decoded;
    try {
      decoded = await ctx.decodeAudioData(buffer);
    } catch {
      toast('could not decode that file', true);
      return null;
    } finally {
      ctx.close();
    }
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    const wav = encodeWav({ sampleRate: 16000, samples: rendered.getChannelData(0) });

    finals = [];
    partialText = '';
    renderLive();
    const form = new FormData();
    if (sttModel) form.append('model', sttModel);
    form.append('file', wav, file.name || 'audio.wav');
    const started = performance.now();
    let res;
    try {
      res = await request('/v1/audio/transcriptions', { method: 'POST', body: form });
    } catch {
      return null;
    }
    const latencyMs = performance.now() - started;
    const data = await res.json();
    finals = data.text ? [data.text] : [];
    renderLive();
    renderListenStat(latencyMs);
    return data.text ?? '';
  }

  dropZone.append(
    el('span', {}, ['⇪ drop an audio file or ']),
    (() => {
      const input = el('input', { type: 'file', accept: 'audio/*', style: 'display:none' });
      input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) transcribeFile(file);
        e.target.value = '';
      });
      const browse = el('span', { style: 'color:var(--cyan);cursor:pointer' }, ['browse']);
      browse.addEventListener('click', () => input.click());
      return el('span', {}, [browse, input]);
    })(),
  );
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('over');
    const file = e.dataTransfer.files[0];
    if (file) transcribeFile(file);
  });

  // --- Boot ----------------------------------------------------------------

  async function refresh() {
    if (disposed) return;
    let models;
    try {
      models = (await getJson('/v1/models')).data;
    } catch {
      return;
    }
    if (disposed) return;
    // Only what's actually loadable here: an unpulled model has no voices
    // and no /v1/audio/speech|transcriptions to try it against, so it
    // would just be a picker entry that always errors.
    ttsModels = models.filter((m) => m.x_kind === 'tts' && m.x_pulled);
    sttModels = models.filter((m) => m.x_kind === 'stt' && m.x_pulled);
    if (!speakModel && ttsModels.length) speakModel = ttsModels[0].id;
    if (!sttModel && sttModels.length) sttModel = sttModels[0].id;
    onSpeakModelChanged();
    renderSttModelBar();
  }

  window.naruAdmin = window.naruAdmin ?? {};
  window.naruAdmin.playground = { speak: doSpeak, transcribeFile };

  refresh();
  return () => {
    disposed = true;
    stopRecording();
    if (window.naruAdmin) delete window.naruAdmin.playground;
  };
}
