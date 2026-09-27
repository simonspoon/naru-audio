// Clone tab (mockup: na-admin-clone): pick a clone-capable model, get a
// single-speaker sample (drop/browse a file, record one, or pick one from
// the samples library — naru task 1463), trim it to 5-15 s (the server
// accepts 3-30), auto-transcribe the trimmed clip, name it, and create the
// voice. The right panel plays the original and previews the clone saying
// a test line before anything is saved.
//
// A dropped/recorded file is a *raw* clip: it is uploaded to
// `POST /v1/audio/prep/clips` and handed to the samples tab
// (`#samples?clip=<id>&return=clone&model=<model>`) to go through voice-prep
// first, rather than trimmed here directly. Picking a sample from the
// library, by contrast, is already a cleaned single-speaker clip: its audio
// is fetched and loaded through the same local decode/trim path as before,
// and its saved transcript pre-fills the transcript box (skipping
// auto-transcribe) unless the sample has none, in which case auto-transcribe
// runs as it does for a fresh recording.
//
// Exposes `window.naruAdmin.clone.loadFile(file)` and
// `window.naruAdmin.clone.pickSample(id)` so a headless test without a file
// picker or a live samples library can drive either path directly.

import { getJson, request } from './api.mjs';
import { el, toast, encodeWav, waveformSvg, peaks, playButton } from './ui.mjs';

const MIN_S = 3;
const MAX_S = 30;
const TEST_LINE = 'Warm coffee, a quiet morning, and nothing on fire yet.';

function averageToMono(buffer) {
  const { numberOfChannels, length } = buffer;
  const out = new Float32Array(length);
  for (let c = 0; c < numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) out[i] += data[i] / numberOfChannels;
  }
  return out;
}

export function mount(view, params) {
  let disposed = false;
  let models = [];
  let selectedModel = params.model ?? null;
  // The decoded original sample and the current trim, in seconds.
  let sample = null; // {sampleRate, samples}
  let trimStart = 0;
  let trimEnd = 0;
  let transcript = '';
  let mediaRecorder = null;
  let recordedChunks = [];
  let sampleLibrary = []; // [{id, name, transcript, ...}] from GET /v1/audio/samples
  const presetSample = params.sample ?? null;
  // Bumped on every pickSample() call so a slower, superseded fetch can
  // tell it lost the race and skip applying its (now stale) result.
  let pickGen = 0;
  let pickedSampleId = null;

  const modelBar = el('div', { class: 'filt' });
  const sampleSelect = el('select', {
    class: 'inp',
    onchange: (e) => {
      if (e.target.value) pickSample(e.target.value);
    },
  });
  const sampleHint = el('div', { style: 'color:var(--muted);font-size:11px' });
  const dropZone = el('div', { class: 'drop' });
  const waveWrap = el('div', { class: 'trim-wrap' });
  const transcriptBox = el('textarea', { class: 'inp', rows: 2 });
  const nameInput = el('input', { class: 'inp' });
  const permCheck = el('input', { type: 'checkbox' });
  const createBtn = el('button', { class: 'big', style: 'margin:0 0 0 auto;max-width:200px' }, ['CREATE VOICE']);
  const originalPlay = el('span', { class: 'cmp' }, ['no sample loaded']);
  const previewLine = el('div', { class: 'inp', style: 'font-size:12px' }, [TEST_LINE]);
  const previewBtn = el('button', { class: 'big' }, ['▶ PREVIEW CLONE']);

  const leftCard = el('div', { class: 'card' }, [
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['1']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['MODEL']),
        modelBar,
        el('span', { style: 'color:var(--muted);font-size:11px' }, [' only models that can clone']),
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['2']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['SAMPLE · ONE SPEAKER · 5-15 S']),
        el('div', { class: 'row' }, [sampleSelect, sampleHint]),
        dropZone,
        waveWrap,
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['3']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['TRANSCRIPT · AUTO · EDIT IF WRONG']),
        transcriptBox,
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['4']),
      el('div', { style: 'flex:1;display:flex;gap:14px;align-items:center;flex-wrap:wrap' }, [
        el('div', {}, [el('div', { class: 'lab' }, ['NAME']), nameInput]),
        el('label', { class: 'chk' }, [permCheck, 'I have permission to clone this voice']),
        createBtn,
      ]),
    ]),
  ]);

  const rightCard = el('div', { class: 'card' }, [
    el('h3', {}, ['BEFORE YOU SAVE']),
    el('div', { class: 'lab' }, ['ORIGINAL']),
    originalPlay,
    el('div', { class: 'lab', style: 'margin-top:10px' }, ['CLONE SAYS']),
    previewLine,
    previewBtn,
    el('div', { style: 'color:var(--muted);font-size:11px;margin-top:10px' }, [
      'Sounds off? Trim tighter, fix the transcript, try again. Nothing is saved until you create it.',
    ]),
  ]);

  view.append(el('div', { class: 'side' }, [leftCard, rightCard]));

  function renderModelBar() {
    modelBar.replaceChildren(
      ...models.map((m) =>
        el(
          'span',
          {
            class: `chip${m.id === selectedModel ? ' on' : ''}`,
            onclick: () => {
              selectedModel = m.id;
              renderModelBar();
            },
          },
          [m.id],
        ),
      ),
    );
  }

  function trimmedSamples() {
    const from = Math.floor(trimStart * sample.sampleRate);
    const to = Math.min(sample.samples.length, Math.ceil(trimEnd * sample.sampleRate));
    return sample.samples.slice(from, to);
  }

  function trimmedWavBlob() {
    return encodeWav({ sampleRate: sample.sampleRate, samples: trimmedSamples() });
  }

  function renderWave() {
    waveWrap.replaceChildren();
    if (!sample) return;
    const duration = sample.samples.length / sample.sampleRate;
    const svg = waveformSvg(peaks(sample.samples, 140));
    waveWrap.append(svg);
    const startHandle = el('div', {
      class: 'trim-handle',
      style: `left:${(trimStart / duration) * 100}%`,
    });
    const endHandle = el('div', {
      class: 'trim-handle',
      style: `left:${(trimEnd / duration) * 100}%`,
    });
    waveWrap.append(startHandle, endHandle);
    wireDrag(startHandle, 'start', duration);
    wireDrag(endHandle, 'end', duration);
    waveWrap.append(
      el('div', { style: 'display:flex;justify-content:space-between;color:var(--muted);font-size:10px' }, [
        `${(trimEnd - trimStart).toFixed(1)} s selected`,
        'drag the cyan handles to trim',
      ]),
    );
  }

  function wireDrag(handle, which, duration) {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      const rect = waveWrap.getBoundingClientRect();
      const onMove = (ev) => {
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        const t = frac * duration;
        if (which === 'start') trimStart = Math.min(t, trimEnd - 0.5);
        else trimEnd = Math.max(t, trimStart + 0.5);
        renderWave();
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        transcribeSelection();
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });
  }

  async function loadFile(file, opts = {}) {
    const buffer = await file.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    let audioBuffer;
    try {
      audioBuffer = await ctx.decodeAudioData(buffer);
    } catch {
      toast('could not decode that file', true);
      return;
    } finally {
      ctx.close();
    }
    sample = { sampleRate: audioBuffer.sampleRate, samples: averageToMono(audioBuffer) };
    trimStart = 0;
    trimEnd = sample.samples.length / sample.sampleRate;
    renderWave();
    const objectUrl = URL.createObjectURL(file);
    originalPlay.replaceChildren(playButton(() => objectUrl), ` your clip · ${trimEnd.toFixed(1)} s`);
    // A picked sample already has a saved transcript (`opts.presetTranscript`)
    // — use it instead of clobbering it with a fresh auto-transcribe. A
    // sample with no transcript falls back to the same auto-transcribe a
    // fresh recording gets.
    if (opts.presetTranscript != null) {
      transcript = opts.presetTranscript;
      transcriptBox.value = transcript;
    } else {
      await transcribeSelection();
    }
  }

  /** Raw upload (drop/browse/record): routes through voice-prep instead of
   * trimming here directly (naru task 1463) — upload the clip, then hand
   * off to the samples tab to transcribe/crop/clean it. */
  async function uploadRawClip(file) {
    const form = new FormData();
    form.append('file', file, file.name || 'clip');
    let clip;
    try {
      const res = await request('/v1/audio/prep/clips', { method: 'POST', body: form });
      clip = await res.json();
    } catch {
      return;
    }
    const modelPart = selectedModel ? `&model=${encodeURIComponent(selectedModel)}` : '';
    location.hash = `#samples?clip=${encodeURIComponent(clip.id)}&return=clone${modelPart}`;
  }

  function renderSampleSelect() {
    if (!sampleLibrary.length) {
      sampleSelect.replaceChildren();
      sampleSelect.style.display = 'none';
      sampleHint.replaceChildren('no prepped samples yet — ', el('a', { href: '#samples' }, ['prep one']));
      return;
    }
    sampleSelect.style.display = '';
    sampleHint.replaceChildren();
    sampleSelect.replaceChildren(
      el('option', { value: '' }, ['— pick a sample —']),
      ...sampleLibrary.map((s) => el('option', { value: s.id }, [s.name])),
    );
  }

  async function refreshSamples() {
    let ids;
    try {
      ids = (await getJson('/v1/audio/samples')).samples;
    } catch {
      return;
    }
    if (disposed) return;
    sampleLibrary = await Promise.all(
      ids.map((id) => getJson(`/v1/audio/samples/${encodeURIComponent(id)}`, { silent: true }).catch(() => null)),
    ).then((list) => list.filter(Boolean));
    if (disposed) return;
    renderSampleSelect();
    if (presetSample && sampleLibrary.some((s) => s.id === presetSample)) {
      pickSample(presetSample);
    }
  }

  async function pickSample(id) {
    const meta = sampleLibrary.find((s) => s.id === id);
    if (!meta) return;
    const gen = ++pickGen;
    let blob;
    try {
      const res = await fetch(`/v1/audio/samples/${encodeURIComponent(id)}/audio?variant=clean`);
      if (!res.ok) throw new Error('fetch failed');
      blob = await res.blob();
    } catch {
      if (gen !== pickGen) return; // a newer pick already won; don't clobber it
      toast('could not load that sample', true);
      sampleSelect.value = pickedSampleId ?? '';
      return;
    }
    if (gen !== pickGen) return; // a newer pick already won; drop this stale result
    const hasTranscript = !!(meta.transcript && meta.transcript.trim());
    const file = new File([blob], `${meta.name}.wav`, { type: 'audio/wav' });
    await loadFile(file, hasTranscript ? { presetTranscript: meta.transcript } : {});
    if (gen !== pickGen) return;
    pickedSampleId = id;
    sampleSelect.value = id;
  }

  async function transcribeSelection() {
    if (!sample) return;
    const form = new FormData();
    form.append('file', trimmedWavBlob(), 'sample.wav');
    try {
      const res = await request('/v1/audio/transcriptions', { method: 'POST', body: form }, { silent: true });
      const data = await res.json();
      transcript = data.text ?? '';
      transcriptBox.value = transcript;
    } catch {
      // leave whatever the user already typed
    }
  }

  dropZone.append(
    el('span', {}, ['⇪ drop WAV / MP3 or ']),
    (() => {
      const input = el('input', { type: 'file', accept: 'audio/*', style: 'display:none' });
      input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) uploadRawClip(file);
        e.target.value = '';
      });
      const browse = el('span', { style: 'color:var(--cyan);cursor:pointer' }, ['browse']);
      browse.addEventListener('click', () => input.click());
      return el('span', {}, [browse, input]);
    })(),
    el('span', { style: 'color:var(--muted)' }, ['or']),
    (() => {
      const btn = el('button', { class: 'rec' }, ['● RECORD']);
      btn.addEventListener('click', () => toggleRecord(btn));
      return btn;
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
    if (file) uploadRawClip(file);
  });

  async function toggleRecord(btn) {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      toast('microphone recording is not available in this browser', true);
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      toast('microphone permission denied', true);
      return;
    }
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.classList.remove('on');
      btn.textContent = '● RECORD';
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
      await uploadRawClip(new File([blob], 'recording', { type: blob.type }));
    };
    mediaRecorder.start();
    btn.classList.add('on');
    btn.textContent = '■ STOP';
  }

  previewBtn.addEventListener('click', async () => {
    if (!sample || !selectedModel) return;
    const form = new FormData();
    form.append('model', selectedModel);
    form.append('file', trimmedWavBlob(), 'sample.wav');
    form.append('text', transcriptBox.value);
    form.append('input', previewLine.textContent);
    try {
      const res = await request('/api/voices/preview', { method: 'POST', body: form });
      new Audio(URL.createObjectURL(await res.blob())).play();
    } catch {
      // request() already toasted
    }
  });

  createBtn.addEventListener('click', async () => {
    const model = models.find((m) => m.id === selectedModel);
    if (!sample || !model) {
      toast('pick a model and a sample first', true);
      return;
    }
    const name = nameInput.value.trim();
    if (!name) {
      toast('name the voice first', true);
      return;
    }
    if (!permCheck.checked) {
      toast('confirm you have permission to clone this voice', true);
      return;
    }
    if (model.x_clone_requires_transcript && !transcriptBox.value.trim()) {
      toast('this model needs a transcript', true);
      return;
    }
    const duration = trimEnd - trimStart;
    if (duration < MIN_S || duration > MAX_S) {
      toast(`trim to between ${MIN_S} and ${MAX_S} seconds`, true);
      return;
    }
    const form = new FormData();
    form.append('name', name);
    form.append('model', selectedModel);
    form.append('text', transcriptBox.value);
    form.append('file', trimmedWavBlob(), `${name}.wav`);
    try {
      await request('/v1/audio/voices', { method: 'POST', body: form });
      toast(`${name} created`);
    } catch {
      return;
    }
    location.hash = `#voices?model=${selectedModel}`;
  });

  async function refresh() {
    if (disposed) return;
    try {
      models = (await getJson('/v1/models')).data.filter((m) => m.x_kind === 'tts' && m.x_clone);
    } catch {
      return;
    }
    if (disposed) return;
    if (!selectedModel && models.length) selectedModel = models[0].id;
    renderModelBar();
  }

  window.naruAdmin = window.naruAdmin ?? {};
  window.naruAdmin.clone = { loadFile, pickSample };

  renderSampleSelect();
  refresh();
  refreshSamples();
  return () => {
    disposed = true;
    if (window.naruAdmin) delete window.naruAdmin.clone;
  };
}
