// Sample Studio (naru task 1571): perfects one prepped sample before it is
// used as a clone sample. Reached from the samples tab, not the nav bar:
// `#studio?sample=<id>[&draft=1][&model=<model>]`. `draft=1` means the
// samples tab just made the sample for this screen, so saving replaces it
// instead of leaving a second copy behind.
//
// The sample's source clip + range are re-cropped by
// `POST /v1/audio/prep/clips/{id}/render` with the chain on screen, so every
// edit (toggle, knob, reorder, add/remove) re-renders the whole chain after a
// short pause; Solo / Hear render `solo: N` / `until: N` and play the result
// without touching the waveform. The transcript is the sample's verbatim one
// (`/v1/audio/samples/{id}/transcript`, run on first open); its word times
// belong to the saved sample's audio, so they are placed proportionally on
// whatever length is on screen. "Use as clone sample" posts a new sample with
// the chain and the corrected transcript, then opens the clone tab on it.
//
// `#studio?clip=<id>[&model=<model>]` (naru task 1576) opens a raw uploaded
// clip of any length as a sample *project*: `studio-clip.mjs` is the editor
// on top (waveform, speakers, crosstalk, best takes, crop/cuts, undo, saved
// project) and this file renders its kept segments through the same chain
// (`render` with `segments`), as long as they fit `RENDER_MAX_S`. The words
// are the clip transcript's, within the kept segments, placed on the result's
// timeline; the project stores the fixes by word start time.

import { getJson, postJson, request, del } from './api.mjs';
import { el, toast, decodeWav, peaks } from './ui.mjs';
import { createClipEditor, CLONE_MIN_S, CLONE_MAX_S } from './studio-clip.mjs';
import { totalSecs } from './studio-edit.mjs';

const RENDER_DEBOUNCE_MS = 500;
/** Longest kept audio a clip project renders through the chain for preview (the browser decodes the result). */
const RENDER_MAX_S = 60;
/** The words list is a DOM node per word; past this it is not shown. */
const WORDS_MAX = 400;
const BAR_PX = 5;
const KNOB_DRAG_PX = 160; // pixels of vertical drag for a knob's full range
// Meter verdict thresholds (the clone-sample ideal).
const LUFS_OK = [-24, -14];
const PEAK_OK_MAX = -0.5;
const NOISE_OK_MAX = -50;
const SPEECH_OK = [5, 15];

function fmtClock(secs) {
  const s = Math.max(0, secs || 0);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}

function polar(cx, cy, r, deg) {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

function arcPath(cx, cy, r, from, to) {
  const [x0, y0] = polar(cx, cy, r, from);
  const [x1, y1] = polar(cx, cy, r, to);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)}A${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/** A knob's value the way the mockup prints it: `6.5k`, `-45`, `65%`, `120ms`. */
function fmtValue(v, unit) {
  if (unit === 'Hz') return v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : `${Math.round(v)}Hz`;
  if (unit === 'ratio' || unit === 'probability') return `${v.toFixed(unit === 'ratio' && v > 1 ? 1 : 2)}`;
  if (unit === 'ms') return `${Math.round(v)}ms`;
  if (unit === 'semitones') return `${v > 0 ? '+' : ''}${v.toFixed(1)}st`;
  return `${v.toFixed(Math.abs(v) < 10 ? 1 : 0)}`;
}

function paramLabel(name) {
  return name.replace(/_(hz|db|ms|lufs)$/, '').replace(/_/g, ' ').toUpperCase();
}

function stepTitle(type) {
  return type.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
}

export function mount(view, params) {
  let disposed = false;
  const sampleId = params.sample ?? null;
  const clipId = sampleId ? null : (params.clip ?? null);
  const clipMode = !!clipId;
  let editor = null; // clip mode: the raw-clip editor
  let fixes = new Map(); // clip mode: transcript fixes by word start time
  const isDraft = params.draft === '1';
  const cloneModel = params.model ?? null;

  let meta = null; // GET /v1/audio/samples/{id}
  let catalogue = null; // GET /v1/audio/prep/steps
  let steps = []; // the chain on screen
  let rawWave = null; // {sampleRate, samples}
  let rawObjectUrl = null;
  let procWave = null;
  let rawUrl = null;
  let procUrl = null;
  let procObjectUrl = null;
  let analysis = null; // of the full chain, as rendered
  let rawAnalysis = null; // of the crop with no steps, for the "was" hints
  let words = null; // [{start, end, text, value, differs, filler}]
  let wordsDur = 0;
  let transcriptState = 'loading'; // 'loading' | 'ready' | 'failed'
  let ab = 'B';
  let renderGen = 0;
  let renderTimer = null;
  // {kind: 'solo' | 'hear', index, busy}: set from the click until that
  // playback stops; busy while its render is in flight.
  let audition = null;
  let auditionUrl = null;
  let hearGen = 0;
  let hearAbort = null;
  let renderAbort = null;
  let saving = false;
  let dragFrom = null;
  let menuOpen = false;

  // ---- player ---------------------------------------------------------------

  let audio = null;
  let raf = 0;
  let pos = 0; // seconds, for the next play from a stopped state

  function stopAudio() {
    if (audio) {
      audio.pause();
      audio = null;
    }
    cancelAnimationFrame(raf);
    playBtn.textContent = '▶';
    if (auditionUrl) {
      URL.revokeObjectURL(auditionUrl);
      auditionUrl = null;
    }
    if (!audition?.busy) audition = null;
    renderPipeline();
    renderLegend();
  }

  function playUrl(url, from = 0) {
    stopAudio();
    const a = new Audio(url);
    audio = a;
    a.addEventListener('loadedmetadata', () => {
      if (from > 0 && from < a.duration) a.currentTime = from;
    });
    a.addEventListener('ended', () => {
      if (audio === a) {
        pos = 0;
        stopAudio();
        tick();
      }
    });
    a.play().catch(() => {
      if (audio === a) stopAudio();
    });
    playBtn.textContent = '■';
    const loop = () => {
      tick();
      raf = requestAnimationFrame(loop);
    };
    loop();
  }

  function mainUrl() {
    return ab === 'A' ? rawUrl : procUrl;
  }

  function shownWave() {
    return ab === 'A' ? rawWave : procWave;
  }

  function shownDuration() {
    const w = shownWave();
    return w ? w.samples.length / w.sampleRate : 0;
  }

  function tick() {
    const dur = audio && Number.isFinite(audio.duration) ? audio.duration : shownDuration();
    const t = audio ? audio.currentTime : pos;
    const frac = dur ? Math.min(1, t / dur) : 0;
    playhead.style.left = `${frac * 100}%`;
    clock.replaceChildren(el('b', {}, [fmtClock(t)]), ` / ${fmtClock(dur)}`);
  }

  // ---- layout ---------------------------------------------------------------

  const playBtn = el('button', { class: 'studio-play', type: 'button', onclick: onPlay }, ['▶']);
  const clock = el('span', { class: 'studio-clock' });
  const legend = el('div', { class: 'studio-legend' });
  const canvas = el('canvas', { class: 'studio-canvas' });
  const playhead = el('div', { class: 'studio-playhead' });
  const regionHost = el('div', { class: 'studio-regions' });
  const waveWrap = el('div', { class: 'studio-wave', onclick: onWaveClick }, [canvas, regionHost, playhead]);
  const badge = el('div', { class: 'studio-badge' });
  const wordHost = el('div', { class: 'studio-words' });
  const pipelineHost = el('div', { class: 'studio-cards' });
  const meterHost = el('div', { class: 'studio-meters' });
  const subtitle = el('span', { class: 'studio-sub' });
  const abA = el('button', { type: 'button', onclick: () => setAb('A') }, ['A · RAW']);
  const abB = el('button', { type: 'button', onclick: () => setAb('B') }, ['B · PIPELINE']);
  const useBtn = el('button', { class: 'studio-use', type: 'button', onclick: useAsCloneSample }, [
    'USE AS CLONE SAMPLE →',
  ]);
  const clipHost = el('div', { class: 'card studio-card' });

  view.append(
    el('div', { class: 'studio' }, [
      el('div', { class: 'studio-head' }, [
        el('h2', {}, ['SAMPLE STUDIO']),
        subtitle,
        el('div', { class: 'studio-ab' }, [abA, abB]),
        useBtn,
      ]),
      clipMode ? clipHost : null,
      el('div', { class: 'card studio-card' }, [
        clipMode ? el('div', { class: 'lab' }, ['RESULT · THE KEPT AUDIO THROUGH THE PIPELINE']) : null,
        el('div', { class: 'studio-wave-head' }, [playBtn, clock, legend]),
        waveWrap,
        el('div', { class: 'studio-trans-head' }, [
          el('span', { class: 'lab' }, ['TRANSCRIPT · AS SPOKEN · CLICK A WORD TO EDIT']),
          badge,
        ]),
        wordHost,
      ]),
      el('div', { class: 'studio-sect' }, [
        el('h3', {}, ['PIPELINE']),
        el('span', {}, ['one step at a time · drag to reorder · solo to hear just that step']),
      ]),
      pipelineHost,
      meterHost,
    ]),
  );

  function onPlay() {
    if (audio) {
      pos = audio.currentTime;
      stopAudio();
      tick();
      return;
    }
    const url = mainUrl();
    if (url) playUrl(url, pos);
  }

  function onWaveClick(e) {
    const rect = waveWrap.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const dur = audio && Number.isFinite(audio.duration) ? audio.duration : shownDuration();
    pos = frac * dur;
    if (audio) audio.currentTime = pos;
    tick();
  }

  function setAb(next) {
    if (next === ab) return;
    const wasPlaying = !!audio && !audition;
    const t = audio ? audio.currentTime : pos;
    ab = next;
    drawAll();
    if (wasPlaying && mainUrl()) playUrl(mainUrl(), t);
  }

  // ---- waveform + regions ---------------------------------------------------

  function drawWave() {
    const dpr = window.devicePixelRatio || 1;
    const w = waveWrap.clientWidth || 800;
    const h = waveWrap.clientHeight || 120;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const bars = Math.max(1, Math.floor(w / BAR_PX));
    const rawPeaks = rawWave ? peaks(rawWave.samples, bars) : null;
    const procPeaks = procWave ? peaks(procWave.samples, bars) : null;
    const top = Math.max(0.01, ...(rawPeaks ?? []), ...(procPeaks ?? []));
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#00e5ff');
    grad.addColorStop(1, '#b06bff');
    const draw = (list, fill) => {
      ctx.fillStyle = fill;
      list.forEach((p, i) => {
        const bh = Math.max(2, (p / top) * (h - 8));
        ctx.fillRect(i * BAR_PX, (h - bh) / 2, 3, bh);
      });
    };
    if (ab === 'A') {
      if (rawPeaks) draw(rawPeaks, grad);
    } else {
      if (rawPeaks) draw(rawPeaks, 'rgba(93,127,143,0.45)');
      if (procPeaks) draw(procPeaks, grad);
    }
  }

  function renderRegions() {
    regionHost.replaceChildren();
    if (!words || !wordsDur) return;
    for (const w of words) {
      if (!w.differs) continue;
      const left = Math.min(100, (w.start / wordsDur) * 100);
      const width = Math.max(0.6, Math.min(100 - left, ((w.end - w.start) / wordsDur) * 100));
      regionHost.append(el('div', { class: 'studio-region', style: `left:${left}%;width:${width}%` }));
    }
  }

  function renderLegend() {
    const hearing = audition?.kind === 'hear' ? ` ${audition.index + 1}` : '';
    legend.replaceChildren(
      el('span', { class: 'raw' }, ['RAW']),
      el('span', { class: 'proc' }, [audition && !audition.busy && hearing ? `AFTER STEP${hearing}` : 'PIPELINE']),
      el('span', { class: 'look' }, ['NEEDS A LOOK']),
    );
  }

  function drawAll() {
    abA.classList.toggle('on', ab === 'A');
    abB.classList.toggle('on', ab === 'B');
    drawWave();
    renderRegions();
    renderLegend();
    tick();
  }

  // ---- transcript -----------------------------------------------------------

  function unchecked() {
    return words ? words.filter((w) => w.differs).length : 0;
  }

  function renderBadge() {
    const n = unchecked();
    badge.classList.toggle('hidden', !words);
    badge.classList.toggle('ok', n === 0);
    badge.replaceChildren(...(n ? [el('i', {}, [String(n)]), 'SAID ≠ WRITTEN'] : ['✓ ALL CHECKED']));
  }

  function wordNode(w) {
    const node = el(
      'span',
      { class: `studio-word${w.filler ? ' filler' : ''}${w.differs ? ' diff' : ''}`, onclick: () => editWord(w, node) },
      [w.differs ? el('s', {}, [w.text]) : null, w.value],
    );
    return node;
  }

  function editWord(w, node) {
    const input = el('input', { class: 'studio-word-edit', value: w.value, size: Math.max(3, w.value.length + 1) });
    let done = false;
    const commit = (save) => {
      if (done) return;
      done = true;
      if (save) {
        w.value = input.value.trim() || w.value; // emptying a word keeps it
        w.differs = false; // seen and confirmed, edited or not
        if (clipMode) {
          if (w.value === w.text) fixes.delete(w.key);
          else fixes.set(w.key, w.value);
          editor.touch();
        }
      }
      renderWords();
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') commit(true);
      else if (e.key === 'Escape') commit(false);
    });
    input.addEventListener('blur', () => commit(true));
    node.replaceChildren(input);
    input.focus();
    input.select();
  }

  function renderWords() {
    if (clipMode && words && words.length > WORDS_MAX) {
      wordHost.replaceChildren(
        el('span', { class: 'studio-note' }, [`${words.length} words in the kept audio — narrow the edit to read and fix them`]),
      );
    } else if (clipMode && !words) {
      wordHost.replaceChildren(
        el('span', { class: 'studio-note' }, ['no transcript yet — separate speakers (above) to transcribe the clip']),
      );
    } else if (transcriptState === 'loading') {
      wordHost.replaceChildren(el('span', { class: 'studio-note' }, ['transcribing… (first open runs the speech model)']));
    } else if (transcriptState === 'failed') {
      wordHost.replaceChildren(
        el('span', { class: 'studio-note' }, ['transcript unavailable — the sample\'s saved text is used as is. ']),
        el('button', { class: 'btn', type: 'button', onclick: loadTranscript }, ['retry']),
      );
    } else {
      wordHost.replaceChildren(...words.map(wordNode));
    }
    renderBadge();
    renderRegions();
    renderMeters();
  }

  function transcriptText() {
    if (!words) return clipMode ? '' : (meta?.transcript ?? '');
    return words
      .map((w) => w.value)
      .filter(Boolean)
      .join(' ');
  }

  /** Clip mode: the clip transcript's words inside the kept segments, on the result's timeline. */
  function rebuildClipWords() {
    const t = editor.transcript();
    if (!t) {
      words = null;
      transcriptState = 'ready';
      return;
    }
    const kept = editor.kept();
    const out = [];
    let offset = 0;
    let k = 0;
    for (const w of t.words) {
      const mid = (w.start + w.end) / 2;
      while (k < kept.length && kept[k].end <= mid) {
        offset += kept[k].end - kept[k].start;
        k++;
      }
      if (k >= kept.length) break;
      if (mid < kept[k].start) continue;
      const key = w.start.toFixed(2);
      out.push({
        start: offset + Math.max(0, w.start - kept[k].start),
        end: offset + Math.max(0, w.end - kept[k].start),
        text: w.text,
        key,
        value: fixes.get(key) ?? w.text,
        differs: false,
        filler: false,
      });
    }
    words = out;
    wordsDur = totalSecs(kept);
    transcriptState = 'ready';
  }

  async function loadTranscript() {
    transcriptState = 'loading';
    renderWords();
    const base = `/v1/audio/samples/${encodeURIComponent(sampleId)}/transcript`;
    try {
      let t;
      try {
        t = await getJson(base, { silent: true });
      } catch (e) {
        if (e.code !== 'transcript_not_run') throw e;
        t = await postJson(`/v1/audio/samples/${encodeURIComponent(sampleId)}/transcribe`, {});
      }
      if (disposed) return;
      words = t.words.map((w) => ({ ...w, value: w.spoken ?? w.text }));
      wordsDur = procWave ? procWave.samples.length / procWave.sampleRate : (t.words.at(-1)?.end ?? 0);
      transcriptState = 'ready';
    } catch {
      if (disposed) return;
      words = null;
      transcriptState = 'failed';
    }
    renderWords();
  }

  // ---- meters ---------------------------------------------------------------

  function meter(label, value, unit, fill, verdict, hint) {
    return el('div', { class: `studio-meter ${verdict ?? ''}` }, [
      el('div', { class: 'lab' }, [label]),
      el('div', { class: 'val' }, [value, el('small', {}, [unit])]),
      el('div', { class: 'track' }, [el('i', { style: `width:${Math.round(Math.min(1, Math.max(0, fill)) * 100)}%` })]),
      el('div', { class: 'hint' }, [`${verdict === 'ok' ? '✓' : verdict === 'warn' ? '!' : ''} ${hint}`.trim()]),
    ]);
  }

  const num = (v) => typeof v === 'number' && Number.isFinite(v);

  function renderMeters() {
    const a = analysis;
    const n = unchecked();
    const lufs = a?.integrated_lufs;
    const peak = a?.peak_dbfs;
    const noise = a?.noise_floor_dbfs;
    const speech = a?.speech_secs;
    const was = rawAnalysis?.noise_floor_dbfs;
    meterHost.replaceChildren(
      meter(
        'LOUDNESS',
        num(lufs) ? lufs.toFixed(1) : '—',
        'LUFS',
        num(lufs) ? (lufs + 40) / 40 : 0,
        num(lufs) ? (lufs >= LUFS_OK[0] && lufs <= LUFS_OK[1] ? 'ok' : 'warn') : null,
        num(lufs) ? (lufs >= LUFS_OK[0] && lufs <= LUFS_OK[1] ? 'clone range' : `outside ${LUFS_OK[0]}..${LUFS_OK[1]}`) : 'silent',
      ),
      meter(
        'PEAK',
        num(peak) ? peak.toFixed(1) : '—',
        'dBFS',
        num(peak) ? (peak + 40) / 40 : 0,
        num(peak) ? (peak <= PEAK_OK_MAX ? 'ok' : 'warn') : null,
        num(peak) ? (peak <= PEAK_OK_MAX ? 'no clipping' : 'close to clipping') : 'silent',
      ),
      meter(
        'NOISE FLOOR',
        num(noise) ? noise.toFixed(0) : '—',
        'dB',
        num(noise) ? (noise + 90) / 70 : 0,
        num(noise) ? (noise <= NOISE_OK_MAX ? 'ok' : 'warn') : null,
        num(was) ? `was ${was.toFixed(0)}` : num(noise) ? (noise <= NOISE_OK_MAX ? 'quiet' : 'noisy') : 'silent',
      ),
      meter(
        'SPEECH',
        num(speech) ? speech.toFixed(1) : '—',
        num(a?.duration_secs) ? `s of ${a.duration_secs.toFixed(1)}` : 's',
        num(speech) && num(a?.duration_secs) ? speech / a.duration_secs : 0,
        num(speech) ? (speech >= SPEECH_OK[0] && speech <= SPEECH_OK[1] ? 'ok' : 'warn') : null,
        num(speech) ? `${SPEECH_OK[0]}-${SPEECH_OK[1]} s ideal` : 'needs the silero-vad model',
      ),
      meter(
        'TRANSCRIPT MATCH',
        words ? String(n) : '—',
        words ? 'words to check' : '',
        words && words.length ? 1 - n / words.length : 0,
        words ? (n === 0 ? 'ok' : 'warn') : null,
        words ? (n === 0 ? 'all checked' : 'review highlights') : 'no word transcript',
      ),
    );
  }

  // ---- pipeline -------------------------------------------------------------

  function numericSpecs(type) {
    const entry = catalogue.steps.find((s) => s.type === type);
    return (entry?.params ?? []).filter((p) => p.type === 'number');
  }

  /** The knobs a step card shows: its numeric params, or an EQ's band gains. */
  function knobsFor(step) {
    if (step.type === 'eq') {
      const item = catalogue.steps.find((s) => s.type === 'eq').params.find((p) => p.name === 'bands').items;
      const gain = item.find((p) => p.name === 'gain_db');
      return step.bands.slice(0, 3).map((band) => ({
        label: fmtValue(band.freq_hz, 'Hz'),
        spec: gain,
        get: () => band.gain_db,
        set: (v) => (band.gain_db = v),
      }));
    }
    return numericSpecs(step.type).map((spec) => ({
      label: paramLabel(spec.name),
      spec,
      get: () => step[spec.name],
      set: (v) => (step[spec.name] = v),
    }));
  }

  function knobNode(knob, onCommit) {
    const { spec } = knob;
    const span = spec.max - spec.min;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 44 44');
    svg.setAttribute('class', 'studio-knob-svg');
    const valueText = el('div', { class: 'v' });
    const paint = () => {
      const frac = Math.min(1, Math.max(0, (knob.get() - spec.min) / span));
      svg.innerHTML =
        `<path d="${arcPath(22, 22, 17, 135, 405)}" class="trk"/>` +
        (frac > 0.002 ? `<path d="${arcPath(22, 22, 17, 135, 135 + 270 * frac)}" class="val"/>` : '');
      valueText.textContent = fmtValue(knob.get(), spec.unit);
    };
    paint();
    const wrap = el('div', { class: 'studio-knob', title: `${spec.name} (${spec.min}..${spec.max} ${spec.unit}) — drag, double-click resets` }, [
      svg,
      valueText,
      el('div', { class: 'l' }, [knob.label]),
    ]);
    svg.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      svg.setPointerCapture(e.pointerId);
      const y0 = e.clientY;
      const v0 = knob.get();
      const move = (ev) => {
        const raw = v0 + ((y0 - ev.clientY) / KNOB_DRAG_PX) * span;
        knob.set(Math.min(spec.max, Math.max(spec.min, Math.round(raw * 1000) / 1000)));
        paint();
      };
      const up = () => {
        svg.removeEventListener('pointermove', move);
        svg.removeEventListener('pointerup', up);
        svg.removeEventListener('pointercancel', up);
        svg.removeEventListener('lostpointercapture', up);
        if (knob.get() !== v0) onCommit();
      };
      svg.addEventListener('pointermove', move);
      svg.addEventListener('pointerup', up);
      svg.addEventListener('pointercancel', up);
      svg.addEventListener('lostpointercapture', up);
    });
    svg.addEventListener('dblclick', () => {
      knob.set(spec.default);
      paint();
      onCommit();
    });
    return wrap;
  }

  function engineTag(step) {
    const engine = meta.engines?.find((e) => e.stage === step.type);
    return step.model ?? engine?.model ?? null;
  }

  function cardNode(step, i) {
    const knobs = knobsFor(step);
    const tag = engineTag(step);
    const isSolo = audition?.kind === 'solo' && audition.index === i;
    const isHear = audition?.kind === 'hear' && audition.index === i;
    const actionBtn = (kind, text, active) =>
      el(
        'button',
        {
          type: 'button',
          class: `studio-act${active ? ' on' : ''}`,
          disabled: (audition?.busy && audition.index === i && audition.kind === kind) || undefined,
          onclick: () => hearStep(kind, i),
        },
        [audition?.busy && audition.index === i && audition.kind === kind ? '…' : text],
      );
    const card = el(
      'div',
      {
        class: `studio-cardstep${step.enabled ? '' : ' off'}${isSolo || isHear ? ' hot' : ''}`,
        draggable: 'true',
        style: `min-width:${Math.max(180, 56 + knobs.length * 58)}px`,
        ondragstart: (e) => {
          dragFrom = i;
          e.dataTransfer.effectAllowed = 'move';
          e.dataTransfer.setData('text/plain', String(i));
        },
        ondragover: (e) => e.preventDefault(),
        ondrop: (e) => {
          e.preventDefault();
          if (dragFrom == null || dragFrom === i) return;
          const [moved] = steps.splice(dragFrom, 1);
          steps.splice(i, 0, moved);
          dragFrom = null;
          chainChanged();
        },
      },
      [
        el('div', { class: 'top' }, [
          el('span', { class: 'grip' }, ['⠿']),
          el('span', { class: 'n' }, [String(i + 1)]),
          el('b', {}, [stepTitle(step.type)]),
          el('button', {
            class: `studio-switch${step.enabled ? ' on' : ''}`,
            type: 'button',
            title: step.enabled ? 'on' : 'off',
            onclick: () => {
              step.enabled = !step.enabled;
              chainChanged();
            },
          }),
        ]),
        tag ? el('div', { class: 'tag' }, [tag]) : null,
        el('div', { class: 'knobs' }, knobs.length ? knobs.map((k) => knobNode(k, chainChanged)) : [el('div', { class: 'studio-note' }, ['no settings'])]),
        el('div', { class: 'acts' }, [
          actionBtn('solo', 'SOLO', isSolo),
          actionBtn('hear', '▶ HEAR', isHear),
          el('button', {
            type: 'button',
            class: 'studio-act x',
            title: 'remove step',
            onclick: () => {
              steps.splice(i, 1);
              chainChanged();
            },
          }, ['✕']),
        ]),
      ],
    );
    return card;
  }

  function availableTypes() {
    const used = new Set(steps.map((s) => s.type));
    return catalogue.steps.filter((s) => !used.has(s.type));
  }

  function newStep(type) {
    const entry = catalogue.steps.find((s) => s.type === type);
    const step = { type, enabled: true };
    for (const p of entry.params) {
      if (p.type === 'number') step[p.name] = p.default;
    }
    if (type === 'eq') {
      const items = entry.params.find((p) => p.name === 'bands').items;
      const dflt = (name) => items.find((p) => p.name === name).default;
      const band = (kind, freq_hz) => ({ kind, freq_hz, gain_db: 0, q: dflt('q') });
      step.bands = [band('low_shelf', 120), band('peak', 3000), band('high_shelf', 8000)];
    }
    return step;
  }

  function renderPipeline() {
    if (!catalogue) return;
    const free = availableTypes();
    const menu = menuOpen
      ? el(
          'div',
          { class: 'studio-menu' },
          free.length
            ? free.map((t) =>
                el(
                  'div',
                  {
                    class: 'item',
                    title: t.description,
                    onclick: () => {
                      menuOpen = false;
                      steps.push(newStep(t.type));
                      chainChanged();
                    },
                  },
                  [stepTitle(t.type)],
                ),
              )
            : [el('div', { class: 'studio-note' }, ['every effect is in the chain'])],
        )
      : null;
    pipelineHost.replaceChildren(
      ...steps.flatMap((s, i) => [cardNode(s, i), el('span', { class: 'arrow' }, ['→'])]),
      el(
        'div',
        {
          class: 'studio-add',
          onclick: () => {
            menuOpen = !menuOpen;
            renderPipeline();
          },
        },
        [el('b', {}, ['+']), el('span', {}, ['ADD STEP']), menu],
      ),
    );
  }

  // ---- rendering ------------------------------------------------------------

  /** An aborted render (superseded by a newer one) rejects without a toast. */
  async function renderChain(extra = {}, signal) {
    if (clipMode && totalSecs(editor.kept()) > RENDER_MAX_S) {
      const err = new Error(`kept audio is over ${RENDER_MAX_S} s — narrow it first`);
      toast(err.message, true);
      throw err;
    }
    let res;
    try {
      res = await request(
        `/v1/audio/prep/clips/${encodeURIComponent(meta.source_clip_id)}/render`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...(clipMode ? { segments: editor.kept() } : { start: meta.range.start, end: meta.range.end }),
            steps,
            ...extra,
          }),
          signal,
        },
        { silent: true },
      );
    } catch (e) {
      if (!signal?.aborted) toast(e.message, true);
      throw e;
    }
    const parsed = JSON.parse(res.headers.get('x-naru-analysis') ?? 'null');
    return { buffer: await res.arrayBuffer(), analysis: parsed };
  }

  function chainChanged() {
    renderPipeline();
    editor?.touch();
    clearTimeout(renderTimer);
    renderTimer = setTimeout(renderFull, RENDER_DEBOUNCE_MS);
    setStatus('chain changed — rendering…');
  }

  /** Length of the processed audio on screen (the saved sample's before it loads). */
  function audioSecs() {
    if (clipMode) return totalSecs(editor?.kept() ?? []);
    if (procWave) return procWave.samples.length / procWave.sampleRate;
    return meta?.analysis?.duration_secs ?? (meta ? meta.range.end - meta.range.start : 0);
  }

  function setStatus(text) {
    subtitle.replaceChildren(
      el('b', {}, [(clipMode ? editor?.name() || editor?.clip()?.original_filename : meta?.name) ?? '']),
      ` · ${audioSecs().toFixed(1)} s · 16 kHz mono`,
      ...(text ? [el('em', {}, [` · ${text}`])] : []),
    );
  }

  async function renderFull() {
    const gen = ++renderGen;
    renderAbort?.abort();
    const ctl = new AbortController();
    renderAbort = ctl;
    let out;
    if (clipMode) {
      const kept = editor.kept();
      if (!kept.length || totalSecs(kept) > RENDER_MAX_S) {
        // Too long to preview in the browser (or nothing kept): leave the result empty until the edit narrows.
        renderAbort = null;
        procWave = rawWave = null;
        procUrl = rawUrl = null;
        analysis = rawAnalysis = null;
        if (audio && !audition) stopAudio();
        setStatus(kept.length ? `kept audio is over ${RENDER_MAX_S} s — narrow it to preview the pipeline` : 'nothing kept');
        drawAll();
        renderMeters();
        return;
      }
    }
    try {
      out = await renderChain({}, ctl.signal);
    } catch {
      if (gen === renderGen && !disposed) setStatus('render failed');
      return;
    }
    if (disposed || gen !== renderGen) return;
    renderAbort = null;
    if (clipMode) {
      // The kept audio with no steps: the A side, and the "was" hints.
      const rawOut = await renderChain({ steps: [] }, ctl.signal).catch(() => null);
      if (disposed || gen !== renderGen) return;
      if (rawOut) {
        rawWave = decodeWav(rawOut.buffer);
        if (rawObjectUrl) URL.revokeObjectURL(rawObjectUrl);
        rawObjectUrl = URL.createObjectURL(new Blob([rawOut.buffer], { type: 'audio/wav' }));
        rawUrl = rawObjectUrl;
        rawAnalysis = rawOut.analysis;
      }
    }
    procWave = decodeWav(out.buffer);
    if (procObjectUrl) URL.revokeObjectURL(procObjectUrl);
    procObjectUrl = URL.createObjectURL(new Blob([out.buffer], { type: 'audio/wav' }));
    procUrl = procObjectUrl;
    analysis = out.analysis;
    if (audio && !audition && ab === 'B') stopAudio();
    setStatus('');
    drawAll();
    renderMeters();
  }

  async function hearStep(kind, i) {
    const wasThis = audition && !audition.busy && audition.kind === kind && audition.index === i;
    const gen = ++hearGen;
    hearAbort?.abort();
    hearAbort = null;
    audition = null;
    stopAudio();
    if (wasThis) return;
    audition = { kind, index: i, busy: true };
    renderPipeline();
    const ctl = new AbortController();
    hearAbort = ctl;
    let out;
    try {
      out = await renderChain(kind === 'solo' ? { solo: i } : { until: i }, ctl.signal);
    } catch {
      if (gen === hearGen && !disposed) {
        audition = null;
        renderPipeline();
      }
      return;
    }
    if (gen !== hearGen || disposed) return;
    hearAbort = null;
    const url = URL.createObjectURL(new Blob([out.buffer], { type: 'audio/wav' }));
    playUrl(url); // its stopAudio() leaves the busy audition alone
    auditionUrl = url;
    audition = { kind, index: i, busy: false };
    renderPipeline();
    renderLegend();
  }

  // ---- save -----------------------------------------------------------------

  async function useClipAsCloneSample() {
    const name = editor.name();
    if (!name) {
      toast('name the project first', true);
      return;
    }
    const kept = editor.kept();
    const secs = totalSecs(kept);
    if (secs < CLONE_MIN_S || secs > CLONE_MAX_S) {
      toast(`a clone sample needs ${CLONE_MIN_S}–${CLONE_MAX_S} s; ${secs.toFixed(1)} s is kept`, true);
      return;
    }
    saving = true;
    useBtn.disabled = true;
    stopAudio();
    clearTimeout(renderTimer);
    try {
      await editor.save();
      const created = await postJson('/v1/audio/samples', {
        clip_id: clipId,
        name,
        segments: kept,
        steps,
        transcript: transcriptText(),
      });
      toast(`${name} saved as a clone sample`);
      const modelPart = cloneModel ? `model=${encodeURIComponent(cloneModel)}&` : '';
      location.hash = `#clone?${modelPart}sample=${encodeURIComponent(created.id)}`;
    } catch {
      saving = false;
      useBtn.disabled = false;
    }
  }

  async function useAsCloneSample() {
    if (clipMode) {
      if (!saving && editor?.clip()) await useClipAsCloneSample();
      return;
    }
    if (saving || !meta) return;
    saving = true;
    useBtn.disabled = true;
    stopAudio();
    clearTimeout(renderTimer);
    try {
      const created = await postJson('/v1/audio/samples', {
        clip_id: meta.source_clip_id,
        name: meta.name,
        start: meta.range.start,
        end: meta.range.end,
        speaker: meta.speaker ?? undefined,
        steps,
        transcript: transcriptText(),
      });
      if (isDraft) await del(`/v1/audio/samples/${encodeURIComponent(sampleId)}`).catch(() => {});
      toast(`${meta.name} saved as a clone sample`);
      const modelPart = cloneModel ? `model=${encodeURIComponent(cloneModel)}&` : '';
      location.hash = `#clone?${modelPart}sample=${encodeURIComponent(created.id)}`;
    } catch {
      saving = false;
      useBtn.disabled = false;
    }
  }

  // ---- boot -----------------------------------------------------------------

  async function bootClip() {
    try {
      catalogue = await getJson('/v1/audio/prep/steps');
      editor = createClipEditor({
        clipId,
        getExtras: () => ({ steps, transcript: transcriptText(), fixes: Object.fromEntries(fixes) }),
        onChange: clipChanged,
      });
      clipHost.append(el('div', { class: 'lab' }, ['CLIP EDITOR · RAW AUDIO']), editor.root);
      await editor.ready;
    } catch {
      if (!disposed) {
        view.replaceChildren(el('div', { class: 'placeholder' }, ['could not open that clip — ', el('a', { href: '#samples' }, ['back to samples'])]));
      }
      return;
    }
    if (disposed) return;
    const project = editor.project();
    steps = structuredClone(project?.steps?.length ? project.steps : catalogue.default_chain);
    const saved = project?.takes?.fixes;
    if (saved && typeof saved === 'object') fixes = new Map(Object.entries(saved));
    meta = { name: editor.name(), source_clip_id: clipId, engines: [] };
    renderPipeline();
    clipChanged();
  }

  /** Clip mode: the edit (or the transcript) changed what is kept. */
  function clipChanged() {
    if (!meta || disposed) return;
    rebuildClipWords();
    renderWords();
    clearTimeout(renderTimer);
    renderTimer = setTimeout(renderFull, RENDER_DEBOUNCE_MS);
    setStatus('rendering…');
  }

  async function boot() {
    if (clipMode) return bootClip();
    if (!sampleId) {
      view.replaceChildren(el('div', { class: 'placeholder' }, ['open a sample from the ', el('a', { href: '#samples' }, ['samples tab'])]));
      return;
    }
    try {
      [meta, catalogue] = await Promise.all([
        getJson(`/v1/audio/samples/${encodeURIComponent(sampleId)}`),
        getJson('/v1/audio/prep/steps'),
      ]);
    } catch {
      return;
    }
    if (disposed) return;
    steps = structuredClone(meta.steps?.length ? meta.steps : catalogue.default_chain);
    analysis = meta.analysis;
    rawUrl = `/v1/audio/samples/${encodeURIComponent(sampleId)}/audio?variant=cropped`;
    procUrl = `/v1/audio/samples/${encodeURIComponent(sampleId)}/audio?variant=clean`;
    setStatus('');
    renderPipeline();
    renderMeters();
    try {
      const [raw, proc] = await Promise.all(
        [rawUrl, procUrl].map((u) => fetch(u).then((r) => r.arrayBuffer()).then(decodeWav)),
      );
      rawWave = raw;
      procWave = proc;
    } catch {
      toast('could not load the sample audio', true);
    }
    if (disposed) return;
    setStatus('');
    drawAll();
    loadTranscript();
    // The crop with no steps: what the audio was before the chain ("was -41").
    renderChain({ steps: [] }).then(
      (out) => {
        rawAnalysis = out.analysis;
        renderMeters();
      },
      () => {},
    );
  }

  const onResize = () => drawWave();
  window.addEventListener('resize', onResize);
  renderWords();
  renderLegend();
  tick();
  boot();

  return () => {
    disposed = true;
    clearTimeout(renderTimer);
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', onResize);
    renderAbort?.abort();
    hearAbort?.abort();
    if (audio) audio.pause();
    if (auditionUrl) URL.revokeObjectURL(auditionUrl);
    if (procObjectUrl) URL.revokeObjectURL(procObjectUrl);
    if (rawObjectUrl) URL.revokeObjectURL(rawObjectUrl);
    editor?.dispose();
  };
}
