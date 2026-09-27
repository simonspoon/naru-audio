// Samples tab (naru task 1462, board 126): the voice-prep pipeline (feat
// 0315e3b, `/v1/audio/prep/clips*` and `/v1/audio/samples*`) turns a raw
// upload into a model-agnostic clean clip. Library view lists what's
// already prepped (name, length, cleaned-or-raw); "+ new clip" walks
// upload -> transcript (words + speakers) -> crop (click words or drag the
// waveform) -> speaker -> isolate + clean -> A/B preview -> keep/discard.
//
// Exposes `window.naruAdmin.samples.loadFile(file)` so a headless test
// without a file picker can drive the upload path directly.
//
// `#samples?clip=<id>&return=clone&model=<model>` (naru task 1463): the
// clone tab routes a raw drop/recording here first instead of trimming it
// itself. `clip` opens the prep flow directly on that already-uploaded
// clip (reusing `loadClipWave`/`runTranscribe`, no re-upload); once the
// user keeps the processed sample, `return=clone` sends them back to
// `#clone?model=<model>&sample=<new id>` instead of the library.

import { getJson, postJson, request, del } from './api.mjs';
import { el, toast, decodeWav, waveformSvg, peaks, playButton } from './ui.mjs';

const SPEAKER_COLORS = ['var(--cyan)', 'var(--violet)', 'var(--amber)', 'var(--green)', 'var(--magenta)'];

function fmtMmSs(secs) {
  if (secs == null || Number.isNaN(secs)) return '—';
  const s = Math.max(0, Math.round(secs));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

function speakerColor(id) {
  return SPEAKER_COLORS[((id % SPEAKER_COLORS.length) + SPEAKER_COLORS.length) % SPEAKER_COLORS.length];
}

/** Which diarized speaker (if any) a word's midpoint falls inside. */
function speakerForWord(word, speakers) {
  const mid = (word.start + word.end) / 2;
  const span = speakers.find((s) => mid >= s.start && mid < s.end);
  return span ? span.speaker : null;
}

export function mount(view, params) {
  let disposed = false;

  // ---- library state ------------------------------------------------------
  let samples = []; // [{id, name, range, engines, ...}]
  let mode = 'library'; // 'library' | 'prep'
  let selectedId = params.sample ?? null;
  let renameValue = '';
  // Set when we arrived from the clone tab's raw-upload hand-off: keeping
  // the processed sample sends the user back there instead of the library.
  const returnTo = params.return === 'clone' ? { model: params.model ?? null } : null;

  // ---- prep-flow state ------------------------------------------------------
  let clip = null; // ClipMeta json
  let transcript = null; // {words, speakers}
  let clipWave = null; // {sampleRate, samples} decoded working.wav, for the crop fallback
  let wordSel = null; // {start: idx, end: idx} into transcript.words
  let wordAnchor = null;
  let range = null; // {start, end} in seconds, the crop actually used
  let selectedSpeaker = null;
  let flags = { isolate: true, denoise: true, trim_silence: true, normalize: true };
  let sampleName = '';
  let created = null; // SampleMeta json once POST /v1/audio/samples has run
  let busy = false;
  // Bumped by every entry point into the prep flow (startPrep, loadFile,
  // startPrepForClip) so a stale continuation — e.g. params.clip's own
  // fetch still in flight when the user hits "+ new clip" or drops a file
  // — can tell it is no longer the current attempt and bail out instead of
  // overwriting clip/clipWave/transcript out from under the newer one.
  let prepGen = 0;

  const listCard = el('div', { class: 'card' });
  const sidePanel = el('div', { class: 'card' });
  const prepCard = el('div', { class: 'card' });
  // Persistent across renderPrep() calls (unlike prepCard's other children,
  // which are rebuilt wholesale each render): a drag needs the wrap it
  // measures with getBoundingClientRect() to stay attached to the DOM for
  // the whole gesture, the same way clone.mjs's own `waveWrap` does.
  const waveWrap = el('div', { class: 'trim-wrap' });
  view.append(el('div', { class: 'side' }, [listCard, sidePanel]));

  // ---- library --------------------------------------------------------------

  function statusOf(s) {
    return s.engines?.length ? '✓ cleaned' : 'raw, not cleaned yet';
  }

  /** `node.replaceChildren(...)` is the native DOM method, not `el()`'s own
   * child list (ui.mjs:21, which drops `null`/`undefined` entries) — passed
   * a bare `null` from a `cond ? el(...) : null` ternary, it stringifies it
   * to the literal text "null" instead of skipping it. Every top-level
   * `replaceChildren` call in this tab builds its list that way, so they all
   * go through this filter (naru task 1462 review). */
  function setChildren(node, children) {
    node.replaceChildren(...children.filter((c) => c != null));
  }

  /** A `.row` that hosts a `playButton()` needs `position:relative` — `.play`
   * positions itself `absolute` against the nearest positioned ancestor, and
   * plain `.row` (unlike `.vc`/`.cmp`, which other tabs use for this) has
   * none, so without this every play button on the tab stacks at the page's
   * top-right corner instead of next to its own label. */
  function playRow(getSrc, label) {
    return el('div', { class: 'row', style: 'position:relative;padding-right:30px' }, [playButton(getSrc), label]);
  }

  function libraryRow(s) {
    const tr = el(
      'div',
      { class: `row${s.id === selectedId ? ' sel' : ''}`, style: 'cursor:pointer', onclick: () => selectSample(s.id) },
      [
        el('b', {}, [s.name]),
        el('span', { style: 'color:var(--muted);width:50px;text-align:right' }, [
          fmtMmSs(s.range ? s.range.end - s.range.start : null),
        ]),
        el('span', { style: `color:${s.engines?.length ? 'var(--green)' : 'var(--amber)'};font-size:11px` }, [
          statusOf(s),
        ]),
      ],
    );
    return tr;
  }

  function renderList() {
    setChildren(listCard, [
      el('h3', {}, ['SAMPLES']),
      el(
        'button',
        { class: 'big', onclick: startPrep },
        ['+ new clip'],
      ),
      ...samples.map(libraryRow),
      samples.length ? null : el('div', { class: 'placeholder' }, ['no samples yet']),
    ]);
  }

  function kv(label, value) {
    return el('div', { class: 'kv' }, [el('span', {}, [label]), el('span', {}, [String(value)])]);
  }

  function renderSide() {
    if (mode === 'prep') {
      sidePanel.replaceChildren(prepCard);
      renderPrep();
      return;
    }
    const s = samples.find((x) => x.id === selectedId);
    if (!s) {
      sidePanel.replaceChildren(el('div', { class: 'placeholder' }, ['select a sample']));
      return;
    }
    renameValue = renameValue || s.name;
    const renameInput = el('input', { class: 'inp', value: renameValue, oninput: (e) => (renameValue = e.target.value) });
    const cleanUrl = `/v1/audio/samples/${encodeURIComponent(s.id)}/audio?variant=clean`;
    const rawUrl = `/v1/audio/samples/${encodeURIComponent(s.id)}/audio?variant=cropped`;
    setChildren(sidePanel, [
      el('h3', {}, [s.name]),
      kv('length', fmtMmSs(s.range.end - s.range.start)),
      kv('status', statusOf(s)),
      kv('speaker', s.speaker ?? '—'),
      s.warnings?.length ? el('div', { class: 'warn' }, [s.warnings.join('; ')]) : null,
      el('div', { class: 'lab', style: 'margin-top:10px' }, ['A/B PREVIEW']),
      playRow(() => rawUrl, 'raw crop'),
      playRow(() => cleanUrl, 'cleaned'),
      el('div', { class: 'lab', style: 'margin-top:10px' }, ['RENAME']),
      el('div', { class: 'row' }, [
        renameInput,
        el('button', { class: 'btn', onclick: () => renameSample(s) }, ['save']),
      ]),
      el('button', { class: 'big r', onclick: () => deleteSample(s) }, ['DELETE']),
    ]);
  }

  function selectSample(id) {
    mode = 'library';
    selectedId = id;
    renameValue = '';
    renderList();
    renderSide();
  }

  async function renameSample(s) {
    const name = renameValue.trim();
    if (!name || name === s.name) return;
    try {
      await request(`/v1/audio/samples/${encodeURIComponent(s.id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      toast(`renamed to ${name}`);
    } catch {
      return;
    }
    refresh();
  }

  async function deleteSample(s) {
    if (!confirm(`Delete sample "${s.name}"? This cannot be undone.`)) return;
    try {
      await del(`/v1/audio/samples/${encodeURIComponent(s.id)}`);
      toast(`${s.name} deleted`);
    } catch {
      return;
    }
    selectedId = null;
    refresh();
  }

  // ---- prep flow: 1. upload --------------------------------------------------

  function resetPrep() {
    clip = null;
    transcript = null;
    clipWave = null;
    wordSel = null;
    wordAnchor = null;
    range = null;
    selectedSpeaker = null;
    flags = { isolate: true, denoise: true, trim_silence: true, normalize: true };
    sampleName = '';
    created = null;
    busy = false;
  }

  function startPrep() {
    mode = 'prep';
    prepGen++;
    resetPrep();
    renderList();
    renderSide();
  }

  /** Opens the prep flow on a clip the clone tab already uploaded
   * (`#samples?clip=<id>`, naru task 1463) — reuses `loadClipWave`/
   * `runTranscribe` same as a fresh upload, but skips `loadFile`'s own
   * `POST /v1/audio/prep/clips` since the clip already exists.
   *
   * This runs automatically from `params.clip` and can race a user who
   * hits "+ new clip" or drops a file while its own fetch is still in
   * flight; `gen` lets it notice it's been superseded and bail rather than
   * overwrite `clip`/`clipWave`/`transcript` out from under the newer
   * attempt. */
  async function startPrepForClip(clipId) {
    mode = 'prep';
    const gen = ++prepGen;
    resetPrep();
    renderList();
    renderSide();
    busy = true;
    renderPrep();
    let fetchedClip;
    try {
      fetchedClip = await getJson(`/v1/audio/prep/clips/${encodeURIComponent(clipId)}`);
    } catch {
      if (gen !== prepGen) return;
      busy = false;
      renderPrep();
      return;
    }
    if (gen !== prepGen) return;
    clip = fetchedClip;
    busy = false;
    await loadClipWave();
    if (gen !== prepGen) return;
    await runTranscribe();
  }

  function backToLibrary() {
    mode = 'library';
    renderList();
    refresh();
  }

  /** "Keep" on a freshly processed sample: back to the library normally,
   * or back to the clone tab with this sample preselected when we arrived
   * via its raw-upload hand-off (naru task 1463). */
  function keepSample() {
    if (returnTo && created) {
      const modelPart = returnTo.model ? `model=${encodeURIComponent(returnTo.model)}&` : '';
      location.hash = `#clone?${modelPart}sample=${encodeURIComponent(created.id)}`;
      return;
    }
    backToLibrary();
  }

  async function loadFile(file) {
    prepGen++;
    const form = new FormData();
    form.append('file', file, file.name || 'clip');
    busy = true;
    renderPrep();
    try {
      // A multipart upload needs `request` directly; `postJson` always sends JSON.
      const res = await request('/v1/audio/prep/clips', { method: 'POST', body: form });
      clip = await res.json();
      toast(`uploaded · ${fmtMmSs(clip.duration_secs)}`);
    } catch {
      busy = false;
      renderPrep();
      return;
    }
    busy = false;
    await loadClipWave();
    await runTranscribe();
  }

  async function loadClipWave() {
    if (!clip) return;
    try {
      const res = await fetch(`/v1/audio/prep/clips/${encodeURIComponent(clip.id)}/audio`);
      if (!res.ok) return;
      const buffer = await res.arrayBuffer();
      clipWave = decodeWav(buffer);
    } catch {
      clipWave = null;
    }
    renderPrep();
  }

  // ---- prep flow: 2. transcribe ----------------------------------------------

  async function runTranscribe() {
    if (!clip) return;
    busy = true;
    renderPrep();
    try {
      transcript = await request(`/v1/audio/prep/clips/${encodeURIComponent(clip.id)}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      }).then((r) => r.json());
    } catch {
      transcript = null;
    }
    busy = false;
    renderPrep();
  }

  // ---- prep flow: 3. crop -----------------------------------------------------

  function setRangeFromWords() {
    if (!wordSel || !transcript) return;
    const words = transcript.words.slice(wordSel.start, wordSel.end + 1);
    if (!words.length) return;
    range = { start: words[0].start, end: words[words.length - 1].end };
    renderPrep();
  }

  function onWordClick(idx, shiftKey) {
    if (shiftKey && wordAnchor != null) {
      wordSel = { start: Math.min(wordAnchor, idx), end: Math.max(wordAnchor, idx) };
    } else {
      wordAnchor = idx;
      wordSel = { start: idx, end: idx };
    }
    setRangeFromWords();
  }

  function renderTranscript() {
    if (!transcript) return el('div', { class: 'placeholder' }, ['transcribing…']);
    return el(
      'div',
      { class: 'cmp' },
      transcript.words.map((w, i) => {
        const speaker = speakerForWord(w, transcript.speakers);
        const inSel = wordSel && i >= wordSel.start && i <= wordSel.end;
        return el(
          'span',
          {
            style: `cursor:pointer;padding:1px 2px;margin-right:2px;display:inline-block;${
              speaker != null ? `color:${speakerColor(speaker)};` : ''
            }${inSel ? 'background:rgba(0,229,255,0.18);' : ''}`,
            onclick: (e) => onWordClick(i, e.shiftKey),
          },
          [w.text],
        );
      }),
    );
  }

  function renderWaveformCrop() {
    if (!clipWave) return el('div', { class: 'placeholder' }, ['no waveform yet']);
    const duration = clipWave.samples.length / clipWave.sampleRate;
    const start = range?.start ?? 0;
    const end = range?.end ?? duration;
    const startHandle = el('div', { class: 'trim-handle', style: `left:${(start / duration) * 100}%` });
    const endHandle = el('div', { class: 'trim-handle', style: `left:${(end / duration) * 100}%` });
    wireDrag(startHandle, 'start', duration);
    wireDrag(endHandle, 'end', duration);
    // Mutates the persistent `waveWrap` in place — a fresh `trim-wrap` div
    // here would detach mid-drag the moment the first pointermove calls
    // renderPrep() (naru task 1462 review).
    waveWrap.replaceChildren(
      waveformSvg(peaks(clipWave.samples, 140)),
      startHandle,
      endHandle,
      el('div', { style: 'display:flex;justify-content:space-between;color:var(--muted);font-size:10px' }, [
        el('span', {}, [`${(end - start).toFixed(1)} s selected`]),
        el('span', {}, ['drag the cyan handles to crop (fallback for clicking words)']),
      ]),
    );
    return waveWrap;
  }

  function wireDrag(handle, which, duration) {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      const onMove = (ev) => {
        const rect = waveWrap.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        const t = frac * duration;
        const start = range?.start ?? 0;
        const end = range?.end ?? duration;
        if (which === 'start') range = { start: Math.min(t, end - 0.2), end };
        else range = { start, end: Math.max(t, start + 0.2) };
        wordSel = null; // dragging the waveform supersedes a word selection
        renderPrep();
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });
  }

  // ---- prep flow: 4. speaker --------------------------------------------------

  function speakerIds() {
    if (!transcript) return [];
    return [...new Set(transcript.speakers.map((s) => s.speaker))].sort((a, b) => a - b);
  }

  function pickSpeaker(id) {
    selectedSpeaker = selectedSpeaker === id ? null : id;
    if (selectedSpeaker != null && transcript) {
      const spans = transcript.speakers.filter((s) => s.speaker === selectedSpeaker);
      const start = Math.min(...spans.map((s) => s.start));
      const end = Math.max(...spans.map((s) => s.end));
      range = { start, end };
      wordSel = null;
    }
    renderPrep();
  }

  function renderSpeakerChips() {
    const ids = speakerIds();
    if (!ids.length) return el('div', { style: 'color:var(--muted);font-size:11px' }, ['no diarized speakers']);
    return el(
      'div',
      { class: 'filt' },
      ids.map((id) =>
        el(
          'span',
          {
            class: id === selectedSpeaker ? 'on' : '',
            style: `color:${speakerColor(id)};border-color:${speakerColor(id)}`,
            onclick: () => pickSpeaker(id),
          },
          [`speaker ${id}`],
        ),
      ),
    );
  }

  // ---- prep flow: 5. isolate + clean -----------------------------------------

  function flagCheck(key, label) {
    return el('label', { class: 'chk' }, [
      el('input', {
        type: 'checkbox',
        checked: flags[key] || undefined,
        onchange: (e) => (flags[key] = e.target.checked),
      }),
      label,
    ]);
  }

  async function runProcess() {
    if (!clip || !range) {
      toast('crop a range first', true);
      return;
    }
    const name = sampleName.trim();
    if (!name) {
      toast('name the sample first', true);
      return;
    }
    busy = true;
    renderPrep();
    try {
      created = await postJson('/v1/audio/samples', {
        clip_id: clip.id,
        name,
        start: range.start,
        end: range.end,
        speaker: selectedSpeaker,
        ...flags,
      });
      toast(`${name} processed`);
    } catch {
      created = null;
    }
    busy = false;
    renderPrep();
  }

  function renderPreview() {
    if (!created) return null;
    const cleanUrl = `/v1/audio/samples/${encodeURIComponent(created.id)}/audio?variant=clean`;
    const rawUrl = `/v1/audio/samples/${encodeURIComponent(created.id)}/audio?variant=cropped`;
    return el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['6']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['A/B PREVIEW']),
        playRow(() => rawUrl, 'raw crop'),
        playRow(() => cleanUrl, 'cleaned'),
        created.warnings?.length ? el('div', { class: 'warn' }, [created.warnings.join('; ')]) : null,
        el('div', { style: 'display:flex;gap:8px;margin-top:8px' }, [
          el('button', { class: 'big', style: 'margin:0', onclick: keepSample }, ['✓ KEEP IN LIBRARY']),
          el(
            'button',
            {
              class: 'big r',
              style: 'margin:0',
              onclick: async () => {
                await del(`/v1/audio/samples/${encodeURIComponent(created.id)}`).catch(() => {});
                created = null;
                renderPrep();
              },
            },
            ['✕ DISCARD, REDO'],
          ),
        ]),
      ]),
    ]);
  }

  function renderPrep() {
    setChildren(prepCard, [
      el('h3', {}, ['+ NEW CLIP']),
      el('div', { class: 'step' }, [
        el('span', { class: 'num' }, ['1']),
        el('div', { style: 'flex:1' }, [
          el('div', { class: 'lab' }, ['UPLOAD · WAV / MP3 / M4A']),
          clip
            ? playRow(
                () => `/v1/audio/prep/clips/${encodeURIComponent(clip.id)}/audio`,
                `${clip.original_filename ?? clip.id} · ${fmtMmSs(clip.duration_secs)}`,
              )
            : el('div', { class: 'drop' }, [
                el('span', {}, ['⇪ drop an audio file or ']),
                (() => {
                  const input = el('input', { type: 'file', accept: 'audio/*', style: 'display:none' });
                  input.addEventListener('change', (e) => {
                    const file = e.target.files[0];
                    if (file) loadFile(file);
                    e.target.value = '';
                  });
                  const browse = el('span', { style: 'color:var(--cyan);cursor:pointer' }, ['browse']);
                  browse.addEventListener('click', () => input.click());
                  return el('span', {}, [browse, input]);
                })(),
              ]),
        ]),
      ]),
      clip
        ? el('div', { class: 'step' }, [
            el('span', { class: 'num' }, ['2']),
            el('div', { style: 'flex:1' }, [
              el('div', { class: 'lab' }, ['TRANSCRIPT · WORDS + SPEAKER']),
              busy && !transcript ? el('div', { class: 'placeholder' }, ['working…']) : renderTranscript(),
            ]),
          ])
        : null,
      transcript
        ? el('div', { class: 'step' }, [
            el('span', { class: 'num' }, ['3']),
            el('div', { style: 'flex:1' }, [
              el('div', { class: 'lab' }, ['CROP · CLICK A WORD, SHIFT-CLICK THE LAST ONE']),
              renderWaveformCrop(),
            ]),
          ])
        : null,
      transcript
        ? el('div', { class: 'step' }, [
            el('span', { class: 'num' }, ['4']),
            el('div', { style: 'flex:1' }, [el('div', { class: 'lab' }, ['SPEAKER']), renderSpeakerChips()]),
          ])
        : null,
      transcript && !created
        ? el('div', { class: 'step' }, [
            el('span', { class: 'num' }, ['5']),
            el('div', { style: 'flex:1;display:flex;gap:14px;align-items:center;flex-wrap:wrap' }, [
              el('div', {}, [
                el('div', { class: 'lab' }, ['NAME']),
                el('input', {
                  class: 'inp',
                  value: sampleName,
                  oninput: (e) => (sampleName = e.target.value),
                }),
              ]),
              flagCheck('isolate', 'isolate'),
              flagCheck('denoise', 'denoise'),
              flagCheck('trim_silence', 'trim silence'),
              flagCheck('normalize', 'normalize'),
              el(
                'button',
                { class: 'big', style: 'margin:0 0 0 auto;max-width:220px', disabled: busy || undefined, onclick: runProcess },
                [busy ? 'working…' : '▶ ISOLATE + CLEAN'],
              ),
            ]),
          ])
        : null,
      renderPreview(),
      el('button', { class: 'btn x', style: 'margin-top:10px', onclick: backToLibrary }, ['‹ back to library']),
    ]);
  }

  // ---- boot -------------------------------------------------------------------

  async function refresh() {
    if (disposed) return;
    let ids;
    try {
      ids = (await getJson('/v1/audio/samples')).samples;
    } catch {
      return;
    }
    samples = await Promise.all(
      ids.map((id) => getJson(`/v1/audio/samples/${encodeURIComponent(id)}`, { silent: true }).catch(() => null)),
    ).then((list) => list.filter(Boolean));
    if (disposed) return;
    renderList();
    renderSide();
  }

  window.naruAdmin = window.naruAdmin ?? {};
  window.naruAdmin.samples = { loadFile };

  if (params.clip) {
    startPrepForClip(params.clip);
  } else {
    renderList();
    renderSide();
  }
  refresh();
  return () => {
    disposed = true;
    if (window.naruAdmin) delete window.naruAdmin.samples;
  };
}
