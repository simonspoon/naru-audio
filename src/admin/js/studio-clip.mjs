// Sample Studio, raw-clip mode (naru task 1576): `#studio?clip=<id>`. The
// editor on top of the studio's pipeline -- a sample *project* on an
// uploaded clip of any length. It owns the edit model (`studio-edit.mjs`:
// crop, cuts, speaker, crosstalk, best takes, with undo/redo), the waveform
// (`studio-wave.mjs`), the speakers and best-takes panels, playback of the
// clip's own audio stream, and saving the project
// (`PUT /v1/audio/prep/clips/{id}/project`, autosaved ~2 s after an edit once
// it has a name). `studio.mjs` renders the kept segments through the chain
// and supplies what the project also stores (steps, transcript fixes).

import { getJson, postJson, putJson } from './api.mjs';
import { el, toast } from './ui.mjs';
import { createWave, createSegmentPlayer } from './studio-wave.mjs';
import {
  History,
  newModel,
  keptSegments,
  speakerStats,
  speakerSpans,
  speakerColor,
  mergeIntervals,
  subtractIntervals,
  intersectIntervals,
  totalSecs,
} from './studio-edit.mjs';

const AUTOSAVE_MS = 2000;
/** The clone voice endpoint takes 3-30 s; 10-20 s is the sweet spot. */
export const CLONE_MIN_S = 3;
export const CLONE_MAX_S = 30;
const IDEAL = [10, 20];

function fmt(t) {
  const s = Math.max(0, t);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}

/**
 * @param {{clipId: string, getExtras: () => {steps: object[], transcript: string, fixes: object},
 *   onChange: () => void}} opts
 */
export function createClipEditor({ clipId, getExtras, onChange }) {
  const base = `/v1/audio/prep/clips/${encodeURIComponent(clipId)}`;
  let disposed = false;
  let clip = null;
  let project = null; // the saved project when this clip had one
  let transcript = null;
  let transcriptState = 'none'; // 'none' | 'running' | 'ready'
  let stats = [];
  let model = null;
  let keptCache = [];
  const history = new History();
  let selection = null;
  let playhead = 0;
  let name = '';
  let savedOnce = false;
  let saveText = '';
  let saveTimer = null;
  let saving = false;
  let takesBusy = false;
  let target = { min: IDEAL[0], max: IDEAL[1] };
  let sepTimer = null;
  let sepSecs = 0;
  let numSpeakers = null; // forced speaker count for the next separation; null = auto
  let wave = null;
  let lastPlay = null; // {list, tag} of the running playback, to restart it on a seek
  const player = createSegmentPlayer(`${base}/audio`, {
    onTime: (t) => {
      playhead = t;
      wave?.setPlayhead(t, true);
    },
    onStop: () => {
      lastPlay = null;
      renderToolbar();
      renderSpeakers();
      renderTakes();
    },
  });

  const root = el('div', { class: 'sc' }, [el('div', { class: 'studio-note' }, ['loading the clip…'])]);
  const toolbar = el('div', { class: 'sc-bar' });
  const readout = el('div', { class: 'sc-readout' });
  const speakersHost = el('div', { class: 'sc-panel' });
  const takesHost = el('div', { class: 'sc-panel' });
  const projectHost = el('div', { class: 'sc-project' });

  /** `replaceChildren` stringifies a bare `null` (see `el()`, which skips it); this drops them. */
  function setKids(node, ...kids) {
    node.replaceChildren(...kids.filter((k) => k != null));
  }

  // ---- model ---------------------------------------------------------------

  function recompute() {
    keptCache = keptSegments(model, transcript);
  }

  /** Runs `fn` on the model as one undoable step. */
  function mutate(fn) {
    history.push(model);
    fn(model);
    afterChange();
  }

  function afterChange() {
    recompute();
    refreshWave();
    renderToolbar();
    renderSpeakers();
    renderTakes();
    renderReadout();
    renderProject();
    onChange();
    scheduleSave();
  }

  function refreshWave() {
    wave.setData({
      crop: model.crop,
      cuts: model.cuts,
      lanes: stats.map((s) => ({ speaker: s.speaker, spans: s.spans })),
      overlaps: transcript?.overlaps ?? [],
      excludeOverlaps: model.excludeOverlaps,
      takes: model.takes?.takes ?? null,
      kept: keptCache,
      activeSpeaker: model.speaker,
    });
  }

  function undo() {
    const prev = history.undo(model);
    if (prev) {
      model = prev;
      afterChange();
    }
  }

  function redo() {
    const next = history.redo(model);
    if (next) {
      model = next;
      afterChange();
    }
  }

  function cutSelection() {
    if (!selection) return;
    const sel = selection;
    selection = null;
    wave.setSelection(null);
    mutate((m) => (m.cuts = mergeIntervals([...m.cuts, sel])));
  }

  function uncutSelection() {
    if (!selection) return;
    const sel = selection;
    mutate((m) => (m.cuts = subtractIntervals(mergeIntervals(m.cuts), [sel])));
  }

  function cropToSelection() {
    if (!selection) return;
    const sel = selection;
    selection = null;
    wave.setSelection(null);
    mutate((m) => (m.crop = { start: sel.start, end: sel.end }));
  }

  // ---- playback ------------------------------------------------------------

  function startPlay(list, tag, from = null) {
    if (!list.length) {
      toast('nothing to play there', true);
      return;
    }
    lastPlay = { list, tag };
    player.play(list, tag, from);
    renderToolbar();
    renderSpeakers();
    renderTakes();
  }

  function togglePlay(list, tag) {
    if (player.playingTag() === tag) player.stop();
    else startPlay(list, tag, tag === 'kept' && playhead > 0 ? playhead : null);
  }

  const playKept = () => togglePlay(keptCache, 'kept');
  const playSelection = () => selection && togglePlay([selection], 'selection');

  function onSeek(t) {
    playhead = t;
    wave.setPlayhead(t);
    if (lastPlay && player.playingTag()) startPlay(lastPlay.list, lastPlay.tag, t);
  }

  // ---- toolbar -------------------------------------------------------------

  function tb(label, onclick, opts = {}) {
    return el(
      'button',
      {
        class: `btn${opts.cls ? ` ${opts.cls}` : ''}`,
        type: 'button',
        title: opts.title,
        disabled: opts.disabled || undefined,
        onclick,
      },
      [label],
    );
  }

  function renderToolbar() {
    if (!wave) return;
    const sel = !!selection;
    const playing = player.playingTag();
    setKids(toolbar,
      tb(playing === 'kept' ? '■ STOP' : '▶ PLAY KEPT', playKept, { title: 'plays only what is kept (Space)' }),
      tb(playing === 'selection' ? '■ STOP' : '▶ SELECTION', playSelection, { disabled: !sel }),
      el('span', { class: 'sc-sep' }),
      tb('✂ CUT', cutSelection, { disabled: !sel, title: 'delete the selected region (Delete)' }),
      tb('UNCUT', uncutSelection, { disabled: !sel || !model.cuts.length, cls: 'x', title: 'bring cut audio in the selection back' }),
      tb('KEEP ONLY SELECTION', cropToSelection, { disabled: !sel, title: 'crop to the selection' }),
      el('span', { class: 'sc-sep' }),
      tb('↶', undo, { disabled: !history.canUndo(), cls: 'x', title: 'undo (Ctrl/Cmd-Z)' }),
      tb('↷', redo, { disabled: !history.canRedo(), cls: 'x', title: 'redo (Shift-Cmd-Z / Ctrl-Y)' }),
      tb('RESET', () => mutate((m) => Object.assign(m, newModel(clip.duration_secs), { takes: m.takes })), {
        cls: 'x',
        title: 'drop every crop, cut and speaker choice',
      }),
      el('span', { class: 'sc-sep' }),
      tb('−', () => wave.zoomBy(0.5), { cls: 'x', title: 'zoom out (Ctrl/Cmd-wheel)' }),
      tb('+', () => wave.zoomBy(2), { cls: 'x', title: 'zoom in (Ctrl/Cmd-wheel)' }),
      tb('FIT', () => wave.fitAll(), { cls: 'x' }),
      tb('◂', () => wave.panBy(-0.4), { cls: 'x', title: 'scroll left (shift-drag also pans)' }),
      tb('▸', () => wave.panBy(0.4), { cls: 'x' }),
    );
  }

  function renderReadout() {
    const secs = totalSecs(keptCache);
    let verdict = el('span', { class: 'ok2' }, ['✓ ideal clone length']);
    if (!keptCache.length) verdict = el('span', { class: 'warn-t' }, ['nothing kept']);
    else if (secs < CLONE_MIN_S) verdict = el('span', { class: 'warn-t' }, [`! too short — a clone sample needs ${CLONE_MIN_S}–${CLONE_MAX_S} s`]);
    else if (secs > CLONE_MAX_S) verdict = el('span', { class: 'warn-t' }, [`! too long — a clone sample needs ${CLONE_MIN_S}–${CLONE_MAX_S} s (pick best takes or crop)`]);
    else if (secs < IDEAL[0] || secs > IDEAL[1]) verdict = el('span', { class: 'studio-note' }, [`ok — ${IDEAL[0]}–${IDEAL[1]} s is ideal`]);
    setKids(readout,
      el('span', { class: 'lab' }, ['KEPT']),
      el('b', {}, [`${secs.toFixed(1)} s`]),
      el('span', { class: 'studio-note' }, [` in ${keptCache.length} segment${keptCache.length === 1 ? '' : 's'} of ${fmt(clip.duration_secs)}`]),
      verdict,
      selection ? el('span', { class: 'studio-note' }, [`selection ${fmt(selection.start)} – ${fmt(selection.end)} (${(selection.end - selection.start).toFixed(1)} s)`]) : null,
    );
  }

  // ---- speakers ------------------------------------------------------------

  async function separateSpeakers() {
    transcriptState = 'running';
    sepSecs = 0;
    clearInterval(sepTimer);
    sepTimer = setInterval(() => {
      sepSecs++;
      renderSpeakers();
    }, 1000);
    renderSpeakers();
    try {
      await postJson(`${base}/transcribe`, numSpeakers ? { num_speakers: numSpeakers } : {});
      if (disposed) return;
      await loadTranscript();
      // Cluster ids are not stable across runs: a kept speaker or its takes may not exist now.
      if (model.speaker != null && !stats.some((s) => s.speaker === model.speaker)) {
        model.speaker = null;
        model.takes = null;
        model.useTakes = false;
      }
    } catch {
      // postJson already toasted
    }
    clearInterval(sepTimer);
    if (disposed) return;
    transcriptState = transcript ? 'ready' : 'none';
    afterChange();
  }

  async function loadTranscript() {
    try {
      transcript = await getJson(`${base}/transcript`, { silent: true });
      transcriptState = 'ready';
      stats = speakerStats(transcript);
      // A clip with one voice has nothing to separate; with several, keep them all until the user picks.
    } catch {
      transcript = null;
      transcriptState = 'none';
      stats = [];
    }
  }

  /** "speakers: auto / 1-8" — how many voices the separation should find. */
  function speakerCountSelect() {
    return el('label', { class: 'studio-note' }, [
      'speakers: ',
      el(
        'select',
        {
          class: 'inp',
          title: 'force the number of speakers, e.g. 2 when auto splits one voice into several',
          onchange: (e) => {
            numSpeakers = e.target.value ? Number(e.target.value) : null;
          },
        },
        ['auto', 1, 2, 3, 4, 5, 6, 7, 8].map((n) => el('option', { value: n === 'auto' ? '' : n, selected: (n === 'auto' ? numSpeakers == null : n === numSpeakers) || undefined }, [n])),
      ),
    ]);
  }

  function pickSpeaker(id) {
    if (model.speaker === id) return;
    mutate((m) => {
      m.speaker = id;
      m.takes = null; // scored for the previous speaker
      m.useTakes = false;
    });
  }

  function auditionSpans(id) {
    return intersectIntervals(speakerSpans(transcript, id), [model.crop]);
  }

  function renderSpeakers() {
    if (!wave) return;
    const head = el('div', { class: 'lab' }, ['SPEAKERS']);
    if (transcriptState === 'none') {
      setKids(speakersHost,
        head,
        el('div', { class: 'studio-note' }, ['Separate speakers to see who talks when, keep one voice and drop crosstalk. Runs speech recognition over the whole clip, so a long clip takes a while — you can keep editing meanwhile.']),
        speakerCountSelect(),
        tb('SEPARATE SPEAKERS', separateSpeakers, { cls: 'v' }),
      );
      return;
    }
    if (transcriptState === 'running') {
      setKids(speakersHost, head, el('div', { class: 'studio-note' }, [`separating speakers… ${sepSecs} s (the clip stays editable)`]));
      return;
    }
    const overlapSecs = totalSecs(mergeIntervals(transcript.overlaps ?? []));
    const radio = (id, label, extra) =>
      el('label', { class: `sc-spk${model.speaker === id ? ' on' : ''}` }, [
        el('input', { type: 'radio', name: `sc-keep-${clipId}`, checked: model.speaker === id || undefined, onchange: () => pickSpeaker(id) }),
        ...extra,
        label,
      ]);
    setKids(speakersHost,
      head,
      ...(stats.length
        ? [
            el('div', { class: 'studio-note' }, ['KEEP one speaker — the others are dropped']),
            ...[...stats].sort((a, b) => b.secs - a.secs).map((s) =>
              el('div', { class: 'sc-row' }, [
                radio(s.speaker, `SPEAKER ${s.speaker}`, [el('i', { class: 'sc-dot', style: `background:${speakerColor(s.speaker)}` })]),
                el('span', { class: 'studio-note' }, [`${s.secs.toFixed(1)} s`]),
                tb(player.playingTag() === `speaker:${s.speaker}` ? '■' : '▶', () => togglePlay(auditionSpans(s.speaker), `speaker:${s.speaker}`), {
                  title: `audition speaker ${s.speaker}`,
                }),
              ]),
            ),
            el('div', { class: 'sc-row' }, [radio(null, 'ALL SPEAKERS', [])]),
            el('div', { class: 'sc-row' }, [speakerCountSelect(), tb('RE-SEPARATE', separateSpeakers, { title: 'run speaker separation again with this speaker count' })]),
          ]
        : [el('div', { class: 'studio-note' }, ['no speech found'])]),
      el('label', { class: 'chk sc-cross' }, [
        el('input', {
          type: 'checkbox',
          checked: model.excludeOverlaps || undefined,
          onchange: (e) => mutate((m) => (m.excludeOverlaps = e.target.checked)),
        }),
        `drop crosstalk (${(transcript.overlaps ?? []).length} overlaps · ${overlapSecs.toFixed(1)} s)`,
      ]),
      el('div', { class: 'studio-note' }, [model.excludeOverlaps ? 'overlapping speech is removed from the kept audio' : 'crosstalk is kept — shown hatched red on the waveform']),
    );
  }

  // ---- best takes ----------------------------------------------------------

  async function pickTakes() {
    if (!transcript) return;
    const from = subtractIntervals([model.crop], mergeIntervals(model.cuts));
    if (!from.length) {
      toast('nothing left to pick takes from', true);
      return;
    }
    takesBusy = true;
    renderTakes();
    let result;
    try {
      result = await postJson(`${base}/takes`, {
        speaker: model.speaker ?? undefined,
        segments: from,
        target_min: target.min,
        target_max: target.max,
        exclude_overlaps: model.excludeOverlaps,
      });
    } catch {
      takesBusy = false;
      renderTakes();
      return;
    }
    takesBusy = false;
    if (disposed) return;
    mutate((m) => {
      m.takes = result;
      m.useTakes = true;
    });
    if (!result.takes.length) toast('no takes found in what is left', true);
  }

  function toggleTake(i) {
    mutate((m) => {
      m.takes.takes[i].picked = !m.takes.takes[i].picked;
      m.takes.picked_secs = m.takes.takes.filter((t) => t.picked).reduce((s, t) => s + (t.end - t.start), 0);
    });
  }

  function chip(label, value, bad) {
    return el('span', { class: `sc-chip${bad ? ' bad' : ''}` }, [`${label} ${value}`]);
  }

  function takeRow(t, i) {
    const m = t.metrics ?? {};
    const tag = `take:${i}`;
    return el('div', { class: `sc-take${t.picked ? ' on' : ''}` }, [
      el('input', { type: 'checkbox', checked: t.picked || undefined, title: 'in / out of the pick', onchange: () => toggleTake(i) }),
      el('span', { class: 'sc-take-t', onclick: () => wave.showRange(t.start, t.end) }, [`${fmt(t.start)}–${fmt(t.end)} · ${(t.end - t.start).toFixed(1)} s`]),
      el('b', { class: 'sc-score', title: 'score (higher is cleaner)' }, [t.score.toFixed(2)]),
      m.noise_floor_dbfs != null ? chip('noise', `${m.noise_floor_dbfs.toFixed(0)} dB`, m.noise_floor_dbfs > -50) : null,
      chip('clip', `${((m.clipped_ratio ?? 0) * 100).toFixed(1)}%`, (m.clipped_ratio ?? 0) > 0.001),
      chip('overlap', `${((m.overlap_ratio ?? 0) * 100).toFixed(0)}%`, (m.overlap_ratio ?? 0) > 0.05),
      m.confidence != null ? chip('conf', m.confidence.toFixed(2), m.confidence < 0.7) : null,
      m.words_per_sec != null ? chip('pace', `${m.words_per_sec.toFixed(1)}/s`, false) : null,
      tb(player.playingTag() === tag ? '■' : '▶', () => togglePlay([{ start: t.start, end: t.end }], tag), { title: 'audition this take' }),
    ]);
  }

  function renderTakes() {
    if (!wave) return;
    const head = el('div', { class: 'lab' }, ['BEST TAKES']);
    if (!transcript) {
      setKids(takesHost, head, el('div', { class: 'studio-note' }, ['Takes are scored on the transcript and speakers — separate speakers first.']));
      return;
    }
    const num = (key) =>
      el('input', {
        class: 'inp sc-num',
        type: 'number',
        min: '1',
        max: '120',
        value: target[key],
        onchange: (e) => {
          const v = Number(e.target.value);
          if (v > 0) target[key] = v;
          if (target.min > target.max) target.max = target.min;
          renderTakes();
        },
      });
    const list = model.takes?.takes ?? [];
    const pickedSecs = list.filter((t) => t.picked).reduce((s, t) => s + (t.end - t.start), 0);
    setKids(takesHost,
      head,
      el('div', { class: 'sc-row' }, [
        el('span', { class: 'studio-note' }, ['target']),
        num('min'),
        el('span', { class: 'studio-note' }, ['–']),
        num('max'),
        el('span', { class: 'studio-note' }, ['s']),
        tb(takesBusy ? 'SCORING…' : '★ PICK BEST TAKES', pickTakes, { cls: 'v', disabled: takesBusy }),
      ]),
      list.length
        ? el('div', { class: 'sc-takes' }, list.map(takeRow))
        : el('div', { class: 'studio-note' }, [model.takes ? 'no takes found' : 'Assembles the cleanest stretches of the kept speaker into a sample. Edit the pick afterwards.']),
      list.length
        ? el('div', { class: 'sc-row' }, [
            el('label', { class: 'chk' }, [
              el('input', { type: 'checkbox', checked: model.useTakes || undefined, onchange: (e) => mutate((m) => (m.useTakes = e.target.checked)) }),
              'use picked takes',
            ]),
            el('span', { class: 'studio-note' }, [`${list.filter((t) => t.picked).length} of ${list.length} picked · ${pickedSecs.toFixed(1)} s`]),
          ])
        : null,
    );
  }

  // ---- project -------------------------------------------------------------

  function projectBody() {
    const x = getExtras();
    return {
      name: name.trim(),
      speaker: model.speaker,
      segments: keptCache,
      cuts: model.cuts,
      exclude_overlaps: model.excludeOverlaps,
      steps: x.steps,
      transcript: x.transcript,
      // `takes` holds the editor state the server's own fields do not: the crop the kept segments came from, the takes result, transcript fixes.
      takes: { v: 1, crop: model.crop, useTakes: model.useTakes, result: model.takes, fixes: x.fixes },
    };
  }

  async function save() {
    if (!name.trim()) {
      toast('name the project first', true);
      return false;
    }
    if (saving) return false;
    saving = true;
    saveText = 'saving…';
    renderProject();
    try {
      await putJson(`${base}/project`, projectBody());
      savedOnce = true;
      saveText = `saved ${new Date().toLocaleTimeString()}`;
    } catch {
      saveText = 'save failed';
    }
    saving = false;
    if (!disposed) renderProject();
    return savedOnce;
  }

  function scheduleSave() {
    if (!savedOnce || !name.trim() || disposed) return;
    clearTimeout(saveTimer);
    saveText = 'edits pending…';
    saveTimer = setTimeout(save, AUTOSAVE_MS);
    renderProject();
  }

  function renderProject() {
    if (!wave) return;
    setKids(projectHost,
      el('span', { class: 'lab' }, ['PROJECT']),
      el('input', {
        class: 'inp sc-name',
        placeholder: 'name this project',
        value: name,
        oninput: (e) => (name = e.target.value),
        onchange: () => scheduleSave(),
      }),
      tb('SAVE PROJECT', save, { disabled: saving }),
      el('span', { class: 'studio-note' }, [saveText || (savedOnce ? '' : 'not saved yet — saving turns on autosave')]),
    );
  }

  // ---- keyboard ------------------------------------------------------------

  function onKey(e) {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    const mod = e.metaKey || e.ctrlKey;
    const k = e.key.toLowerCase();
    if (mod && k === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    } else if (mod && k === 'y') {
      e.preventDefault();
      redo();
    } else if ((e.key === 'Delete' || e.key === 'Backspace') && selection) {
      e.preventDefault();
      cutSelection();
    } else if (e.key === ' ' && !mod && tag !== 'BUTTON') {
      e.preventDefault();
      if (player.playingTag()) player.stop();
      else if (selection) playSelection();
      else playKept();
    }
  }

  // ---- boot ----------------------------------------------------------------

  function restore() {
    model = newModel(clip.duration_secs);
    name = clip.original_filename?.replace(/\.[^.]+$/, '') ?? '';
    if (!project) return;
    name = project.name;
    savedOnce = true;
    saveText = 'project loaded — edits autosave';
    const ed = project.takes && typeof project.takes === 'object' ? project.takes : {};
    model.speaker = project.speaker ?? null;
    model.cuts = project.cuts ?? [];
    model.excludeOverlaps = !!project.exclude_overlaps;
    const segs = project.segments ?? [];
    model.crop = ed.crop ?? (segs.length ? { start: segs[0].start, end: segs[segs.length - 1].end } : model.crop);
    model.takes = ed.result ?? null;
    model.useTakes = !!ed.useTakes && !!model.takes;
  }

  const ready = (async () => {
    clip = await getJson(base);
    try {
      project = await getJson(`${base}/project`, { silent: true });
    } catch {
      project = null;
    }
    await loadTranscript();
    if (disposed) return;
    restore();
    recompute();
    wave = createWave({
      clipId,
      duration: clip.duration_secs,
      onSeek,
      onSelect: (sel) => {
        selection = sel;
        renderToolbar();
        renderReadout();
      },
      onCrop: (crop) => mutate((m) => (m.crop = crop)),
    });
    root.replaceChildren(
      toolbar,
      wave.root,
      readout,
      el('div', { class: 'sc-panels' }, [speakersHost, takesHost]),
      projectHost,
    );
    window.addEventListener('keydown', onKey);
    refreshWave();
    renderToolbar();
    renderSpeakers();
    renderTakes();
    renderReadout();
    renderProject();
  })();

  return {
    root,
    ready,
    clip: () => clip,
    project: () => project,
    transcript: () => transcript,
    kept: () => keptCache,
    name: () => name.trim(),
    /** Steps or transcript fixes changed: autosave the project if it is saved. */
    touch: scheduleSave,
    save,
    dispose() {
      disposed = true;
      clearTimeout(saveTimer);
      clearInterval(sepTimer);
      player.stop();
      wave?.destroy();
      window.removeEventListener('keydown', onKey);
    },
  };
}
