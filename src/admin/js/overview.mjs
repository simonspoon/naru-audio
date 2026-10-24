// Overview tab (mockup: na-admin-v2): a models card (running models on top
// with memory, the rest folded; dots with load/unload/pull) and a voice grid
// for the default TTS model, with real waveforms and playback pulled from
// `GET /api/voices/{model}/{voice}/sample` (added alongside this by the voice
// API subtask). Built-in voices with no cached preview show a dashed
// placeholder and a "generate preview" action; we never auto-generate one.

import { getJson, postJson, putJson, streamNdjson } from './api.mjs';
import { el, toast, playButton, waveformSvg, peaks, decodeWav, fmtBytes } from './ui.mjs';

export function mount(view) {
  let disposed = false;
  const modelsCard = el('div', { class: 'card' }, [el('h3', {}, ['Models'])]);
  const voicesTitle = el('h3', {}, ['Voices']);
  const voicesGrid = el('div', { class: 'vg' });
  const voicesCard = el('div', { class: 'card' }, [voicesTitle, voicesGrid]);
  view.append(el('div', { class: 'grid' }, [modelsCard, voicesCard]));

  async function refresh() {
    if (disposed) return;
    let list;
    try {
      list = (await getJson('/v1/models')).data;
    } catch {
      return;
    }
    if (disposed) return;
    renderModels(list);
    const ttsDefault = list.find((m) => m.x_kind === 'tts' && m.x_default);
    await renderVoices(ttsDefault);
  }

  // Running models first, with their memory; everything else folds under
  // "N more on disk". The fold's open state survives the re-render.
  let foldOpen = false;
  function renderModels(list) {
    const running = list.filter((m) => m.x_loaded);
    const rest = list.filter((m) => !m.x_loaded);
    const onDisk = rest.filter((m) => m.x_pulled).length;
    modelsCard.replaceChildren(el('h3', {}, [`Models · ${running.length} running`]));
    if (running.length === 0) {
      modelsCard.append(el('div', { class: 'placeholder' }, ['Nothing running. Load a model below.']));
    }
    for (const m of running) {
      modelsCard.append(modelRow(m));
    }
    if (rest.length === 0) return;
    let summary = `${onDisk} more on disk`;
    if (rest.length > onDisk) summary += ` · ${rest.length - onDisk} not pulled`;
    const fold = el('details', { class: 'fold', open: foldOpen }, [el('summary', {}, [summary])]);
    fold.addEventListener('toggle', () => (foldOpen = fold.open));
    for (const m of rest) {
      fold.append(modelRow(m));
    }
    modelsCard.append(fold);
  }

  function modelRow(m) {
    let dotStyle;
    let action;
    if (m.x_loaded) {
      dotStyle = 'background:var(--green)';
      action = el('button', { class: 'btn', onclick: () => unload(m) }, ['Unload']);
    } else if (m.x_pulled) {
      dotStyle = 'background:var(--muted)';
      action = el('button', { class: 'btn', onclick: () => load(m) }, ['Load']);
    } else {
      dotStyle = 'border:1px solid var(--muted);background:transparent';
      action = el(
        'button',
        { class: 'btn', onclick: (e) => pull(m, e.currentTarget) },
        [`Pull ${fmtBytes(m.x_size_bytes)}`],
      );
    }
    const label = [m.id];
    if (m.x_default) label.push(' ', el('span', { class: 'star' }, ['★']));
    if (m.x_kind !== 'tts') label.push(` · ${m.x_kind.toUpperCase()}`);
    if (m.x_non_commercial) label.push(' ', el('small', { style: 'color:var(--amber)' }, ['NC']));
    return el('div', { class: 'row' }, [
      el('span', { class: 'dot', style: dotStyle }),
      el('b', {}, label),
      m.x_loaded ? el('small', { class: 'mem' }, [fmtBytes(m.x_size_bytes)]) : null,
      action,
    ]);
  }

  async function load(m) {
    try {
      await postJson('/api/load', { model: m.id, kind: m.x_kind });
      toast(`${m.id} loaded`);
    } catch {
      return;
    }
    refresh();
  }

  async function unload(m) {
    try {
      await postJson('/api/load', { model: m.id, kind: m.x_kind, keep_alive: 0 });
    } catch {
      return;
    }
    refresh();
  }

  async function pull(m, btn) {
    btn.disabled = true;
    try {
      await streamNdjson('/api/pull', { model: m.id }, (line) => {
        if (line.model_total) {
          const pct = Math.round((100 * line.model_completed) / line.model_total);
          btn.textContent = `${pct}%`;
        }
        if (line.error) throw new Error(line.error.message ?? 'pull failed');
      });
      toast(`${m.id} pulled`);
    } catch (e) {
      toast(e.message ?? 'pull failed', true);
    }
    refresh();
  }

  // Bumped per render so an overlapping refresh (e.g. set-default while the
  // grid is still fetching samples) drops the stale run instead of appending.
  let voicesGen = 0;
  async function renderVoices(ttsDefault) {
    const gen = ++voicesGen;
    voicesGrid.replaceChildren();
    if (!ttsDefault) {
      voicesTitle.textContent = 'Voices';
      voicesGrid.append(el('div', { class: 'placeholder' }, ['no default TTS model']));
      return;
    }
    voicesTitle.textContent = `Voices · ${ttsDefault.id}`;
    let voices;
    try {
      voices = (await getJson(`/v1/audio/voices?model=${encodeURIComponent(ttsDefault.id)}`, {
        silent: true,
      })).voices;
    } catch {
      voices = [];
    }
    if (gen !== voicesGen) return;
    for (const v of voices) {
      const card = await voiceCard(ttsDefault.id, v);
      if (gen !== voicesGen) return;
      voicesGrid.append(card);
    }
    voicesGrid.append(addCard(ttsDefault.id));
  }

  async function voiceCard(model, voice) {
    const kindLabel = voice.origin ?? (voice.cloned ? 'clone' : 'built-in');
    const card = el('div', { class: voice.default ? 'vc default' : 'vc' }, [
      voice.id,
      voice.default ? el('span', { class: 'star', title: 'Default voice' }, [' ★']) : null,
      el('small', {}, [kindLabel]),
    ]);
    // Non-default cards get a set-default action, shown on hover/focus.
    if (!voice.default) {
      card.append(
        el(
          'button',
          {
            class: 'btn set-default',
            type: 'button',
            title: 'Set as default voice',
            onclick: async (e) => {
              e.stopPropagation();
              const btn = e.currentTarget;
              btn.disabled = true;
              try {
                await putJson('/api/defaults', { voices: { [model]: voice.id } });
                toast(`${voice.id} is now default for ${model}`);
              } catch {
                btn.disabled = false;
                return;
              }
              refresh();
            },
          },
          ['★ Set default'],
        ),
      );
    }
    const sampleUrl = `/api/voices/${encodeURIComponent(model)}/${encodeURIComponent(voice.id)}/sample`;
    let objectUrl = null;
    const play = playButton(async () => {
      if (objectUrl) return objectUrl;
      try {
        const res = await fetch(sampleUrl);
        if (!res.ok) throw new Error('no preview');
        objectUrl = URL.createObjectURL(await res.blob());
        return objectUrl;
      } catch {
        toast('No preview available for this voice', true);
        return null;
      }
    });
    card.prepend(play);
    try {
      const res = await fetch(sampleUrl);
      if (res.ok) {
        const buffer = await res.arrayBuffer();
        objectUrl = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
        card.append(waveformSvg(peaks(decodeWav(buffer).samples)));
      } else {
        const placeholder = el('div', { class: 'wave-placeholder' });
        card.append(placeholder);
        if (!voice.cloned) {
          card.append(
            el(
              'button',
              {
                class: 'btn',
                onclick: async (e) => {
                  e.stopPropagation();
                  const btn = e.currentTarget;
                  btn.disabled = true;
                  try {
                    const genRes = await fetch(sampleUrl, { method: 'POST' });
                    if (!genRes.ok) throw new Error('generate failed');
                    const buffer = await genRes.arrayBuffer();
                    objectUrl = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
                    placeholder.replaceWith(waveformSvg(peaks(decodeWav(buffer).samples)));
                    btn.remove();
                  } catch {
                    toast('failed to generate preview', true);
                    btn.disabled = false;
                  }
                },
              },
              ['Generate preview'],
            ),
          );
        }
      }
    } catch {
      card.append(el('div', { class: 'wave-placeholder' }));
    }
    return card;
  }

  function addCard(model) {
    const card = el('div', { class: 'vc split' });
    card.append(
      el(
        'div',
        { class: 'split-half', onclick: () => (location.hash = `#clone?model=${model}`) },
        ['+ clone'],
      ),
      el(
        'div',
        { class: 'split-half', onclick: () => (location.hash = `#design?model=${model}`) },
        ['design'],
      ),
    );
    return card;
  }

  refresh();
  return () => {
    disposed = true;
  };
}
