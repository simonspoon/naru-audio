// Overview tab (mockup: na-admin-v2): a models card (running/on-disk/not-
// pulled dots with load/unload/pull) and a voice grid for the default TTS
// model, with real waveforms and playback pulled from `GET
// /api/voices/{model}/{voice}/sample` (added alongside this by the voice
// API subtask). Built-in voices with no cached preview show a dashed
// placeholder and a "generate preview" action; we never auto-generate one.

import { getJson, postJson, streamNdjson } from './api.mjs';
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

  function renderModels(list) {
    modelsCard.replaceChildren(el('h3', {}, ['Models']));
    for (const m of list) {
      modelsCard.append(modelRow(m));
    }
  }

  function modelRow(m) {
    let dotStyle;
    let action;
    if (m.x_loaded) {
      dotStyle = 'background:var(--green)';
      action = el('button', { class: 'btn x', onclick: () => unload(m) }, ['unload']);
    } else if (m.x_pulled) {
      dotStyle = 'background:var(--muted)';
      action = el('button', { class: 'btn', onclick: () => load(m) }, ['load']);
    } else {
      dotStyle = 'border:1px solid var(--muted);background:transparent';
      action = el(
        'button',
        { class: 'btn', onclick: (e) => pull(m, e.currentTarget) },
        [`pull ${fmtBytes(m.x_size_bytes)}`],
      );
    }
    const label = [m.id];
    if (m.x_default) label.push(' ', el('span', { class: 'star' }, ['★']));
    if (m.x_kind !== 'tts') label.push(` · ${m.x_kind.toUpperCase()}`);
    if (m.x_non_commercial) label.push(' ', el('small', { style: 'color:var(--amber)' }, ['NC']));
    return el('div', { class: 'row' }, [
      el('span', { class: 'dot', style: dotStyle }),
      el('b', {}, label),
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

  async function renderVoices(ttsDefault) {
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
    for (const v of voices) {
      voicesGrid.append(await voiceCard(ttsDefault.id, v));
    }
    voicesGrid.append(addCard(ttsDefault.id));
  }

  async function voiceCard(model, voice) {
    const kindLabel = voice.origin ?? (voice.cloned ? 'clone' : 'built-in');
    const card = el('div', { class: 'vc' }, [
      voice.id,
      el('small', {}, [kindLabel]),
    ]);
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
                class: 'btn v',
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
              ['generate preview'],
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
