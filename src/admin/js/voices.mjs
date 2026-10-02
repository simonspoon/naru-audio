// Voices tab (mockup: na-admin-voices): voices belong to a model, so the
// tab opens with a model picker, then a filterable card grid (built-in /
// cloned / designed) for that model plus a side panel for the selected
// voice: transcript, a "try it" line you can speak, set-default, export,
// rename and delete. Import restores a voice from an exported `.json`.
//
// Exposes `window.naruAdmin.voices.loadImportFile(file)` so a headless
// test without a file picker can drive the import path directly.

import { getJson, postJson, putJson, request, del } from './api.mjs';
import { el, toast, playButton, waveformSvg, peaks, decodeWav } from './ui.mjs';

const FILTERS = ['All', 'Built-in', 'Cloned', 'Designed'];
const TRY_IT_DEFAULT = 'Warm coffee, a quiet morning, and nothing on fire yet.';

function originOf(v) {
  return v.origin ?? (v.cloned ? 'cloned' : 'builtin');
}

function matchesFilter(v, filter) {
  switch (filter) {
    case 'Built-in':
      return originOf(v) === 'builtin';
    case 'Cloned':
      return originOf(v) === 'cloned';
    case 'Designed':
      return originOf(v) === 'designed';
    default:
      return true;
  }
}

export function mount(view, params) {
  let disposed = false;
  let ttsModels = [];
  let selectedModel = params.model ?? null;
  let voiceList = [];
  let selectedVoiceId = null;
  let filter = 'All';

  const modelBar = el('div', { class: 'filt' });
  const filtBar = el('div', { class: 'filt' });
  const grid = el('div', { class: 'vg' });
  const listCard = el('div', { class: 'card' }, [
    el('h3', {}, ['Voices']),
    modelBar,
    filtBar,
    grid,
  ]);
  const sidePanel = el('div', { class: 'card' });
  view.append(el('div', { class: 'side' }, [listCard, sidePanel]));

  const importInput = el('input', {
    type: 'file',
    accept: '.json,application/json',
    style: 'display:none',
    onchange: (e) => {
      const file = e.target.files[0];
      if (file) importFile(file);
      e.target.value = '';
    },
  });
  view.append(importInput);

  function renderModelBar() {
    modelBar.replaceChildren(
      ...ttsModels.map((m) =>
        el(
          'span',
          {
            class: m.id === selectedModel ? 'on' : '',
            onclick: () => {
              selectedModel = m.id;
              selectedVoiceId = null;
              refreshVoices();
            },
          },
          [m.id],
        ),
      ),
      el(
        'button',
        { class: 'btn', style: 'margin-left:auto', onclick: () => importInput.click() },
        ['import'],
      ),
    );
  }

  function renderFilters() {
    const counts = { All: voiceList.length, 'Built-in': 0, Cloned: 0, Designed: 0 };
    for (const v of voiceList) {
      const label = { builtin: 'Built-in', cloned: 'Cloned', designed: 'Designed' }[originOf(v)];
      if (label) counts[label]++;
    }
    filtBar.replaceChildren(
      ...FILTERS.map((f) =>
        el(
          'span',
          {
            class: f === filter ? 'on' : '',
            onclick: () => {
              filter = f;
              renderGrid();
            },
          },
          [`${f} (${counts[f]})`],
        ),
      ),
    );
  }

  async function voiceCard(v) {
    const card = el('div', {
      class: `vc${v.id === selectedVoiceId ? ' sel' : ''}`,
      style: v.id === selectedVoiceId ? 'border-color:var(--cyan)' : '',
      onclick: () => {
        selectedVoiceId = v.id;
        renderGrid();
        renderSide();
      },
    });
    card.append(v.id, el('small', {}, [originOf(v)]));
    const sampleUrl = `/api/voices/${encodeURIComponent(selectedModel)}/${encodeURIComponent(v.id)}/sample`;
    let objectUrl = null;
    card.prepend(
      playButton(async () => {
        if (objectUrl) return objectUrl;
        const res = await fetch(sampleUrl);
        if (!res.ok) return null;
        objectUrl = URL.createObjectURL(await res.blob());
        return objectUrl;
      }),
    );
    try {
      const res = await fetch(sampleUrl);
      if (res.ok) {
        const buffer = await res.arrayBuffer();
        objectUrl = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
        card.append(waveformSvg(peaks(decodeWav(buffer).samples)));
      } else {
        card.append(el('div', { class: 'wave-placeholder' }));
      }
    } catch {
      card.append(el('div', { class: 'wave-placeholder' }));
    }
    return card;
  }

  function addCard() {
    const card = el('div', { class: 'vc split' });
    card.append(
      el(
        'div',
        { class: 'split-half', onclick: () => (location.hash = `#clone?model=${selectedModel}`) },
        ['+ clone'],
      ),
      el(
        'div',
        { class: 'split-half', onclick: () => (location.hash = `#design?model=${selectedModel}`) },
        ['design'],
      ),
    );
    return card;
  }

  async function renderGrid() {
    renderFilters();
    grid.replaceChildren();
    const rows = voiceList.filter((v) => matchesFilter(v, filter));
    for (const v of rows) grid.append(await voiceCard(v));
    grid.append(addCard());
  }

  function kv(label, value) {
    return el('div', { class: 'kv' }, [el('span', {}, [label]), el('span', {}, [String(value)])]);
  }

  async function renderSide() {
    const v = voiceList.find((x) => x.id === selectedVoiceId);
    if (!v) {
      sidePanel.replaceChildren(el('div', { class: 'placeholder' }, ['select a voice']));
      return;
    }
    const origin = originOf(v);
    const nodes = [
      el('h3', {}, [v.id]),
      kv('type', origin),
      kv('model', selectedModel),
      kv('duration', v.duration != null ? `${v.duration.toFixed(1)} s` : '—'),
      kv('description', v.description ?? '—'),
    ];
    if (v.has_transcript) {
      const transcriptRow = el('div', { class: 'kv' }, [el('span', {}, ['transcript']), el('span', {}, ['loading…'])]);
      nodes.push(transcriptRow);
      request(`/v1/audio/voices/${encodeURIComponent(v.id)}`, {}, { silent: true })
        .then((r) => r.json())
        .then((data) => {
          transcriptRow.lastChild.textContent = data.text ?? '—';
        })
        .catch(() => {
          transcriptRow.lastChild.textContent = '—';
        });
    }
    const tryText = el('input', { class: 'inp', value: TRY_IT_DEFAULT });
    nodes.push(
      el('div', { class: 'lab', style: 'margin-top:10px' }, ['Try it']),
      tryText,
      el(
        'button',
        {
          class: 'big',
          onclick: async () => {
            try {
              const res = await request('/v1/audio/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: selectedModel,
                  voice: v.id,
                  input: tryText.value,
                  response_format: 'wav',
                  stream: false,
                }),
              });
              new Audio(URL.createObjectURL(await res.blob())).play();
            } catch {
              // request() already toasted
            }
          },
        },
        ['▶ Speak'],
      ),
    );
    if (!v.default) {
      nodes.push(
        el(
          'button',
          {
            class: 'big',
            onclick: async () => {
              try {
                await putJson('/api/defaults', { voices: { [selectedModel]: v.id } });
                toast(`${v.id} is now default for ${selectedModel}`);
              } catch {
                return;
              }
              refreshVoices();
            },
          },
          ['★ Set default'],
        ),
      );
    }
    nodes.push(
      el(
        'button',
        {
          class: 'big',
          onclick: async () => {
            try {
              const res = await request(`/v1/audio/voices/${encodeURIComponent(v.id)}`);
              const data = await res.json();
              const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
              const a = el('a', { href: URL.createObjectURL(blob), download: `${v.id}.json` });
              a.click();
            } catch {
              // request() already toasted
            }
          },
        },
        ['Export'],
      ),
    );
    if (origin !== 'builtin') {
      const renameInput = el('input', { class: 'inp', value: v.id });
      nodes.push(
        el('div', { class: 'lab', style: 'margin-top:10px' }, ['Rename']),
        el('div', { class: 'row' }, [
          renameInput,
          el(
            'button',
            {
              class: 'btn',
              onclick: async () => {
                const name = renameInput.value.trim();
                if (!name || name === v.id) return;
                try {
                  await request(`/v1/audio/voices/${encodeURIComponent(v.id)}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                  });
                  toast(`renamed to ${name}`);
                } catch {
                  return;
                }
                selectedVoiceId = name;
                refreshVoices();
              },
            },
            ['save'],
          ),
        ]),
        el(
          'button',
          {
            class: 'big r',
            onclick: async () => {
              if (!confirm(`Delete voice "${v.id}"? This cannot be undone.`)) return;
              try {
                await del(`/v1/audio/voices/${encodeURIComponent(v.id)}`);
                toast(`${v.id} deleted`);
              } catch {
                return;
              }
              selectedVoiceId = null;
              refreshVoices();
            },
          },
          ['Delete'],
        ),
      );
    }
    sidePanel.replaceChildren(...nodes);
  }

  async function importFile(file) {
    let data;
    try {
      data = JSON.parse(await file.text());
    } catch {
      toast('not a valid voice export', true);
      return;
    }
    try {
      const bin = atob(data.wav_base64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const form = new FormData();
      form.append('name', data.name);
      form.append('model', data.model);
      form.append('text', data.text ?? '');
      if (data.description) form.append('description', data.description);
      form.append('file', new Blob([bytes], { type: 'audio/wav' }), `${data.name}.wav`);
      await request('/v1/audio/voices', { method: 'POST', body: form });
      toast(`imported ${data.name}`);
    } catch {
      return;
    }
    if (data.model !== selectedModel) {
      selectedModel = data.model;
    }
    selectedVoiceId = data.name;
    refreshVoices();
  }

  async function refreshVoices() {
    if (disposed || !selectedModel) return;
    renderModelBar();
    try {
      voiceList = (
        await getJson(`/v1/audio/voices?model=${encodeURIComponent(selectedModel)}`, { silent: true })
      ).voices;
    } catch {
      voiceList = [];
    }
    if (disposed) return;
    if (!selectedVoiceId && voiceList.length) selectedVoiceId = voiceList[0].id;
    renderGrid();
    renderSide();
  }

  async function refresh() {
    if (disposed) return;
    try {
      ttsModels = (await getJson('/v1/models')).data.filter((m) => m.x_kind === 'tts');
    } catch {
      return;
    }
    if (disposed) return;
    if (!selectedModel && ttsModels.length) selectedModel = ttsModels[0].id;
    await refreshVoices();
  }

  window.naruAdmin = window.naruAdmin ?? {};
  window.naruAdmin.voices = { loadImportFile: importFile };

  refresh();
  return () => {
    disposed = true;
    if (window.naruAdmin) delete window.naruAdmin.voices;
  };
}
