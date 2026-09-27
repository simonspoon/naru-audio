// Design tab (mockup: na-admin-design): pick a design-capable model, build
// a description from quick-pick chips (formatted per that model's
// `x_prompt_format.style`), generate three takes of a shared test line,
// pick one, name it, and save it as a voice on `x_design_voice_model` (the
// model that can actually hold the voice — VoiceDesign models can't clone
// themselves).

import { getJson, request } from './api.mjs';
import { el, toast, playButton, waveformSvg, peaks, decodeWav } from './ui.mjs';

const TEST_LINE = 'Warm coffee, a quiet morning, and nothing on fire yet.';
const RECENT_KEY = 'naru-audio-design-recent';
const CHIP_GROUPS = [
  { key: 'voice', label: 'VOICE', options: ['male', 'female', 'neutral'] },
  { key: 'age', label: 'AGE', options: ['young', 'adult', 'older'] },
  { key: 'pitch', label: 'PITCH', options: ['low', 'mid', 'high'] },
  { key: 'pace', label: 'PACE', options: ['slow', 'steady', 'quick'] },
  { key: 'mood', label: 'MOOD', options: ['warm', 'dry', 'bright', 'gravelly'] },
];

/** Builds the instructions text from the quick-pick attributes, formatted
 * per the model's declared `x_prompt_format.style`. `attributes` models
 * get a plain key:value list; everything else (including the common
 * `free_text` style) gets a natural-language sentence, matching the
 * mockup. */
function formatDescription(style, picks) {
  const chosen = CHIP_GROUPS.filter((g) => picks[g.key]);
  if (style === 'attributes') {
    return chosen.map((g) => `${g.key}: ${picks[g.key]}`).join(', ');
  }
  const subject = [picks.age, picks.voice].filter(Boolean).join(' ') || 'voice';
  const qualities = [
    picks.pitch && `${picks.pitch} pitch`,
    picks.pace && `${picks.pace} pace`,
    picks.mood && `${picks.mood} tone`,
  ].filter(Boolean);
  let text = `A ${subject}`;
  if (qualities.length) text += ` with a ${qualities.join(', ')}`;
  return `${text}.`;
}

function recentPrompts() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]');
  } catch {
    return [];
  }
}

function pushRecentPrompt(text) {
  const list = [text, ...recentPrompts().filter((p) => p !== text)].slice(0, 8);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list));
}

export function mount(view, params) {
  let disposed = false;
  let models = [];
  let selectedModel = params.model ?? null;
  const picks = {};
  let takes = []; // {blob, buffer}
  let pickedIndex = null;

  const modelBar = el('div', { class: 'filt' });
  const chipsWrap = el('div', {});
  const descBox = el('textarea', { class: 'inp', rows: 4 });
  const takesGrid = el('div', { class: 'vg3' });
  const nameInput = el('input', { class: 'inp' });
  const saveHint = el('span', { style: 'color:var(--muted);font-size:11px' });
  const createBtn = el('button', { class: 'big', style: 'margin:0 0 0 auto;max-width:200px' }, ['CREATE VOICE']);
  const testLineInput = el('input', { class: 'inp', value: TEST_LINE });
  const recentList = el('div', {});

  const leftCard = el('div', { class: 'card' }, [
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['1']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['MODEL']),
        modelBar,
        el('span', { style: 'color:var(--muted);font-size:11px' }, [' only models that can design']),
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['2']),
      el('div', { style: 'flex:1;display:grid;grid-template-columns:1fr 1.2fr;gap:14px' }, [
        el('div', {}, [el('div', { class: 'lab' }, ['QUICK PICKS']), chipsWrap]),
        el('div', {}, [el('div', { class: 'lab' }, ['DESCRIBE IT · PICKS FILL THIS IN']), descBox]),
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['3']),
      el('div', { style: 'flex:1' }, [
        el('div', { style: 'display:flex;align-items:center' }, [
          el('div', { class: 'lab', style: 'margin:0' }, ['TAKES · SAME DESCRIPTION, THREE TRIES']),
          el(
            'button',
            { class: 'btn', style: 'margin-left:auto', onclick: generateTakes },
            ['↻ GENERATE 3'],
          ),
        ]),
        takesGrid,
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['4']),
      el('div', { style: 'flex:1;display:flex;gap:14px;align-items:center;flex-wrap:wrap' }, [
        el('div', {}, [el('div', { class: 'lab' }, ['NAME']), nameInput]),
        saveHint,
        createBtn,
      ]),
    ]),
  ]);

  const rightCard = el('div', { class: 'card' }, [
    el('h3', {}, ['TEST LINE']),
    testLineInput,
    el('div', { style: 'color:var(--muted);font-size:11px;margin-top:8px' }, [
      'Every take speaks this line, so you compare like with like.',
    ]),
    el('div', { class: 'lab', style: 'margin-top:14px' }, ['RECENT PROMPTS']),
    recentList,
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
              renderSaveHint();
            },
          },
          [m.id],
        ),
      ),
    );
  }

  function renderChips() {
    chipsWrap.replaceChildren(
      ...CHIP_GROUPS.map((g) =>
        el('div', { class: 'chip-row' }, [
          el('span', { class: 'lab' }, [g.label]),
          ...g.options.map((opt) =>
            el(
              'span',
              {
                class: `chip${picks[g.key] === opt ? ' on' : ''}`,
                onclick: () => {
                  picks[g.key] = picks[g.key] === opt ? null : opt;
                  renderChips();
                  const model = models.find((m) => m.id === selectedModel);
                  descBox.value = formatDescription(model?.x_prompt_format?.style, picks);
                },
              },
              [opt],
            ),
          ),
        ]),
      ),
    );
  }

  function renderRecent() {
    recentList.replaceChildren(
      ...recentPrompts().map((p) =>
        el(
          'div',
          { class: 'kv', style: 'cursor:pointer', onclick: () => (descBox.value = p) },
          [el('span', {}, [p])],
        ),
      ),
    );
  }

  function renderSaveHint() {
    const model = models.find((m) => m.id === selectedModel);
    const target = model?.x_design_voice_model ?? selectedModel;
    saveHint.textContent =
      pickedIndex == null
        ? 'pick a take first'
        : `saves take ${String.fromCharCode(65 + pickedIndex)} as a voice on ${target}`;
  }

  function renderTakes() {
    takesGrid.replaceChildren(
      ...takes.map((t, i) => {
        const card = el('div', { class: `vc${i === pickedIndex ? ' picked' : ''}` }, [
          `take ${String.fromCharCode(65 + i)}`,
          i === pickedIndex ? el('span', { class: 'ok2' }, [' ✓ picked']) : null,
        ]);
        card.prepend(playButton(() => URL.createObjectURL(t.blob)));
        card.append(waveformSvg(peaks(t.samples)));
        card.addEventListener('click', (e) => {
          if (e.target.closest('.play')) return;
          pickedIndex = i;
          renderTakes();
          renderSaveHint();
        });
        return card;
      }),
    );
  }

  async function generateTakes() {
    if (!selectedModel) return;
    const description = descBox.value.trim();
    if (!description) {
      toast('describe the voice first', true);
      return;
    }
    pushRecentPrompt(description);
    renderRecent();
    takes = [];
    pickedIndex = null;
    renderTakes();
    for (let i = 0; i < 3; i++) {
      try {
        const res = await request('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModel,
            input: testLineInput.value,
            instructions: description,
            response_format: 'wav',
            stream: false,
          }),
        });
        const arrayBuffer = await res.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        takes.push({ blob, samples: decodeWav(arrayBuffer).samples });
      } catch {
        break;
      }
      renderTakes();
    }
  }

  createBtn.addEventListener('click', async () => {
    const model = models.find((m) => m.id === selectedModel);
    if (!model || pickedIndex == null) {
      toast('generate and pick a take first', true);
      return;
    }
    const name = nameInput.value.trim();
    if (!name) {
      toast('name the voice first', true);
      return;
    }
    const target = model.x_design_voice_model ?? selectedModel;
    const form = new FormData();
    form.append('name', name);
    form.append('model', target);
    form.append('text', testLineInput.value);
    form.append('description', descBox.value.trim());
    form.append('file', takes[pickedIndex].blob, `${name}.wav`);
    try {
      await request('/v1/audio/voices', { method: 'POST', body: form });
      toast(`${name} created`);
    } catch {
      return;
    }
    location.hash = `#voices?model=${target}`;
  });

  async function refresh() {
    if (disposed) return;
    try {
      models = (await getJson('/v1/models')).data.filter((m) => m.x_kind === 'tts' && m.x_design);
    } catch {
      return;
    }
    if (disposed) return;
    if (!selectedModel && models.length) selectedModel = models[0].id;
    renderModelBar();
    renderSaveHint();
  }

  renderChips();
  renderRecent();
  refresh();
  return () => {
    disposed = true;
  };
}
