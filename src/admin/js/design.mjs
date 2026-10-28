// Design tab (mockup: na-admin-design): pick a design-capable model, build
// a description with that model's own wizard form (fields, presets and tips
// from design-models.mjs, with a live preview of the exact `instructions`
// string), generate takes of a shared test line (how many is picked, each
// on its own seed so they differ),
// pick one, name it, and save it as a voice on `x_design_voice_model` (the
// model that can actually hold the voice).

import { getJson, request } from './api.mjs';
import { el, toast, playButton, waveformSvg, peaks, decodeWav } from './ui.mjs';
import { designModel } from './design-models.mjs';

const TEST_LINE = 'Warm coffee, a quiet morning, and nothing on fire yet.';
// Ready-made test lines, one click each: a spread of length and mood to hear
// how a designed voice reacts.
const SAMPLE_LINES = [
  'Yes.',
  'Hey, are you still there? I thought I lost you for a second.',
  'Wait, you actually got the job? That is amazing, I am so happy for you!',
  'Take a slow breath in, and let it go. There is no rush at all.',
  'I kept the porch light on for weeks, hoping he would come home.',
  'No. I told you three times already, and I am done repeating myself.',
  'Come closer, I will tell you a secret. Just between the two of us.',
  'Sleep well, little one. I will be right here when you wake up.',
  'Ha! I did not see that coming at all. Okay, okay, you win this round.',
  'Could you remind me what time the train leaves tomorrow morning?',
  'The storm rolled in just after dusk. By midnight the river had crept over the old stone bridge, and the whole village gathered at the church on the hill, sharing blankets and stories until the sun came up.',
];

export function mount(view, params) {
  let disposed = false;
  let models = [];
  let selectedModel = params.model ?? null;
  let values = {};
  const openFields = new Set(); // chip fields the user has unfolded
  let notesOpen = false;
  let takes = []; // {blob, buffer}
  let pickedIndex = null;

  const modelBar = el('div', { class: 'filt' });
  const notesWrap = el('div', {});
  const presetsWrap = el('div', { class: 'chip-row wrap' });
  const fieldsWrap = el('div', {});
  const tagsWrap = el('div', { class: 'chip-row wrap' });
  const descBox = el('textarea', { class: 'inp', rows: 4 });
  const previewBox = el('pre', { class: 'design-preview' });
  const warnBox = el('div', { class: 'design-warn' });
  const takesGrid = el('div', { class: 'vg3' });
  const nameInput = el('input', { class: 'inp' });
  const saveHint = el('span', { style: 'color:var(--muted);font-size:13px' });
  const createBtn = el('button', { class: 'btn primary', style: 'margin-left:auto' }, ['Create voice']);
  const testLineInput = el('input', { class: 'inp', value: TEST_LINE });
  const countSelect = el(
    'select',
    { class: 'inp design-select', style: 'margin-left:auto;width:auto;min-width:0', title: 'How many takes to generate' },
    [1, 2, 3, 4, 5].map((n) => el('option', { value: n }, [String(n)])),
  );
  const samplesList = el('div', {});

  const leftCard = el('div', { class: 'card' }, [
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['1']),
      el('div', { style: 'flex:1' }, [
        el('div', { class: 'lab' }, ['Model']),
        modelBar,
        el('span', { style: 'color:var(--muted);font-size:13px' }, [' only models that can design']),
        notesWrap,
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['2']),
      el('div', { style: 'flex:1;display:grid;grid-template-columns:1fr 1.2fr;gap:14px;align-items:start' }, [
        el('div', {}, [
          el('div', { class: 'lab' }, ['Starter themes · fill the whole form']),
          presetsWrap,
          el('div', { class: 'lab', style: 'margin-top:10px' }, ['Quick picks']),
          fieldsWrap,
        ]),
        // Sticky so the description and preview stay in view while picking.
        el('div', { class: 'side-sticky design-sticky' }, [
          el('div', { class: 'lab' }, ['Describe it · edit freely, form picks rewrite this']),
          descBox,
          el('div', { class: 'lab', style: 'margin-top:8px' }, ['Preview · exactly what is sent']),
          previewBox,
          warnBox,
        ]),
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['3']),
      el('div', { style: 'flex:1' }, [
        el('div', { style: 'display:flex;align-items:center' }, [
          el('div', { class: 'lab', style: 'margin:0' }, ['Takes · same description, a new seed each']),
          countSelect,
          el('button', { class: 'btn', style: 'margin-left:8px', onclick: generateTakes }, ['↻ Generate']),
        ]),
        takesGrid,
      ]),
    ]),
    el('div', { class: 'step' }, [
      el('span', { class: 'num' }, ['4']),
      el('div', { style: 'flex:1;display:flex;gap:14px;align-items:center;flex-wrap:wrap' }, [
        el('div', {}, [el('div', { class: 'lab' }, ['Name']), nameInput]),
        saveHint,
        createBtn,
      ]),
    ]),
  ]);

  const rightCard = el('div', { class: 'card' }, [
    el('h3', {}, ['Test line']),
    testLineInput,
    tagsWrap,
    el('div', { style: 'color:var(--muted);font-size:13px;margin-top:8px' }, [
      'Every take speaks this line, so you compare like with like.',
    ]),
    el('div', { class: 'lab', style: 'margin-top:14px' }, ['Sample lines · click to use']),
    samplesList,
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
              values = {};
              renderModelBar();
              renderSaveHint();
              renderModelUi();
            },
          },
          [m.id],
        ),
      ),
    );
  }

  function currentModel() {
    const m = models.find((x) => x.id === selectedModel);
    return designModel(selectedModel, m?.x_prompt_format?.hint);
  }

  /** Rebuilds the description from the form, then the preview. */
  function applyForm() {
    descBox.value = currentModel().build(values);
    renderPreview();
  }

  function renderPreview() {
    const dm = currentModel();
    const text = descBox.value.trim();
    descBox.placeholder = dm.placeholder ?? '';
    const lines = [`instructions: ${JSON.stringify(text)}`];
    previewBox.textContent = lines.join('\n');
    const warnings = [...(dm.check ? dm.check(text) : [])];
    if (dm.charLimit) {
      const note = `${text.length}/${dm.charLimit} characters`;
      if (text.length > dm.charLimit) warnings.push(`${note}: over the ${dm.charLimit}-character guideline.`);
      else previewBox.textContent += `\n${note}`;
    }
    warnBox.replaceChildren(...warnings.map((w) => el('div', {}, [`⚠ ${w}`])));
  }

  // Tips and pitfalls fold away; the summary line and licence stay visible.
  function renderNotes() {
    const dm = currentModel();
    const list = (items) => el('ul', { class: 'design-list' }, items.map((t) => el('li', {}, [t])));
    const tips = el('details', { class: 'fold', open: notesOpen }, [
      el('summary', {}, ['Tips and pitfalls']),
      list(dm.tips),
      dm.pitfalls?.length ? el('div', { class: 'lab', style: 'margin-top:6px' }, ['Pitfalls']) : null,
      dm.pitfalls?.length ? list(dm.pitfalls) : null,
    ]);
    tips.addEventListener('toggle', () => (notesOpen = tips.open));
    notesWrap.replaceChildren(
      el('div', { class: 'design-note' }, [
        el('b', {}, [dm.label]),
        ` ${dm.summary}`,
        dm.license ? el('div', { class: 'design-license' }, [`Licence: ${dm.license}`]) : null,
        tips,
      ]),
    );
  }

  function applyPreset(p) {
    values = { ...p.values };
    renderFields();
    applyForm();
    if (p.text) {
      descBox.value = p.text;
      renderPreview();
    }
    if (p.line) testLineInput.value = p.line;
    renderPreview();
  }

  function renderPresets() {
    presetsWrap.replaceChildren(
      ...currentModel().presets.map((p) =>
        el('span', { class: 'chip', onclick: () => applyPreset(p) }, [p.name]),
      ),
    );
  }

  function setValue(key, value) {
    values[key] = value || null;
    renderFields();
    applyForm();
  }

  function renderFields() {
    fieldsWrap.replaceChildren(
      ...currentModel().fields.map((f) => {
        let control;
        if (f.type === 'select') {
          control = el(
            'select',
            { class: 'inp design-select', onchange: (e) => setValue(f.key, e.target.value) },
            [
              el('option', { value: '' }, ['—']),
              ...f.options.map((o) => el('option', { value: o, selected: values[f.key] === o }, [o])),
            ],
          );
        } else if (f.type === 'text') {
          control = el('input', {
            class: 'inp',
            value: values[f.key] ?? '',
            placeholder: f.placeholder ?? '',
            oninput: (e) => {
              values[f.key] = e.target.value;
              applyForm();
            },
          });
        } else {
          // Chip fields fold into one line showing the pick; open to change it.
          const fold = el('details', { class: 'fold design-fold', open: openFields.has(f.key) }, [
            el('summary', {}, [
              el('span', { class: 'lab' }, [f.label]),
              el('span', { class: `design-pick${values[f.key] ? ' on' : ''}` }, [values[f.key] || '—']),
            ]),
            el(
              'div',
              { class: 'chip-row wrap', style: 'margin:0' },
              f.options.map((opt) =>
                el(
                  'span',
                  {
                    class: `chip${values[f.key] === opt ? ' on' : ''}`,
                    onclick: () => setValue(f.key, values[f.key] === opt ? null : opt),
                  },
                  [opt],
                ),
              ),
            ),
            f.hint ? el('div', { class: 'design-hint' }, [f.hint]) : null,
          ]);
          fold.addEventListener('toggle', () => {
            if (fold.open) openFields.add(f.key);
            else openFields.delete(f.key);
          });
          return fold;
        }
        return el('div', { class: 'chip-row design-field' }, [
          el('span', { class: 'lab' }, [f.label]),
          control,
          f.hint ? el('span', { class: 'design-hint' }, [f.hint]) : null,
        ]);
      }),
    );
  }

  function insertTag(tag) {
    const i = testLineInput.selectionStart ?? testLineInput.value.length;
    const j = testLineInput.selectionEnd ?? i;
    const v = testLineInput.value;
    testLineInput.value = `${v.slice(0, i)}${tag}${v.slice(j)}`;
    testLineInput.focus();
    testLineInput.setSelectionRange(i + tag.length, i + tag.length);
    renderPreview();
  }

  function renderTags() {
    const dm = currentModel();
    tagsWrap.replaceChildren(
      ...(dm.tags.length ? [el('span', { class: 'lab' }, [dm.tagLabel ?? 'Insert a tag'])] : []),
      ...dm.tags.map((t) => el('span', { class: 'chip', onclick: () => insertTag(t.insert) }, [t.label])),
    );
  }

  /** Re-renders everything that depends on the selected model. */
  function renderModelUi() {
    renderNotes();
    renderPresets();
    renderFields();
    renderTags();
    renderPreview();
  }

  function renderSamples() {
    samplesList.replaceChildren(
      ...SAMPLE_LINES.map((line) =>
        el(
          'div',
          {
            class: 'kv',
            style: 'cursor:pointer',
            onclick: () => {
              testLineInput.value = line;
              renderPreview();
            },
          },
          [el('span', {}, [line])],
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
    takes = [];
    pickedIndex = null;
    renderTakes();
    // The server's default seed is fixed, so takes sent without one come out
    // identical: each take gets its own, consecutive from a random base.
    const base = crypto.getRandomValues(new Uint32Array(1))[0];
    const count = Number(countSelect.value);
    for (let i = 0; i < count; i++) {
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
            seed: base + i,
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
    renderModelUi();
  }

  descBox.addEventListener('input', renderPreview);
  testLineInput.addEventListener('input', renderPreview);
  renderModelUi();
  renderSamples();
  refresh();
  return () => {
    disposed = true;
  };
}
