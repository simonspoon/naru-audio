// Models tab (mockup: na-admin-models): a filterable/searchable table of
// every catalog model plus a side panel for the selected one. Pull
// progress and cancel poll `GET /api/pulls`; everything per-model comes
// from `GET /v1/models`' `x_*` fields, never hardcoded.

import { getJson, postJson, putJson, del, streamNdjson } from './api.mjs';
import { el, toast, fmtBytes, fmtUptime } from './ui.mjs';

const FILTERS = ['All', 'TTS', 'STT', 'On disk', 'Running'];

export function mount(view, params) {
  let disposed = false;
  let models = [];
  let selectedId = params.model ?? null;
  let filter = 'All';
  let search = '';
  let pullTimer = null;

  const filtBar = el('div', { class: 'filt' });
  const table = el('table', { class: 'tb' });
  const listCard = el('div', { class: 'card' }, [filtBar, table]);
  const sidePanel = el('div', { class: 'card' });
  view.append(el('div', { class: 'side' }, [listCard, sidePanel]));

  function matchesFilter(m) {
    switch (filter) {
      case 'TTS':
        return m.x_kind === 'tts';
      case 'STT':
        return m.x_kind === 'stt';
      case 'On disk':
        return m.x_pulled;
      case 'Running':
        return m.x_loaded;
      default:
        return true;
    }
  }

  function renderFilters() {
    filtBar.replaceChildren(
      ...FILTERS.map((f) =>
        el(
          'span',
          {
            class: f === filter ? 'on' : '',
            onclick: () => {
              filter = f;
              renderAll();
            },
          },
          [f],
        ),
      ),
      el('input', {
        type: 'search',
        placeholder: '⌕ search models',
        value: search,
        oninput: (e) => {
          search = e.target.value;
          renderTable();
        },
      }),
    );
  }

  function renderTable() {
    const rows = models
      .filter(matchesFilter)
      .filter((m) => !search || m.id.toLowerCase().includes(search.toLowerCase()));
    table.replaceChildren(
      el('tr', {}, [
        el('th', {}, ['Model']),
        el('th', {}, ['Kind']),
        el('th', {}, ['Can do']),
        el('th', {}, ['License']),
        el('th', {}, ['State']),
        el('th', {}, ['']),
      ]),
      ...rows.map(modelRow),
    );
    if (selectedId && !rows.some((m) => m.id === selectedId)) {
      // Selection fell out of the filtered view; keep it selected (the
      // panel still shows it) but nothing in the table highlights.
    }
  }

  function capChips(m) {
    if (m.x_kind !== 'tts') return [];
    return [
      cap('clone', m.x_clone),
      cap('design', m.x_design),
      cap('transcript', m.x_clone_requires_transcript),
    ];
  }

  function cap(label, on) {
    return el('span', { class: `cap${on ? ' y' : ''}` }, [label]);
  }

  function licenseChip(m) {
    const nc = m.x_non_commercial;
    const color = nc ? 'var(--amber)' : 'var(--green)';
    return el('span', { class: 'lic', style: `color:${color};border-color:${color}` }, [
      nc ? 'NC' : m.x_license ?? '—',
    ]);
  }

  function stateCell(m) {
    const pull = pulls.get(m.id);
    if (pull) {
      const pct = pull.model_total ? Math.round((100 * pull.model_completed) / pull.model_total) : 0;
      return el('td', { class: 'st', style: 'color:var(--cyan)' }, [
        '↓ pulling ',
        el('span', { class: 'bar' }, [el('i', { style: `width:${pct}%` })]),
        ` ${pct}%`,
      ]);
    }
    if (m.x_loaded) {
      return el('td', { class: 'st', style: 'color:var(--green)' }, [
        `● running · ${fmtBytes(m.x_size_bytes)}`,
      ]);
    }
    if (m.x_pulled) {
      return el('td', { class: 'st', style: 'color:var(--muted)' }, [
        `● on disk · ${fmtBytes(m.x_size_bytes)}`,
      ]);
    }
    return el('td', { class: 'st', style: 'color:var(--muted)' }, [
      `○ not pulled · ${fmtBytes(m.x_size_bytes)}`,
    ]);
  }

  function actionCell(m) {
    const pull = pulls.get(m.id);
    if (pull) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn x',
            onclick: async (e) => {
              e.stopPropagation();
              await del(`/api/pulls/${encodeURIComponent(m.id)}`, { silent: true }).catch(() => {});
            },
          },
          ['cancel'],
        ),
      ]);
    }
    if (m.x_loaded) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn x',
            onclick: (e) => {
              e.stopPropagation();
              unload(m);
            },
          },
          ['unload'],
        ),
      ]);
    }
    if (m.x_pulled) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn',
            onclick: (e) => {
              e.stopPropagation();
              load(m);
            },
          },
          ['load'],
        ),
      ]);
    }
    return el('td', {}, [
      el(
        'button',
        {
          class: 'btn',
          onclick: (e) => {
            e.stopPropagation();
            startPull(m);
          },
        },
        ['pull'],
      ),
    ]);
  }

  function modelRow(m) {
    const tr = el(
      'tr',
      { class: m.id === selectedId ? 'sel' : '', onclick: () => selectModel(m.id) },
      [
        el('td', { style: 'color:var(--text-h)' }, [
          m.id,
          m.x_default ? ' ' : null,
          m.x_default ? el('span', { class: 'star' }, ['★']) : null,
        ]),
        el('td', {}, [m.x_kind.toUpperCase()]),
        el('td', {}, capChips(m)),
        el('td', {}, [licenseChip(m)]),
      ],
    );
    tr.append(stateCell(m), actionCell(m));
    return tr;
  }

  function selectModel(id) {
    selectedId = id;
    renderTable();
    renderSide();
  }

  function renderSide() {
    const m = models.find((x) => x.id === selectedId);
    if (!m) {
      sidePanel.replaceChildren(el('div', { class: 'placeholder' }, ['select a model']));
      return;
    }
    const rows = [
      kv('state', m.x_loaded ? `running${m.x_loaded_at ? ' ' + fmtUptime(m.x_loaded_at) : ''}` : m.x_pulled ? 'on disk' : 'not pulled'),
      kv('memory', m.x_loaded ? fmtBytes(m.x_size_bytes) : '—'),
      kv('voices', m.x_kind === 'tts' ? `${m.x_voice_count} · ${m.x_cloned_voice_count} cloned` : '—'),
      kv('languages', m.x_languages ? m.x_languages.length : '—'),
      kv('source', m.x_source ?? '—'),
    ];
    const nodes = [el('h3', {}, [m.id]), ...rows];
    if (m.x_kind === 'tts' || m.x_kind === 'stt') {
      nodes.push(
        el(
          'button',
          { class: 'big', onclick: () => setDefault(m) },
          [m.x_default ? '★ Default' : `★ Default for ${m.x_kind.toUpperCase()}`],
        ),
      );
    }
    if (m.x_loaded) {
      nodes.push(el('button', { class: 'big', onclick: () => unload(m) }, ['Unload']));
    }
    if (m.x_pulled) {
      nodes.push(
        el('button', { class: 'big r', onclick: () => removeModel(m) }, ['Delete from disk']),
      );
    }
    sidePanel.replaceChildren(...nodes);
  }

  function kv(label, value) {
    return el('div', { class: 'kv' }, [el('span', {}, [label]), el('span', {}, [String(value)])]);
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

  async function removeModel(m) {
    if (!confirm(`Delete "${m.id}" from disk? This cannot be undone.`)) return;
    try {
      await del(`/api/models/${encodeURIComponent(m.id)}`);
      toast(`${m.id} deleted`);
    } catch {
      return;
    }
    refresh();
  }

  async function setDefault(m) {
    try {
      await putJson('/api/defaults', { [m.x_kind]: m.id });
      toast(`${m.id} is now default ${m.x_kind}`);
    } catch {
      return;
    }
    refresh();
  }

  const pulls = new Map(); // name -> {model_completed, model_total}

  function startPull(m) {
    streamNdjson('/api/pull', { model: m.id }, (line) => {
      if (line.model_total) {
        pulls.set(m.id, line);
        renderTable();
        if (m.id === selectedId) renderSide();
      }
      if (line.error) {
        pulls.delete(m.id);
        toast(line.error.message ?? 'pull failed', true);
        renderTable();
        return;
      }
      if (line.status === 'success') {
        pulls.delete(m.id);
      }
    })
      .catch((e) => {
        pulls.delete(m.id);
        if (e.code !== 'pull_in_progress') toast(e.message ?? 'pull failed', true);
      })
      .finally(() => refresh());
  }

  async function pollPulls() {
    try {
      const list = await getJson('/api/pulls', { silent: true });
      const seen = new Set();
      for (const p of list) {
        seen.add(p.model);
        pulls.set(p.model, p);
      }
      for (const name of [...pulls.keys()]) {
        if (!seen.has(name)) pulls.delete(name);
      }
    } catch {
      // offline; leave last-known state
    }
    renderTable();
  }

  function renderAll() {
    renderFilters();
    renderTable();
    renderSide();
  }

  async function refresh() {
    if (disposed) return;
    try {
      models = (await getJson('/v1/models')).data;
    } catch {
      return;
    }
    if (disposed) return;
    if (!selectedId && models.length) selectedId = models[0].id;
    renderAll();
  }

  refresh();
  pullTimer = setInterval(pollPulls, 3000);
  return () => {
    disposed = true;
    clearInterval(pullTimer);
  };
}
