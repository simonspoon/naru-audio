// Models tab (mockup: na-admin-models): a filterable/searchable table of
// every catalog model plus a side panel for the selected one. Pull
// progress and cancel poll `GET /api/pulls`; everything per-model comes
// from `GET /v1/models`' `x_*` fields, never hardcoded.

import { getJson, postJson, putJson, del, streamNdjson } from './api.mjs';
import { el, toast, fmtBytes, fmtUptime, confirmModal } from './ui.mjs';

const FILTERS = ['All', 'TTS', 'STT', 'On disk', 'Running'];

export function mount(view, params) {
  let disposed = false;
  let models = [];
  let selectedId = params.model ?? null;
  let filter = 'All';
  let search = '';
  let sortKey = null; // a SORTS key; null keeps catalog order
  let sortDir = 'desc';
  let pullTimer = null;
  let budgetBytes = 0; // /health profile.budget_bytes; 0 until fetched
  const busy = new Map(); // id -> 'loading' | 'unloading' while /api/load is in flight

  const filtBar = el('div', { class: 'filt' });
  const table = el('table', { class: 'tb' });
  const listCard = el('div', { class: 'card' }, [filtBar, el('div', { class: 'tb-wrap' }, [table])]);
  const sidePanel = el('div', { class: 'card side-sticky' });
  const memCard = el('div', { class: 'card mem-card' });
  view.append(memCard, el('div', { class: 'side' }, [listCard, sidePanel]));

  function renderMem() {
    const used = models.reduce((sum, m) => sum + (m.x_loaded ? (m.x_size_bytes ?? 0) : 0), 0);
    const pct = budgetBytes ? Math.min(100, Math.round((100 * used) / budgetBytes)) : 0;
    memCard.replaceChildren(
      el('div', { class: 'kv' }, [
        el('span', {}, ['memory budget']),
        el('span', {}, [`${fmtBytes(used)} / ${budgetBytes ? fmtBytes(budgetBytes) : '—'}`]),
      ]),
      el('div', { class: `mem-track${pct >= 85 ? ' warn' : ''}` }, [el('i', { style: `width:${pct}%` })]),
    );
  }

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
    if (sortKey) {
      const by = SORTS[sortKey];
      rows.sort((a, b) => {
        const av = by(a);
        const bv = by(b);
        const cmp = av < bv ? -1 : av > bv ? 1 : 0;
        return sortDir === 'asc' ? cmp : -cmp;
      });
    }
    // Headers are rebuilt on every render (sort, 3s poll); keep keyboard focus.
    const focused = table.contains(document.activeElement) ? document.activeElement.dataset.sort : null;
    table.replaceChildren(
      el('tr', {}, [
        th('Model', 'name'),
        th('Kind', 'kind'),
        el('th', {}, ['Can do']),
        el('th', {}, ['License']),
        th('Size', 'size', 'size'),
        th('State', 'state'),
        el('th', {}, ['']),
      ]),
      ...rows.map(modelRow),
    );
    if (focused) table.querySelector(`th[data-sort="${focused}"]`)?.focus();
    if (selectedId && !rows.some((m) => m.id === selectedId)) {
      // Selection fell out of the filtered view; keep it selected (the
      // panel still shows it) but nothing in the table highlights.
    }
  }

  // Same behaviour as Naru's ccTable: first click sorts descending, the
  // next flips it; Tab reaches a header, Enter or Space sorts.
  const SORTS = {
    name: (m) => m.id.toLowerCase(),
    kind: (m) => m.x_kind,
    size: (m) => m.x_size_bytes ?? -1,
    state: (m) => (m.x_loaded ? 3 : pulls.has(m.id) ? 2 : m.x_pulled ? 1 : 0),
  };

  function th(label, key, cls = '') {
    const on = key === sortKey;
    const click = () => {
      if (on) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      else {
        sortKey = key;
        sortDir = 'desc';
      }
      renderTable();
    };
    return el(
      'th',
      {
        class: `${cls} sortable`.trim(),
        tabindex: '0',
        'data-sort': key,
        'aria-sort': on ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none',
        onclick: click,
        onkeydown: (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            click();
          }
        },
      },
      [label, on ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''],
    );
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
    // Wraps only between words ("MIT AND Apache-2.0"), never at a hyphen.
    const words = (nc ? 'NC' : m.x_license ?? '—').split(' ');
    return el(
      'span',
      { class: 'lic', style: `color:${color}` },
      words.flatMap((w, i) => [i ? ' ' : null, el('span', { class: 'nw' }, [w])]),
    );
  }

  function stateCell(m) {
    const pull = pulls.get(m.id);
    if (busy.has(m.id)) {
      return el('td', { class: 'st', style: 'color:var(--cyan)' }, [
        `◌ ${busy.get(m.id)} `,
        el('span', { class: 'bar indet' }, [el('i')]),
      ]);
    }
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
        '● running',
      ]);
    }
    if (m.x_pulled) {
      return el('td', { class: 'st', style: 'color:var(--muted)' }, [
        '● on disk',
      ]);
    }
    return el('td', { class: 'st', style: 'color:var(--muted)' }, [
      '○ not pulled',
    ]);
  }

  function actionCell(m) {
    const pull = pulls.get(m.id);
    if (pull) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn',
            onclick: async (e) => {
              e.stopPropagation();
              await del(`/api/pulls/${encodeURIComponent(m.id)}`, { silent: true }).catch(() => {});
            },
          },
          ['Cancel'],
        ),
      ]);
    }
    if (m.x_loaded) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn',
            disabled: busy.has(m.id),
            onclick: (e) => {
              e.stopPropagation();
              unload(m);
            },
          },
          busyLabel(m, 'Unload'),
        ),
      ]);
    }
    if (m.x_pulled) {
      return el('td', {}, [
        el(
          'button',
          {
            class: 'btn',
            disabled: busy.has(m.id),
            onclick: (e) => {
              e.stopPropagation();
              load(m);
            },
          },
          busyLabel(m, 'Load'),
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
        ['Pull'],
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
        el('td', { class: 'kind' }, [m.x_kind.toUpperCase()]),
        el('td', {}, capChips(m)),
        el('td', { class: 'license' }, [licenseChip(m)]),
        el('td', { class: 'size' }, [fmtBytes(m.x_size_bytes)]),
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
      kv('state', busy.has(m.id) ? busy.get(m.id) : m.x_loaded ? `running${m.x_loaded_at ? ' ' + fmtUptime(m.x_loaded_at) : ''}` : m.x_pulled ? 'on disk' : 'not pulled'),
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
          { class: 'btn primary wide', onclick: () => setDefault(m) },
          [m.x_default ? '★ Default' : `★ Default for ${m.x_kind.toUpperCase()}`],
        ),
      );
    }
    if (m.x_loaded) {
      nodes.push(
        el('button', { class: 'btn wide', disabled: busy.has(m.id), onclick: () => unload(m) }, busyLabel(m, 'Unload')),
      );
    }
    if (m.x_pulled) {
      nodes.push(
        el('button', { class: 'btn danger wide', onclick: () => removeModel(m) }, ['Delete from disk']),
      );
    }
    sidePanel.replaceChildren(...nodes);
  }

  function kv(label, value) {
    return el('div', { class: 'kv' }, [el('span', {}, [label]), el('span', {}, [String(value)])]);
  }

  // The API reports no byte progress for a load, so the in-flight state is
  // indeterminate: a spinner label and an animated bar, never a percentage.
  function busyLabel(m, idle) {
    const what = busy.get(m.id);
    if (!what) return [idle];
    return [el('span', { class: 'spin' }), what === 'loading' ? 'Loading…' : 'Unloading…'];
  }

  async function setBusy(m, what, body, done) {
    if (busy.has(m.id)) return;
    busy.set(m.id, what);
    renderTable();
    if (m.id === selectedId) renderSide();
    try {
      await postJson('/api/load', body);
      if (done) toast(done);
      await refresh();
    } catch {
      // postJson already toasted the failure
    } finally {
      busy.delete(m.id);
      if (!disposed) {
        renderTable();
        if (m.id === selectedId) renderSide();
      }
    }
  }

  function load(m) {
    return setBusy(m, 'loading', { model: m.id, kind: m.x_kind }, `${m.id} loaded`);
  }

  function unload(m) {
    return setBusy(m, 'unloading', { model: m.id, kind: m.x_kind, keep_alive: 0 });
  }

  async function removeModel(m) {
    if (!(await confirmModal(`Delete "${m.id}" from disk? This cannot be undone.`))) return;
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
    renderMem();
    renderFilters();
    renderTable();
    renderSide();
  }

  async function refresh() {
    if (disposed) return;
    const health = getJson('/health', { silent: true }).catch(() => null);
    try {
      models = (await getJson('/v1/models')).data;
    } catch {
      return;
    }
    budgetBytes = (await health)?.profile?.budget_bytes ?? budgetBytes; // keep last-known on failure
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
