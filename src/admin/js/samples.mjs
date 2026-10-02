// Samples tab (naru task 1462, board 126; reworked in task 1576): the
// voice-prep library. Picking, dropping or recording a voice file of any
// length uploads it as a raw clip (`POST /v1/audio/prep/clips`) and opens it
// straight in Sample Studio (`studio.mjs`, `#studio?clip=<id>`), which does
// the speaker separation, crop/cuts, best takes and cleaning chain. The tab
// lists the saved sample projects (`GET /v1/audio/prep/projects`: reopen or
// delete) and the finished samples, with the A/B preview, rename and delete.
//
// Exposes `window.naruAdmin.samples.loadFile(file)` so a headless test
// without a file picker can drive the upload path directly.
//
// `#samples?clip=<id>&model=<model>` is the old hand-off from the clone tab;
// it now forwards to the studio.

import { getJson, request, del, uploadClip } from './api.mjs';
import { el, toast, playButton } from './ui.mjs';

/** A sample's audio length: its kept segments when it has them (its `range` then only brackets them). */
function sampleSecs(s) {
  if (s.segments?.length) return s.segments.reduce((sum, g) => sum + (g.end - g.start), 0);
  return s.range ? s.range.end - s.range.start : null;
}

function fmtMmSs(secs) {
  if (secs == null || Number.isNaN(secs)) return '—';
  const s = Math.max(0, Math.round(secs));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

export function mount(view, params) {
  // The clone tab's old hand-off: forward to the studio.
  if (params.clip) {
    const modelPart = params.model ? `&model=${encodeURIComponent(params.model)}` : '';
    location.hash = `#studio?clip=${encodeURIComponent(params.clip)}${modelPart}`;
    return () => {};
  }

  let disposed = false;
  let samples = []; // [{id, name, range, engines, ...}]
  let projects = []; // [{clip_id, name, duration, updated_at, speaker, segment_count, has_transcript}]
  let selectedId = params.sample ?? null;
  let renameValue = '';
  let uploading = false;
  // The clone tab's model, handed on to the studio.
  const cloneModel = params.model ?? null;

  const uploadCard = el('div', { class: 'card' });
  const projectsCard = el('div', { class: 'card' });
  const listCard = el('div', { class: 'card' });
  const sidePanel = el('div', { class: 'card' });
  view.append(el('div', { class: 'side' }, [el('div', {}, [uploadCard, projectsCard, listCard]), sidePanel]));

  /** `node.replaceChildren(...)` is the native DOM method, not `el()`'s own
   * child list (ui.mjs:21, which drops `null`/`undefined` entries) — passed
   * a bare `null` from a `cond ? el(...) : null` ternary, it stringifies it
   * to the literal text "null" instead of skipping it. Every top-level
   * `replaceChildren` call in this tab builds its list that way, so they all
   * go through this filter (naru task 1462 review). */
  function setChildren(node, children) {
    node.replaceChildren(...children.filter((c) => c != null));
  }

  function openStudioClip(clipId) {
    const modelPart = cloneModel ? `&model=${encodeURIComponent(cloneModel)}` : '';
    location.hash = `#studio?clip=${encodeURIComponent(clipId)}${modelPart}`;
  }

  function openStudio(id) {
    const modelPart = cloneModel ? `&model=${encodeURIComponent(cloneModel)}` : '';
    location.hash = `#studio?sample=${encodeURIComponent(id)}${modelPart}`;
  }

  // ---- upload ---------------------------------------------------------------

  /** Any length: the clip is streamed up, then opened raw in the studio. */
  async function loadFile(file) {
    if (uploading) return;
    uploading = true;
    renderUpload();
    try {
      const clip = await uploadClip(file);
      toast(`uploaded · ${fmtMmSs(clip.duration_secs)}`);
      openStudioClip(clip.id);
    } catch {
      uploading = false;
      if (!disposed) renderUpload();
    }
  }

  function renderUpload() {
    const input = el('input', { type: 'file', accept: 'audio/*', style: 'display:none' });
    input.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) loadFile(file);
      e.target.value = '';
    });
    const browse = el('span', { style: 'color:var(--cyan);cursor:pointer' }, ['browse']);
    browse.addEventListener('click', () => input.click());
    const drop = el('div', { class: 'drop' }, [
      uploading
        ? el('span', {}, ['uploading… the studio opens when it is in'])
        : el('span', {}, ['⇪ drop a voice file (any length) or ', browse, input]),
    ]);
    drop.addEventListener('dragover', (e) => {
      e.preventDefault();
      drop.classList.add('over');
    });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('over');
      const file = e.dataTransfer.files[0];
      if (file) loadFile(file);
    });
    setChildren(uploadCard, [
      el('h3', {}, ['New sample project']),
      drop,
      el('div', { class: 'studio-note', style: 'margin-top:6px' }, [
        'WAV / MP3 / M4A, up to 4 h. It opens raw in Sample Studio — separate speakers, cut, pick the best takes, clean.',
      ]),
    ]);
  }

  // ---- projects -------------------------------------------------------------

  async function deleteProjectFor(p) {
    if (!confirm(`Delete the project "${p.name}"? The uploaded clip stays; its edits are lost.`)) return;
    try {
      await del(`/v1/audio/prep/clips/${encodeURIComponent(p.clip_id)}/project`);
      toast(`${p.name} deleted`);
    } catch {
      return;
    }
    refresh();
  }

  function projectRow(p) {
    return el('div', { class: 'row' }, [
      el('b', {}, [p.name]),
      el('span', { style: 'color:var(--muted);font-size:13px' }, [
        `${fmtMmSs(p.duration)} · ${p.speaker != null ? `speaker ${p.speaker}` : 'all speakers'} · ${p.segment_count} segment${p.segment_count === 1 ? '' : 's'} · ${
          p.updated_at ? new Date(p.updated_at).toLocaleString() : ''
        }`,
      ]),
      el('button', { class: 'btn', onclick: () => openStudioClip(p.clip_id) }, ['open']),
      el('button', { class: 'btn r', onclick: () => deleteProjectFor(p) }, ['delete']),
    ]);
  }

  function renderProjects() {
    setChildren(projectsCard, [
      el('h3', {}, ['Sample projects']),
      ...projects.map(projectRow),
      projects.length ? null : el('div', { class: 'placeholder' }, ['no saved projects yet']),
    ]);
  }

  // ---- samples library --------------------------------------------------------

  function statusOf(s) {
    return s.engines?.length ? '✓ cleaned' : 'raw, not cleaned yet';
  }

  function libraryRow(s) {
    return el(
      'div',
      { class: `row${s.id === selectedId ? ' sel' : ''}`, style: 'cursor:pointer', onclick: () => selectSample(s.id) },
      [
        el('b', {}, [s.name]),
        el('span', { style: 'color:var(--muted);width:50px;text-align:right' }, [
          fmtMmSs(sampleSecs(s)),
        ]),
        el('span', { style: `color:${s.engines?.length ? 'var(--green)' : 'var(--amber)'};font-size:13px` }, [
          statusOf(s),
        ]),
      ],
    );
  }

  function renderList() {
    setChildren(listCard, [
      el('h3', {}, ['Samples']),
      ...samples.map(libraryRow),
      samples.length ? null : el('div', { class: 'placeholder' }, ['no samples yet']),
    ]);
  }

  /** A `.row` that hosts a `playButton()` needs `position:relative` — `.play`
   * positions itself `absolute` against the nearest positioned ancestor, and
   * plain `.row` (unlike `.vc`/`.cmp`, which other tabs use for this) has
   * none, so without this every play button on the tab stacks at the page's
   * top-right corner instead of next to its own label. */
  function playRow(getSrc, label) {
    return el('div', { class: 'row', style: 'position:relative;padding-right:30px' }, [playButton(getSrc), label]);
  }

  function kv(label, value) {
    return el('div', { class: 'kv' }, [el('span', {}, [label]), el('span', {}, [String(value)])]);
  }

  function renderSide() {
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
      kv('length', fmtMmSs(sampleSecs(s))),
      kv('status', statusOf(s)),
      kv('speaker', s.speaker ?? '—'),
      s.warnings?.length ? el('div', { class: 'warn' }, [s.warnings.join('; ')]) : null,
      el('div', { class: 'lab', style: 'margin-top:10px' }, ['A/B preview']),
      playRow(() => rawUrl, 'raw crop'),
      playRow(() => cleanUrl, 'cleaned'),
      el('button', { class: 'big', onclick: () => openStudio(s.id) }, ['Open in Sample Studio']),
      el('div', { class: 'lab', style: 'margin-top:10px' }, ['Rename']),
      el('div', { class: 'row' }, [
        renameInput,
        el('button', { class: 'btn', onclick: () => renameSample(s) }, ['save']),
      ]),
      el('button', { class: 'big r', onclick: () => deleteSample(s) }, ['Delete']),
    ]);
  }

  function selectSample(id) {
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

  // ---- boot -------------------------------------------------------------------

  async function refresh() {
    if (disposed) return;
    getJson('/v1/audio/prep/projects')
      .then((r) => {
        if (disposed) return;
        projects = r.projects;
        renderProjects();
      })
      .catch(() => {});
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

  renderUpload();
  renderProjects();
  renderList();
  renderSide();
  refresh();
  return () => {
    disposed = true;
    if (window.naruAdmin) delete window.naruAdmin.samples;
  };
}
