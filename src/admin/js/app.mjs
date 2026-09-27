// Admin shell: top bar (status/defaults/memory, polled), tab nav, and hash
// routing (`#tab` or `#tab?key=value&...`). Each tab is a module exported
// from `./<tab>.mjs` with the shape:
//
//   export function mount(view, params) { ...; return () => {/* cleanup */}; }
//
// `view` is the `<main id="view">` element, cleared before mount; `params`
// is the parsed query string from the hash (e.g. `#clone?model=X` ->
// `{model: "X"}`). `mount` returns an unmount function called before the
// next tab replaces it, so a module can clear its own timers/listeners.

import { getJson } from './api.mjs';
import * as overview from './overview.mjs';
import * as models from './models.mjs';
import * as voices from './voices.mjs';
import * as clone from './clone.mjs';
import * as design from './design.mjs';
import * as playground from './playground.mjs';
import * as samples from './samples.mjs';

const TABS = { overview, models, voices, clone, design, playground, samples };
const DEFAULT_TAB = 'overview';
const POLL_MS = 3000;

let currentUnmount = null;
let pollTimer = null;

function parseHash() {
  const hash = location.hash.replace(/^#/, '');
  const [tab, query] = hash.split('?');
  const params = {};
  if (query) {
    for (const [k, v] of new URLSearchParams(query)) params[k] = v;
  }
  return { tab: TABS[tab] ? tab : DEFAULT_TAB, params };
}

function renderTabs(active) {
  for (const a of document.querySelectorAll('#tabs a')) {
    a.classList.toggle('on', a.dataset.tab === active);
  }
}

function route() {
  const { tab, params } = parseHash();
  renderTabs(tab);
  if (typeof currentUnmount === 'function') currentUnmount();
  const view = document.getElementById('view');
  view.replaceChildren();
  currentUnmount = TABS[tab].mount(view, params) ?? null;
}

async function pollTopbar() {
  try {
    const [health, ps] = await Promise.all([
      getJson('/health', { silent: true }),
      getJson('/api/ps', { silent: true }),
    ]);
    const online = document.getElementById('online-pill');
    online.textContent = `● ONLINE :${location.port || 80}`;
    online.classList.add('ok');
    online.classList.remove('bad');
    document.getElementById('tts-pill').textContent = `TTS: ${health.tts.default}`;
    document.getElementById('stt-pill').textContent = `STT: ${health.stt.default}`;
    const used = ps.reduce((sum, m) => sum + (m.resident_bytes ?? 0), 0);
    const budget = health.profile.budget_bytes;
    document.getElementById('mem-pill').textContent =
      `MEM ${(used / 1e9).toFixed(1)} / ${(budget / 1e9).toFixed(0)} GB`;
  } catch {
    const online = document.getElementById('online-pill');
    online.textContent = '● OFFLINE';
    online.classList.add('bad');
    online.classList.remove('ok');
  }
}

function startPolling() {
  pollTopbar();
  pollTimer = setInterval(pollTopbar, POLL_MS);
}

// Module scripts execute after the document is parsed, i.e. after
// `DOMContentLoaded` has already fired — waiting for that event here would
// never run. Start directly.
window.addEventListener('hashchange', route);
startPolling();
route();
