// Settings tab (naru task 2869): the config.toml keys that are not default
// models or voices, over `GET`/`PUT /api/settings`. One card per setting
// with its value in force, default, env override and when a save applies.
// Keep recent is a `serve` flag, so it is shown read-only.

import { getJson, putJson } from './api.mjs';
import { el, toast, fmtBytes } from './ui.mjs';

const INFO = {
  'defaults.keep_alive': {
    label: 'Keep alive',
    what: 'How long an idle model stays loaded. 0 unloads it once idle, -1 never unloads.',
    placeholder: 'e.g. 5m, 1h, 0, -1',
  },
  'memory.max_resident': {
    label: 'Memory budget',
    what: 'Most memory loaded models may use; idle ones are unloaded, oldest first, to fit.',
    placeholder: 'e.g. 8GiB, 512MiB',
    bytes: true,
  },
  'mic.device': {
    label: 'Microphone',
    what: "The ambient listener's input device. Unset means the built-in mic.",
  },
  'naru.url': {
    label: 'Naru address',
    what: "Where naru serve runs, for the ambient listener's check-in.",
    placeholder: 'http://127.0.0.1:7777',
  },
  'mlx.breeze_depth_cache': {
    label: 'Breeze depth cache',
    what: 'Breeze only: about 30% faster, but its samples differ from the reference.',
    bool: true,
  },
};

const APPLIES = {
  restart: 'Takes effect when naru-audio restarts.',
  listener_start: 'Takes effect the next time the ambient listener turns on.',
  model_load: 'Takes effect the next time Breeze loads (unload it to apply now).',
};

function show(key, v) {
  if (v === null || v === undefined) {
    if (key === 'mic.device') return 'built-in mic';
    if (key === 'naru.url') return 'none (check-in off)';
    return 'unset';
  }
  if (INFO[key].bytes && typeof v === 'number') return fmtBytes(v);
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return String(v);
}

export function mount(view) {
  let disposed = false;
  const notes = el('div');
  const list = el('div', { class: 'set-list' });
  view.append(notes, list);

  async function refresh() {
    let data;
    try {
      data = await getJson('/api/settings');
    } catch {
      return;
    }
    if (disposed) return;
    render(data);
  }

  async function save(body, btn) {
    btn.disabled = true;
    try {
      const data = await putJson('/api/settings', body);
      toast('Saved to config.toml');
      if (!disposed) render(data);
    } catch {
      btn.disabled = false;
    }
  }

  function render(data) {
    notes.replaceChildren();
    notes.append(
      el('p', { class: 'set-intro' }, [
        'Saved to ',
        el('code', {}, [data.config_path]),
        '. Default models and voices are set on the Models and Voices tabs.',
      ]),
    );
    if (data.error) notes.append(el('p', { class: 'set-warn' }, [`config.toml problem: ${data.error}`]));
    list.replaceChildren();
    for (const [key, s] of Object.entries(data.settings)) {
      list.append(card(key, s, data.mic_devices));
    }
    list.append(keepRecentCard(data.keep_recent));
  }

  function card(key, s, micDevices) {
    const info = INFO[key];
    let input;
    let read;
    if (info.bool) {
      input = el('input', { type: 'checkbox', id: `set-${key}` });
      input.checked = s.configured === true;
      read = () => input.checked;
    } else if (key === 'mic.device' && micDevices.length) {
      const names = [...micDevices];
      if (s.configured && !names.includes(s.configured)) names.push(s.configured);
      input = el('select', { class: 'inp', id: `set-${key}` }, [
        el('option', { value: '' }, ['Built-in mic (default)']),
        ...names.map((n) => el('option', { value: n }, [n])),
      ]);
      input.value = s.configured ?? '';
      read = () => input.value || null;
    } else {
      input = el('input', {
        class: 'inp',
        id: `set-${key}`,
        placeholder: info.placeholder ?? '',
      });
      input.value = s.configured == null ? '' : String(s.configured);
      read = () => input.value.trim() || null;
    }
    const saveBtn = el('button', { class: 'btn primary', type: 'button' }, ['Save']);
    saveBtn.addEventListener('click', () => save({ [key]: read() }, saveBtn));
    const resetBtn = el('button', { class: 'btn', type: 'button' }, ['Use default']);
    resetBtn.disabled = s.configured === null;
    resetBtn.addEventListener('click', () => save({ [key]: null }, resetBtn));

    const status = [];
    if (s.env_override) {
      status.push(
        el('p', { class: 'set-warn' }, [
          el('code', {}, [s.env]),
          ' is set, so it wins over config.toml; saving here changes the file but not what runs.',
        ]),
      );
    }
    if (s.restart_needed) {
      status.push(el('p', { class: 'set-warn' }, ['Restart needed: config.toml differs from what is running.']));
    }
    return el('div', { class: 'card set-card' }, [
      el('label', { class: 'set-label', for: `set-${key}` }, [info.label]),
      el('p', { class: 'set-what' }, [info.what]),
      el('div', { class: 'set-control' }, [
        info.bool ? el('label', { class: 'set-check', for: `set-${key}` }, [input, ' On']) : input,
        saveBtn,
        resetBtn,
      ]),
      el('div', { class: 'set-facts' }, [
        el('span', {}, ['In use: ', el('b', {}, [show(key, s.effective)])]),
        el('span', {}, ['Default: ', el('b', {}, [show(key, s.default)])]),
        el('span', {}, [el('code', {}, [`[${key.replace('.', '] ')}`])]),
      ]),
      el('p', { class: 'set-applies' }, [APPLIES[s.applies]]),
      ...status,
    ]);
  }

  function keepRecentCard(kr) {
    return el('div', { class: 'card set-card' }, [
      el('div', { class: 'set-label' }, ['Keep recent replies']),
      el('p', { class: 'set-what' }, [
        'A debugging aid: saves the audio of the newest 50 spoken replies. Off by default, so nothing spoken is stored.',
      ]),
      el('div', { class: 'set-facts' }, [el('span', {}, ['In use: ', el('b', {}, [kr.on ? 'on' : 'off'])])]),
      el('p', { class: 'set-applies' }, [
        'Not a config.toml setting. Turn it on by starting naru-audio with ',
        el('code', {}, ['serve --keep-recent']),
        ' or ',
        el('code', {}, [`${kr.env}=1`]),
        ', then restart.',
      ]),
    ]);
  }

  refresh();
  return () => {
    disposed = true;
  };
}
