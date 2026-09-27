// Playground tab -- placeholder for a later subtask (plan.md section 4 ST5/ST6).
// Same mount contract as every tab module (see app.mjs): `mount(view,
// params)` returns an unmount function.

import { el } from './ui.mjs';

export function mount(view, params) {
  view.append(
    el('div', { class: 'card' }, [
      el('h3', {}, ['PLAYGROUND']),
      el('div', { class: 'placeholder' }, [
        params.model ? `playground · ${params.model} -- coming soon` : 'coming soon',
      ]),
    ]),
  );
  return () => {};
}
