// Sample Studio's clip waveform (naru task 1576): an overview strip over the
// whole clip with a draggable viewport window, and a zoomable main view with
// the edit overlays. Neither ever decodes the clip's audio: both draw server
// peaks (`GET /v1/audio/prep/clips/{id}/peaks`) -- the overview's fetched once,
// the main view's re-fetched (debounced, abortable) for the range on screen.
// Also here: `createSegmentPlayer`, which plays a list of spans of the clip's
// own `<audio>` stream by seeking between them.

import { getJson } from './api.mjs';
import { el } from './ui.mjs';
import { speakerColor } from './studio-edit.mjs';

const OVERVIEW_BUCKETS = 2000;
const OVERVIEW_H = 52;
const RULER_H = 16;
const TAKES_H = 18;
// The waveform grows with the window.
const waveH = () => Math.round(Math.min(380, Math.max(150, window.innerHeight * 0.3)));
const LANE_H = 9;
const MIN_VIEW_S = 0.5;
const MIN_CROP_S = 0.2;
const PEAKS_DEBOUNCE_MS = 150;
const DRAG_PX = 4;

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(s < 10 ? 1 : 0).padStart(s < 10 ? 4 : 2, '0')}`;
}

/** Plays spans of `src` back to back: seeks to the next span's start when one ends. */
export function createSegmentPlayer(src, { onTime, onStop }) {
  let audio = null;
  let segs = [];
  let idx = 0;
  let raf = 0;
  let tag = null;

  function stop() {
    cancelAnimationFrame(raf);
    if (audio) {
      audio.pause();
      audio = null;
    }
    const was = tag;
    tag = null;
    if (was != null) onStop?.();
  }

  function loop() {
    if (!audio) return;
    const t = audio.currentTime;
    onTime?.(t);
    if (t >= segs[idx].end - 0.02 || audio.ended) {
      idx++;
      if (idx >= segs.length) {
        stop();
        return;
      }
      audio.currentTime = segs[idx].start;
    } else if (t < segs[idx].start - 0.05) {
      audio.currentTime = segs[idx].start;
    }
    raf = requestAnimationFrame(loop);
  }

  /** `list`: sorted spans; `from` (optional) starts inside the span that holds it. `label` identifies this playback to `playingTag()`. */
  function play(list, label, from = null) {
    stop();
    if (!list.length) return;
    segs = list;
    idx = 0;
    if (from != null) {
      const at = segs.findIndex((s) => from < s.end);
      if (at >= 0) idx = at;
    }
    tag = label ?? '';
    const a = new Audio(src);
    audio = a;
    const start = from != null && from >= segs[idx].start && from < segs[idx].end ? from : segs[idx].start;
    const seekAndPlay = () => {
      a.currentTime = start;
      a.play().catch(() => {
        if (audio === a) stop();
      });
      raf = requestAnimationFrame(loop);
    };
    if (a.readyState >= 1) seekAndPlay();
    else a.addEventListener('loadedmetadata', seekAndPlay, { once: true });
    a.addEventListener('error', () => {
      if (audio === a) stop();
    });
  }

  return { play, stop, playingTag: () => tag };
}

/**
 * @param {{clipId: string, duration: number, onSeek: (t: number) => void,
 *   onSelect: (sel: {start: number, end: number} | null) => void,
 *   onCrop: (crop: {start: number, end: number}) => void}} opts
 */
export function createWave({ clipId, duration, onSeek, onSelect, onCrop }) {
  const peaksUrl = `/v1/audio/prep/clips/${encodeURIComponent(clipId)}/peaks`;
  let view = { start: 0, end: duration };
  let overview = null; // peaks response for the whole clip
  let top = 0.01; // loudest peak in the overview: the waveform's full scale
  let detail = null; // peaks response for (a bit more than) the view
  let detailAbort = null;
  let detailTimer = null;
  let data = emptyData();
  let selection = null;
  let playhead = 0;
  let cropDraft = null; // the crop while a handle is being dragged

  function emptyData() {
    return { crop: { start: 0, end: duration }, cuts: [], lanes: [], overlaps: [], excludeOverlaps: true, takes: null, kept: [], activeSpeaker: null };
  }

  // ---- layout ---------------------------------------------------------------

  const ovCanvas = el('canvas', { class: 'sw-canvas' });
  const ovWindow = el('div', { class: 'sw-window' });
  const overviewBox = el('div', { class: 'sw-overview', style: `height:${OVERVIEW_H}px` }, [ovCanvas, ovWindow]);
  const canvas = el('canvas', { class: 'sw-canvas' });
  const selBox = el('div', { class: 'sw-sel hidden' });
  const startHandle = el('div', { class: 'sw-handle start', title: 'crop start — drag' });
  const endHandle = el('div', { class: 'sw-handle end', title: 'crop end — drag' });
  const playheadEl = el('div', { class: 'sw-playhead' });
  const main = el('div', { class: 'sw-main' }, [canvas, selBox, startHandle, endHandle, playheadEl]);
  const status = el('div', { class: 'sw-status' });
  const root = el('div', { class: 'sw' }, [overviewBox, main, status]);

  function hasTakes() {
    return !!data.takes?.length;
  }

  function mainHeight() {
    return RULER_H + (hasTakes() ? TAKES_H : 0) + waveH() + data.lanes.length * LANE_H + 2;
  }

  const clampT = (t) => Math.min(duration, Math.max(0, t));
  const tToX = (t, w) => ((t - view.start) / (view.end - view.start)) * w;
  const xToT = (x, w) => view.start + (x / w) * (view.end - view.start);

  // ---- peaks ----------------------------------------------------------------

  /** Min/max of a peaks response over `[t0, t1)`. */
  function peakRange(src, t0, t1) {
    const n = src.min.length;
    const bw = (src.end - src.start) / n;
    const i0 = Math.max(0, Math.min(n - 1, Math.floor((t0 - src.start) / bw)));
    const i1 = Math.max(i0, Math.min(n - 1, Math.ceil((t1 - src.start) / bw) - 1));
    let lo = 0;
    let hi = 0;
    for (let i = i0; i <= i1; i++) {
      if (src.min[i] < lo) lo = src.min[i];
      if (src.max[i] > hi) hi = src.max[i];
    }
    return [lo, hi];
  }

  function detailCovers() {
    return detail && detail.start <= view.start + 1e-6 && detail.end >= view.end - 1e-6;
  }

  async function fetchOverview() {
    try {
      overview = await getJson(`${peaksUrl}?buckets=${OVERVIEW_BUCKETS}`, { silent: true });
      top = Math.max(0.01, ...overview.max, ...overview.min.map((v) => -v));
    } catch {
      status.textContent = 'waveform unavailable';
    }
    draw();
  }

  function scheduleDetail() {
    clearTimeout(detailTimer);
    detailTimer = setTimeout(fetchDetail, PEAKS_DEBOUNCE_MS);
  }

  async function fetchDetail() {
    const len = view.end - view.start;
    if (len >= duration * 0.9 || detailCovers()) return;
    // A bit of margin each side so a small pan reuses this fetch.
    const start = clampT(view.start - len * 0.25);
    const end = clampT(view.end + len * 0.25);
    const buckets = Math.min(8192, Math.max(200, Math.round((main.clientWidth || 800) * 1.5 * ((end - start) / len))));
    detailAbort?.abort();
    const ctl = new AbortController();
    detailAbort = ctl;
    try {
      const url = `${peaksUrl}?start=${start.toFixed(3)}&end=${end.toFixed(3)}&buckets=${buckets}`;
      const res = await fetch(url, { signal: ctl.signal });
      if (!res.ok) return;
      const body = await res.json();
      if (ctl.signal.aborted) return;
      detail = body;
      draw();
    } catch {
      // aborted by a newer view, or the peaks call failed: the overview stays on screen
    }
  }

  // ---- drawing --------------------------------------------------------------

  function sizeCanvas(c, w, h) {
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function hatch() {
    const p = document.createElement('canvas');
    p.width = 8;
    p.height = 8;
    const c = p.getContext('2d');
    c.strokeStyle = 'rgba(255,59,92,0.85)';
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(0, 8);
    c.lineTo(8, 0);
    c.moveTo(-2, 2);
    c.lineTo(2, -2);
    c.moveTo(6, 10);
    c.lineTo(10, 6);
    c.stroke();
    return p;
  }
  const hatchTile = hatch();

  function drawColumns(ctx, w, y0, h, src) {
    const mid = y0 + h / 2;
    const k = (h / 2 - 3) / top;
    const len = view.end - view.start;
    const step = 2;
    for (let x = 0; x < w; x += step) {
      const [lo, hi] = peakRange(src, view.start + (x / w) * len, view.start + ((x + step) / w) * len);
      const y1 = mid - hi * k;
      ctx.fillRect(x, y1, 1.4, Math.max(1.5, (hi - lo) * k));
    }
  }

  function draw() {
    drawOverview();
    drawMain();
    placeOverlays();
  }

  function drawMain() {
    const w = main.clientWidth || 800;
    const h = mainHeight();
    main.style.height = `${h}px`;
    const ctx = sizeCanvas(canvas, w, h);
    const crop = cropDraft ?? data.crop;
    const takesH = hasTakes() ? TAKES_H : 0;
    const wy = RULER_H + takesH;
    const x = (t) => tToX(t, w);

    // ruler
    ctx.font = '12px "Share Tech Mono", monospace';
    ctx.fillStyle = '#5d7f8f';
    ctx.strokeStyle = '#16384a';
    const len = view.end - view.start;
    const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];
    const stepT = nice.find((s) => len / s <= 10) ?? 3600;
    for (let t = Math.ceil(view.start / stepT) * stepT; t < view.end; t += stepT) {
      const tx = Math.round(x(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(tx, RULER_H - 4);
      ctx.lineTo(tx, h);
      ctx.stroke();
      ctx.fillText(fmtTime(t), tx + 3, 12);
    }

    // speaker tint behind the waveform
    for (const lane of data.lanes) {
      ctx.fillStyle = `${speakerColor(lane.speaker)}${data.activeSpeaker === lane.speaker ? '30' : '14'}`;
      for (const s of lane.spans) {
        if (s.end < view.start || s.start > view.end) continue;
        ctx.fillRect(x(s.start), wy, Math.max(1, x(s.end) - x(s.start)), waveH());
      }
    }

    // waveform: dim everywhere, bright where kept
    const src = detailCovers() ? detail : overview;
    if (src) {
      ctx.fillStyle = 'rgba(93,127,143,0.55)';
      drawColumns(ctx, w, wy, waveH(), src);
      ctx.save();
      ctx.beginPath();
      for (const s of data.kept) {
        if (s.end < view.start || s.start > view.end) continue;
        ctx.rect(x(s.start), 0, Math.max(1, x(s.end) - x(s.start)), h);
      }
      ctx.clip();
      const grad = ctx.createLinearGradient(0, wy, 0, wy + waveH());
      grad.addColorStop(0, '#00e5ff');
      grad.addColorStop(1, '#b06bff');
      ctx.fillStyle = grad;
      drawColumns(ctx, w, wy, waveH(), src);
      ctx.restore();
    } else {
      ctx.fillStyle = '#5d7f8f';
      ctx.fillText('loading waveform…', 8, wy + waveH() / 2);
    }

    // outside the crop
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    if (crop.start > view.start) ctx.fillRect(0, wy, Math.min(w, x(crop.start)), waveH());
    if (crop.end < view.end) ctx.fillRect(Math.max(0, x(crop.end)), wy, w, waveH());

    // cuts: dimmed and struck through
    for (const c of data.cuts) {
      if (c.end < view.start || c.start > view.end) continue;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(x(c.start), wy, Math.max(1, x(c.end) - x(c.start)), waveH());
      ctx.fillStyle = 'rgba(255,59,92,0.9)';
      ctx.fillRect(x(c.start), wy + waveH() / 2 - 1, Math.max(1, x(c.end) - x(c.start)), 2);
    }

    // crosstalk: hatched red; a stronger wash and top bar when it is not being dropped
    const pattern = ctx.createPattern(hatchTile, 'repeat');
    for (const o of data.overlaps) {
      if (o.end < view.start || o.start > view.end) continue;
      const ox = x(o.start);
      const ow = Math.max(2, x(o.end) - ox);
      ctx.fillStyle = data.excludeOverlaps ? 'rgba(255,59,92,0.10)' : 'rgba(255,59,92,0.28)';
      ctx.fillRect(ox, wy, ow, waveH());
      ctx.globalAlpha = data.excludeOverlaps ? 0.45 : 1;
      ctx.fillStyle = pattern;
      ctx.fillRect(ox, wy, ow, waveH());
      ctx.globalAlpha = 1;
      if (!data.excludeOverlaps) {
        ctx.fillStyle = '#ff3b5c';
        ctx.fillRect(ox, wy, ow, 3);
      }
    }

    // best takes: outlined with their score
    if (hasTakes()) {
      for (const t of data.takes) {
        if (t.end < view.start || t.start > view.end) continue;
        const tx = x(t.start);
        const tw = Math.max(2, x(t.end) - tx);
        ctx.strokeStyle = t.picked ? '#2bff9c' : 'rgba(93,127,143,0.8)';
        ctx.lineWidth = t.picked ? 2 : 1;
        ctx.strokeRect(tx + 0.5, RULER_H + 1, tw, TAKES_H - 2 + waveH());
        ctx.lineWidth = 1;
        ctx.fillStyle = t.picked ? '#2bff9c' : '#5d7f8f';
        if (tw > 26) ctx.fillText(t.score.toFixed(2), tx + 4, RULER_H + 13);
      }
    }

    // speaker lanes under the waveform
    data.lanes.forEach((lane, i) => {
      const ly = wy + waveH() + i * LANE_H;
      ctx.fillStyle = speakerColor(lane.speaker);
      ctx.globalAlpha = data.activeSpeaker == null || data.activeSpeaker === lane.speaker ? 1 : 0.35;
      for (const s of lane.spans) {
        if (s.end < view.start || s.start > view.end) continue;
        ctx.fillRect(x(s.start), ly + 1, Math.max(1, x(s.end) - x(s.start)), LANE_H - 2);
      }
      ctx.globalAlpha = 1;
    });
  }

  function drawOverview() {
    const w = overviewBox.clientWidth || 800;
    const h = OVERVIEW_H;
    const ctx = sizeCanvas(ovCanvas, w, h);
    const x = (t) => (t / duration) * w;
    if (overview) {
      const mid = h / 2;
      const k = (h / 2 - 2) / top;
      ctx.fillStyle = 'rgba(0,229,255,0.7)';
      for (let px = 0; px < w; px += 2) {
        const [lo, hi] = peakRange(overview, (px / w) * duration, ((px + 2) / w) * duration);
        ctx.fillRect(px, mid - hi * k, 1.4, Math.max(1, (hi - lo) * k));
      }
    }
    for (const lane of data.lanes) {
      ctx.fillStyle = speakerColor(lane.speaker);
      for (const s of lane.spans) ctx.fillRect(x(s.start), h - 4, Math.max(1, x(s.end) - x(s.start)), 3);
    }
    ctx.fillStyle = 'rgba(255,59,92,0.9)';
    for (const o of data.overlaps) ctx.fillRect(x(o.start), 0, Math.max(1, x(o.end) - x(o.start)), 3);
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    const crop = cropDraft ?? data.crop;
    ctx.fillRect(0, 0, x(crop.start), h);
    ctx.fillRect(x(crop.end), 0, w - x(crop.end), h);
    for (const c of data.cuts) ctx.fillRect(x(c.start), 0, Math.max(1, x(c.end) - x(c.start)), h);
  }

  function placeOverlays() {
    const w = main.clientWidth || 800;
    const h = mainHeight();
    const crop = cropDraft ?? data.crop;
    const place = (node, t) => {
      const px = tToX(t, w);
      node.style.display = px < -2 || px > w + 2 ? 'none' : '';
      node.style.left = `${px}px`;
    };
    startHandle.style.height = `${h}px`;
    endHandle.style.height = `${h}px`;
    place(startHandle, crop.start);
    place(endHandle, crop.end);
    place(playheadEl, playhead);
    if (selection) {
      const x0 = Math.max(0, tToX(selection.start, w));
      const x1 = Math.min(w, tToX(selection.end, w));
      selBox.classList.toggle('hidden', x1 <= x0);
      selBox.style.left = `${x0}px`;
      selBox.style.width = `${Math.max(1, x1 - x0)}px`;
    } else {
      selBox.classList.add('hidden');
    }
    const ow = overviewBox.clientWidth || 800;
    ovWindow.style.left = `${(view.start / duration) * ow}px`;
    ovWindow.style.width = `${Math.max(6, ((view.end - view.start) / duration) * ow)}px`;
    status.textContent = `${fmtTime(view.start)} – ${fmtTime(view.end)}  ·  ${(view.end - view.start).toFixed(1)} s of ${fmtTime(duration)} in view${
      !detailCovers() && view.end - view.start < duration * 0.9 ? '  ·  loading detail…' : ''
    }`;
  }

  // ---- view -----------------------------------------------------------------

  function setView(start, end) {
    const minLen = Math.min(MIN_VIEW_S, duration);
    let len = Math.min(duration, Math.max(minLen, end - start));
    let s = Math.min(duration - len, Math.max(0, start));
    view = { start: s, end: s + len };
    draw();
    scheduleDetail();
  }

  function zoomBy(factor, centerT = (view.start + view.end) / 2) {
    const len = view.end - view.start;
    const next = len / factor;
    const frac = (centerT - view.start) / len;
    setView(centerT - frac * next, centerT - frac * next + next);
  }

  function panBy(frac) {
    const len = view.end - view.start;
    setView(view.start + len * frac, view.end + len * frac);
  }

  function showRange(start, end) {
    const pad = Math.max(0.5, (end - start) * 0.1);
    setView(start - pad, end + pad);
  }

  // ---- pointer input ----------------------------------------------------------

  main.addEventListener('pointerdown', (e) => {
    if (e.target === startHandle || e.target === endHandle) return;
    e.preventDefault();
    main.setPointerCapture(e.pointerId);
    const rect = main.getBoundingClientRect();
    const x0 = e.clientX - rect.left;
    const t0 = xToT(x0, rect.width);
    const pan = e.shiftKey || e.altKey;
    const viewAtStart = { ...view };
    let moved = false;
    const move = (ev) => {
      const x = ev.clientX - rect.left;
      if (Math.abs(x - x0) > DRAG_PX) moved = true;
      if (!moved) return;
      if (pan) {
        const dt = ((x0 - x) / rect.width) * (viewAtStart.end - viewAtStart.start);
        setView(viewAtStart.start + dt, viewAtStart.end + dt);
      } else {
        const t1 = clampT(xToT(x, rect.width));
        selection = { start: Math.min(t0, t1), end: Math.max(t0, t1) };
        placeOverlays();
      }
    };
    const up = () => {
      main.removeEventListener('pointermove', move);
      main.removeEventListener('pointerup', up);
      main.removeEventListener('pointercancel', up);
      if (pan) return;
      if (moved && selection) onSelect(selection);
      else {
        selection = null;
        placeOverlays();
        onSelect(null);
        onSeek(clampT(t0));
      }
    };
    main.addEventListener('pointermove', move);
    main.addEventListener('pointerup', up);
    main.addEventListener('pointercancel', up);
  });

  main.addEventListener(
    'wheel',
    (e) => {
      const rect = main.getBoundingClientRect();
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        zoomBy(Math.exp(-e.deltaY * 0.01), xToT(e.clientX - rect.left, rect.width));
      } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault();
        panBy(e.deltaX / rect.width);
      }
    },
    { passive: false },
  );

  function dragHandle(handle, which) {
    handle.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      handle.setPointerCapture(e.pointerId);
      const rect = main.getBoundingClientRect();
      const move = (ev) => {
        const t = clampT(xToT(ev.clientX - rect.left, rect.width));
        const base = cropDraft ?? data.crop;
        cropDraft =
          which === 'start'
            ? { start: Math.min(t, base.end - MIN_CROP_S), end: base.end }
            : { start: base.start, end: Math.max(t, base.start + MIN_CROP_S) };
        draw();
      };
      const up = () => {
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', up);
        handle.removeEventListener('pointercancel', up);
        const crop = cropDraft;
        cropDraft = null;
        if (crop && (crop.start !== data.crop.start || crop.end !== data.crop.end)) onCrop(crop);
        else draw();
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', up);
      handle.addEventListener('pointercancel', up);
    });
  }
  dragHandle(startHandle, 'start');
  dragHandle(endHandle, 'end');

  overviewBox.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    overviewBox.setPointerCapture(e.pointerId);
    const rect = overviewBox.getBoundingClientRect();
    const len = view.end - view.start;
    const tAt = (ev) => ((ev.clientX - rect.left) / rect.width) * duration;
    // Grabbing the window keeps the grab point; clicking elsewhere centres it there.
    const t0 = tAt(e);
    const inside = t0 >= view.start && t0 <= view.end;
    const grab = inside ? t0 - view.start : len / 2;
    const place = (ev) => setView(tAt(ev) - grab, tAt(ev) - grab + len);
    if (!inside) place(e);
    const move = (ev) => place(ev);
    const up = () => {
      overviewBox.removeEventListener('pointermove', move);
      overviewBox.removeEventListener('pointerup', up);
      overviewBox.removeEventListener('pointercancel', up);
    };
    overviewBox.addEventListener('pointermove', move);
    overviewBox.addEventListener('pointerup', up);
    overviewBox.addEventListener('pointercancel', up);
  });

  const onResize = () => draw();
  window.addEventListener('resize', onResize);

  fetchOverview();
  draw();

  return {
    root,
    redraw: draw,
    setData(next) {
      data = { ...emptyData(), ...next };
      draw();
    },
    getView: () => ({ ...view }),
    setView,
    zoomBy,
    panBy,
    showRange,
    fitAll: () => setView(0, duration),
    getSelection: () => selection,
    setSelection(sel) {
      selection = sel;
      placeOverlays();
    },
    /** `follow`: scroll the view when the playhead leaves it (during playback). */
    setPlayhead(t, follow = false) {
      playhead = t;
      if (follow && (t > view.end || t < view.start)) setView(t - (view.end - view.start) * 0.1, t + (view.end - view.start) * 0.9);
      else placeOverlays();
    },
    destroy() {
      window.removeEventListener('resize', onResize);
      detailAbort?.abort();
      clearTimeout(detailTimer);
    },
  };
}
