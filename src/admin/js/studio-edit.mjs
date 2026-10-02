// Sample Studio's edit model for a raw clip (naru task 1576): pure data, no
// DOM. Intervals are `{start, end}` in seconds of the clip's working audio.
//
// The model is what the user edits: a `crop`, a list of `cuts`, the chosen
// `speaker` (or null for everyone), whether crosstalk is dropped
// (`excludeOverlaps`), the last best-takes result (`takes`) and whether it is
// applied (`useTakes`). `keptSegments()` derives from it, plus the clip's
// diarization, the exact list the render / sample / project requests send.
// `History` is the undo/redo stack over snapshots of the model.

export const MAX_HISTORY = 100;
/** Pieces shorter than this are dropped from the kept list (no audible content, and the server caps a request at 2048 segments). */
const MIN_PIECE_S = 0.1;

/** Colours for speaker lanes: distinct, readable on the dark theme. */
export const SPEAKER_COLORS = ['#00e5ff', '#ffb444', '#b06bff', '#2bff9c', '#ff2bd6', '#6aa8ff'];

export function speakerColor(id) {
  const n = SPEAKER_COLORS.length;
  return SPEAKER_COLORS[((id % n) + n) % n];
}

/** Sorted, with overlapping/touching intervals merged. */
export function mergeIntervals(list) {
  const sorted = list
    .filter((s) => s.end > s.start)
    .map((s) => ({ start: s.start, end: s.end }))
    .sort((a, b) => a.start - b.start);
  const out = [];
  for (const s of sorted) {
    const last = out[out.length - 1];
    if (last && s.start <= last.end) last.end = Math.max(last.end, s.end);
    else out.push(s);
  }
  return out;
}

/** `a` minus `b` (both merged and sorted). */
export function subtractIntervals(a, b) {
  const out = [];
  for (const s of a) {
    let from = s.start;
    for (const c of b) {
      if (c.end <= from) continue;
      if (c.start >= s.end) break;
      if (c.start > from) out.push({ start: from, end: c.start });
      from = Math.max(from, c.end);
      if (from >= s.end) break;
    }
    if (from < s.end) out.push({ start: from, end: s.end });
  }
  return out;
}

/** `a` ∩ `b` (both merged and sorted). */
export function intersectIntervals(a, b) {
  const out = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const start = Math.max(a[i].start, b[j].start);
    const end = Math.min(a[i].end, b[j].end);
    if (end > start) out.push({ start, end });
    if (a[i].end < b[j].end) i++;
    else j++;
  }
  return out;
}

export function totalSecs(list) {
  return list.reduce((sum, s) => sum + (s.end - s.start), 0);
}

/** A fresh model over a clip of `duration` seconds. */
export function newModel(duration) {
  return {
    crop: { start: 0, end: duration },
    cuts: [],
    speaker: null,
    excludeOverlaps: true,
    takes: null, // {takes: [{start, end, score, picked, metrics}], picked_secs}
    useTakes: false,
  };
}

/** A diarized speaker's spans, merged: `transcript.speakers` filtered by id. */
export function speakerSpans(transcript, speaker) {
  if (!transcript) return [];
  return mergeIntervals(transcript.speakers.filter((s) => s.speaker === speaker));
}

/** The regions of the clip the model keeps, spliced in this order. */
export function keptSegments(model, transcript) {
  let kept = [{ start: model.crop.start, end: model.crop.end }];
  kept = subtractIntervals(kept, mergeIntervals(model.cuts));
  if (model.speaker != null && transcript) kept = intersectIntervals(kept, speakerSpans(transcript, model.speaker));
  if (model.excludeOverlaps && transcript?.overlaps?.length) {
    kept = subtractIntervals(kept, mergeIntervals(transcript.overlaps));
  }
  if (model.useTakes && model.takes) {
    kept = intersectIntervals(kept, mergeIntervals(model.takes.takes.filter((t) => t.picked)));
  }
  return kept.filter((s) => s.end - s.start >= MIN_PIECE_S);
}

/** Per-speaker `{speaker, secs, spans}` from the diarization, by id. */
export function speakerStats(transcript) {
  if (!transcript) return [];
  const ids = [...new Set(transcript.speakers.map((s) => s.speaker))].sort((a, b) => a - b);
  return ids.map((speaker) => {
    const spans = speakerSpans(transcript, speaker);
    return { speaker, secs: totalSecs(spans), spans };
  });
}

/** Undo/redo over JSON-able snapshots of the model. */
export class History {
  constructor() {
    this.past = [];
    this.future = [];
  }

  /** Call with the model's state *before* a change. */
  push(snapshot) {
    this.past.push(JSON.stringify(snapshot));
    if (this.past.length > MAX_HISTORY) this.past.shift();
    this.future = [];
  }

  canUndo() {
    return this.past.length > 0;
  }

  canRedo() {
    return this.future.length > 0;
  }

  /** The state to restore, given the current one; null when there is nothing to undo. */
  undo(current) {
    if (!this.past.length) return null;
    this.future.push(JSON.stringify(current));
    return JSON.parse(this.past.pop());
  }

  redo(current) {
    if (!this.future.length) return null;
    this.past.push(JSON.stringify(current));
    return JSON.parse(this.future.pop());
  }
}
