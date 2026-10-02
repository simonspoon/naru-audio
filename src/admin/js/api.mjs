// Fetch wrapper for naru-audio's own API. Same-origin only (§2.1) — the
// admin UI is served by this daemon, so relative paths reach it directly.
// Every non-2xx response is the §2.6 `{"error":{"message","type","code",
// "param"}}` envelope; on failure we toast `error.message` and reject with
// an Error carrying `.code` and `.status` so callers can branch on it.

import { toast } from './ui.mjs';

/**
 * @param {string} path e.g. "/v1/models"
 * @param {RequestInit} [init]
 * @param {{silent?: boolean}} [opts] silent: true skips the error toast
 *   (the caller handles the failure itself, e.g. a 404 that means "no
 *   preview cached yet").
 */
export async function request(path, init = {}, opts = {}) {
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    const err = new Error(`network error: ${e.message}`);
    err.code = 'network_error';
    if (!opts.silent) toast(err.message, true);
    throw err;
  }
  if (res.ok) return res;
  let body = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error body; fall through with a generic message
  }
  const message = body?.error?.message ?? `${res.status} ${res.statusText}`;
  const err = new Error(message);
  err.code = body?.error?.code ?? 'unknown_error';
  err.status = res.status;
  if (!opts.silent) toast(message, true);
  throw err;
}

export async function getJson(path, opts) {
  return (await request(path, {}, opts)).json();
}

export async function putJson(path, body, opts) {
  return (
    await request(
      path,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      opts,
    )
  ).json();
}

export async function postJson(path, body, opts) {
  return (
    await request(
      path,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      opts,
    )
  ).json();
}

/** Uploads a voice file (any length) as a prep clip: `POST /v1/audio/prep/clips`, multipart. Resolves with the clip JSON. */
export async function uploadClip(file) {
  const form = new FormData();
  form.append('file', file, file.name || 'clip');
  return (await request('/v1/audio/prep/clips', { method: 'POST', body: form })).json();
}

export async function del(path, opts) {
  return request(path, { method: 'DELETE' }, opts);
}

/**
 * Streams an NDJSON response (used by `POST /api/pull`), calling `onLine`
 * with each parsed JSON object as it arrives.
 */
export async function streamNdjson(path, body, onLine, opts) {
  const res = await request(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    opts,
  );
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onLine(JSON.parse(line));
    }
  }
  const rest = buf.trim();
  if (rest) onLine(JSON.parse(rest));
}
