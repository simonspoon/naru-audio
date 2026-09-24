"""The sidecar's framed protocol, stdlib only (docs/design.md §5.3).

Every message, in both directions, is one frame:

    4 bytes      N, a u32 little-endian: the header's length
    N bytes      the header, a UTF-8 JSON object
    4 * S bytes  S f32 little-endian samples, when the header has "samples": S

N is at most MAX_HEADER and S at most MAX_SAMPLES; a frame over either is
an error, and the connection is closed.

Requests, one at a time, each answered before the next is sent:

    {"op": "load", "model": NAME, "dir": PATH}
    {"op": "unload", "model": NAME}
    {"op": "transcribe", "model": NAME, "samples": S} + S samples at 16 kHz
    {"op": "stats"}

Answers are {"ok": true, ...} or {"ok": false, "error": MESSAGE}. A
`transcribe` answer adds "segments": [{"start", "end", "text"}] (seconds
into the samples sent); a `stats` answer adds "active_bytes".

The sidecar serves one connection, the daemon's, and exits when it closes
or when stdin (a pipe from the daemon) reaches EOF, so it never outlives
the daemon.
"""

import json
import os
import socket
import struct
import sys
import threading

MAX_HEADER = 1 << 20
MAX_SAMPLES = 16_000 * 60 * 60


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("the connection closed mid-frame")
    return data


def read_frame(f):
    """(header, samples as bytes or None), or None at a clean EOF."""
    prefix = f.read(4)
    if not prefix:
        return None
    if len(prefix) != 4:
        raise EOFError("the connection closed mid-frame")
    (n,) = struct.unpack("<I", prefix)
    if n > MAX_HEADER:
        raise ValueError(f"header of {n} bytes is over {MAX_HEADER}")
    header = json.loads(read_exact(f, n))
    samples = header.get("samples")
    if samples is None:
        return header, None
    if not isinstance(samples, int) or not 0 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples {samples!r} is not a count up to {MAX_SAMPLES}")
    return header, read_exact(f, 4 * samples)


def write_frame(f, header):
    body = json.dumps(header).encode()
    f.write(struct.pack("<I", len(body)) + body)
    f.flush()


def _exit_with_daemon():
    sys.stdin.buffer.read()
    os._exit(0)


def serve(socket_path, handle):
    """Serves the daemon's connection; `handle(header, samples)` answers a
    request with a dict, or raises for an error answer."""
    threading.Thread(target=_exit_with_daemon, daemon=True).start()
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(1)
    conn, _ = listener.accept()
    listener.close()
    os.unlink(socket_path)
    f = conn.makefile("rwb")
    while True:
        frame = read_frame(f)
        if frame is None:
            break
        header, samples = frame
        try:
            answer = dict(handle(header, samples), ok=True)
        except Exception as e:  # the answer carries it; the sidecar stays up
            answer = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        write_frame(f, answer)
    os._exit(0)
